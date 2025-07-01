# streamlit_app.py
import os, datetime as dt
import requests, requests_cache
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

"""
🔋 Stromspeicher-Simulator v4.2 – Contractor, Inflation, THG-Quote
-----------------------------------------------------------------
• Live-Day-Ahead- & CO₂-Daten (ENTSO-E API + Retry & 24-h-Cache)
• Fester EV-Ladepreis  +  jährliche Inflation (EV-Preis % / Opex %)
• THG-Quote (Bonus €/kWh) als zusätzliche Einnahme
• Gewinnaufteilung Contractor / Standort
• KPIs: NPV @ Diskont, IRR, Break-Even, Zyklen
"""

# ---------------------------------------------------------------------------
# 🔗  ENT­SO-E  API  –  Cache + Retry
# ---------------------------------------------------------------------------
CACHE_DB = "/tmp/entsoe_cache.sqlite"
requests_cache.install_cache(CACHE_DB, expire_after=60 * 60 * 24)      # 24 h
session = requests_cache.CachedSession()
session.mount(
    "https://",
    requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(total=3, backoff_factor=1)),
)

ENTSOE_URL   = "https://transparency.entsoe.eu/api"
BIDDING_ZONE = "10Y1001A1001A83"          # DE-LU

def _get_xml(params: dict) -> str:
    r = session.get(ENTSOE_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.text

def fetch_day_ahead(start: dt.datetime, end: dt.datetime, api_key: str) -> pd.DataFrame:
    params = {
        "documentType": "A44", "processType": "A01",
        "in_Domain": BIDDING_ZONE, "out_Domain": BIDDING_ZONE,
        "periodStart": start.strftime("%Y%m%d%H%M"),
        "periodEnd"  : end.strftime("%Y%m%d%H%M"),
        "securityToken": api_key,
    }
    xml = _get_xml(params)
    df  = pd.read_xml(xml, xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int) - 1, unit="h", origin=start)
    df.rename(columns={"price.amount": "Preis_EUR_MWh"}, inplace=True)
    return df[["Zeit", "Preis_EUR_MWh"]]

def fetch_co2_intensity(start: dt.datetime, end: dt.datetime, api_key: str) -> pd.DataFrame:
    params = {
        "documentType": "A75", "processType": "A16",
        "in_Domain": BIDDING_ZONE,
        "periodStart": start.strftime("%Y%m%d%H%M"),
        "periodEnd"  : end.strftime("%Y%m%d%H%M"),
        "securityToken": api_key,
    }
    xml = _get_xml(params)
    df  = pd.read_xml(xml, xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int) - 1, unit="h", origin=start)
    df.rename(columns={"quantity": "CO2_g_per_kWh"}, inplace=True)
    return df[["Zeit", "CO2_g_per_kWh"]]

# ---------------------------------------------------------------------------
# 🔋  Simulation  (Inflation + THG Quote)
# ---------------------------------------------------------------------------
def simulate(prices: pd.DataFrame, co2: pd.DataFrame, cfg: dict):
    prices = prices.copy()
    prices["Datum"] = prices["Zeit"].dt.date

    # Verkaufsfenster-Flag
    def in_sell_window(ts: pd.Timestamp) -> bool:
        if cfg["sell_start"] < cfg["sell_end"]:
            return cfg["sell_start"] <= ts.time() <= cfg["sell_end"]
        return ts.time() >= cfg["sell_start"] or ts.time() <= cfg["sell_end"]

    prices["Sell"] = prices["Zeit"].apply(in_sell_window)
    if not co2.empty:
        prices = prices.merge(co2, on="Zeit", how="left")
    else:
        prices["CO2_g_per_kWh"] = np.nan

    need_hours = int(np.ceil(cfg["cap"] / cfg["conn"]))
    daily, cycles = [], 0

    # -------- Jahr 0 (Basisjahr) Tagesgewinne --------
    for d, grp in prices.groupby("Datum"):
        load_hours = grp.nsmallest(need_hours, "Preis_EUR_MWh")
        sell_hours = grp[grp["Sell"]]
        if sell_hours.empty:
            continue

        buy_p   = load_hours["Preis_EUR_MWh"].mean()
        sell_pMWh = (cfg["ev_price"] + cfg["thg"]) * 1_000     # €/MWh

        e_charge = cfg["cap"] * (1 + cfg["closs"])
        e_sell   = cfg["cap"] * (1 - cfg["dloss"])

        revenue  = e_sell * sell_pMWh
        energy_c = e_charge * buy_p

        peak_ratio = sell_hours["Zeit"].dt.hour.between(8, 20).mean()
        net_cost   = e_charge * (peak_ratio * cfg["net_peak"] +
                                 (1 - peak_ratio) * cfg["net_off"])

        profit = revenue - energy_c - net_cost
        cycles += 1
        daily.append({"Datum": pd.to_datetime(d), "Gewinn": profit})

    df = pd.DataFrame(daily)

    # -------- Kosten & Cashflow Jahr 0 --------
    deg_cost = cycles * (cfg["capex"] / cfg["max_cycles"])
    opex_y0  = cfg["opex"]

    gross_y0 = df["Gewinn"].sum()
    net_y0   = gross_y0 - deg_cost - opex_y0

    contractor_y0 = net_y0 * (1 - cfg["share"])
    cashflows = [-cfg["capex"], contractor_y0]

    # -------- Jahre 1 … n  (mit Inflation) --------
    for year in range(1, cfg["years"] + 1):
        infl_factor_ev  = (1 + cfg["infl_ev"]) ** year
        infl_factor_opx = (1 + cfg["infl_op"]) ** year

        revenue_y = gross_y0 * infl_factor_ev
        opex_y    = cfg["opex"] * infl_factor_opx

        net_y = revenue_y - deg_cost - opex_y
        contractor_y = net_y * (1 - cfg["share"])
        cashflows.append(contractor_y)

    # -------- KPIs --------
    disc = cfg["disc"]
    npv = sum(cf / (1 + disc) ** t for t, cf in enumerate(cashflows))
    try:
        irr = np.irr(cashflows)
    except Exception:
        irr = np.nan
    breakeven = next((t for t, c in enumerate(np.cumsum(cashflows)) if c >= 0), None)

    summary = dict(NPV=npv, IRR=irr, BE=breakeven, cycles=cycles)
    return df, summary

# ---------------------------------------------------------------------------
# 🚀  Streamlit Frontend
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Contractor-Simulator v4.2", layout="wide")
    st.title("🔋 Contractor-Ladepark – Inflation, THG-Quote, Retry-Cache")

    # -------- Sidebar: Basis --------
    api_key = st.sidebar.text_input("ENTSO-E API-Key",
                                    os.getenv("ENTSOE_API_KEY", ""),
                                    type="password")
    year = st.sidebar.selectbox("Startjahr", [2023, 2024, 2025], 0)

    # -------- Batterie --------
    st.sidebar.header("🔋 Batterie")
    cap  = st.sidebar.number_input("Speicher (MWh)", 0.5, 20.0, 3.5, 0.1)
    conn = st.sidebar.number_input("Netzanschluss (MW)", 0.1, 5.0, 0.35, 0.05)
    closs = st.sidebar.slider("Ladeverlust %", 0, 20, 5) / 100
    dloss = st.sidebar.slider("Entladeverlust %", 0, 20, 5) / 100

    # -------- EV-Preis + Inflation --------
    st.sidebar.header("🚗 EV-Preis & Inflation")
    ev_price = st.sidebar.number_input("EV-Preis €/kWh", 0.10, 1.00, 0.285, 0.005)
    infl_ev  = st.sidebar.slider("Inflation EV-Preis %/a", 0.0, 10.0, 2.0) / 100

    # -------- Opex + Inflation --------
    st.sidebar.header("⚙️ Opex & Inflation")
    opex   = st.sidebar.number_input("Opex €/Jahr", 0, 50_000, 5_000, 500)
    infl_op = st.sidebar.slider("Inflation Opex %/a", 0.0, 10.0, 2.0) / 100

    # -------- THG-Quote --------
    st.sidebar.header("🌱 THG-Quote")
    thg_bonus = st.sidebar.number_input("THG-Bonus €/kWh", 0.0, 0.5, 0.20, 0.01)

    # -------- CAPEX & Laufzeit --------
    st.sidebar.header("💰 CAPEX & Laufzeit")
    capex      = st.sidebar.number_input("CAPEX €", 10_000, 2_000_000, 350_000, 5_000)
    max_cycles = st.sidebar.number_input("Max. Zyklen", 1_000, 15_000, 6_000, 100)
    years      = st.sidebar.number_input("Vertragsjahre", 5, 20, 10, 1)
    discount   = st.sidebar.number_input("Diskont % p.a.", 0.0, 15.0, 6.0) / 100
    share      = st.sidebar.slider("Anteil Standort %", 0, 50, 20) / 100

    # -------- Netzentgelt --------
    st.sidebar.header("🔌 Netzentgelt")
    net_peak = st.sidebar.number_input("Peak €/MWh", 0, 200, 75, 5)
    net_off  = st.sidebar.number_input("Off-Peak €/MWh", 0, 200, 40, 5)

    # -------- Verkauf --------
    st.sidebar.header("🕑 Verkaufsfenster")
    sell_start = st.sidebar.time_input("Start", dt.time(0, 0))
    sell_end   = st.sidebar.time_input("Ende",  dt.time(23, 59))

    # -------- Simulation --------
    if st.button("Simulation starten"):
        start = dt.datetime(year, 1, 1)
        end   = dt.datetime(year, 12, 31, 23)

        # Preise
        try:
            with st.spinner("Day-Ahead-Preise laden …"):
                prices = fetch_day_ahead(start, end, api_key)
        except Exception as e:
            st.error(f"🚫 API-Fehler: {e}")
            st.stop()

        # CO₂ (optional)
        try:
            co2 = fetch_co2_intensity(start, end, api_key)
        except Exception:
            co2 = pd.DataFrame()

        cfg = dict(
            cap=cap, conn=conn, closs=closs, dloss=dloss,
            ev_price=ev_price, infl_ev=infl_ev,
            opex=opex, infl_op=infl_op,
            thg=thg_bonus,
            capex=capex, max_cycles=max_cycles,
            years=years, disc=discount, share=share,
            net_peak=net_peak, net_off=net_off,
            sell_start=sell_start, sell_end=sell_end
        )

        df, summ = simulate(prices, co2, cfg)
        st.success("✅ Simulation fertig")

        k1, k2, k3 = st.columns(3)
        k1.metric("NPV €", f"{summ['NPV']:,.0f}")
        k2.metric("IRR %", f"{summ['IRR']*100:,.2f}" if not np.isnan(summ["IRR"]) else "n/a")
        k3.metric("Break-Even Jahr", summ["BE"] if summ["BE"] is not None else "> Laufzeit")

        st.subheader("Tagesgewinne (Jahr 0)")
        st.dataframe(df)

        st.download_button("CSV herunterladen",
                           df.to_csv(index=False).encode(),
                           file_name="contractor_detail.csv")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
