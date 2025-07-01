# streamlit_app.py
# ===============================================================
# 🔋 Contractor-Ladepark-Simulator  –  Spot-Arbitrage | Kreditplan
# Version 4.3  |  enthält
#   • Live-Day-Ahead-Preise (ENTSO-E)  + Retry + 24 h-Cache
#   • Fester EV-Preis  +  jährliche Inflation EV-Preis / Opex
#   • THG-Bonus (€/kWh)  als Zusatzumsatz
#   • Wahl: 100 % Eigenkapital  ODER  Kredit-Finanzierung
#        - Kreditanteil %, Zins %, Laufzeit J
#        - Vollständiger Amortisationsplan (Zins / Tilgung / Restschuld)
#   • KPIs:  NPV @ Diskont,  IRR,  Break-Even-Jahr,  Zyklen
# ===============================================================

import os, datetime as dt
import numpy as np, pandas as pd
import requests, requests_cache
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1)  ENT-SO-E API  — Cache & Retry
# ------------------------------------------------------------------
CACHE_DB = "/tmp/entsoe_cache.sqlite"
requests_cache.install_cache(CACHE_DB, expire_after=60 * 60 * 24)   # 24 h

from requests.adapters import HTTPAdapter, Retry
session = requests_cache.CachedSession()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))

ENTSOE_URL   = "https://transparency.entsoe.eu/api"
BIDDING_ZONE = "10Y1001A1001A83"           # DE-LU

def _xml(params: dict) -> str:
    r = session.get(ENTSOE_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.text

def fetch_day_ahead(start: dt.datetime, end: dt.datetime, token: str) -> pd.DataFrame:
    p = dict(documentType="A44", processType="A01",
             in_Domain=BIDDING_ZONE, out_Domain=BIDDING_ZONE,
             periodStart=start.strftime("%Y%m%d%H%M"),
             periodEnd=end.strftime("%Y%m%d%H%M"),
             securityToken=token)
    df = pd.read_xml(_xml(p), xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int) - 1, unit="h", origin=start)
    return df[["Zeit"]].assign(Preis_EUR_MWh=df["price.amount"])

# ------------------------------------------------------------------
# 2)  Darlehens-Amortisationsplan
# ------------------------------------------------------------------
def amort_schedule(principal: float, rate: float, years: int) -> pd.DataFrame:
    """gibt DataFrame mit Jahr, Zins, Tilgung, Rate, Restschuld"""
    if principal <= 0 or years <= 0:
        return pd.DataFrame()
    ann = principal * (rate * (1 + rate) ** years) / ((1 + rate) ** years - 1)
    rows, bal = [], principal
    for y in range(1, years + 1):
        interest = bal * rate
        principal_pay = ann - interest
        bal = max(bal - principal_pay, 0)
        rows.append(dict(Jahr=y, Zins=interest, Tilgung=principal_pay,
                         Rate=ann, Restschuld=bal))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# 3)  Haupt-Simulation   (Inflation + THG + Finanzierung)
# ------------------------------------------------------------------
def simulate(prices: pd.DataFrame, cfg: dict):
    prices = prices.copy()
    prices["Datum"] = prices["Zeit"].dt.date

    # Flag „Sell“ nach Verkaufsfenster
    def in_window(t: dt.time) -> bool:
        if cfg["sell_start"] < cfg["sell_end"]:
            return cfg["sell_start"] <= t <= cfg["sell_end"]
        return t >= cfg["sell_start"] or t <= cfg["sell_end"]

    prices["Sell"] = prices["Zeit"].dt.time.apply(in_window)

    need_h = int(np.ceil(cfg["cap_MWh"] / cfg["conn_MW"]))
    cycles, daily = 0, []

    for d, g in prices.groupby("Datum"):
        load = g.nsmallest(need_h, "Preis_EUR_MWh")
        sell = g[g["Sell"]]
        if sell.empty:
            continue
        buy_price = load["Preis_EUR_MWh"].mean()
        sell_price_MWh = (cfg["ev_price"] + cfg["thg_bonus"]) * 1_000

        e_in  = cfg["cap_MWh"] * (1 + cfg["loss_in"])
        e_out = cfg["cap_MWh"] * (1 - cfg["loss_out"])

        peak_ratio = sell["Zeit"].dt.hour.between(8, 20).mean()
        net_cost   = e_in * (peak_ratio * cfg["net_peak"] +
                             (1 - peak_ratio) * cfg["net_off"])

        profit = e_out * sell_price_MWh - e_in * buy_price - net_cost
        cycles += 1
        daily.append(dict(Datum=pd.to_datetime(d), Gewinn=profit))

    df = pd.DataFrame(daily)

    # ---------- Jahr 0 ----------
    deg_cost = cycles * (cfg["capex"] / cfg["max_cycles"])
    opex0    = cfg["opex"]
    gross0   = df["Gewinn"].sum()
    net0     = gross0 - deg_cost - opex0             # vor Gewinn­aufteilung

    # ---------- Finanzierung ----------
    if cfg["fin_mode"] == "Kredit":
        schedule = amort_schedule(cfg["loan_principal"],
                                  cfg["loan_rate"],
                                  cfg["loan_years"])
        equity_out = cfg["capex"] - cfg["loan_principal"]
    else:                                            # 100 % Eigenkapital
        schedule = pd.DataFrame()
        equity_out = cfg["capex"]

    cashflows = [-equity_out]                    # t = 0

    # ---------- Jahres-Cash-Flows ----------
    for y in range(1, cfg["years"] + 1):
        infl_ev   = (1 + cfg["infl_ev"])  ** y
        infl_opex = (1 + cfg["infl_op"]) ** y
        gross_y   = gross0 * infl_ev
        opex_y    = opex0 * infl_opex
        net_y     = gross_y - deg_cost - opex_y

        # Kreditrate?
        rate_y = schedule.loc[schedule["Jahr"] == y, "Rate"].iloc[0] if not schedule.empty and y <= cfg["loan_years"] else 0
        contractor_share = (net_y - rate_y) * (1 - cfg["share_site"])
        cashflows.append(contractor_share)

    # ---------- KPIs ----------
    disc = cfg["discount"]
    npv = sum(cf / (1 + disc) ** t for t, cf in enumerate(cashflows))
    try:
        irr = np.irr(cashflows)
    except Exception:
        irr = np.nan
    be_year = next((t for t, c in enumerate(np.cumsum(cashflows)) if c >= 0), None)

    summary = dict(NPV=npv, IRR=irr, BE=be_year, cycles=cycles, schedule=schedule)
    return df, summary

# ------------------------------------------------------------------
# 4)  Streamlit Frontend
# ------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Contractor v4.3", layout="wide")
    st.title("🔋 Contractor-Simulator – Kreditplan, Inflation, THG")

    # -------- API & Jahr --------
    api_key = st.sidebar.text_input("ENTSO-E API-Key", os.getenv("ENTSOE_API_KEY", ""), type="password")
    year    = st.sidebar.selectbox("Startjahr", [2023, 2024, 2025], 0)

    # -------- Batterie --------
    st.sidebar.header("🔋 Batterie")
    cap_MWh  = st.sidebar.number_input("Speicher (MWh)", 0.5, 20.0, 3.5, 0.1)
    conn_MW  = st.sidebar.number_input("Netzanschluss (MW)", 0.1, 5.0, 0.35, 0.05)
    loss_in  = st.sidebar.slider("Ladeverluste % ", 0, 20, 5) / 100
    loss_out = st.sidebar.slider("Entladeverluste %", 0, 20, 5) / 100

    # -------- Preise & Inflation --------
    st.sidebar.header("🚗 EV-Preis & THG")
    ev_price = st.sidebar.number_input("EV-Preis €/kWh", 0.10, 1.00, 0.285, 0.005)
    infl_ev  = st.sidebar.slider("Inflation EV-Preis %/a", 0.0, 10.0, 2.0) / 100
    thg_bonus= st.sidebar.number_input("THG-Bonus €/kWh", 0.0, 0.50, 0.20, 0.01)

    # -------- Opex & Inflation --------
    st.sidebar.header("⚙️ Betriebskosten")
    opex    = st.sidebar.number_input("Opex €/Jahr", 0, 50_000, 5_000, 500)
    infl_op = st.sidebar.slider("Inflation Opex %/a", 0.0, 10.0, 2.0) / 100

    # -------- CAPEX & Finanzierung --------
    st.sidebar.header("💰 CAPEX & Finanzierung")
    capex      = st.sidebar.number_input("CAPEX €", 10_000, 2_000_000, 350_000, 5_000)
    max_cycles = st.sidebar.number_input("Max. Zyklen", 1_000, 15_000, 6_000, 100)
    fin_mode   = st.sidebar.selectbox("Finanzierung", ["Eigenkapital", "Kredit"], 0)

    if fin_mode == "Kredit":
        loan_share   = st.sidebar.slider("Fremdkapitalanteil %", 10, 100, 70)
        loan_rate    = st.sidebar.number_input("Zins % p.a.", 0.0, 15.0, 4.0, 0.1) / 100
        loan_years   = st.sidebar.number_input("Laufzeit J", 1, 20, 10, 1)
        loan_principal = capex * loan_share / 100
    else:
        loan_principal = 0
        loan_rate      = 0
        loan_years     = 0

    # -------- Vertrag --------
    st.sidebar.header("📑 Vertrag & Diskont")
    years    = st.sidebar.number_input("Vertragsjahre", 5, 20, 10, 1)
    discount = st.sidebar.number_input("Diskont % p.a.", 0.0, 15.0, 6.0) / 100
    share_site = st.sidebar.slider("Gewinnanteil Standort %", 0, 50, 20) / 100

    # -------- Netz --------
    st.sidebar.header("🔌 Netzentgelt")
    net_peak = st.sidebar.number_input("Peak €/MWh", 0, 200, 75, 5)
    net_off  = st.sidebar.number_input("Off-Peak €/MWh", 0, 200, 40, 5)

    # -------- Verkauf --------
    st.sidebar.header("🕑 Verkaufsfenster")
    sell_start = st.sidebar.time_input("Start", dt.time(0, 0))
    sell_end   = st.sidebar.time_input("Ende",  dt.time(23, 59))

    # -------- Simulation drücken --------
    if st.button("Simulation starten"):
        start = dt.datetime(year, 1, 1)
        end   = dt.datetime(year, 12, 31, 23)

        # Preise holen
        try:
            with st.spinner("Day-Ahead-Preise laden …"):
                prices = fetch_day_ahead(start, end, api_key)
        except Exception as e:
            st.error(f"🚫 API-Fehler: {e}")
            st.stop()

        cfg = dict(
            cap_MWh=cap_MWh, conn_MW=conn_MW,
            loss_in=loss_in, loss_out=loss_out,
            ev_price=ev_price, infl_ev=infl_ev,
            opex=opex, infl_op=infl_op,
            thg_bonus=thg_bonus,
            capex=capex, max_cycles=max_cycles,
            fin_mode=fin_mode,
            loan_principal=loan_principal,
            loan_rate=loan_rate,
            loan_years=loan_years,
            years=years, discount=discount,
            share_site=share_site,
            net_peak=net_peak, net_off=net_off,
            sell_start=sell_start, sell_end=sell_end
        )

        df, summ = simulate(prices, cfg)

        st.success("✅ Simulation abgeschlossen")

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("NPV €", f"{summ['NPV']:,.0f}")
        k2.metric("IRR %", f"{summ['IRR']*100:,.2f}" if not np.isnan(summ["IRR"]) else "n/a")
        k3.metric("Break-Even Jahr", summ["BE"] if summ["BE"] is not None else "> Laufzeit")

        # Kreditplan anzeigen
        if not summ["schedule"].empty:
            st.subheader("📑 Darlehensplan")
            st.dataframe(
                summ["schedule"].style.format(
                    {"Zins": "{:,.0f}", "Tilgung": "{:,.0f}",
                     "Rate": "{:,.0f}", "Restschuld": "{:,.0f}"}
                )
            )

        st.subheader("Tagesgewinne Jahr 0")
        st.dataframe(df)

        st.download_button("CSV herunterladen",
                           df.to_csv(index=False).encode(),
                           file_name="contractor_detail.csv")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
