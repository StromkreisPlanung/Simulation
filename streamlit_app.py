# streamlit_app.py
import os
import requests
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt

"""
🔋 Stromspeicher-Simulator v4.0 – Contractor-Ladepark
====================================================
• Live-Day-Ahead-Preise & CO₂-Intensität (ENTSO-E-API)
• Fester EV-Ladepreis (€/kWh) statt Mark-up
• Finanzierungsmodelle: Kauf ▸ Kredit (Annuität) ▸ Leasing
• 10-Jahres-Cashflow → NPV @ Diskont, IRR, Break-Even-Jahr
• Heatmap täglicher Gewinne + Monats-Cashflow (Jahr 1)
• Robustes Fehler-Handling für API-Aufrufe
"""

# --------------------------------------------------------------------------
# 🔗 ENTSO-E API-Layer
# --------------------------------------------------------------------------
ENTSOE_ENDPOINT = "https://transparency.entsoe.eu/api"
BIDDING_ZONE    = "10Y1001A1001A83"  # Deutschland / Luxemburg

def fetch_day_ahead_prices(start: dt.datetime,
                           end: dt.datetime,
                           api_key: str,
                           domain: str = BIDDING_ZONE) -> pd.DataFrame:
    params = {
        "documentType": "A44",
        "processType" : "A01",
        "in_Domain"   : domain,
        "out_Domain"  : domain,
        "periodStart" : start.strftime("%Y%m%d%H%M"),
        "periodEnd"   : end.strftime("%Y%m%d%H%M"),
        "securityToken": api_key,
    }
    r = requests.get(ENTSOE_ENDPOINT, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_xml(r.text, xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int) - 1,
                                unit="h", origin=start)
    df.rename(columns={"price.amount": "Preis_EUR_MWh"}, inplace=True)
    return df[["Zeit", "Preis_EUR_MWh"]]


def fetch_co2_intensity(start: dt.datetime,
                        end: dt.datetime,
                        api_key: str,
                        domain: str = BIDDING_ZONE) -> pd.DataFrame:
    params = {
        "documentType": "A75",
        "processType" : "A16",
        "in_Domain"   : domain,
        "periodStart" : start.strftime("%Y%m%d%H%M"),
        "periodEnd"   : end.strftime("%Y%m%d%H%M"),
        "securityToken": api_key,
    }
    r = requests.get(ENTSOE_ENDPOINT, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_xml(r.text, xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int) - 1,
                                unit="h", origin=start)
    df.rename(columns={"quantity": "CO2_g_per_kWh"}, inplace=True)
    return df[["Zeit", "CO2_g_per_kWh"]]

# --------------------------------------------------------------------------
# 🔋 Simulation-Kernel – angepasst an EV-Preis
# --------------------------------------------------------------------------
def simulate(prices: pd.DataFrame,
             co2: pd.DataFrame,
             cfg: dict) -> tuple[pd.DataFrame, dict]:
    prices = prices.copy()
    prices["Datum"] = prices["Zeit"].dt.date

    def in_sell_window(ts: pd.Timestamp) -> bool:
        if cfg["sell_start"] < cfg["sell_end"]:
            return cfg["sell_start"] <= ts.time() <= cfg["sell_end"]
        return ts.time() >= cfg["sell_start"] or ts.time() <= cfg["sell_end"]

    prices["Sell"] = prices["Zeit"].apply(in_sell_window)

    if not co2.empty:
        prices = prices.merge(co2, on="Zeit", how="left")
    else:
        prices["CO2_g_per_kWh"] = np.nan

    results, cycles = [], 0
    need_hours = int(np.ceil(cfg["capacity"] / cfg["connection"]))

    for d, grp in prices.groupby("Datum"):
        load_hours  = grp.nsmallest(need_hours, "Preis_EUR_MWh")
        sell_window = grp[grp["Sell"]]
        if sell_window.empty:
            continue

        buy_price  = load_hours["Preis_EUR_MWh"].mean()
        sell_price = cfg["ev_price_kwh"] * 1000      # €/kWh  → €/MWh

        charged = cfg["capacity"] * (1 + cfg["charge_loss"])
        sold    = cfg["capacity"] * (1 - cfg["discharge_loss"])

        revenue     = sold * sell_price
        energy_cost = charged * buy_price

        peak_ratio  = sell_window["Zeit"].dt.hour.between(8, 20).mean()
        net_cost    = charged * (peak_ratio * cfg["net_peak"] +
                                 (1 - peak_ratio) * cfg["net_off"])

        daily_profit = revenue - energy_cost - net_cost

        # CO₂-Bilanz
        if (not load_hours["CO2_g_per_kWh"].isna().all() and
                not sell_window["CO2_g_per_kWh"].isna().all()):
            co2_load = (load_hours["CO2_g_per_kWh"].mean() * charged) / 1000
            co2_sell = (sell_window["CO2_g_per_kWh"].mean() * sold) / 1000
            co2_saved = max(co2_sell - co2_load, 0)
        else:
            co2_saved = np.nan

        cycles += 1
        results.append({
            "Datum"        : pd.to_datetime(d),
            "Gewinn_EUR"   : daily_profit,
            "CO2_saved_kg" : co2_saved,
        })

    df = pd.DataFrame(results)

    # ---- Kosten & Finanzierung Jahr 1 ----
    cyc_cost = cfg["capex"] / cfg["max_cycles"]
    deg_cost = cycles * cyc_cost

    if cfg["fin_model"] == "Kauf":
        fin_annual = 0
        init_cash  = -cfg["capex"]
    elif cfg["fin_model"] == "Kredit":
        r, n = cfg["loan_rate"], cfg["loan_years"]
        fin_annual = cfg["capex"] * (r * (1 + r)**n) / ((1 + r)**n - 1)
        init_cash  = 0
    else:  # Leasing
        fin_annual = cfg["lease_month"] * 12
        init_cash  = 0

    op_cost           = cfg["opex"]
    total_extra_year  = deg_cost + fin_annual + op_cost

    gross_profit = df["Gewinn_EUR"].sum()
    net_profit_y1 = gross_profit - total_extra_year

    # ---- Cashflow über Vertragslaufzeit ----
    years      = cfg["contract_years"]
    cashflows  = [init_cash] + [net_profit_y1] * years
    disc_rate  = cfg["discount"]
    npv        = sum(cf / (1 + disc_rate)**t for t, cf in enumerate(cashflows))
    try:
        irr = np.irr(cashflows)
    except Exception:
        irr = np.nan
    breakeven  = next((t for t, cum in enumerate(np.cumsum(cashflows))
                       if cum >= 0), None)

    summary = dict(
        cycles           = cycles,
        gross_profit     = gross_profit,
        net_profit_first = net_profit_y1,
        NPV              = npv,
        IRR              = irr,
        breakeven_year   = breakeven,
        co2_saved_total  = df["CO2_saved_kg"].sum()
    )
    return df, summary

# --------------------------------------------------------------------------
# 🚀 Streamlit-Frontend
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="🔋 Ladepark Contractor-Simulator", layout="wide")
    st.title("🔋 Ladepark Contractor-Modell – Spotmarkt & Finanzierung")

    # ----------- Sidebar: Basis -----------
    api_key = st.sidebar.text_input("ENTSO-E API-Key",
                                    os.getenv("ENTSOE_API_KEY", ""),
                                    type="password")
    year = st.sidebar.selectbox("Startjahr", [2023, 2024, 2025], 0)

    # ----------- Batterie -----------
    st.sidebar.header("🔋 Batterie")
    cap      = st.sidebar.number_input("Speichergröße (MWh)", 0.5, 20.0, 3.5, 0.1)
    conn     = st.sidebar.number_input("Netzanschluss (MW)",  0.1, 5.0, 0.35, 0.05)
    ch_loss  = st.sidebar.slider("Ladeverlust %",    0, 20, 5) / 100
    dis_loss = st.sidebar.slider("Entladeverlust %", 0, 20, 5) / 100

    # ----------- EV-Preis -----------
    st.sidebar.header("🚗 EV-Ladepreis")
    ev_price = st.sidebar.number_input("Preis an Fahrer (€/kWh)",
                                       0.1, 1.0, 0.285, 0.005)

    # ----------- Finanzierung -----------
    st.sidebar.header("💰 Finanzierung")
    fin_model  = st.sidebar.selectbox("Modell", ["Kauf", "Kredit", "Leasing"])
    capex      = st.sidebar.number_input("Capex Batterie €",
                                         10_000, 2_000_000, 350_000, 5_000)
    max_cycles = st.sidebar.number_input("Max. Zyklen", 1000, 15000, 6000, 100)
    opex       = st.sidebar.number_input("Opex/Jahr €", 0, 50_000, 5_000, 500)

    loan_rate  = st.sidebar.number_input("Kredit-Zins %", 0.0, 15.0, 4.0, 0.1)/100 \
                 if fin_model == "Kredit" else 0
    loan_years = st.sidebar.number_input("Kredit-Laufzeit Jahre", 1, 20, 10, 1) \
                 if fin_model == "Kredit" else 0
    lease_month= st.sidebar.number_input("Leasingrate €/Monat", 0, 20_000, 3_000, 100) \
                 if fin_model == "Leasing" else 0

    # ----------- Vertrag & Netzentgelt -----------
    st.sidebar.header("📄 Vertrag & Netzentgelt")
    contract_years = st.sidebar.number_input("Vertragslaufzeit Jahre", 5, 20, 10, 1)
    discount       = st.sidebar.number_input("Diskontsatz %", 0.0, 15.0, 6.0, 0.1)/100
    net_peak       = st.sidebar.number_input("Netzentgelt Peak €/MWh",     0, 200, 75, 5)
    net_off        = st.sidebar.number_input("Netzentgelt Off-Peak €/MWh", 0, 200, 40, 5)

    # ----------- Verkauf -----------
    st.sidebar.header("🕑 Verkauf (Ertrag wird über EV-Preis bestimmt)")
    sell_start = st.sidebar.time_input("Start", dt.time(0, 0))
    sell_end   = st.sidebar.time_input("Ende",  dt.time(23, 59))

    # ----------- Simulation starten -----------
    if st.button("Simulation starten"):
        start = dt.datetime(year, 1, 1)
        end   = dt.datetime(year, 12, 31, 23)

        # Preise-Abruf
        try:
            with st.spinner("Day-Ahead-Preise abrufen …"):
                prices = fetch_day_ahead_prices(start, end, api_key)
        except requests.HTTPError as e:
            st.error(f"❌ API-Fehler ({e.response.status_code} – {e.response.reason}).")
            st.stop()

        # CO₂-Abruf
        try:
            with st.spinner("CO₂-Intensität abrufen …"):
                co2 = fetch_co2_intensity(start, end, api_key)
        except Exception:
            st.warning("⚠️ CO₂-Daten nicht verfügbar – Simulation ohne CO₂.")
            co2 = pd.DataFrame()

        cfg = dict(
            capacity      = cap,
            connection    = conn,
            charge_loss   = ch_loss,
            discharge_loss= dis_loss,
            ev_price_kwh  = ev_price,
            sell_start    = sell_start,
            sell_end      = sell_end,
            capex         = capex,
            max_cycles    = max_cycles,
            opex          = opex,
            fin_model     = fin_model,
            loan_rate     = loan_rate,
            loan_years    = loan_years,
            lease_month   = lease_month,
            net_peak      = net_peak,
            net_off       = net_off,
            contract_years= contract_years,
            discount      = discount
        )

        df, summary = simulate(prices, co2, cfg)
        st.success("Simulation abgeschlossen ✔️")

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("NPV €",          f"{summary['NPV']:,.0f}")
        k2.metric("IRR %",          f"{summary['IRR']*100:,.2f}" if not np.isnan(summary['IRR']) else "n/a")
        k3.metric("Break-Even Jahr", summary['breakeven_year'] if summary['breakeven_year'] is not None else "> Laufzeit")

        # Monats-Cashflow (Jahr 1)
        st.subheader("📊 Monats-Cashflow (Jahr 1)")
        df_m = df.set_index("Datum").resample("M").sum()
        monthly_cost = summary['net_profit_first'] - df['Gewinn_EUR'].sum()
        df_m["Net_Cashflow"] = df_m["Gewinn_EUR"] - monthly_cost
        fig_cf, ax_cf = plt.subplots(figsize=(9, 4))
        ax_cf.bar(df_m.index.strftime("%b"), df_m["Net_Cashflow"], color="tab:blue")
        ax_cf.axhline(0, color="black", linewidth=0.8)
        st.pyplot(fig_cf)

        # Heatmap Tagesgewinne (Jahr 1)
        st.subheader("🔥 Heatmap täglicher Gewinne (Jahr 1)")
        heat = df.copy()
        heat["Monat"] = heat["Datum"].dt.month
        heat["Tag"]   = heat["Datum"].dt.day
        pivot = heat.pivot_table(index="Monat", columns="Tag",
                                 values="Gewinn_EUR", aggfunc="sum")
        fig_hm, ax_hm = plt.subplots(figsize=(12, 4))
        im = ax_hm.imshow(pivot, aspect="auto", cmap="RdYlGn")
        ax_hm.set_yticks(range(12))
        ax_hm.set_yticklabels([dt.date(1900, m, 1).strftime("%b")
                               for m in range(1, 13)])
        ax_hm.set_xlabel("Tag")
        fig_hm.colorbar(im, ax=ax_hm, label="€")
        st.pyplot(fig_hm)

        # Detail-Tabelle & Download
        st.subheader("🔍 Detailtabelle (Jahr 1)")
        st.dataframe(df.round(2))
        st.download_button("CSV herunterladen",
                           df.to_csv(index=False).encode(),
                           file_name="contractor_detail.csv")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
