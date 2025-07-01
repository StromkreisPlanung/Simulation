import os
import requests
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt

"""
Stromspeicher Simulator –– Live‑API v3
=====================================
Erweiterungen:
1. **Finanzierungs­module**: Kauf, Kredit (Annuität), Leasing
2. **Heatmaps / Monats‑Cashflows**: tägliche & monatliche Visualisierung
3. Weiterhin: Live‑Preise & CO₂‑Daten (ENTSO‑E)

Hinweis: ENTSO‑E‑API‑Key als Umgebungs­variable `ENTSOE_API_KEY` oder im Sidebar angeben.
"""

# -----------------------------------------------------------------------------
# 🔗 ENTSO‑E API Layer
# -----------------------------------------------------------------------------
ENTSOE_ENDPOINT = "https://transparency.entsoe.eu/api"
BIDDING_ZONE = "10Y1001A1001A83"  # DE‑LU

def fetch_day_ahead_prices(start: dt.datetime, end: dt.datetime, api_key: str, domain: str = BIDDING_ZONE) -> pd.DataFrame:
    params = {
        "documentType": "A44",
        "processType": "A01",
        "in_Domain": domain,
        "out_Domain": domain,
        "periodStart": start.strftime("%Y%m%d%H%M"),
        "periodEnd": end.strftime("%Y%m%d%H%M"),
        "securityToken": api_key,
    }
    r = requests.get(ENTSOE_ENDPOINT, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_xml(r.text, xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int) - 1, unit="h", origin=start)
    df.rename(columns={"price.amount": "Preis_EUR_MWh"}, inplace=True)
    return df[["Zeit", "Preis_EUR_MWh"]]

def fetch_co2_intensity(start: dt.datetime, end: dt.datetime, api_key: str, domain: str = BIDDING_ZONE) -> pd.DataFrame:
    params = {
        "documentType": "A75",
        "processType": "A16",
        "in_Domain": domain,
        "periodStart": start.strftime("%Y%m%d%H%M"),
        "periodEnd": end.strftime("%Y%m%d%H%M"),
        "securityToken": api_key,
    }
    r = requests.get(ENTSOE_ENDPOINT, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_xml(r.text, xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int) - 1, unit="h", origin=start)
    df.rename(columns={"quantity": "CO2_g_per_kWh"}, inplace=True)
    return df[["Zeit", "CO2_g_per_kWh"]]

# -----------------------------------------------------------------------------
# 🔋 Simulation Core
# -----------------------------------------------------------------------------

def simulate(prices: pd.DataFrame,
             co2: pd.DataFrame,
             cfg: dict) -> tuple[pd.DataFrame, dict]:
    prices = prices.copy()
    prices["Datum"] = prices["Zeit"].dt.date

    # Verkaufszeitfenster
    def in_sell_window(ts: pd.Timestamp) -> bool:
        if cfg["sell_start"] < cfg["sell_end"]:
            return cfg["sell_start"] <= ts.time() <= cfg["sell_end"]
        return ts.time() >= cfg["sell_start"] or ts.time() <= cfg["sell_end"]

    prices["Sell"] = prices["Zeit"].apply(in_sell_window)
    prices = prices.merge(co2, on="Zeit", how="left")

    results = []
    cycles = 0
    need_hours = int(np.ceil(cfg["capacity"] / cfg["connection"]))

    for d, grp in prices.groupby("Datum"):
        load_hours = grp.nsmallest(need_hours, "Preis_EUR_MWh")
        sell_window = grp[grp["Sell"]]
        if sell_window.empty:
            continue

        buy_price = load_hours["Preis_EUR_MWh"].mean()
        sell_price = sell_window["Preis_EUR_MWh"].mean() + cfg["markup"]

        charged = cfg["capacity"] * (1 + cfg["charge_loss"])
        sold = cfg["capacity"] * (1 - cfg["discharge_loss"])

        revenue = sold * sell_price
        energy_cost = charged * buy_price

        # Netzentgelt (Peak = 08‑20 Uhr)
        peak_ratio = sell_window["Zeit"].dt.hour.between(8, 20).mean()
        net_cost = charged * (peak_ratio * cfg["net_peak"] + (1 - peak_ratio) * cfg["net_off"])

        daily_profit = revenue - energy_cost - net_cost

        # CO₂
        co2_load = (load_hours["CO2_g_per_kWh"].mean() * charged) / 1000 if not load_hours["CO2_g_per_kWh"].isna().all() else np.nan
        co2_sell = (sell_window["CO2_g_per_kWh"].mean() * sold) / 1000 if not sell_window["CO2_g_per_kWh"].isna().all() else np.nan
        co2_saved = max(co2_sell - co2_load, 0) if (not np.isnan(co2_load) and not np.isnan(co2_sell)) else np.nan

        cycles += 1
        results.append({
            "Datum": pd.to_datetime(d),
            "Gewinn_EUR": daily_profit,
            "CO2_saved_kg": co2_saved,
        })

    df = pd.DataFrame(results)

    # Kosten & Finanzierung
    cy_cost = cfg["capex"] / cfg["max_cycles"]
    deg_cost = cycles * cy_cost

    if cfg["fin_model"] == "Kauf":
        fin_annual = 0  # Capex bereits berücksichtigt im Deg-Kosten
    elif cfg["fin_model"] == "Kredit":
        r = cfg["loan_rate"]
        n = cfg["loan_years"]
        annuity = cfg["capex"] * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
        fin_annual = annuity
    else:  # Leasing
        fin_annual = cfg["lease_month"] * 12

    op_cost = cfg["opex"]
    total_extra_cost = deg_cost + fin_annual + op_cost

    summary = {
        "cycles": cycles,
        "deg_cost": deg_cost,
        "fin_annual": fin_annual,
        "opex": op_cost,
        "total_extra_cost": total_extra_cost,
        "gross_profit": df["Gewinn_EUR"].sum(),
        "net_profit": df["Gewinn_EUR"].sum() - total_extra_cost,
        "co2_saved_total": df["CO2_saved_kg"].sum()
    }

    return df, summary

# -----------------------------------------------------------------------------
# 🚀 Streamlit Frontend
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="🔋 Stromspeicher – Finanz & Heatmaps", layout="wide")
    st.title("🔋 Stromspeicher Simulator v3 – Live‑Preise, Finanzierung & Heatmaps")

    # Sidebar Basics
    api_key = st.sidebar.text_input("ENTSO‑E API‑Key", os.getenv("ENTSOE_API_KEY", ""), type="password")
    year = st.sidebar.selectbox("Jahr", [2023, 2024, 2025], index=0)

    # Batterie
    st.sidebar.header("🔋 Batterie")
    cap = st.sidebar.number_input("Speichergröße (MWh)", 0.5, 20.0, 3.5, 0.1)
    conn = st.sidebar.number_input("Netzanschluss (MW)", 0.1, 5.0, 0.35, 0.05)
    ch_loss = st.sidebar.slider("Ladeverlust %", 0, 20, 5) / 100
    dis_loss = st.sidebar.slider("Entladeverlust %", 0, 20, 5) / 100
    markup = st.sidebar.number_input("Aufschlag EUR/MWh", 0, 500, 240, 10)

    # Finanzierung
    st.sidebar.header("💰 Finanzierung")
    fin_model = st.sidebar.selectbox("Modell", ["Kauf", "Kredit", "Leasing"], index=0)
    capex = st.sidebar.number_input("Capex Batterie (€)", 10000, 2000000, 350000, 5000)
    max_cycles = st.sidebar.number_input("Max Zyklen", 1000, 15000, 6000, 100)
    opex = st.sidebar.number_input("Opex/Jahr (€)", 0, 50000, 5000, 500)

    loan_rate = st.sidebar.number_input("Kredit‑Zins %", 0.0, 15.0, 4.0, 0.1)/100 if fin_model == "Kredit" else 0
    loan_years = st.sidebar.number_input("Kredit‑Laufzeit (J)", 1, 20, 10, 1) if fin_model == "Kredit" else 0
    lease_month = st.sidebar.number_input("Leasingrate €/Monat", 0, 20000, 3000, 100) if fin_model == "Leasing" else 0

    # Netzentgelt
    st.sidebar.header("🔌 Netzentgelte")
    net_peak = st.sidebar.number_input("Peak (€/MWh)", 0, 200, 75, 5)
    net_off = st.sidebar.number_input("Off‑Peak (€/MWh)", 0, 200, 40, 5)

    # Verkauf
    st.sidebar.header("🕑 Verkauf")
    sell_start = st.sidebar.time_input("Start", dt.time(16, 30))
    sell_end = st.sidebar.time_input("Ende", dt.time(6, 0))

    if st.button("Simulation starten"):
        start = dt.datetime(year, 1, 1)
        end = dt.datetime(year, 12, 31, 23, 0)
        with st.spinner("API‑Daten abrufen …"):
            prices = fetch_day_ahead_prices(start, end, api_key)
            co2 = fetch_co2_intensity(start, end, api_key)

        cfg = {
            "capacity": cap,
            "connection": conn,
            "charge_loss": ch_loss,
            "discharge_loss": dis_loss,
            "markup": markup,
            "sell_start": sell_start,
            "sell_end": sell_end,
            "capex": capex,
            "max_cycles": max_cycles,
            "opex": opex,
            "fin_model": fin_model,
            "loan_rate": loan_rate,
            "loan_years": loan_years,
            "lease_month": lease_month,
            "net_peak": net_peak,
            "net_off": net_off,
        }

        df, summary = simulate(prices, co2, cfg)
        st.success("Simulation fertig ✔️")

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Brutto‑Gewinn €", f"{summary['gross_profit']:,.0f}")
        k2.metric("Kosten €", f"{summary['total_extra_cost']:,.0f}")
        k3.metric("Netto‑Gewinn €", f"{summary['net_profit']:,.0f}")
        k4.metric("Zyklen", summary['cycles'])

        # Monats‑Cashflow
        df_month = df.set_index("Datum").resample("M").sum()
        monthly_cost = summary['total_extra_cost'] / 12
        df_month["Net_Cashflow"] = df_month["Gewinn_EUR"] - monthly_cost

        st.subheader("📊 Monats‑Cashflow")
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.bar(df_month.index.strftime("%b"), df_month["Net_Cashflow"], color="tab:blue")
        ax2.set_ylabel("€")
        ax2.set_title("Netto‑Cashflow pro Monat")
        ax2.axhline(0, color="black", linewidth=0.8)
        st.pyplot(fig2)

        # Heatmap täglicher Gewinne
        st.subheader("🔥 Heatmap Tages­gewinne")
        df_heat = df.copy()
        df_heat["Monat"] = df_heat["Datum"].dt.month
        df_heat["Tag"] = df_heat["Datum"].dt.day
        pivot = df_heat.pivot_table(index="Monat", columns="Tag", values="Gewinn_EUR", aggfunc="sum")
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        im = ax3.imshow(pivot, aspect="auto", cmap="RdYlGn")
        ax3.set_yticks(np.arange(0, 12))
        ax3.set_yticklabels([dt.date(1900, m, 1).strftime("%b") for m in range(1, 13)])
        ax3.set_xlabel("Tag im Monat")
        ax3.set_title("Tägliche Gewinne – Heatmap")
        fig3.colorbar(im, ax=ax3, label="€")
        st.pyplot(fig3)

        st.subheader("🔍 Detailtabelle")
        st.dataframe(df.round(2))
        st.download_button("CSV herunterladen", df.to_csv(index=False).encode(), file_name="sim_detail.csv")

if __name__ == "__main__":
    main()
