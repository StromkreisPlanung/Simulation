import os
import requests
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt

"""
Stromspeicher Spotmarkt Simulator – Version API & CO₂
====================================================
Diese Version bindet Live‑Daten der ENTSO‑E Transparency Platform ein und kombiniert sie
mit einem erweiterten Batteriemodell (Verluste, Degradation) sowie CO₂‑ und
Netzentgelt‑Berechnung. Finanzierungsmodelle folgen in Phase 4.

⚠️ Wichtig: Der Nutzer muss einen gültigen ENTSO‑E API‑Key als Umgebungsvariable
    ENTSOE_API_KEY setzen (oder im Sidebar‑Feld eingeben).
"""

# -----------------------------------------------------------------------------
# 🗝️ API Utility Layer
# -----------------------------------------------------------------------------
ENTSOE_ENDPOINT = "https://transparency.entsoe.eu/api"
# Bidding‑Zone für Deutschland/Luxemburg (DE‑LU)
BIDDING_ZONE = "10Y1001A1001A83"


def fetch_day_ahead_prices(start: dt.datetime, end: dt.datetime, domain: str = BIDDING_ZONE, api_key: str | None = None) -> pd.DataFrame:
    """Holt Day‑Ahead‑Preise (HOURLY) von ENTSO‑E und gibt DataFrame zurück."""
    if api_key is None:
        raise ValueError("ENTSOE API‑Key fehlt – bitte Umgebungsvariable ENTSOE_API_KEY setzen oder im Sidebar eingeben.")

    params = {
        "documentType": "A44",          # Day‑Ahead prices
        "processType": "A01",            # Realtime process
        "in_Domain": domain,
        "out_Domain": domain,
        "periodStart": start.strftime("%Y%m%d%H%M"),  # im Format YYYYMMDDHHMM
        "periodEnd": end.strftime("%Y%m%d%H%M"),
        "securityToken": api_key,
    }

    r = requests.get(ENTSOE_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    # ENTSO‑E liefert XML → mit pandas read_xml auslesen
    df = pd.read_xml(r.text, xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int) - 1, unit="h", origin=start)
    df.rename(columns={"price.amount": "Preis_EUR_MWh"}, inplace=True)
    return df[["Zeit", "Preis_EUR_MWh"]]


def fetch_co2_intensity(start: dt.datetime, end: dt.datetime, domain: str = BIDDING_ZONE, api_key: str | None = None) -> pd.DataFrame:
    """Holt CO₂‑Intensität des Strommix (gCO2/kWh) von ENTSO‑E."""
    if api_key is None:
        raise ValueError("ENTSOE API‑Key fehlt – bitte Umgebungsvariable ENTSOE_API_KEY setzen oder im Sidebar eingeben.")

    params = {
        "documentType": "A75",          # Emissions per production type
        "processType": "A16",            # Day‑ahead forecast
        "in_Domain": domain,
        "periodStart": start.strftime("%Y%m%d%H%M"),
        "periodEnd": end.strftime("%Y%m%d%H%M"),
        "securityToken": api_key,
    }
    r = requests.get(ENTSOE_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    df = pd.read_xml(r.text, xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int) - 1, unit="h", origin=start)
    df.rename(columns={"quantity": "CO2_g_per_kWh"}, inplace=True)
    return df[["Zeit", "CO2_g_per_kWh"]]

# -----------------------------------------------------------------------------
# 🔋 Batterie‑ & Business‑Logik
# -----------------------------------------------------------------------------

def battery_simulation(prices: pd.DataFrame,
                       co2: pd.DataFrame | None,
                       speicher_mwh: float,
                       anschluss_mw: float,
                       ladeverlust: float,
                       entladeverlust: float,
                       aufschlag_eur_mwh: float,
                       verkauf_von: dt.time,
                       verkauf_bis: dt.time,
                       max_zyklen: int,
                       batteriekosten: float,
                       betriebskosten: float,
                       degradation_pro_zyklus: float,
                       netzentgelt_peak: float,
                       netzentgelt_offpeak: float) -> pd.DataFrame:
    """Simuliert Lade‑/Verkaufsstrategie + Kosten & CO₂."""
    prices = prices.copy()
    prices["Datum"] = prices["Zeit"].dt.date

    def is_sell_window(ts: pd.Timestamp) -> bool:
        if verkauf_von < verkauf_bis:
            return verkauf_von <= ts.time() <= verkauf_bis
        return ts.time() >= verkauf_von or ts.time() <= verkauf_bis

    prices["Verkaufszeit"] = prices["Zeit"].apply(is_sell_window)

    if co2 is not None and not co2.empty:
        prices = prices.merge(co2, on="Zeit", how="left")
    else:
        prices["CO2_g_per_kWh"] = np.nan

    results = []
    zyklen = 0
    for datum, grp in prices.groupby("Datum"):
        stunden_needed = int(np.ceil(speicher_mwh / anschluss_mw))
        # günstigste Lade‑Stunden
        lade_hours = grp.nsmallest(stunden_needed, "Preis_EUR_MWh")
        avg_buy_price = lade_hours["Preis_EUR_MWh"].mean()
        energy_charged = speicher_mwh * (1 + ladeverlust)

        # Verkauf innerhalb Fensters
        sell_window = grp[grp["Verkaufszeit"]]
        if sell_window.empty:
            continue
        avg_sell_price = sell_window["Preis_EUR_MWh"].mean() + aufschlag_eur_mwh
        energy_sold = speicher_mwh * (1 - entladeverlust)

        umsatz = energy_sold * avg_sell_price
        energiekosten = energy_charged * avg_buy_price

        # Dynamisches Netzentgelt
        peak_mask = sell_window["Zeit"].dt.hour.between(8, 20)
        net_cost = (energy_charged * netzentgelt_offpeak + energy_charged * peak_mask.mean() * (netzentgelt_peak - netzentgelt_offpeak))

        tagesgewinn = umsatz - energiekosten - net_cost

        # CO₂ Bilanz
        if not lade_hours["CO2_g_per_kWh"].isna().all() and not sell_window["CO2_g_per_kWh"].isna().all():
            co2_load = (lade_hours["CO2_g_per_kWh"].mean() * energy_charged) / 1000  # kg
            co2_sell = (sell_window["CO2_g_per_kWh"].mean() * energy_sold) / 1000
            co2_saved = max(co2_sell - co2_load, 0)
        else:
            co2_saved = np.nan

        zyklen += 1
        results.append({
            "Datum": datum,
            "Einkaufspreis": avg_buy_price,
            "Verkaufspreis": avg_sell_price,
            "Tagesgewinn_EUR": tagesgewinn,
            "CO2_saved_kg": co2_saved
        })

    df_result = pd.DataFrame(results)

    # Batterie‑Kosten linear über Zyklen
    zyklus_cost = batteriekosten / max_zyklen
    deg_cost = zyklen * zyklus_cost
    total_costs = deg_cost + betriebskosten
    df_result["Netto_Jahresgewinn_EUR"] = df_result["Tagesgewinn_EUR"].sum() - total_costs
    return df_result

# -----------------------------------------------------------------------------
# 🚀 Streamlit App (main)
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="🔋 Stromspeicher Simulator – Live API", layout="wide")
    st.title("🔋 Stromspeicher Simulator – Live Spotmarkt & CO₂")

    st.sidebar.header("🔑 API & Basisdaten")
    api_key_input = st.sidebar.text_input("ENTSO‑E API‑Key", value=os.getenv("ENTSOE_API_KEY", ""), type="password")

    jahr = st.sidebar.selectbox("Jahr auswählen", options=[2023, 2024, 2025], index=0)
    start_date = dt.datetime(jahr, 1, 1)
    end_date = dt.datetime(jahr, 12, 31, 23, 59)

    with st.sidebar.expander("🔋 Batterieparameter"):
        speicher_mwh = st.number_input("Speichergröße (MWh)", 0.5, 20.0, 3.5, 0.1)
        anschluss_mw = st.number_input("Netzanschluss (MW)", 0.1, 5.0, 0.35, 0.05)
        ladeverlust = st.slider("Ladeverlust %", 0, 20, 5) / 100
        entladeverlust = st.slider("Entladeverlust %", 0, 20, 5) / 100
        aufschlag_eur_mwh = st.number_input("Verkaufsaufschlag (EUR/MWh)", 0, 500, 240, 10)

    with st.sidebar.expander("📦 Finanzierung & Kosten"):
        batteriekosten = st.number_input("Batteriekosten (€)", 10000, 1000000, 350000, 5000)
        max_zyklen = st.number_input("Max. Ladezyklen", 1000, 15000, 6000, 100)
        betriebskosten = st.number_input("Betriebskosten/Jahr (€)", 0, 50000, 5000, 500)
        degradation_pro_zyklus = st.slider("Degradation pro Zyklus %", 0.0, 0.2, 0.05) / 100

    with st.sidebar.expander("💡 Netzentgeltmodell"):
        net_peak = st.number_input("Netzentgelt Peak (€/MWh)", 0, 200, 75, 5)
        net_off = st.number_input("Netzentgelt Off‑Peak (€/MWh)", 0, 200, 40, 5)

    verkauf_von = st.sidebar.time_input("Verkauf ab", value=dt.time(16, 30))
    verkauf_bis = st.sidebar.time_input("Verkauf bis", value=dt.time(6, 0))

    if st.button("Simulation starten"):
        with st.spinner("Daten werden geladen …"):
            prices = fetch_day_ahead_prices(start_date, end_date, api_key=api_key_input)
            co2 = fetch_co2_intensity(start_date, end_date, api_key=api_key_input)

        df_result = battery_simulation(prices, co2, speicher_mwh, anschluss_mw, ladeverlust,
                                        entladeverlust, aufschlag_eur_mwh, verkauf_von, verkauf_bis,
                                        max_zyklen, batteriekosten, betriebskosten,
                                        degradation_pro_zyklus, net_peak, net_off)

        st.success("Simulation abgeschlossen")

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Ø Tagesgewinn", f"{df_result['Tagesgewinn_EUR'].mean():.2f} €")
        kpi2.metric("Gesamter Gewinn", f"{df_result['Tagesgewinn_EUR'].sum():.0f} €")
        kpi3.metric("CO₂‑Einsparung (kg)", f"{df_result['CO2_saved_kg'].sum():.0f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df_result["Tagesgewinn_EUR"], bins=30, edgecolor="black")
        ax.set_title("Verteilung Tagesgewinne")
        ax.set_xlabel("Gewinn €")
        ax.set_ylabel("Anzahl Tage")
        st.pyplot(fig)

        st.subheader("Detailergebnisse")
        st.dataframe(df_result)

        st.download_button("CSV herunterladen", df_result.to_csv(index=False).encode(), file_name="simulation_ergebnisse.csv")


if __name__ == "__main__":
    main()
