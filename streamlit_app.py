# streamlit_app.py
import os
import requests
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt

"""
🔋 Stromspeicher-Simulator v3.1
==============================
• Live-Day-Ahead-Preise & CO₂-Intensität per ENTSO-E-API  
• Finanzierungs­modelle: Kauf, Kredit (Annuität), Leasing  
• Batterie-Verluste, Degradation, Netzentgelte Peak / Off-Peak  
• Heatmap täglicher Gewinne + Monats-Cashflow  
• Robustes Fehler-Handling (Statuscodes & Hinweise)  
"""

# --------------------------------------------------------------------------
# 🔗 ENTSO-E API-Layer
# --------------------------------------------------------------------------
ENTSOE_ENDPOINT = "https://transparency.entsoe.eu/api"
BIDDING_ZONE    = "10Y1001A1001A83"        # Deutschland / Luxemburg

def fetch_day_ahead_prices(start: dt.datetime,
                           end: dt.datetime,
                           api_key: str,
                           domain: str = BIDDING_ZONE) -> pd.DataFrame:
    """Stündliche Day-Ahead-Preise (€ /MWh) als DataFrame"""
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
    """Stündliche CO₂-Intensität (g CO₂ /kWh) als DataFrame"""
    params = {
        "documentType": "A75",          # Emissions forecast
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
# 🔋 Simulation-Kernel
# --------------------------------------------------------------------------
def simulate(prices: pd.DataFrame,
             co2: pd.DataFrame,
             cfg: dict) -> tuple[pd.DataFrame, dict]:
    """
    Simuliert ein Jahr Lade/Verkaufs-Betrieb + Kosten, CO₂ & Cashflows.
    Gibt (tägliches Ergebnis-DF, Jahres-Summary) zurück.
    """
    prices = prices.copy()
    prices["Datum"] = prices["Zeit"].dt.date

    # Flag, ob Stunde im Verkaufs­zeitfenster liegt
    def in_sell_window(ts: pd.Timestamp) -> bool:
        if cfg["sell_start"] < cfg["sell_end"]:
            return cfg["sell_start"] <= ts.time() <= cfg["sell_end"]
        return ts.time() >= cfg["sell_start"] or ts.time() <= cfg["sell_end"]

    prices["Sell"] = prices["Zeit"].apply(in_sell_window)

    # CO₂ anreichern (falls verfügbar)
    if not co2.empty:
        prices = prices.merge(co2, on="Zeit", how="left")
    else:
        prices["CO2_g_per_kWh"] = np.nan

    results, cycles = [], 0
    need_hours = int(np.ceil(cfg["capacity"] / cfg["connection"]))

    for d, grp in prices.groupby("Datum"):
        # günstigste Lade-Stunden
        load_hours  = grp.nsmallest(need_hours, "Preis_EUR_MWh")
        sell_window = grp[grp["Sell"]]
        if sell_window.empty:
            continue                                  # kein Verkaufsfenster → Tag überspringen

        buy_price  = load_hours["Preis_EUR_MWh"].mean()
        sell_price = sell_window["Preis_EUR_MWh"].mean() + cfg["markup"]

        charged = cfg["capacity"] * (1 + cfg["charge_loss"])
        sold    = cfg["capacity"] * (1 - cfg["discharge_loss"])

        revenue     = sold * sell_price
        energy_cost = charged * buy_price

        peak_ratio = sell_window["Zeit"].dt.hour.between(8, 20).mean()
        net_cost   = charged * (peak_ratio * cfg["net_peak"] +
                                (1 - peak_ratio) * cfg["net_off"])

        daily_profit = revenue - energy_cost - net_cost

        # CO₂-Bilanz
        if (not load_hours["CO2_g_per_kWh"].isna().all() and
                not sell_window["CO2_g_per_kWh"].isna().all()):
            co2_load = (load_hours["CO2_g_per_kWh"].mean() * charged) / 1000  # kg
            co2_sell = (sell_window["CO2_g_per_kWh"].mean() * sold) / 1000
            co2_saved = max(co2_sell - co2_load, 0)
        else:
            co2_saved = np.nan

        cycles += 1
        results.append({
            "Datum": pd.to_datetime(d),
            "Gewinn_EUR": daily_profit,
            "CO2_saved_kg": co2_saved,
        })

    df = pd.DataFrame(results)

    # -------- Kosten & Finanzierung --------
    cyc_cost  = cfg["capex"] / cfg["max_cycles"]          # €/Zyklus
    deg_cost  = cycles * cyc_cost

    if cfg["fin_model"] == "Kauf":
        fin_annual = 0
    elif cfg["fin_model"] == "Kredit":
        r, n = cfg["loan_rate"], cfg["loan_years"]
        fin_annual = cfg["capex"] * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    else:                                                 # Leasing
        fin_annual = cfg["lease_month"] * 12

    op_cost           = cfg["opex"]
    total_extra_cost  = deg_cost + fin_annual + op_cost

    summary = {
        "cycles"         : cycles,
        "deg_cost"       : deg_cost,
        "fin_annual"     : fin_annual,
        "opex"           : op_cost,
        "total_extra_cost": total_extra_cost,
        "gross_profit"   : df["Gewinn_EUR"].sum(),
        "net_profit"     : df["Gewinn_EUR"].sum() - total_extra_cost,
        "co2_saved_total": df["CO2_saved_kg"].sum()
    }
    return df, summary

# --------------------------------------------------------------------------
# 🚀 Streamlit-Frontend
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="🔋 Stromspeicher – Live-Simulator", layout="wide")
    st.title("🔋 Stromspeicher Simulator v3.1 – Live-Preise, Finanzierung, Heatmaps")

    # ---------- Sidebar: Basis ----------
    api_key = st.sidebar.text_input("ENTSO-E API-Key",
                                    os.getenv("ENTSOE_API_KEY", ""),
                                    type="password")
    year = st.sidebar.selectbox("Simulationsjahr", [2023, 2024, 2025], index=0)

    # ---------- Sidebar: Batterie ----------
    st.sidebar.header("🔋 Batterie")
    cap      = st.sidebar.number_input("Speichergröße (MWh)",      0.5, 20.0, 3.5, 0.1)
    conn     = st.sidebar.number_input("Netzanschluss (MW)",       0.1, 5.0, 0.35, 0.05)
    ch_loss  = st.sidebar.slider       ("Ladeverlust %",            0, 20, 5) / 100
    dis_loss = st.sidebar.slider       ("Entladeverlust %",         0, 20, 5) / 100
    markup   = st.sidebar.number_input("Verkaufs-Aufschlag €/MWh",  0, 500, 240, 10)

    # ---------- Sidebar: Finanzierung ----------
    st.sidebar.header("💰 Finanzierung")
    fin_model = st.sidebar.selectbox("Modell", ["Kauf", "Kredit", "Leasing"])
    capex      = st.sidebar.number_input("Capex Batterie €",         10000, 2_000_000, 350_000, 5_000)
    max_cycles = st.sidebar.number_input("Max. Zyklen",              1000, 15000, 6000, 100)
    opex       = st.sidebar.number_input("Opex/Jahr €",              0, 50_000, 5_000, 500)

    loan_rate  = st.sidebar.number_input("Kredit-Zins %",            0.0, 15.0, 4.0, 0.1)/100 if fin_model == "Kredit" else 0
    loan_years = st.sidebar.number_input("Kredit-Laufzeit Jahre",    1, 20, 10, 1)            if fin_model == "Kredit" else 0
    lease_month= st.sidebar.number_input("Leasingrate €/Monat",      0, 20_000, 3_000, 100)    if fin_model == "Leasing" else 0

    # ---------- Sidebar: Netzentgelte ----------
    st.sidebar.header("🔌 Netzentgelte")
    net_peak = st.sidebar.number_input("Peak €/MWh",     0, 200, 75, 5)
    net_off  = st.sidebar.number_input("Off-Peak €/MWh", 0, 200, 40, 5)

    # ---------- Sidebar: Verkauf ----------
    st.sidebar.header("🕑 Verkauf")
    sell_start = st.sidebar.time_input("Start", dt.time(16, 30))
    sell_end   = st.sidebar.time_input("Ende",  dt.time(6, 0))

    # ---------- Simulation ----------
    if st.button("Simulation starten"):
        start = dt.datetime(year, 1, 1)
        end   = dt.datetime(year, 12, 31, 23)

        # --- Preise laden ---
        try:
            with st.spinner("Day-Ahead-Preise abrufen …"):
                prices = fetch_day_ahead_prices(start, end, api_key)
        except requests.HTTPError as err:
            st.error(f"❌ API-Fehler Day-Ahead ({err.response.status_code} – {err.response.reason}). "
                     "Bitte Key & Zeitraum prüfen.")
            st.stop()
        except Exception as ex:
            st.error(f"❌ Unbekannter Fehler beim Preis-Abruf: {ex}")
            st.stop()

        # --- CO₂ laden ---
        try:
            with st.spinner("CO₂-Intensität abrufen …"):
                co2 = fetch_co2_intensity(start, end, api_key)
        except requests.HTTPError:
            st.warning("⚠️ CO₂-Daten konnten nicht geladen werden – Simulation ohne CO₂-Bilanz.")
            co2 = pd.DataFrame()
        except Exception as ex:
            st.warning(f"⚠️ CO₂-Abruf-Fehler: {ex}")
            co2 = pd.DataFrame()

        cfg = {
            "capacity"      : cap,
            "connection"    : conn,
            "charge_loss"   : ch_loss,
            "discharge_loss": dis_loss,
            "markup"        : markup,
            "sell_start"    : sell_start,
            "sell_end"      : sell_end,
            "capex"         : capex,
            "max_cycles"    : max_cycles,
            "opex"          : opex,
            "fin_model"     : fin_model,
            "loan_rate"     : loan_rate,
            "loan_years"    : loan_years,
            "lease_month"   : lease_month,
            "net_peak"      : net_peak,
            "net_off"       : net_off,
        }

        # --- Simulation ---
        df, summary = simulate(prices, co2, cfg)
        st.success("Simulation abgeschlossen ✔️")

        # ---------- KPIs ----------
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Brutto-Gewinn €", f"{summary['gross_profit']:,.0f}")
        k2.metric("Kosten €",        f"{summary['total_extra_cost']:,.0f}")
        k3.metric("Netto-Gewinn €",  f"{summary['net_profit']:,.0f}")
        k4.metric("Zyklen", summary['cycles'])

        # ---------- Monats-Cashflow ----------
        st.subheader("📊 Monats-Cashflow")
        df_m = df.set_index("Datum").resample("M").sum()
        monthly_cost = summary["total_extra_cost"] / 12
        df_m["Net_Cashflow"] = df_m["Gewinn_EUR"] - monthly_cost

        fig_cf, ax_cf = plt.subplots(figsize=(9, 4))
        ax_cf.bar(df_m.index.strftime("%b"), df_m["Net_Cashflow"], color="tab:blue")
        ax_cf.set_ylabel("€")
        ax_cf.set_title("Netto-Cashflow pro Monat")
        ax_cf.axhline(0, color="black", linewidth=0.8)
        st.pyplot(fig_cf)

        # ---------- Heatmap ----------
        st.subheader("🔥 Heatmap täglicher Gewinne")
        heat = df.copy()
        heat["Monat"] = heat["Datum"].dt.month
        heat["Tag"]   = heat["Datum"].dt.day
        pivot = heat.pivot_table(index="Monat", columns="Tag",
                                 values="Gewinn_EUR", aggfunc="sum")
        fig_hm, ax_hm = plt.subplots(figsize=(12, 4))
        im = ax_hm.imshow(pivot, aspect="auto", cmap="RdYlGn")
        ax_hm.set_yticks(range(0, 12))
        ax_hm.set_yticklabels([dt.date(1900, m, 1).strftime("%b")
                               for m in range(1, 13)])
        ax_hm.set_xlabel("Tag im Monat")
        ax_hm.set_title("Tägliche Gewinne – Heatmap")
        fig_hm.colorbar(im, ax=ax_hm, label="€")
        st.pyplot(fig_hm)

        # ---------- Detailtabelle ----------
        st.subheader("🔍 Detailtabelle")
        st.dataframe(df.round(2))
        st.download_button("CSV herunterladen",
                           df.to_csv(index=False).encode(),
                           file_name="sim_detail.csv")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
