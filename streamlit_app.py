
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="🔋 Stromspeicher Spotmarkt Simulator V2", layout="wide")
st.title("🔋 Stromspeicher Spotmarkt Simulator mit Preisdaten & Batterie")

st.sidebar.header("⚙️ Parameter einstellen")

# Batterieparameter
speicher_groesse = st.sidebar.number_input("Speichergröße (MWh)", 0.5, 10.0, 3.5, 0.1)
anschlussleistung = st.sidebar.number_input("Netzanschluss (MW)", 0.1, 2.0, 0.35, 0.05)
ladeverlust = st.sidebar.number_input("Ladeverlust (%)", 0, 20, 5, 1)/100
entladeverlust = st.sidebar.number_input("Entladeverlust (%)", 0, 20, 5, 1)/100
max_zyklen = st.sidebar.number_input("Maximale Ladezyklen Batterie", 1000, 15000, 6000, 100)
batteriekosten = st.sidebar.number_input("Batteriekosten (€)", 10000, 1000000, 350000, 5000)
betriebskosten = st.sidebar.number_input("Betriebskosten pro Jahr (€)", 0, 50000, 5000, 500)
degradation_pro_zyklus = st.sidebar.number_input("Degradation pro Zyklus (%)", 0.0, 0.2, 0.05, 0.01)/100

# Verkaufsparameter
aufschlag = st.sidebar.number_input("Verkaufsaufschlag (EUR/MWh)", 100, 500, 240, 10)
verkauf_von = st.sidebar.time_input("Verkauf ab", value=pd.to_datetime("16:30").time())
verkauf_bis = st.sidebar.time_input("Verkauf bis", value=pd.to_datetime("06:00").time())

st.sidebar.markdown("---")
preisdatei = st.sidebar.file_uploader("📥 CSV mit Preisdaten hochladen", type=['csv'])

if preisdatei:
    daten = pd.read_csv(preisdatei, parse_dates=['Zeit'])
else:
    st.info('Nutze Beispiel-Daten. Lade eigene CSV hoch für echte Preise.')
    daten = pd.read_csv('beispiel_preise.csv', parse_dates=['Zeit'])

daten['Datum'] = daten['Zeit'].dt.date

def ist_verkaufszeit(zeit):
    start = pd.to_datetime(verkauf_von.strftime('%H:%M')).time()
    ende = pd.to_datetime(verkauf_bis.strftime('%H:%M')).time()
    if start < ende:
        return start <= zeit.time() <= ende
    else:
        return zeit.time() >= start or zeit.time() <= ende

daten['Verkaufszeit'] = daten['Zeit'].apply(ist_verkaufszeit)

ergebnisse = []
zyklen_counter = 0

for datum, gruppe in daten.groupby('Datum'):
    benoetigte_stunden = int(speicher_groesse / anschlussleistung)
    gruppe_sorted = gruppe.sort_values('Preis_EUR_MWh')
    ladezeiten = gruppe_sorted.head(benoetigte_stunden)
    einkaufspreis = ladezeiten['Preis_EUR_MWh'].mean()

    geladen = speicher_groesse * (1 + ladeverlust)

    verkaufsfenster = gruppe[gruppe['Verkaufszeit'] == True]
    if not verkaufsfenster.empty:
        verkaufspreis = verkaufsfenster['Preis_EUR_MWh'].mean() + aufschlag
    else:
        verkaufspreis = np.nan

    verkauft = speicher_groesse * (1 - entladeverlust)

    if not np.isnan(verkaufspreis):
        umsatz = verkauft * verkaufspreis
        kosten = geladen * einkaufspreis
        tagesgewinn = umsatz - kosten
        zyklen_counter += 1
    else:
        tagesgewinn = np.nan

    ergebnisse.append({
        'Datum': datum,
        'Einkaufspreis': einkaufspreis,
        'Verkaufspreis': verkaufspreis,
        'Tagesgewinn_EUR': tagesgewinn
    })

result_df = pd.DataFrame(ergebnisse).dropna()

# Batterieabschreibung
zyklus_kosten = batteriekosten / max_zyklen
degradation_kosten = zyklen_counter * zyklus_kosten
gesamt_betriebskosten = degradation_kosten + betriebskosten

# 📊 KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Ø Tagesgewinn", f"{result_df['Tagesgewinn_EUR'].mean():.2f} EUR")
col2.metric("Max Tagesgewinn", f"{result_df['Tagesgewinn_EUR'].max():.2f} EUR")
col3.metric("Jahresgewinn nach Kosten", f"{(result_df['Tagesgewinn_EUR'].sum()-gesamt_betriebskosten):,.2f} EUR")


st.subheader(f"🔋 Batteriezyklen im Jahr: {zyklen_counter}")
st.subheader(f"💰 Betriebskosten (inkl. Deg.): {gesamt_betriebskosten:,.2f} EUR")

# Histogramm
st.subheader("🔍 Verteilung der Tagesgewinne")
fig, ax = plt.subplots(figsize=(10,6))
ax.hist(result_df['Tagesgewinn_EUR'], bins=30, edgecolor='black')
ax.set_title('Verteilung der Tagesgewinne')
ax.set_xlabel('Tagesgewinn in EUR')
ax.set_ylabel('Anzahl Tage')
ax.grid(True)
st.pyplot(fig)

st.subheader("📅 Detaillierte Tagesergebnisse")
st.dataframe(result_df)

st.download_button(
    label="📥 Ergebnisse als CSV herunterladen",
    data=result_df.to_csv(index=False).encode('utf-8'),
    file_name='stromspeicher_ergebnisse.csv',
    mime='text/csv'
)

st.caption("Simulator mit echten Preisdaten und Batteriemodell.")
