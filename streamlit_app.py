import os
import requests
import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt

"""
Stromspeicher-Simulator v4.1 – Reines Contractor-Modell
=======================================================
• Live-Day-Ahead & CO₂ (ENTSO-E)
• Fester EV-Ladepreis (€/kWh)
• Contractor investiert CAPEX, trägt Opex, erhält X % der Jahresgewinne
• Standortbesitzer erhält (100 – X) % Gewinnbeteiligung – keine Miete
• KPIs (für Contractor): NPV @ Diskont, IRR, Break‑Even
• Monats-Cashflow & Heatmap (Jahr 1)
"""

# -----------------------------------------------------------------------------
# API-Funktionen (unverändert)
# -----------------------------------------------------------------------------
ENTSOE_ENDPOINT = "https://transparency.entsoe.eu/api"
BIDDING_ZONE    = "10Y1001A1001A83"

def fetch_day_ahead_prices(start, end, api_key, domain=BIDDING_ZONE):
    params = {
        "documentType": "A44", "processType": "A01",
        "in_Domain": domain, "out_Domain": domain,
        "periodStart": start.strftime("%Y%m%d%H%M"),
        "periodEnd":   end.strftime("%Y%m%d%H%M"),
        "securityToken": api_key,
    }
    r = requests.get(ENTSOE_ENDPOINT, params=params, timeout=60); r.raise_for_status()
    df = pd.read_xml(r.text, xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int)-1, unit="h", origin=start)
    df.rename(columns={"price.amount":"Preis_EUR_MWh"}, inplace=True)
    return df[["Zeit","Preis_EUR_MWh"]]

def fetch_co2_intensity(start, end, api_key, domain=BIDDING_ZONE):
    params = {
        "documentType": "A75", "processType": "A16", "in_Domain": domain,
        "periodStart": start.strftime("%Y%m%d%H%M"), "periodEnd": end.strftime("%Y%m%d%H%M"),
        "securityToken": api_key,
    }
    r=requests.get(ENTSOE_ENDPOINT, params=params, timeout=60); r.raise_for_status()
    df=pd.read_xml(r.text, xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int)-1, unit="h", origin=start)
    df.rename(columns={"quantity":"CO2_g_per_kWh"}, inplace=True)
    return df[["Zeit","CO2_g_per_kWh"]]

# -----------------------------------------------------------------------------
# Simulation für Contractor-Modell
# -----------------------------------------------------------------------------

def simulate(prices: pd.DataFrame, co2: pd.DataFrame, cfg: dict):
    prices = prices.copy(); prices["Datum"] = prices["Zeit"].dt.date
    prices["Sell"] = prices["Zeit"].apply(lambda ts: cfg["sell_start"]<=ts.time()<=cfg["sell_end"] if cfg["sell_start"]<cfg["sell_end"] else ts.time()>=cfg["sell_start"] or ts.time()<=cfg["sell_end"])
    if not co2.empty:
        prices = prices.merge(co2, on="Zeit", how="left")
    else:
        prices["CO2_g_per_kWh"] = np.nan

    need_hours = int(np.ceil(cfg["capacity"]/cfg["connection"]))
    cycles, daily_rows = 0, []
    for d,g in prices.groupby("Datum"):
        load = g.nsmallest(need_hours, "Preis_EUR_MWh")
        sell = g[g["Sell"]]
        if sell.empty: continue
        buy_p  = load["Preis_EUR_MWh"].mean()
        sell_p = cfg["ev_price"]*1000  # €/MWh
        charged = cfg["capacity"]*(1+cfg["closs"]); sold = cfg["capacity"]*(1-cfg["dloss"])
        revenue = sold*sell_p; cost_energy = charged*buy_p
        peak = sell["Zeit"].dt.hour.between(8,20).mean(); net_cost = charged*(peak*cfg["npeak"]+(1-peak)*cfg["noff"])
        profit = revenue - cost_energy - net_cost
        cycles +=1
        daily_rows.append({"Datum":pd.to_datetime(d),"Gewinn_EUR":profit})
    df = pd.DataFrame(daily_rows)

    deg_cost = cycles*(cfg["capex"]/cfg["max_cycles"])
    annual_net = df["Gewinn_EUR"].sum() - deg_cost - cfg["opex"]

    contractor_gain = annual_net*(1-cfg["owner_share"])
    cashflows = [-cfg["capex"]] + [contractor_gain]*cfg["years"]
    npv = sum(cf/((1+cfg["disc"])**t) for t,cf in enumerate(cashflows))
    try: irr = np.irr(cashflows)
    except: irr=np.nan
    be = next((t for t,c in enumerate(np.cumsum(cashflows)) if c>=0), None)

    summary = dict(NPV=npv, IRR=irr, BE=be, cycles=cycles, contractor_gain=contractor_gain)
    return df, summary

# -----------------------------------------------------------------------------
# Streamlit Frontend
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="🔋 Contractor-Ladepark", layout="wide")
    st.title("🔋 Contractor-Ladepark Simulator – Spot & Gewinnbeteiligung")

    api = st.sidebar.text_input("ENTSO-E API-Key", os.getenv("ENTSOE_API_KEY",""), type="password")
    year= st.sidebar.selectbox("Startjahr", [2023,2024,2025],0)

    st.sidebar.header("🔋 Batterie")
    cap=st.sidebar.number_input("Speicher MWh",0.5,20.0,3.5,0.1)
    conn=st.sidebar.number_input("Anschluss MW",0.1,5.0,0.35,0.05)
    closs=st.sidebar.slider("Ladeverlust %",0,20,5)/100
    dloss=st.sidebar.slider("Entladeverlust %",0,20,5)/100

    st.sidebar.header("🚗 EV-Ladepreis")
    ev_price=st.sidebar.number_input("Preis €/kWh",0.1,1.0,0.285,0.005)

    st.sidebar.header("📄 Vertrag & Kosten")
    capex=st.sidebar.number_input("CAPEX €",10000,2_000_000,350000,5000)
    max_cycles=st.sidebar.number_input("Max Zyklen",1000,15000,6000,100)
    opex=st.sidebar.number_input("Opex €/Jahr",0,50000,5000,500)
    years=st.sidebar.number_input("Vertragsjahre",5,20,10,1)
    disc =st.sidebar.number_input("Diskont %",0.0,15.0,6.0,0.1)/100
    owner_share=st.sidebar.slider("Anteil Standortbesitz (%)",0,50,20)/100

    st.sidebar.header("🔌 Netzentgelt")
    npeak=st.sidebar.number_input("Peak €/MWh",0,200,75,5)
    noff =st.sidebar.number_input("Off-Peak €/MWh",0,200,40,5)

    st.sidebar.header("🕑 Verkauf")
    s_start=st.sidebar.time_input("Start",dt.time(0,0))
    s_end  =st.sidebar.time_input("Ende", dt.time(23,59))

    if st.button("Simulation starten"):
        start=dt.datetime(year,1,1); end=dt.datetime(year,12,31,23)
        with st.spinner("Preise laden …"):
            try: prices=fetch_day_ahead_prices(start,end,api)
            except Exception as e: st.error(f"API-Error {e}"); st.stop()
        try: co2=fetch_co2_intensity(start,end,api)
        except: co2=pd.DataFrame()

        cfg=dict(capacity=cap,connection=conn,closs=closs,dloss=dloss,ev_price=ev_price,
                 sell_start=s_start,sell_end=s_end,capex=capex,max_cycles=max_cycles,opex=opex,
                 years=years,disc=disc,owner_share=owner_share,npeak=npeak,noff=noff)
        df,summary=simulate(prices,co2,cfg)
        st.success("Fertig ✔️")

        k1,k2,k3=st.columns(3)
        k1.metric("NPV €",f"{summary['NPV']:,.0f}")
        k2.metric("IRR %",f"{summary['IRR']*100:,.2f}" if not np.isnan(summary['IRR']) else "n/a")
        k3.metric("Break-Even Jahr", summary['BE'] if summary['BE'] is not None else "> Laufzeit")

        st.subheader("📊 Monats-Cashflow (Jahr 1)")
        m=df.set_index("Datum").resample("M").sum()
        contr_share=(1-owner_share); monthly_contr=(summary['contractor_gain']/12)
        fig,ax=plt.subplots(figsize=(9,4)); ax.bar(m.index.strftime("%b"), m["Gewinn_EUR"]*contr_share, color="tab:blue"); ax.axhline(0,color="black"); st.pyplot(fig)

        st.subheader("🔥 Heatmap Gewinne (Jahr 1)")
        heat=df.copy(); heat["Monat"]=heat["Datum"].dt.month; heat["Tag"]=heat["Datum"].dt.day
        piv=heat.pivot_table(index="Monat",columns="Tag",values="Gewinn_EUR"); fig2,ax2=plt.subplots(figsize=(12,4)); im=ax2.imshow(piv,cmap="RdYlGn",aspect="auto"); ax2.set_yticks(range(12)); ax2.set_yticklabels([dt.date(1900,m,1).strftime("%b") for m in range(1,13)]); st.pyplot(fig2)

        st.dataframe(df)
        st.download_button("CSV",df.to_csv(index=False).encode(),"contractor_detail.csv")

if __name__=="__main__":
    main()
