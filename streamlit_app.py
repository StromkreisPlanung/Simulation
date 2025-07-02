# streamlit_app.py  –  Contractor-Simulator  v4.6.1
# -------------------------------------------------
# • Neuer ENTSO-E-Endpoint  https://web-api.tp.entsoe.eu/api
# • Bearer-Header-Auth  +  Whitespace-Trim des Tokens
# • Token-Check im UI  → klare Fehlermeldung statt 401
# • Bidding-Zone-Auswahl  (DE, AT, FR, NL, CH …)
# • Inflation, THG-Bonus, Kreditplan, KPIs

import os, datetime as dt
import numpy as np, pandas as pd
import requests, requests_cache
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1)  Transparency-API  –  Cache + Retry + Bearer
# ------------------------------------------------------------------
BASE_URL = "https://web-api.tp.entsoe.eu/api"

requests_cache.install_cache("/tmp/entsoe_cache.sqlite", expire_after=60*60*24)
session = requests_cache.CachedSession()
from requests.adapters import HTTPAdapter, Retry
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))

def _get_xml(params: dict, token: str) -> str:
    """GET-Request mit Bearer-Header.
    Leer-/Zeilenumbrüche im Token werden entfernt."""
    token_clean = token.strip().split()[0]            # nur erstes Token-Wort
    headers = {"Authorization": f"Bearer {token_clean}"}
    r = session.get(BASE_URL, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.text

def fetch_day_ahead(start: dt.datetime, end: dt.datetime, token: str, zone: str) -> pd.DataFrame:
    p = dict(documentType="A44", processType="A01",
             in_Domain=zone, out_Domain=zone,
             periodStart=start.strftime("%Y%m%d%H%M"),
             periodEnd=end.strftime("%Y%m%d%H%M"))
    df = pd.read_xml(_get_xml(p, token), xpath="//Point")
    df["Zeit"] = pd.to_datetime(df["position"].astype(int)-1, unit="h", origin=start)
    df.rename(columns={"price.amount": "Preis_EUR_MWh"}, inplace=True)
    return df[["Zeit", "Preis_EUR_MWh"]]

# ------------------------------------------------------------------
# 2)  Kredit-Amortisations­plan
# ------------------------------------------------------------------
def amort_schedule(principal: float, rate: float, years: int) -> pd.DataFrame:
    if principal <= 0 or years <= 0:
        return pd.DataFrame()
    ann = principal * (rate*(1+rate)**years) / ((1+rate)**years - 1)
    rows, bal = [], principal
    for y in range(1, years+1):
        interest = bal * rate
        repay    = ann - interest
        bal      = max(bal - repay, 0)
        rows.append(dict(Jahr=y, Zins=interest, Tilgung=repay,
                         Rate=ann, Restschuld=bal))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# 3)  Haupt-Simulation  (unverändert)
# ------------------------------------------------------------------
def simulate(prices: pd.DataFrame, cfg: dict):
    prices = prices.copy()
    prices["Datum"] = prices["Zeit"].dt.date

    def in_window(t):  # Verkaufsfenster
        if cfg["sell_start"] < cfg["sell_end"]:
            return cfg["sell_start"] <= t <= cfg["sell_end"]
        return t >= cfg["sell_start"] or t <= cfg["sell_end"]

    prices["Sell"] = prices["Zeit"].dt.time.apply(in_window)
    need = int(np.ceil(cfg["cap_MWh"] / cfg["conn_MW"]))
    cycles, rec = 0, []

    for d, g in prices.groupby("Datum"):
        load = g.nsmallest(need, "Preis_EUR_MWh")
        sell = g[g["Sell"]]
        if sell.empty:
            continue
        buy = load["Preis_EUR_MWh"].mean()
        sell_p = (cfg["ev_price"] + cfg["thg_bonus"]) * 1_000
        e_in  = cfg["cap_MWh"] * (1 + cfg["loss_in"])
        e_out = cfg["cap_MWh"] * (1 - cfg["loss_out"])
        peak  = sell["Zeit"].dt.hour.between(8, 20).mean()
        net_c = e_in * (peak*cfg["net_peak"] + (1-peak)*cfg["net_off"])
        profit = e_out*sell_p - e_in*buy - net_c
        cycles += 1
        rec.append(dict(Datum=pd.to_datetime(d), Gewinn=profit))

    df = pd.DataFrame(rec)

    deg = cycles * (cfg["capex"]/cfg["max_cycles"])
    opex0 = cfg["opex"]
    gross0 = df["Gewinn"].sum()
    net0   = gross0 - deg - opex0

    if cfg["fin_mode"] == "Kredit":
        sched = amort_schedule(cfg["loan_principal"], cfg["loan_rate"], cfg["loan_years"])
        equity = cfg["capex"] - cfg["loan_principal"]
    else:
        sched = pd.DataFrame(); equity = cfg["capex"]

    cash = [-equity]
    for y in range(1, cfg["years"]+1):
        gross_y = gross0 * (1+cfg["infl_ev"])**y
        opex_y  = opex0  * (1+cfg["infl_op"])**y
        net_y   = gross_y - deg - opex_y
        rate_y  = sched.loc[sched["Jahr"]==y,"Rate"].iloc[0] if not sched.empty and y<=cfg["loan_years"] else 0
        cash.append((net_y - rate_y) * (1 - cfg["share_site"]))

    npv = sum(cf/((1+cfg["discount"])**t) for t,cf in enumerate(cash))
    irr = np.irr(cash) if len(cash)>1 else np.nan
    be  = next((t for t,c in enumerate(np.cumsum(cash)) if c>=0), None)
    return df, dict(NPV=npv, IRR=irr, BE=be, schedule=sched, cycles=cycles)

# ------------------------------------------------------------------
# 4)  Streamlit Frontend
# ------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Contractor v4.6.1", layout="wide")
    st.title("🔋 Contractor-Simulator – Bidding-Zone & Token-Check")

    token = st.sidebar.text_input("Bearer-Token", os.getenv("ENTSOE_API_KEY",""), type="password")

    zones = {
        "DE-LU (Deutschland)": "10Y1001A1001A83",
        "AT (Österreich)":     "10YAT-APG------L",
        "FR (Frankreich)":     "10YFR-RTE------C",
        "NL (Niederlande)":    "10YNL----------L",
        "CH (Schweiz)":        "10YCH-SWISSGRIDC",
    }
    z_label = st.sidebar.selectbox("Bidding-Zone", list(zones.keys()), 0)
    z_id    = zones[z_label]

    year = st.sidebar.selectbox("Startjahr", [2023,2024,2025], 0)

    # technische + finanzielle Eingaben (wie zuvor) -----------------
    st.sidebar.header("🔋 Batterie")
    cap  = st.sidebar.number_input("Speicher (MWh)",0.5,20.0,3.5,0.1)
    conn = st.sidebar.number_input("Netzanschluss (MW)",0.1,5.0,0.35,0.05)
    lin  = st.sidebar.slider("Ladeverluste %",0,20,5)/100
    lout = st.sidebar.slider("Entladeverluste %",0,20,5)/100

    st.sidebar.header("🚗 Preis & THG")
    ev   = st.sidebar.number_input("EV-Preis €/kWh",0.10,1.00,0.285,0.005)
    infl_ev = st.sidebar.slider("Inflation EV-Preis %/a",0.0,10.0,2.0)/100
    thg  = st.sidebar.number_input("THG-Bonus €/kWh",0.0,0.5,0.20,0.01)

    st.sidebar.header("⚙️ Betriebskosten")
    opex = st.sidebar.number_input("Opex €/Jahr",0,50_000,5_000,500)
    infl_op = st.sidebar.slider("Inflation Opex %/a",0.0,10.0,2.0)/100

    st.sidebar.header("💰 CAPEX & Finanzierung")
    capex = st.sidebar.number_input("CAPEX €",10_000,2_000_000,350_000,5_000)
    max_cy = st.sidebar.number_input("Max. Zyklen",1_000,15_000,6_000,100)
    fin_mode = st.sidebar.selectbox("Finanzierung",["Eigenkapital","Kredit"],0)
    if fin_mode=="Kredit":
        share=st.sidebar.slider("FK-Anteil %",10,100,70)
        loan_p = capex*share/100
        loan_r = st.sidebar.number_input("Zins % p.a.",0.0,15.0,4.0,0.1)/100
        loan_y = st.sidebar.number_input("Laufzeit J",1,20,10,1)
    else:
        loan_p=loan_r=loan_y=0

    st.sidebar.header("📑 Vertrag")
    years=st.sidebar.number_input("Vertragsjahre",5,20,10,1)
    disc =st.sidebar.number_input("Diskont % p.a.",0.0,15.0,6.0)/100
    share_site=st.sidebar.slider("Gewinn Standort %",0,50,20)/100

    st.sidebar.header("🔌 Netzentgelt")
    npk = st.sidebar.number_input("Peak €/MWh",0,200,75,5)
    nof = st.sidebar.number_input("Off-Peak €/MWh",0,200,40,5)

    st.sidebar.header("🕑 Verkauf")
    s_start = st.sidebar.time_input("Start",dt.time(0,0))
    s_end   = st.sidebar.time_input("Ende", dt.time(23,59))

    # ---------- Simulation ----------
    if st.button("Simulation starten"):
        if not token.strip():
            st.error("Bitte gültigen Bearer-Token eingeben (Sidebar).")
            st.stop()

        start,end = dt.datetime(year,1,1), dt.datetime(year,12,31,23)
        try:
            st.spinner("Preise laden …")
            prices = fetch_day_ahead(start,end,token,z_id)
        except requests.HTTPError as e:
            st.error(f"HTTP {e.response.status_code} – {e.response.reason}")
            st.stop()

        cfg=dict(cap_MWh=cap, conn_MW=conn, loss_in=lin, loss_out=lout,
                 ev_price=ev, infl_ev=infl_ev, thg_bonus=thg,
                 opex=opex, infl_op=infl_op,
                 capex=capex, max_cycles=max_cy,
                 fin_mode=fin_mode, loan_principal=loan_p, loan_rate=loan_r, loan_years=loan_y,
                 years=years, discount=disc, share_site=share_site,
                 net_peak=npk, net_off=nof, sell_start=s_start, sell_end=s_end)

        df,summ = simulate(prices,cfg)

        st.success("✅ Simulation fertig")
        st.caption(f"Marktgebiet: **{z_label}**")
        k1,k2,k3 = st.columns(3)
        k1.metric("NPV €",f"{summ['NPV']:,.0f}")
        k2.metric("IRR %",f"{summ['IRR']*100:,.2f}" if not np.isnan(summ['IRR']) else "n/a")
        k3.metric("Break-Even",summ['BE'] if summ['BE'] is not None else "> Laufzeit")

        if not summ["schedule"].empty:
            st.subheader("📑 Darlehensplan")
            st.dataframe(summ["schedule"].style.format({c:"{:,.0f}" for c in ["Zins","Tilgung","Rate","Restschuld"]}))

        st.subheader("Tagesgewinne Jahr 0")
        st.dataframe(df)
        st.download_button("CSV herunterladen",
                           df.to_csv(index=False).encode(),
                           file_name="contractor_detail.csv")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
