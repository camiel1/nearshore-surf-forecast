
from __future__ import annotations
import sys
# from pathlib import Path
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from src.features.build import nearshore_surf_proxy
from src.models.baseline import make_baseline_forecast

st.set_page_config(page_title="Nearshore Surf Forecast (MVP)", page_icon="🌊", layout="wide")
st.title("🌊 Nearshore Surf Forecast — MVP")

def load_sample() -> pd.DataFrame:
    sample_path = Path("data/sample/sample_ndbc_socal.csv")
    return pd.read_csv(sample_path, parse_dates=["time_utc"])

def find_latest_raw() -> Path | None:
    d = Path("data/raw")
    if not d.exists():
        return None
    paths = sorted(d.glob("ndbc_*_realtime_*.csv"))
    return paths[-1] if paths else None

mode = st.sidebar.radio("Data source", ["Auto", "Sample only", "Latest raw (if any)"], index=0)

if mode == "Sample only":
    df = load_sample(); source = "Sample"
elif mode == "Latest raw (if any)":
    latest = find_latest_raw()
    if latest is not None:
        df = pd.read_csv(latest, parse_dates=["time_utc"]); source = f"Raw: {latest.name}"
    else:
        df = load_sample(); source = "Sample (fallback; no raw found)"
else:
    latest = find_latest_raw()
    if latest is not None:
        df = pd.read_csv(latest, parse_dates=["time_utc"]); source = f"Raw: {latest.name}"
    else:
        df = load_sample(); source = "Sample"

st.caption(f"Source: {source}")
st.dataframe(df.tail(10), use_container_width=True)

df_proxy = nearshore_surf_proxy(df)
horizon = st.sidebar.slider("Forecast horizon (hours)", 1, 12, 6)
df_fc = make_baseline_forecast(df_proxy, horizon_steps=horizon)

latest_row = df_proxy.sort_values("time_utc").iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Surf face proxy", f"{latest_row.get('surf_face_proxy_ft', float('nan')):0.1f} ft")
c2.metric("Hs (sig.)", f"{latest_row.get('Hs_ft', float('nan')):0.1f} ft")
c3.metric("Tp (dominant)", f"{latest_row.get('Tp_s', float('nan')):0.0f} s")
c4.metric("Wind", f"{latest_row.get('wind_mph', float('nan')):0.0f} mph")

st.subheader("History & Forecast")
fig, ax = plt.subplots(figsize=(10,4))
hist = df_fc[df_fc["is_forecast"] == False]
fut  = df_fc[df_fc["is_forecast"] == True]
ax.plot(hist["time_utc"], hist["surf_face_proxy_ft"], label="History (proxy ft)")
if not fut.empty:
    ax.plot(fut["time_utc"], fut["surf_face_proxy_ft"], linestyle="--", label="Forecast (EMA)")
ax.set_ylabel("Surf face (ft)")
ax.set_xlabel("UTC time")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig, clear_figure=True)

st.markdown("Next: add tide & wind-shelter adjustments, CDIP labels, and an ML model.")
