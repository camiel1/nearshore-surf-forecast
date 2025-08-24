
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
        
# normalize and sort
# after we build `df` and `source`
df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
df = (
    df.dropna(subset=["time_utc"])
      .drop_duplicates(subset=["time_utc"])
      .sort_values("time_utc")        # always ASC for processing
)


# st.caption(f"Source: {source}")
# st.dataframe(df.tail(10), use_container_width=True)
rows_to_show = 12  # tweak if you like (or make it a sidebar slider)
recent = (
    df.sort_values("time_utc", ascending=False)  # newest first
      .head(rows_to_show)                        # take N newest
      .sort_values("time_utc")                   # show them in time order
)
st.caption(f"Source: {source} — showing last {len(recent)} rows by time (UTC)")
st.dataframe(recent, use_container_width=True)



# build features + forecast
# df_proxy = nearshore_surf_proxy(df)
df_proxy = nearshore_surf_proxy(df)
df_proxy_sorted = df_proxy.sort_values("time_utc")

horizon = st.sidebar.slider("Forecast horizon (hours)", 1, 12, 6)
df_fc = make_baseline_forecast(df_proxy, horizon_steps=horizon)

def latest_nonnull(col):
    s = df_proxy_sorted.dropna(subset=[col])
    return float(s.iloc[-1][col]) if not s.empty else float("nan")

surf_val = latest_nonnull("surf_face_proxy_ft")
hs_val   = latest_nonnull("Hs_ft")
tp_val   = latest_nonnull("Tp_s")
wind_val = latest_nonnull("wind_mph")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Surf face proxy", f"{surf_val:0.1f} ft" if not pd.isna(surf_val) else "—")
c2.metric("Hs (sig.)",       f"{hs_val:0.1f} ft"   if not pd.isna(hs_val) else "—")
c3.metric("Tp (dominant)",   f"{tp_val:0.0f} s"    if not pd.isna(tp_val) else "—")
c4.metric("Wind",            f"{wind_val:0.0f} mph"if not pd.isna(wind_val) else "—")

# chart — drop NaNs from history
st.subheader("History & Forecast")
fig, ax = plt.subplots(figsize=(10,4))
hist = df_fc[(df_fc["is_forecast"] == False) & df_fc["surf_face_proxy_ft"].notna()]
fut  = df_fc[(df_fc["is_forecast"] == True)  & df_fc["surf_face_proxy_ft"].notna()]
if not hist.empty:
    ax.plot(hist["time_utc"], hist["surf_face_proxy_ft"], label="History (proxy ft)")
if not fut.empty:
    ax.plot(fut["time_utc"], fut["surf_face_proxy_ft"], linestyle="--", label="Forecast (EMA)")
ax.set_ylabel("Surf face (ft)")
ax.set_xlabel("UTC time")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig, clear_figure=True)

if hist.empty:
    st.info("No valid history yet—this buoy reports missing period (`MM`) often. We now fall back to APD and carry forward values; new data points will fill in soon.")


st.markdown("Next: add tide & wind-shelter adjustments, CDIP labels, and an ML model.")
