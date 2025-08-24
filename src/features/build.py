from __future__ import annotations
import pandas as pd
import numpy as np

def add_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "WVHT" in out.columns:
        out["Hs_m"] = out["WVHT"].astype(float)
        out["Hs_ft"] = out["Hs_m"] * 3.28084
    if "DPD" in out.columns:
        out["Tp_s"] = out["DPD"].astype(float)
    if "MWD" in out.columns:
        out["Mwdir_deg"] = out["MWD"].astype(float)
    if "WSPD" in out.columns:
        out["wind_ms"] = out["WSPD"].astype(float)
        out["wind_mph"] = out["wind_ms"] * 2.23694
        out["wind_kt"] = out["wind_ms"] * 1.94384
    return out

def nearshore_surf_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = add_units(df)
    if "Hs_ft" not in out or "Tp_s" not in out:
        return out
    tp = out["Tp_s"].clip(lower=3, upper=22)
    hs = out["Hs_ft"].clip(lower=0.0)
    period_boost = np.sqrt((tp - 3) / (22 - 3) * 0.8 + 0.8)
    out["surf_face_proxy_ft"] = (hs * period_boost * 1.2).round(1)
    return out

def ema_forecast(df: pd.DataFrame, horizon_steps: int = 6) -> pd.DataFrame:
    if "surf_face_proxy_ft" not in df.columns:
        df = nearshore_surf_proxy(df)
    out = df.copy().sort_values("time_utc")
    y = out["surf_face_proxy_ft"].astype(float)
    trend = y.ewm(span=6, adjust=False).mean()
    last_t = out["time_utc"].iloc[-1]
    last_val = float(trend.iloc[-1])
    future_idx = pd.date_range(last_t + pd.Timedelta(hours=1),
                               periods=horizon_steps, freq="H", tz="UTC")
    fc = pd.DataFrame({"time_utc": future_idx,
                       "surf_face_proxy_ft": [last_val]*horizon_steps,
                       "is_forecast": True})
    hist = out.copy()
    hist["is_forecast"] = False
    return pd.concat([hist, fc], ignore_index=True)
