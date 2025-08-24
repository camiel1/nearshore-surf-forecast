from __future__ import annotations
import pandas as pd
import numpy as np

def add_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # WVHT: significant wave height (m) → feet
    if "WVHT" in out.columns:
        out["Hs_m"]  = pd.to_numeric(out["WVHT"], errors="coerce")
        out["Hs_ft"] = out["Hs_m"] * 3.28084
    # WSPD: wind (m/s) → mph & kt
    if "WSPD" in out.columns:
        out["wind_ms"]  = pd.to_numeric(out["WSPD"], errors="coerce")
        out["wind_mph"] = out["wind_ms"] * 2.23694
        out["wind_kt"]  = out["wind_ms"] * 1.94384
    return out

def _compute_period(out: pd.DataFrame) -> pd.DataFrame:
    """Build Tp_s using DPD, else APD, then forward/back-fill and clip."""
    tp = None
    if "DPD" in out.columns:
        tp = pd.to_numeric(out["DPD"], errors="coerce")
    if tp is None or tp.isna().all():
        tp = pd.Series(np.nan, index=out.index)

    if "APD" in out.columns:
        apd = pd.to_numeric(out["APD"], errors="coerce")
        tp = tp.fillna(apd)

    # carry last known period forward/backward to avoid NaN at the latest row
    tp = tp.ffill().bfill()
    # keep in a reasonable ocean swell range
    tp = tp.clip(lower=3, upper=22)
    out["Tp_s"] = tp
    return out

def nearshore_surf_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline ‘surf face’ proxy (ft). Uses Hs (ft) and period; period = DPD->APD->filled.
    """
    out = add_units(df)
    out = _compute_period(out)

    if "Hs_ft" not in out.columns or "Tp_s" not in out.columns:
        return out

    hs = pd.to_numeric(out["Hs_ft"], errors="coerce")
    tp = pd.to_numeric(out["Tp_s"], errors="coerce")

    # gentle period boost: ~0.9x..1.6x across 3..22 s
    period_boost = np.sqrt((tp - 3) / (22 - 3) * 0.8 + 0.8)
    proxy = hs * period_boost * 1.2

    # only keep values where both hs and tp are present
    proxy = proxy.where(hs.notna() & tp.notna())
    out["surf_face_proxy_ft"] = proxy.round(1)
    return out

def ema_forecast(df: pd.DataFrame, horizon_steps: int = 6) -> pd.DataFrame:
    """
    Short-term forecast using EMA of the proxy; falls back to Hs_ft if proxy is all NaN.
    """
    if "surf_face_proxy_ft" not in df.columns:
        df = nearshore_surf_proxy(df)

    out = df.copy().sort_values("time_utc")
    valid = out.dropna(subset=["surf_face_proxy_ft"])
    if valid.empty:
        # fallback: use Hs_ft * 1.2 if available
        if "Hs_ft" in out.columns and out["Hs_ft"].notna().any():
            last_t = out["time_utc"].dropna().iloc[-1]
            last_val = float(out["Hs_ft"].dropna().iloc[-1] * 1.2)
        else:
            hist = out.copy()
            hist["is_forecast"] = False
            return hist
    else:
        y = valid["surf_face_proxy_ft"].astype(float)
        last_t = valid["time_utc"].iloc[-1]
        last_val = float(y.ewm(span=6, adjust=False).mean().iloc[-1])

    future_idx = pd.date_range(last_t + pd.Timedelta(hours=1),
                               periods=horizon_steps, freq="H", tz="UTC")
    fc = pd.DataFrame({
        "time_utc": future_idx,
        "surf_face_proxy_ft": [last_val] * horizon_steps,
        "is_forecast": True
    })
    hist = out.copy()
    hist["is_forecast"] = False
    return pd.concat([hist, fc], ignore_index=True)
