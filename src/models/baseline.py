from __future__ import annotations
import pandas as pd
from src.features.build import nearshore_surf_proxy, ema_forecast

def make_baseline_forecast(df: pd.DataFrame, horizon_steps: int = 6) -> pd.DataFrame:
    df2 = nearshore_surf_proxy(df)
    return ema_forecast(df2, horizon_steps=horizon_steps)
