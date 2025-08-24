from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

out = Path("data/sample"); out.mkdir(parents=True, exist_ok=True)
np.random.seed(7)
hours = pd.date_range(datetime.now(timezone.utc) - timedelta(hours=47), periods=48, freq="H")
base_hs_m = np.clip(np.sin(np.linspace(0, 3*np.pi, len(hours))) * 0.5 + 1.2, 0.2, None)
noise_hs = np.random.normal(0, 0.08, len(hours))
wvht = (base_hs_m + noise_hs).round(2)
dpd  = np.round(np.clip(8 + 2*np.sin(np.linspace(0, 2*np.pi, len(hours))) + np.random.normal(0,0.4,len(hours)), 5, 17), 1)
mwd  = (220 + 10*np.sin(np.linspace(0, 4*np.pi, len(hours)))) % 360
wspd = np.clip(np.random.normal(5, 2, len(hours)), 0, None)
pd.DataFrame({"time_utc": hours, "WVHT": wvht, "DPD": dpd, "MWD": mwd, "WSPD": wspd}).to_csv(out/"sample_ndbc_socal.csv", index=False)
print("Wrote data/sample/sample_ndbc_socal.csv")
