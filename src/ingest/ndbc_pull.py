from __future__ import annotations
import sys
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime, timezone

REALTIME_URL = "https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"

def parse_ndbc_realtime(txt: str) -> pd.DataFrame:
    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise ValueError("No data lines found.")
    header = lines[0].split()
    rows = [ln.split() for ln in lines[1:]]
    df = pd.DataFrame(rows, columns=header)
    if all(k in df.columns for k in ["YY","MM","DD","hh","mm"]):
        comp = dict(year=df["YY"].astype(int)+2000,
                    month=df["MM"].astype(int),
                    day=df["DD"].astype(int),
                    hour=df["hh"].astype(int),
                    minute=df["mm"].astype(int))
    else:
        comp = dict(year=df["YY"].astype(int)+2000,
                    month=df["MM"].astype(int),
                    day=df["DD"].astype(int),
                    hour=df["hh"].astype(int),
                    minute=0)
    ts = pd.to_datetime(comp, utc=True, errors="coerce")
    df.insert(0, "time_utc", ts)
    for col in df.columns:
        if col != "time_utc":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["time_utc"]).reset_index(drop=True)

def fetch_and_save(buoy_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = REALTIME_URL.format(station=buoy_id)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    df = parse_ndbc_realtime(resp.text)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = out_dir / f"ndbc_{buoy_id}_realtime_{ts}.csv"
    df.to_csv(p, index=False)
    return p

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.ndbc_pull <BUOY_ID> [out_dir]", file=sys.stderr)
        sys.exit(1)
    buoy = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("data/raw")
    path = fetch_and_save(buoy, out)
    print(f"Wrote {path}")
