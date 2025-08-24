from __future__ import annotations
import sys
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime, timezone

REALTIME_URL = "https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"

def parse_ndbc_realtime(txt: str) -> pd.DataFrame:
    """
    Parse NDBC realtime2 *.txt into a tidy DataFrame.

    Robust to:
      - Header line starting with '#'
      - Year as 'YY' (2-digit) or 'YYYY' (4-digit)
      - Optional minute column 'mm'
      - Extra commented/unit lines before/after header
    """
    lines = [ln.rstrip("\n") for ln in txt.splitlines()]
    if not lines:
        raise ValueError("Empty NDBC response.")

    # 1) Locate the header line (allow it to start with '#')
    header_idx = None
    header_tokens = None
    KNOWN_DATA_COLS = {"WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD", "PRES", "ATMP", "WTMP", "DEWP", "VIS", "PTDY", "TIDE"}
    for i, ln in enumerate(lines[:80]):  # look near the top
        s = ln.strip()
        if not s:
            continue
        toks = s.lstrip("#").split()
        has_date = (("YY" in toks or "YYYY" in toks) and "MM" in toks and "DD" in toks and ("hh" in toks or "HH" in toks))
        has_data = any(t in toks for t in KNOWN_DATA_COLS)
        if has_date and has_data:
            header_idx = i
            header_tokens = toks
            break
    if header_idx is None:
        raise ValueError("Could not find NDBC header line — unexpected file format.")

    # 2) Collect data rows below the header; skip blank/comment lines
    data_rows = []
    for ln in lines[header_idx + 1:]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if len(toks) == len(header_tokens):
            data_rows.append(toks)
    if not data_rows:
        raise ValueError("No data rows found after header.")

    # 3) Build DataFrame
    df = pd.DataFrame(data_rows, columns=header_tokens)

    # 4) Identify time columns
    cols = set(df.columns)
    year_col   = "YY" if "YY" in cols else ("YYYY" if "YYYY" in cols else None)
    month_col  = "MM" if "MM" in cols else None
    day_col    = "DD" if "DD" in cols else None
    hour_col   = "hh" if "hh" in cols else ("HH" if "HH" in cols else None)
    minute_col = "mm" if "mm" in cols else None  # optional
    if not (year_col and month_col and day_col and hour_col):
        raise ValueError(f"Unexpected NDBC time columns: {list(df.columns)}")

    # 5) Build UTC timestamp (correctly handle 2-digit vs 4-digit years)
    years_raw = pd.to_numeric(df[year_col], errors="coerce")
    if year_col == "YY":
        # If values look like 4-digit years (>=100), keep as-is. Otherwise pivot 2-digit to 2000+.
        if (years_raw >= 100).any():
            years = years_raw
        else:
            years = years_raw + 2000
    else:
        years = years_raw

    months  = pd.to_numeric(df[month_col], errors="coerce")
    days    = pd.to_numeric(df[day_col], errors="coerce")
    hours   = pd.to_numeric(df[hour_col], errors="coerce")
    minutes = pd.to_numeric(df[minute_col], errors="coerce") if minute_col else 0

    ts = pd.to_datetime(
        dict(year=years, month=months, day=days, hour=hours, minute=minutes),
        errors="coerce", utc=True
    )
    df.insert(0, "time_utc", ts)

    # 6) Convert other columns to numeric where possible
    for col in df.columns:
        if col == "time_utc":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 7) Clean
    df = df.dropna(subset=["time_utc"]).reset_index(drop=True)
    return df




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
