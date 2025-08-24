# nearshore-surf-forecast

MVP Streamlit app that ingests NOAA NDBC buoy data, computes a simple near-shore **surf face proxy (ft)**, and shows a short-term **EMA forecast**.

## Quickstart
```bash
# create env
python -m venv .venv
# win
.venv\Scripts\Activate.ps1
# mac/linux
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/make_sample_data.py
streamlit run src/app/streamlit_app.py
