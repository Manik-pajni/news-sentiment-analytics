# News Sentiment Analytics & Market Impact System

Reproducible pipeline that asks whether **unusually high news volume and extreme GDELT tone**, mapped to listed equities, are associated with **next-trading-day abnormal returns**.


```
GDELT event CSVs ──► SQLite news_articles
         │                    │
         └── entity resolution (aliases → RapidFuzz)
                              │
                              ▼
                    company_mentions + unmatched_mentions
                              │
Yahoo Finance adj. close ──► stock_prices (equities + SPY)
                              │
                     SQL daily_metrics + daily_features
                              │
              HIGH_IMPACT events (train 90th percentile)
                              │
              event study ──► backtest (test period only)
                              │
                         Streamlit app
```

## Technology stack

Python 3.9+, SQLite, Pandas, RapidFuzz, yfinance, SciPy, Streamlit.

## Installation

```bash
cd gdelt_news_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## End-to-end commands

First time: download official GDELT daily event files for the **last 7 days**, ingest every CSV already in `data/raw_gdelt`, then refresh prices and analytics. 

```bash
python -m news_analytics.pipeline --fetch
```

A shorter or longer recent window:

```bash
python -m news_analytics.pipeline --fetch --fetch-days 3
```

Recompute on whatever is already local (**does not download**):

```bash
python main.py
```

Initialize schema only (does **not** drop existing rows unless `--reset`):

```bash
python -m news_analytics.pipeline --steps init
```

Tests:

```bash
python -m pytest -q
```

Dashboard:

```bash
streamlit run app.py
```

