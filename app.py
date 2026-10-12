"""Streamlit dashboard. Launch: streamlit run app.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from news_analytics.config import load_settings
from news_analytics.reporting import mapping_stats, overview, q

st.set_page_config(page_title="News Sentiment Analytics", layout="wide")


def _db() -> Path:
    return load_settings().database_path


def _empty(msg: str) -> None:
    st.info(msg)


def _pct(val) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return f"{val:.2%}"


def page_overview():
    st.header("Overview")
    st.write(
        "This dashboard reads the SQLite database the pipeline fills. "
        "Use it to check coverage, how data landed in the DB, then browse news and one ticker. "
        "Market numbers are on the last page."
    )
    if not _db().exists():
        _empty("Database not found. Run `python -m news_analytics.pipeline --fetch` first.")
        return
    stats = overview(_db())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles", stats["n_articles"])
    c2.metric("Companies mapped", stats["n_tickers_mapped"])
    c3.metric("News from", stats["date_min"] or "n/a")
    c4.metric("News to", stats["date_max"] or "n/a")
    st.caption(f"Last pipeline step: {stats['last_run'] or 'n/a'}. Price rows stored: {stats['n_price_rows']}.")
    with st.expander("How this is calculated"):
        st.write(
            "These counts come from `news_articles`, `company_mentions`, and `pipeline_runs`. "
            "They are live SQL totals, not hardcoded."
        )


def page_pipeline():
    st.header("Pipeline / Data")
    st.caption(
        "How rows got into SQLite: each pipeline step is logged, plus ingest errors, "
        "names we could not map, and a duplicate-article check."
    )
    if not _db().exists():
        _empty("No database.")
        return
    stats = mapping_stats(_db())
    cov = stats["coverage"]
    cov_txt = "n/a" if cov is None else f"{cov:.0%}"
    st.write(
        f"Unique mentions seen: **{stats['unique_mentions']}**. "
        f"Mapped: **{stats['mapped']}**. Unmatched: **{stats['unmatched']}**. "
        f"Match rate: **{cov_txt}**."
    )
    with st.expander("How this is calculated"):
        st.write(
            "GDELT CSVs go into `news_articles`. Actor names are matched to tickers. "
            "Misses go to `unmatched_mentions`. Yahoo fills `stock_prices`. "
            "Each step writes a row in `pipeline_runs`."
        )

    runs = q(_db(), "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT 20")
    st.subheader("Recent pipeline runs")
    if runs.empty:
        _empty("No pipeline runs recorded yet.")
    else:
        st.dataframe(runs, use_container_width=True)

    unmatched = q(
        _db(),
        "SELECT * FROM unmatched_mentions ORDER BY mention_count DESC LIMIT 200",
    )
    st.subheader("Unmatched mentions")
    if unmatched.empty:
        _empty("No unmatched mentions stored.")
    else:
        st.dataframe(unmatched, use_container_width=True)

    errors = q(_db(), "SELECT * FROM ingestion_errors ORDER BY id DESC LIMIT 100")
    st.subheader("Ingestion errors")
    if errors.empty:
        _empty("No ingestion errors logged.")
    else:
        st.dataframe(errors, use_container_width=True)

    dups = q(
        _db(),
        """
        SELECT article_id, COUNT(*) AS n FROM news_articles
        GROUP BY article_id HAVING n > 1
        """,
    )
    st.subheader("Duplicate article IDs")
    if dups.empty:
        st.success("No duplicate article_id values (primary key enforced).")
    else:
        st.dataframe(dups)


def page_news():
    st.header("News Explorer")
    st.caption("Filter mapped articles by ticker, date, and sentiment, then scan the table.")
    if not _db().exists():
        _empty("No database.")
        return
    df = q(
        _db(),
        """
        SELECT na.publish_date, cm.company AS ticker, t.sector, na.sentiment, na.tone,
               cm.match_method, cm.match_confidence, na.source, na.actor1
        FROM news_articles na
        JOIN company_mentions cm ON na.article_id = cm.article_id
        LEFT JOIN tickers t ON t.ticker = cm.company
        """,
    )
    if df.empty:
        _empty("No mapped news yet. Run `python -m news_analytics.pipeline --fetch` to download recent GDELT files.")
        return
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    tickers = ["(all)"] + sorted(df["ticker"].dropna().unique().tolist())
    min_d = df["publish_date"].min()
    max_d = df["publish_date"].max()
    c1, c2, c3 = st.columns(3)
    ticker = c1.selectbox("Ticker", tickers)
    if pd.notna(min_d) and pd.notna(max_d):
        picked = c2.date_input(
            "Date range",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
        )
    else:
        picked = None
    sent = c3.slider("Sentiment range", -1.0, 1.0, (-1.0, 1.0))
    view = df.copy()
    if ticker != "(all)":
        view = view[view["ticker"] == ticker]
    if picked is not None:
        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            start, end = picked
        else:
            start = end = picked
        view = view[
            (view["publish_date"] >= pd.Timestamp(start))
            & (view["publish_date"] <= pd.Timestamp(end))
        ]
    view = view[(view["sentiment"] >= sent[0]) & (view["sentiment"] <= sent[1])]
    st.dataframe(view.sort_values("publish_date", ascending=False).head(500), use_container_width=True)
    with st.expander("How this is calculated"):
        st.write(
            "Sentiment is GDELT AvgTone / 10, clipped to [-1, 1]. "
            "Only articles that matched a ticker are listed."
        )


def page_ticker():
    st.header("Ticker")
    st.caption("Sentiment and news volume over time for one company, plus the events we flagged.")
    if not _db().exists():
        _empty("No database.")
        return
    feat = q(_db(), "SELECT * FROM daily_features")
    if feat.empty:
        _empty("No daily features. Run aggregation after ingesting news and prices.")
        return
    ticker = st.selectbox("Ticker", sorted(feat["ticker"].unique().tolist()))
    sub = feat[feat["ticker"] == ticker].sort_values("date")
    st.subheader("Sentiment")
    st.line_chart(sub.set_index("date")[["mean_sentiment", "rolling_sent_5"]])
    st.subheader("News volume")
    st.bar_chart(sub.set_index("date")["article_count"])
    ev = q(_db(), "SELECT * FROM market_events WHERE company = ?", (ticker,))
    st.subheader("Events")
    if ev.empty:
        _empty("No events for this ticker.")
    else:
        st.dataframe(ev, use_container_width=True)
    with st.expander("How this is calculated"):
        st.write(
            "Daily sentiment is the mean of that day's articles. "
            "Volume is the article count. Events come from the pipeline (unusual volume plus tone)."
        )


def page_market():
    st.header("Market results")
    st.caption(
        "Did high-impact news days move more than the market the next session, "
        "and how did a simple 1-day long-only book do?"
    )
    if not _db().exists():
        _empty("No database.")
        return

    res = q(_db(), "SELECT * FROM event_study_results")
    st.subheader("Event study")
    usable = (
        res[(res["split"] != "none") & (res["window_name"] != "none")]
        if not res.empty
        else res
    )
    if res.empty or usable.empty:
        note = None
        if not res.empty and "notes" in res.columns and res["notes"].notna().any():
            note = str(res["notes"].dropna().iloc[0])
        _empty(note or "Event-study results are empty. The pipeline may lack news, prices, or events.")
    else:
        test = usable[(usable["split"] == "test") & (usable["window_name"] == "next_day")]
        if test.empty:
            _empty("No test-period next-day event-study row yet.")
        else:
            row = test.iloc[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("High-impact events (test)", int(row["n_events"] or 0))
            k2.metric("Average next-day extra return vs market", _pct(row["mean_ar"]))
            k3.metric("Vs other news days", _pct(row["diff_mean"]))
        extra_cols = [
            c
            for c in (
                "split",
                "window_name",
                "n_events",
                "n_tickers",
                "mean_ar",
                "diff_mean",
                "ci_low",
                "ci_high",
                "welch_t",
                "welch_p",
                "mwu_p",
                "notes",
            )
            if c in usable.columns
        ]
        with st.expander("Extra event-study stats"):
            st.dataframe(usable[extra_cols], use_container_width=True)
        obs = q(
            _db(),
            """
            SELECT * FROM event_study_observations
            WHERE window_name='next_day' AND is_event=1 AND split='test'
            """,
        )
        if not obs.empty:
            st.bar_chart(obs.set_index("event_date")["abnormal_return"])

    summary = q(_db(), "SELECT * FROM backtest_summary ORDER BY id DESC LIMIT 5")
    daily = q(_db(), "SELECT * FROM backtest_daily ORDER BY date")
    trades = q(_db(), "SELECT * FROM backtest_trades ORDER BY trade_date")
    st.subheader("Simple backtest")
    if summary.empty:
        _empty("No backtest results. This is expected when there are no test-period signals or prices.")
    else:
        row = summary.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy total return", _pct(row["total_return"]))
        c2.metric("Market (benchmark) return", _pct(row["benchmark_total_return"]))
        c3.metric("Trades", int(row["n_trades"] or 0))
        st.caption(
            f"Costs: {row['transaction_cost_bps']} bps + {row['slippage_bps']} bps slippage."
        )
        with st.expander("Extra backtest stats"):
            st.write(
                {
                    "Sharpe (annualised, rf=0)": None
                    if pd.isna(row["sharpe"])
                    else float(f"{row['sharpe']:.2f}"),
                    "Max drawdown": _pct(row["max_drawdown"]),
                    "Days": row["n_days"],
                }
            )
            if row.get("notes"):
                st.write(row["notes"])
        if not daily.empty:
            st.line_chart(daily.set_index("date")[["cumulative_return", "benchmark_cumulative"]])
        if trades.empty:
            _empty("No trades in the test window.")
        else:
            st.dataframe(trades, use_container_width=True)

    with st.expander("How this is calculated"):
        st.write(
            "Extra return is stock minus market on the next session after the news date. "
            "The backtest goes long high-impact names with positive sentiment for one day, after costs."
        )


def main():
    st.sidebar.title("News Sentiment Analytics")
    page = st.sidebar.radio(
        "Section",
        [
            "Overview",
            "Pipeline / Data",
            "News Explorer",
            "Ticker",
            "Market results",
        ],
    )
    {
        "Overview": page_overview,
        "Pipeline / Data": page_pipeline,
        "News Explorer": page_news,
        "Ticker": page_ticker,
        "Market results": page_market,
    }[page]()


if __name__ == "__main__":
    main()
