"""Recompute on local data only (does not download). Prefer: python -m news_analytics.pipeline --fetch"""

from news_analytics.pipeline import main as pipeline_main


if __name__ == "__main__":
    pipeline_main()
