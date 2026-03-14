#!/usr/bin/env python3
"""
scripts/fetch_data.py
=====================
Standalone script to download all external data sources.

Usage
-----
    python scripts/fetch_data.py
    python scripts/fetch_data.py --app-token YOUR_SOCRATA_TOKEN
    python scripts/fetch_data.py --fbi-key  YOUR_FBI_API_KEY
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.python.data.fetcher import fetch_all
from src.python.utils import get_logger

log = get_logger("fetch_data")


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch all Texas crime data sources")
    p.add_argument("--app-token", default=None,
                   help="Socrata application token for data.texas.gov")
    p.add_argument("--fbi-key",   default="DEMO_KEY",
                   help="FBI Crime Data Explorer API key")
    args = p.parse_args()

    log.info("Starting data fetch…")
    results = fetch_all(app_token=args.app_token, fbi_api_key=args.fbi_key)

    log.info("\n── Downloaded files ──")
    for name, path in results.items():
        log.info("  %-25s  %s", name, path)

    log.info("Done.  Files stored in data/raw/ and data/shapefiles/")


if __name__ == "__main__":
    main()
