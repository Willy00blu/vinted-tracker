"""Background sold-check worker.

A standalone process — symmetric to image_worker and text_worker — that runs
the sold-check drip independently of the scrape monitor. So you can:
  - run the worker alone while the scrape monitor is OFF (still get drops +
    photo_urls + favourites backfill)
  - or both together: the worker uses the same rate-limiter as the monitor,
    so the per-IP budget is naturally shared (no risk of double-load)

Config:
  --db <path>          override shared DB path
  --catalog <id,id,…>  scope to these catalog_ids (default: all)
  --order oldest|favourites|price   sold-check priority order

Run:  python -m core.sold_check_worker
      python -m core.sold_check_worker --once
"""
from __future__ import annotations

import argparse
import threading

from .category_db import shared_db_path
from .sold_check import run_drip


def main():
    ap = argparse.ArgumentParser(description="Vinted sold-check worker")
    ap.add_argument("--db", default=None, help="override DB path")
    ap.add_argument("--catalog", default=None,
                    help="comma-separated catalog_ids to scope the drip")
    ap.add_argument("--order", default="no_photo_first",
                    choices=["oldest", "favourites", "price", "no_photo_first"],
                    help="sold-check priority order (default: no_photo_first, "
                         "which prefers items missing photo_urls — legacy items "
                         "scraped before the photo-capture feature. Self-disables "
                         "once every item has photos: the first sort key becomes "
                         "uniform and the order degrades to oldest-checked first)")
    ap.add_argument("--once", action="store_true",
                    help="exit after one batch (testing)")
    args = ap.parse_args()

    from pathlib import Path
    db = Path(args.db) if args.db else shared_db_path()
    catalog_ids = None
    if args.catalog:
        catalog_ids = [int(x) for x in args.catalog.split(",") if x.strip()]

    print(f"sold_check_worker — started · DB: {db.name} · "
          f"catalog: {catalog_ids or 'all'} · order: {args.order}", flush=True)

    stop = threading.Event()
    if args.once:
        # One-shot: schedule stop after a single short pass.
        import time as _t
        def _stop_after():
            _t.sleep(2)
            stop.set()
        threading.Thread(target=_stop_after, daemon=True).start()

    try:
        run_drip(db, catalog_ids=catalog_ids, order=args.order, stop_event=stop)
    except KeyboardInterrupt:
        print("\nsold_check_worker fermato.")


if __name__ == "__main__":
    main()
