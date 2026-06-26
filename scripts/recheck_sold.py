"""
Re-validate items in a category DB via direct HTTP and update presumed_sold.

For every item (or every item matching --target) the script:
  - GETs /items/{id} on Vinted (streaming, early-abort once we know).
  - Detects "sold" by the presence of `data-testid="item-status"` in the
    HTML body (active items render `item-buy-button` lower down instead).
  - Updates the DB:
      • active  → unflag presumed_sold (false-positive rehab), refresh
                  last_seen_at, clear presumed_sold_favourites.
      • sold    → flag presumed_sold, snapshot favourite_count into
                  presumed_sold_favourites.
      • unknown → just touch last_checked_at; retry on a later run.

Resume-friendly: items are picked by oldest `last_checked_at` first, so
re-running the script naturally continues from where the previous run left
off. If a run is interrupted, partial progress is already committed (each
item is committed individually).

Use:
    python3 scripts/recheck_sold.py data/category_tecnologia.db
    python3 scripts/recheck_sold.py data/category_tecnologia.db --max 500
    python3 scripts/recheck_sold.py data/category_tecnologia.db --target sold
    python3 scripts/recheck_sold.py data/category_tecnologia.db --delay 2
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/Users/wax/Desktop/vinted_scraper-main/src")

from core.category_db   import SCHEMA, _migrate, record_http_check   # noqa: E402
from core.vinted_client import check_item_page_any                   # noqa: E402


DOMAINS_DEFAULT = [
    "https://www.vinted.it",
    "https://www.vinted.fr",
    "https://www.vinted.de",
]


def pick_items(conn: sqlite3.Connection, target: str, limit: int) -> list[dict]:
    """Pick the next batch to validate.
    target: 'all' = every item; 'sold' = only those currently presumed_sold;
            'active' = only those currently active."""
    where = ""
    if target == "sold":
        where = "WHERE presumed_sold = 1"
    elif target == "active":
        where = "WHERE presumed_sold = 0"
    sql = f"""
        SELECT id, presumed_sold
          FROM items
         {where}
         ORDER BY (last_checked_at IS NULL) DESC, last_checked_at ASC, id ASC
         LIMIT ?
    """
    rows = conn.execute(sql, (limit,)).fetchall()
    return [{"id": r[0], "was_sold": bool(r[1])} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", help="path to a category_*.db")
    ap.add_argument("--max", type=int, default=200,
                    help="how many items to validate this run (default 200)")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds between HTTP requests (default 1.5)")
    ap.add_argument("--target", choices=["all", "sold", "active"], default="all",
                    help="restrict to items currently in this state")
    ap.add_argument("--domain", default=None,
                    help="Vinted domain to use (default: random rotation per item)")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    # Same WAL + busy_timeout settings as core.category_db.connect — needed
    # so the script can co-exist with a running category_monitor that holds
    # write locks during each cycle's upsert.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    _migrate(conn)

    items = pick_items(conn, args.target, args.max)
    print(f"Target {args.target}: {len(items)} items to validate from {db_path.name}")
    if not items:
        return

    n_active, n_sold, n_unknown, n_flipped, n_drops = 0, 0, 0, 0, 0
    t0 = time.time()
    domains_for_check = ([args.domain] if args.domain else DOMAINS_DEFAULT)
    for i, it in enumerate(items, 1):
        # Multi-domain fallback + global 30s arithmetic timer baked into
        # check_item_page_any → one configured list, no per-item rotation
        # needed at this layer.
        info   = check_item_page_any(domains_for_check, it["id"])
        status = info["status"]
        price  = info["price"]
        drop   = record_http_check(conn, it["id"], status, price)
        if status == "active":
            n_active += 1
            if it["was_sold"]:
                n_flipped += 1   # false positive rehabilitated
        elif status == "sold":
            n_sold += 1
        else:
            n_unknown += 1
        if drop:
            n_drops += 1
            print(f"  ↓ €{drop['old_price']:.0f}→€{drop['new_price']:.0f} "
                  f"(-{drop['drop_pct']}%) id={drop['item_id']}  "
                  f"{drop['title'][:60]}", flush=True)
        conn.commit()

        if i % 20 == 0 or i == len(items):
            elapsed = time.time() - t0
            rate    = i / elapsed
            eta     = (len(items) - i) / rate
            print(f"  {i}/{len(items)}  "
                  f"active={n_active} sold={n_sold} unknown={n_unknown} "
                  f"unflagged={n_flipped} drops={n_drops}  "
                  f"({elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s)",
                  flush=True)
        time.sleep(args.delay)

    conn.close()
    print(f"\nFinal: active={n_active}  sold={n_sold}  unknown={n_unknown}  "
          f"false-positives rehabilitated={n_flipped}  drops detected={n_drops}")


if __name__ == "__main__":
    main()
