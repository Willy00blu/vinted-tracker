"""
Scan a category DB and delete every item that's listed by a US-based seller.

Vinted lets US sellers list items that appear in the EU catalog feeds (.it,
.fr, .de), but those items can't be bought with the standard checkout flow
and they pollute price medians / cluster stats. This script removes them.

For each item it fetches the public item page once, extracts the seller's
country from the embedded user JSON (`country_title_local`) and DELETEs
when the country is the United States. The price_drops rows that reference
the deleted item are also removed (foreign keys are not enforced by default
on SQLite, so we cascade manually).

Resume-friendly: items are processed in order of oldest `last_checked_at`,
so re-running picks up where we left off. Each row is committed individually.

Side effect: while we're fetching the page anyway, we ALSO update sold/active
status via the same response — kills two birds with one stone if you'd be
running recheck_sold soon afterwards.

Use:
    python3 scripts/cleanup_us_sellers.py data/category_tecnologia.db
    python3 scripts/cleanup_us_sellers.py data/category_tecnologia.db --max 500 --delay 2
    python3 scripts/cleanup_us_sellers.py data/category_tecnologia.db --dry-run
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

from core.category_db   import SCHEMA, _migrate, mark_item_checked   # noqa: E402
from core.vinted_client import check_item_page_any                   # noqa: E402

DOMAINS_DEFAULT = [
    "https://www.vinted.it",
    "https://www.vinted.fr",
    "https://www.vinted.de",
]


def pick_items(conn: sqlite3.Connection, limit: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, title FROM items
         ORDER BY (last_checked_at IS NULL) DESC, last_checked_at ASC, id ASC
         LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [{"id": r[0], "title": r[1]} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", help="path to a category_*.db")
    ap.add_argument("--max",   type=int,   default=500,
                    help="how many items to scan this run (default 500)")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds between HTTP requests")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted but don't touch the DB")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    _migrate(conn)

    items = pick_items(conn, args.max)
    print(f"Scanning {len(items)} items from {db_path.name}"
          + (" [DRY-RUN]" if args.dry_run else ""))
    if not items:
        return

    n_us, n_active, n_sold, n_unknown = 0, 0, 0, 0
    t0 = time.time()
    for i, it in enumerate(items, 1):
        info = check_item_page_any(DOMAINS_DEFAULT, it["id"])
        status, is_us = info["status"], info["is_us_seller"]

        if status == "unknown":
            n_unknown += 1
            # Just touch last_checked_at via mark_item_checked(None) — same
            # effect, lets us resume cleanly.
            if not args.dry_run:
                mark_item_checked(conn, it["id"], None)
        elif is_us:
            n_us += 1
            print(f"  🇺🇸 DELETE id={it['id']}  {it['title'][:60]}", flush=True)
            if not args.dry_run:
                conn.execute("DELETE FROM price_drops WHERE item_id = ?", (it["id"],))
                conn.execute("DELETE FROM items       WHERE id = ?",       (it["id"],))
        else:
            # Non-US seller; still update sold/active state since we paid
            # for the HTML anyway.
            alive = (status == "active")
            if status == "active":
                n_active += 1
            else:
                n_sold += 1
            if not args.dry_run:
                mark_item_checked(conn, it["id"], alive)

        if not args.dry_run:
            conn.commit()

        if i % 20 == 0 or i == len(items):
            elapsed = time.time() - t0
            rate    = i / elapsed
            eta     = (len(items) - i) / rate
            print(f"  {i}/{len(items)}  "
                  f"us-deleted={n_us}  active={n_active}  sold={n_sold}  "
                  f"unknown={n_unknown}  "
                  f"({elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s)",
                  flush=True)

        time.sleep(args.delay)

    conn.close()
    print(f"\nFinal: us-deleted={n_us}  active={n_active}  sold={n_sold}  unknown={n_unknown}")


if __name__ == "__main__":
    main()
