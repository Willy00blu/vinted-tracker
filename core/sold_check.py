"""Sold-check drip loop — extracted from category_monitor so it can also run
as a STANDALONE WORKER (core.sold_check_worker), independent of the scrape
monitor. Same logic in both places.

The drip continuously re-validates active items oldest-first (or configured
order) via the item-page client, recording sold/removed/price-drops + back-
filling photo_urls. Paced by the shared rate limiter (single IP).
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .category_db import connect, items_to_check, record_http_check
from .vinted_client import check_item_page_any

# Defaults — overridable per-call via cfg
DEFAULT_BATCH       = 50
DEFAULT_IDLE_S      = 30
DEFAULT_SOLD_MIN    = 1
DEFAULT_DROP_PCT    = 15
DEFAULT_DROP_ABS    = 5
DEFAULT_ORDER       = "oldest"

DOMAINS_DEFAULT = [
    "www.vinted.it", "www.vinted.fr", "www.vinted.es",
    "www.vinted.de", "www.vinted.nl", "www.vinted.pt",
]


def _pause_flag_path(db_path: Path) -> Path:
    return db_path.parent / ".soldcheck_paused"


def soldcheck_item(conn, item_id: int, *, domains, drop_pct, drop_abs):
    """One HTTP validation, transactional. Returns (status, drop_event|None)."""
    info = check_item_page_any(domains, item_id)
    drop = record_http_check(
        conn, item_id, info["status"], info["price"],
        drop_pct, drop_abs,
        catalog_id=info.get("catalog_id"),
        category=info.get("category"),
        favourites=info.get("favourites"),
        photo_urls=info.get("photo_urls"),
    )
    conn.commit()
    return info["status"], drop


def run_drip(db_path: Path, *,
             domains: list[str] | None = None,
             sold_min: int = DEFAULT_SOLD_MIN,
             drop_pct: float = DEFAULT_DROP_PCT,
             drop_abs: float = DEFAULT_DROP_ABS,
             catalog_ids=None,
             order: str = DEFAULT_ORDER,
             batch: int = DEFAULT_BATCH,
             idle_s: int = DEFAULT_IDLE_S,
             log_prefix: str = "sold-check",
             stop_event=None) -> None:
    """Continuous drip until stop_event is set or process killed. Honors the
    file flag `data/.soldcheck_paused` for a soft pause (no kill needed)."""
    db_path = Path(db_path)
    pause_flag = _pause_flag_path(db_path)
    domains = domains or DOMAINS_DEFAULT
    n_checked = n_sold = n_removed = n_drops = 0
    t0 = time.time()
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            with connect(db_path) as conn:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        return
                    if pause_flag.exists():
                        time.sleep(5)
                        continue
                    batch_items = items_to_check(
                        conn, limit=batch, min_seen_count=sold_min,
                        catalog_ids=catalog_ids, order=order,
                    )
                    if not batch_items:
                        time.sleep(idle_s)
                        continue
                    for c in batch_items:
                        if pause_flag.exists() or (
                            stop_event is not None and stop_event.is_set()
                        ):
                            break
                        status, drop = soldcheck_item(
                            conn, c["id"],
                            domains=domains, drop_pct=drop_pct, drop_abs=drop_abs,
                        )
                        n_checked += 1
                        n_sold    += status == "sold"
                        n_removed += status == "removed"
                        if drop:
                            n_drops += 1
                            print(f"    ↓ €{drop['old_price']:.0f}→€{drop['new_price']:.0f}"
                                  f" -{drop['drop_pct']}%  {drop['title'][:55]}", flush=True)
                        if n_checked % 20 == 0:
                            el = max(1e-6, time.time() - t0)
                            print(f"  🔁 {log_prefix}: {n_checked} checked · "
                                  f"{n_sold} sold · {n_removed} removed · "
                                  f"{n_drops} drops · {n_checked/el:.2f}/s", flush=True)
        except Exception as e:
            print(f"  ⚠ {log_prefix} loop error: {type(e).__name__}: {e}; "
                  f"riprovo tra {idle_s}s", flush=True)
            time.sleep(idle_s)
