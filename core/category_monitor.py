"""
Category monitor — periodically scrape Vinted catalog IDs and build a price
database with clustering, price-drop detection and presumed-sold inference.

Run standalone:
    python -m core.category_monitor --config configs/category_tecnologia.json
or once:
    python -m core.category_monitor --config configs/category_tecnologia.json --once

This is the "generic monitor" pattern requested by the user: instead of
text searches with strict keyword filters (like monitor.py), it browses
a whole category and accumulates everything for statistical analysis.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path

from .category_db   import (
    connect, upsert_items, stats_summary, record_snapshot,
    items_to_check, record_http_check, shared_db_path,
)
from .clustering    import (
    cluster_titles_batch, EMBEDDINGS_AVAILABLE,
)
from .network       import wifi_aware_sleep
from .scraper       import has_meaningful_title
from .vinted_client import (
    get_scraper, safe_search, try_domains, check_item_page_any,
)
from .              import coverage
from .rate_limiter  import global_level


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DOMAINS_DEFAULT = [
    "https://www.vinted.it",
    "https://www.vinted.fr",
    "https://www.vinted.de",
    # vinted.com (US) intentionally dropped — Europe-only by design
]


def _scrape_domain(domain: str, catalog_ids: list[int], pages: int,
                   max_price: float | None, delay: float,
                   order: str = "newest_first") -> list:
    """Uses pooled scraper + safe_search (cookie cached, backoff on 429).
    `order` is the Vinted catalog sort: newest_first / relevance /
    price_low_to_high / price_high_to_low (Vinted has no 'by favourites')."""
    get_scraper(domain)
    catalog_str = ",".join(str(c) for c in catalog_ids)
    out: list   = []
    for page in range(1, pages + 1):
        if page > 1:
            time.sleep(random.uniform(0.8, 1.2) * delay)
        params: dict = {
            "catalog_ids": catalog_str,
            "order":       order,
            "per_page":    96,
            "page":        page,
        }
        if max_price:
            params["price_to"] = max_price
        items = safe_search(domain, params)
        out.extend(items)
        if len(items) < 96:
            break
    return out


def _scrape_all(cfg: dict) -> tuple[list[dict], dict[str, str]]:
    domains   = cfg.get("domains", DOMAINS_DEFAULT)
    catalog   = cfg["catalog_ids"]
    max_price = cfg.get("max_price")
    delay     = cfg.get("request_delay_seconds", 1.0)
    min_price = cfg.get("min_price", 0)

    # Adaptive page count (coverage controller): fetch as many pages as needed to
    # keep overlapping the previous scrape — i.e. to not let new items slip past
    # between cycles — capped at pages_per_domain_max and shrunk under block
    # pressure. pages_per_domain stays the floor / first-run default.
    cov_key   = f"cat_{cfg['id']}"
    pages_floor = cfg.get("pages_per_domain", 2)
    pages_max   = cfg.get("pages_per_domain_max") or max(pages_floor, 8)
    pages     = coverage.pages_for(cov_key, pages_floor, pages_max)

    scrape_order = cfg.get("scrape_order", "newest_first")
    raw: list             = []
    errors: dict[str, str] = {}

    # Single-domain pick with fallback rotation. Item IDs are global on Vinted
    # so one domain per cycle is enough — over a day of cycles we rotate
    # naturally across the eligible domains.
    items, used_domain, fail_errors = try_domains(
        domains,
        lambda d: _scrape_domain(d, catalog, pages, max_price, delay, scrape_order),
    )
    if items is not None:
        raw.extend(items)
    errors.update(fail_errors)
    # Tag the chosen domain in errors dict so caller (run()) can report it
    if used_domain is not None:
        errors["_used_domain"] = used_domain

    # Dedup by id, normalize, apply price range, compute cluster key
    by_id: dict = {}
    for it in raw:
        if it.id is None or it.id in by_id:
            continue
        price = it.price or 0
        if price <= 0 or price < min_price:
            continue
        if max_price and price > max_price:
            continue
        title = it.title or ""
        # Skip single-word titles — useless for clustering and price stats
        if not has_meaningful_title(title):
            continue
        brand = it.brand_title or ""
        by_id[it.id] = {
            "id":              it.id,
            "title":           title,
            "brand":           brand,
            "price":           price,
            "favourite_count": it.favourite_count or 0,
            # Photo URLs (up to 4) captured cheaply from the feed → the regen
            # subprocess downloads+embeds them later (image_emb). Stored now
            # because the CDN URLs are short-lived and absent from old items.
            "photo_urls": ",".join(
                ph.url for ph in (it.photos or [])[:4] if getattr(ph, "url", None)
            ) or None,
            # NOTE: the catalog/search feed does NOT include the item's own
            # catalog_id (the JSON key is absent → always None here). The leaf
            # sub-category + breadcrumb category only exist on the item DETAIL
            # page, so they're filled in later, at sold-check time
            # (record_http_check). Kept here in case the feed ever exposes it.
            "catalog_id":      it.catalog_id,
            "url":             f"https://www.vinted.it/items/{it.id}",
            # cluster_key is filled in later by cluster_titles_batch() once we
            # have the DB connection (needed for centroid persistence)
        }

    # Coverage feedback: did this scrape overlap the previous one? If not, we're
    # missing items between cycles → grow pages next time (unless under block
    # pressure, in which case shrink). Persisted, so it survives restarts.
    under_pressure = (used_domain is None) or global_level() > 0
    cov = coverage.record(cov_key, list(by_id.keys()), pages_max,
                          under_pressure=under_pressure, default=pages_floor)
    errors["_coverage"] = (
        f"overlap {cov['overlap']} (prev {cov['prev']}) · "
        f"pagine {pages}→{cov['pages_next']}/{pages_max} ({cov['reason']})"
    )

    return list(by_id.values()), errors


def run(cfg: dict, once: bool = False):
    label    = cfg.get("label", cfg["id"])
    db_path  = shared_db_path()   # one shared DB; this collector scopes to its catalog_ids

    drop_pct   = cfg.get("price_drop_threshold_pct", 15)
    drop_abs   = cfg.get("price_drop_min_abs",       5)
    sold_min   = cfg.get("presumed_sold_min_seen",     3)

    # Decoupled architecture (2026-05-23): the scrape and the sold-check run as
    # two independent loops (threads). The SCRAPE runs on a tight cadence so new
    # listings are caught fast (it's never blocked behind a long sold-check
    # batch); the SOLD-CHECK is a continuous background DRIP that re-validates
    # active items oldest-first, forever, paced by the item lane + IP bucket in
    # core.rate_limiter. No more per-cycle quota — the drip naturally cycles the
    # whole DB at the max rate the single IP tolerates.
    scrape_interval   = cfg.get("scrape_interval_minutes",
                                cfg.get("check_interval_minutes", 8)) * 60
    domains_for_check = cfg.get("domains") or DOMAINS_DEFAULT
    SOLDCHECK_BATCH   = 50      # items_to_check page size for the drip
    SOLDCHECK_IDLE_S  = 30      # wait when there's nothing to validate
    once_quota        = max(1, cfg.get("presumed_sold_http_quota", 30))
    # Sold-check scope/order: scope the drip to this category's leaf catalog_ids
    # (None = all, e.g. the legacy broad Tecnologia [2994] config) and order by
    # the configured priority (oldest / favourites / price).
    soldcheck_order   = cfg.get("soldcheck_order", "oldest")
    soldcheck_catalog = cfg.get("soldcheck_catalog_ids")   # list[int] | None
    # Cluster every scrape cycle? Default OFF: clustering happens only in the
    # full "Rigenera cluster" pass, so the monitor never loads the embedding
    # model and the heavy e5 model lives only in the regenerate subprocess.
    cluster_on_scrape = cfg.get("cluster_on_scrape", False)

    print(f"{label} — started", flush=True)
    if not once:
        print(f"Catalog: {cfg['catalog_ids']} · scrape every "
              f"{scrape_interval // 60} min · sold-check: continuous drip · "
              f"DB: {db_path.name}", flush=True)
        clustering_mode = ("incremental ogni ciclo" if cluster_on_scrape
                           else "solo a Rigenera cluster (per-ciclo OFF)")
        print(f"Clustering: {clustering_mode} · labels: category · cluster_key\n",
              flush=True)

    def _soldcheck_item(conn, item_id):
        """One HTTP validation (sold/active/removed + price + drop detection).
        Paced by the item lane + IP bucket inside check_item_page_any. Commits so
        the write lock is released between network calls. Returns (status, drop)."""
        info = check_item_page_any(domains_for_check, item_id)
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

    def _soldcheck_once(conn, limit) -> tuple[int, int, int, list[dict]]:
        """Bounded sold-check batch (used by --once / the GUI Test button)."""
        candidates = items_to_check(conn, limit=limit, min_seen_count=sold_min,
                                    catalog_ids=soldcheck_catalog, order=soldcheck_order)
        if not candidates:
            return 0, 0, 0, []
        n_sold = n_removed = 0
        drops: list[dict] = []
        print(f"  HTTP sampling: validating {len(candidates)} items "
              f"(paced by the shared limiter)…", flush=True)
        for idx, c in enumerate(candidates, 1):
            status, drop = _soldcheck_item(conn, c["id"])
            n_sold    += status == "sold"
            n_removed += status == "removed"
            if drop:
                drops.append(drop)
            if idx % 5 == 0 or idx == len(candidates):
                print(f"    [{idx}/{len(candidates)}] sold={n_sold} "
                      f"removed={n_removed} drops={len(drops)}", flush=True)
        return len(candidates), n_sold, n_removed, drops

    # File-flag based pause: lets the GUI suspend the sold-check WITHOUT
    # killing the monitor process. The loop just sleeps while the flag exists.
    from pathlib import Path as _Path
    PAUSE_FLAG = _Path(db_path).parent / ".soldcheck_paused"

    def _soldcheck_loop():
        """Continuous background drip: re-validate active items oldest-checked
        first, forever, paced by the item lane + IP bucket. Decoupled from the
        scrape, so new listings are never stuck behind a long sold-check batch.
        A full pass over the DB takes ~ active_items / sustained_rate (the single
        IP's ceiling).

        Honors `data/.soldcheck_paused` (touch the file from the GUI to pause)."""
        n_checked = n_sold = n_removed = n_drops = 0
        t0 = time.time()
        # Outer loop: survive transient errors (e.g. "database is locked" during
        # a concurrent regenerate) by logging + reopening, instead of stopping
        # the drip permanently.
        while True:
            try:
                with connect(db_path) as conn:
                    while True:
                        # Respect the GUI pause flag — sleep without burning DB
                        # locks or HTTP budget. Re-check the flag every 5s so
                        # resume is responsive.
                        if PAUSE_FLAG.exists():
                            time.sleep(5)
                            continue
                        batch = items_to_check(conn, limit=SOLDCHECK_BATCH,
                                               min_seen_count=sold_min,
                                               catalog_ids=soldcheck_catalog,
                                               order=soldcheck_order)
                        if not batch:
                            time.sleep(SOLDCHECK_IDLE_S)
                            continue
                        for c in batch:
                            if PAUSE_FLAG.exists():
                                break       # interrupt batch on pause

                            status, drop = _soldcheck_item(conn, c["id"])
                            n_checked += 1
                            n_sold    += status == "sold"
                            n_removed += status == "removed"
                            if drop:
                                n_drops += 1
                                print(f"    ↓ €{drop['old_price']:.0f}→€{drop['new_price']:.0f}"
                                      f" -{drop['drop_pct']}%  {drop['title'][:55]}", flush=True)
                            if n_checked % 20 == 0:
                                el = max(1e-6, time.time() - t0)
                                print(f"  🔁 sold-check drip: {n_checked} checked · "
                                      f"{n_sold} sold · {n_removed} removed · "
                                      f"{n_drops} drops · {n_checked/el:.2f}/s", flush=True)
            except Exception as e:
                print(f"  ⚠ sold-check loop error: {type(e).__name__}: {e}; "
                      f"riprovo tra {SOLDCHECK_IDLE_S}s", flush=True)
                time.sleep(SOLDCHECK_IDLE_S)

    def _scrape_once():
        """One catalog scrape: fetch → cluster → upsert → snapshot. Fast; does
        NOT do the sold-check (that's the separate drip thread)."""
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Scraping category — {label}...",
              flush=True)

        items, errors = _scrape_all(cfg)
        used_domain   = errors.get("_used_domain")

        for d in cfg.get("domains", DOMAINS_DEFAULT):
            short = d.replace("https://www.", "")
            if d == used_domain:
                print(f"  ✓ {short}: served this cycle ({len(items)} items)",
                      flush=True)
            elif d in errors:
                print(f"  ✗ {short}: fallback failed — {errors[d]}", flush=True)
            else:
                print(f"  · {short}: idle (not picked)", flush=True)
        if used_domain is None:
            print("  ⚠ all domains failed this cycle", flush=True)
        if errors.get("_coverage"):
            print(f"  📈 coverage: {errors['_coverage']}", flush=True)

        with connect(db_path) as conn:
            # Clustering. By default it runs ONLY in the full "Rigenera cluster"
            # pass (cluster_on_scrape=False): the scrape loop just stores items
            # (new ones get cluster_key=NULL until the next regenerate). This
            # keeps the always-on monitor light — it never loads the embedding
            # model — and lets the (heavy) e5 model live only in the regenerate
            # subprocess. Set cluster_on_scrape=true to cluster every cycle
            # incrementally instead (cheap only with the small MiniLM model).
            if cluster_on_scrape:
                cluster_map = cluster_titles_batch(items, conn=conn)
                for it in items:
                    it["cluster_key"] = cluster_map.get(it["id"], "unknown")
            # upsert detects feed-level price drops (price changed in the catalog
            # between scrapes); the drip thread catches drops on items the feed
            # no longer surfaces.
            n_new, n_upd, drops = upsert_items(conn, items, drop_pct, drop_abs)
            conn.commit()
            record_snapshot(conn)
            stats = stats_summary(conn)

        elapsed = time.time() - t0
        print(
            f"  {len(items)} scraped · {n_new} new · {n_upd} updated · "
            f"{len(drops)} feed-drops  ({elapsed:.1f}s)",
            flush=True,
        )
        print(
            f"  DB: total={stats['total']} active={stats['active']} "
            f"sold={stats['sold']} · median listed €{stats['median_listed']} "
            f"· median sold €{stats['median_sold']}",
            flush=True,
        )
        for d in drops[:5]:
            print(f"    ↓ €{d['old_price']:.0f}→€{d['new_price']:.0f} "
                  f"-{d['drop_pct']}%  {d['title'][:55]}", flush=True)

    if once:
        # One scrape + one bounded sold-check batch (no threads).
        _scrape_once()
        with connect(db_path) as conn:
            nc, ns, nr, dr = _soldcheck_once(conn, once_quota)
        print(f"  sold-check (once): {nc} checked · {ns} sold · {nr} removed · "
              f"{len(dr)} drops", flush=True)
        return

    # SOLD-CHECK is now a SEPARATE WORKER process (core.sold_check_worker), so
    # it can run independently of the scrape monitor (the user can keep doing
    # drops/photo backfill even with scrape off). Start the worker from the GUI
    # toggle. The monitor here only scrapes new catalog pages.
    while True:
        try:
            _scrape_once()
        except Exception as e:
            # Survive transient errors (e.g. "database is locked" during a
            # concurrent regenerate) instead of killing the whole monitor.
            print(f"  ⚠ scrape cycle error: {type(e).__name__}: {e}", flush=True)
        wifi_aware_sleep(scrape_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vinted category monitor")
    parser.add_argument("--config", required=True,
                        help="Path to a category-monitor config JSON")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = BASE_DIR / cfg_path
    cfg = json.loads(cfg_path.read_text())
    run(cfg, once=args.once)
