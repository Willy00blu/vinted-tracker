"""
Category discovery — browse Vinted by macro-category and surface the
most-favourited listings as 'occasioni' (deals).

Unlike the monitor flow this is stateless: no seen-ids, no email, no history.
It's a one-shot scrape used by the GUI "Esplora" tab.

CATEGORIES maps a human macro-category name to a list of Vinted catalog_ids.
To find more IDs: open Vinted in your browser, click a category, and read
the `catalog[]=N` parameter that appears in the URL.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .scraper       import has_meaningful_title
from .vinted_client import get_scraper, safe_search, try_domains


BASE_DIR     = Path(__file__).resolve().parent.parent
CONFIGS_DIR  = BASE_DIR / "configs"
DATA_DIR     = BASE_DIR / "data"


CATEGORIES: dict[str, list[int]] = {
    "Tecnologia":      [2994],   # Elettronica (covers phones, laptops, audio, console, foto)
    "Moda Donna":      [1904],
    "Moda Uomo":       [5],
    "Bambini":         [1193],
    "Casa":            [1918],
    "Intrattenimento": [3245],   # libri, musica, film, giochi
    "Sport":           [4],
    "Animali":         [1453],
}

DOMAINS_DEFAULT = [
    "https://www.vinted.it",
    "https://www.vinted.fr",
    "https://www.vinted.de",
    # vinted.com (US) intentionally dropped — Europe-only by design
]


def subcategories_seen() -> list[tuple[str, int]]:
    """Return [(name, catalog_id)] of the leaf sub-categories actually observed
    in the existing category DBs (real ids that appear on items — exactly what
    the sold-check filter matches on). Used to populate the 'new category' form
    checkboxes; it's self-maintaining (grows as the collector classifies more
    items). The complete Vinted Elettronica tree can be fetched and merged in
    later; the form also lets the user paste extra catalog_ids by hand."""
    import sqlite3
    seen: dict[int, str] = {}
    for db in sorted(DATA_DIR.glob("category_*.db")):
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
            for name, cid in conn.execute(
                "SELECT category, catalog_id FROM items "
                "WHERE category IS NOT NULL AND catalog_id IS NOT NULL "
                "GROUP BY catalog_id"
            ):
                seen.setdefault(int(cid), name)
            conn.close()
        except Exception:
            continue
    return sorted(((name, cid) for cid, name in seen.items()), key=lambda x: x[0])


def _scrape_domain(domain: str, catalog_ids: list[int], pages: int,
                   max_price: float | None, delay: float = 0.4) -> list:
    """Fetch up to `pages` pages from one domain for the given catalog IDs.
    Uses the pooled scraper and safe_search with backoff on 429."""
    get_scraper(domain)
    catalog_str  = ",".join(str(c) for c in catalog_ids)
    out: list    = []
    for page in range(1, pages + 1):
        if page > 1:
            time.sleep(delay)
        params: dict = {
            "catalog_ids": catalog_str,
            "order":       "newest_first",
            "per_page":    96,
            "page":        page,
        }
        if max_price:
            params["price_to"] = max_price
        try:
            items = safe_search(domain, params)
            out.extend(items)
            if len(items) < 96:
                break
        except Exception as e:
            raise RuntimeError(f"{domain}: {e}") from e
    return out


def discover(
    catalog_ids: list[int],
    domains: list[str] | None = None,
    max_price: float | None = None,
    min_favourites: int = 0,
    top_k: int = 30,
    pages_per_domain: int = 1,
    progress_cb=None,
) -> tuple[list[dict], dict[str, str]]:
    """
    Browse Vinted by category across multiple domains and return the most
    favourited listings, ranked by `favourite_count` descending.

    Returns (items, errors):
        items:  list of dicts with id/title/price/favourite_count/brand/url
        errors: { domain: error_message } for any domain that failed
    """
    domains = domains or DOMAINS_DEFAULT
    raw: list                  = []
    errors: dict[str, str]     = {}

    # Single-domain pick with fallback rotation. Used to hit all four domains
    # in sequence — for Esplora that's overkill (the same item IDs surface
    # across domains), and it amplified Vinted's anti-bot pressure.
    items, used_domain, fail_errors = try_domains(
        domains,
        lambda d: _scrape_domain(d, catalog_ids, pages_per_domain, max_price),
    )
    if items is not None:
        raw.extend(items)
        if progress_cb:
            progress_cb(used_domain, ok=True)
        # Report the domains we had to skip past as soft failures
        for d, err in fail_errors.items():
            if d.startswith("_"):     # sentinel keys (e.g. "_giveup"), not domains
                continue
            errors[d] = err
            if progress_cb:
                progress_cb(d, ok=False, error=err)
    else:
        # All domains failed
        for d, err in fail_errors.items():
            if d.startswith("_"):     # sentinel keys (e.g. "_giveup"), not domains
                continue
            errors[d] = err
            if progress_cb:
                progress_cb(d, ok=False, error=err)

    # Dedup by id, filter by min_favourites. Track both numbers so the caller
    # can show "0 risultati (esclusi N con ♥<X)" instead of just "0 risultati".
    pool_total:    set = set()
    filtered_out:  int = 0
    by_id: dict        = {}
    for it in raw:
        if it.id is None or it.id in pool_total:
            continue
        # Skip single-word "iPhone"/"Cuffie" titles — too generic
        if not has_meaningful_title(it.title or ""):
            continue
        pool_total.add(it.id)
        fav = it.favourite_count or 0
        if fav < min_favourites:
            filtered_out += 1
            continue
        by_id[it.id] = {
            "id":              it.id,
            "title":           it.title or "",
            "price":           it.price or 0,
            "favourite_count": fav,
            "brand":           it.brand_title or "",
            "url":             f"https://www.vinted.it/items/{it.id}",
        }

    ranked = sorted(by_id.values(), key=lambda x: x["favourite_count"], reverse=True)
    errors["_pool_total"]   = str(len(pool_total))
    errors["_filtered_out"] = str(filtered_out)
    return ranked[:top_k], errors


# ─── Median comparison (Esplora ↔ category_*.db) ────────────────────────────

def find_category_db(catalog_ids: list[int]) -> Path | None:
    """Return the shared category DB if it exists (all category monitors now
    write to one shared DB; Esplora compares prices against its accumulated
    clusters). `catalog_ids` is accepted for API compatibility but no longer
    selects among per-category DBs."""
    from .category_db import shared_db_path
    db = shared_db_path()
    return db if db.exists() else None


def _pct_tag(price: float, median: float) -> tuple[str, int]:
    """Return (emoji+signed%, diff_pct) for a single price-vs-median delta."""
    diff_pct = round((price - median) / median * 100)
    if   diff_pct <= -30: emoji = "🔥"      # great deal
    elif diff_pct <= -15: emoji = "🟢"      # good
    elif diff_pct <=  15: emoji = "✓"       # in line
    elif diff_pct <=  30: emoji = "🔴"      # overpriced
    else:                 emoji = "🚨"      # very overpriced
    sign = "+" if diff_pct > 0 else ""
    return f"{emoji}{sign}{diff_pct}%", diff_pct


def enrich_with_median(items: list[dict], db_path: Path,
                       min_cluster_size: int = 3) -> dict:
    """
    Mutate `items` in place: for each item, find the closest cluster in the
    category DB and attach a price-vs-median comparison.

    Shows BOTH medians when available:
      - "vnd" = vs mediana VENDUTI  (real clearing price, needs ≥min sold)
      - "att" = vs mediana ATTIVI   (current asking prices, needs ≥min total)
    If only one is available, shows that one; if neither, shows a ⚪ fallback.

    Adds keys to every item dict:
        cluster_match:    cluster_key | None
        cluster_sim:      float        (best cosine similarity, 0..1)
        median_sold:      float | None
        median_listed:    float | None
        diff_sold_pct:    int | None   (positive = sopra mediana venduti)
        diff_listed_pct:  int | None   (positive = sopra mediana attivi)
        price_label:      str          (combined "🔥-45% vnd · 🟢-18% att",
                                        or '⚪ non in DB' / '⚪ campione scarso')

    Returns a small stats dict for the caller to surface in the status bar.
    """
    # Local import avoids paying the sentence-transformers cost when this
    # module is imported but Esplora median comparison is never used.
    from .clustering  import match_cluster_batch
    from .category_db import stats_for_cluster

    stats = {"matched": 0, "no_cluster": 0, "thin_sample": 0,
             "total":   len(items)}
    if not items:
        return stats

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        matches = match_cluster_batch(items, conn)
        for it in items:
            cluster, sim = matches.get(it["id"], (None, 0.0))
            it["cluster_match"]   = cluster
            it["cluster_sim"]     = round(float(sim), 3)
            it["median_sold"]     = None
            it["median_listed"]   = None
            it["diff_sold_pct"]   = None
            it["diff_listed_pct"] = None

            if cluster is None:
                it["price_label"] = "⚪ non in DB"
                stats["no_cluster"] += 1
                continue

            s = stats_for_cluster(conn, cluster)
            sold_med   = s.get("median_sold")
            listed_med = s.get("median_listed")
            n_sold     = s.get("sold", 0)
            n_total    = s.get("total", 0)
            price      = it["price"]

            parts: list[str] = []
            if sold_med and sold_med > 0 and n_sold >= min_cluster_size:
                tag, diff = _pct_tag(price, sold_med)
                it["median_sold"]   = round(sold_med, 2)
                it["diff_sold_pct"] = diff
                parts.append(f"{tag} vnd")
            if listed_med and listed_med > 0 and n_total >= min_cluster_size:
                tag, diff = _pct_tag(price, listed_med)
                it["median_listed"]   = round(listed_med, 2)
                it["diff_listed_pct"] = diff
                parts.append(f"{tag} att")

            if not parts:
                it["price_label"] = "⚪ campione scarso"
                stats["thin_sample"] += 1
                continue

            it["price_label"] = " · ".join(parts)
            stats["matched"] += 1
    finally:
        conn.close()
    return stats
