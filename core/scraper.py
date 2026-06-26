"""
Vinted scraping and item filtering.

item_matches()  — checks if an item passes all config filters
search_all()    — runs every query on every domain and returns matching items
"""

import random
import time

from .storage       import get_price_new, discount_label
from .vinted_client import get_scraper, safe_search, try_domains


def has_meaningful_title(title: str) -> bool:
    """Reject titles consisting of ≤1 meaningful token. Single-word titles
    like "iPhone" or "Cuffie" are too generic to cluster or match keywords
    against — they're noise that pollutes the DB and the medians.
    A "meaningful token" is any space-separated chunk containing at least
    one alphanumeric character (so trailing "." or stray emojis don't count)."""
    if not title:
        return False
    tokens = [t for t in title.split() if any(c.isalnum() for c in t)]
    return len(tokens) >= 2


def _keyword_brand_match(title_lower: str, cfg: dict) -> bool:
    """Title-based filters: exclude keywords, required keywords, required brands."""
    if any(e in title_lower for e in cfg.get("keywords_exclude", [])):
        return False
    required = cfg.get("keywords_required", [])
    if required and not any(k in title_lower for k in required):
        return False
    brands = cfg.get("brands_required", [])
    if brands and not any(b in title_lower for b in brands):
        return False
    return True


def item_matches(title_lower: str, price: float, favourite_count: int, cfg: dict) -> bool:
    """
    Return True if the item passes all filters defined in the config:
      - title has at least 2 meaningful tokens (filters out generic 1-word
        titles like "iPhone" that can't be reliably matched/clustered)
      - price is within range
      - title passes keyword/brand filters
      - favourite_count >= min_favourites (if set)
    """
    if not has_meaningful_title(title_lower):
        return False
    min_price = cfg.get("min_price", 0)
    # Inclusive on both ends — previously min was strict (<) which silently
    # dropped items priced exactly at min_price.
    if not (min_price <= price <= cfg["max_price"]):
        return False
    if not _keyword_brand_match(title_lower, cfg):
        return False
    min_fav = cfg.get("min_favourites", 0)
    if min_fav and favourite_count < min_fav:
        return False
    return True


def _scrape_domain(domain: str, query: str, max_pages: int = 1, delay: float = 2.0, retries: int = 2) -> list:
    """
    Fetch items from a single Vinted domain across up to max_pages pages.
    Stops early if a page returns fewer than 96 items (last page).
    Retries each page on failure with increasing back-off delays.

    Uses the pooled scraper (vinted_client.get_scraper) so the cookie is
    fetched once per process per domain, and safe_search to ride out 429s.
    """
    all_items: list = []
    # Touch the pool early so a broken domain fails fast before per-page loops
    get_scraper(domain)
    for page in range(1, max_pages + 1):
        if page > 1:
            # ±20% jitter: regular timing is a bot fingerprint.
            time.sleep(random.uniform(0.8, 1.2) * delay)
        for attempt in range(retries):
            try:
                items = safe_search(
                    domain,
                    {"search_text": query, "order": "newest_first", "per_page": 96, "page": page},
                )
                all_items.extend(items)
                if len(items) < 96:
                    return all_items
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(3 + attempt * 5)
                else:
                    raise RuntimeError(str(e)) from e
    return all_items


def search_all(cfg: dict, price_db: dict | None) -> tuple[dict, list]:
    """
    Run all configured queries across all configured domains.

    Returns a tuple (found, all_prices):
      - found:      { item_id: item_data } for items passing ALL filters
      - all_prices: list of prices for items passing only keyword/brand filters
                    (used for market median / anomaly detection — ignores price cap)

    Prints a ✓ / ✗ status line per domain for real-time feedback in the GUI log.
    If a price_db is provided, each item is enriched with discount information.
    """
    found: dict = {}
    all_prices: list = []
    seen_for_stats: set = set()  # avoid double-counting same item across queries
    # Per-domain counters for the end-of-cycle status line
    domain_usage:  dict[str, int]  = {d: 0 for d in cfg["domains"]}
    domain_errors: dict[str, list] = {d: [] for d in cfg["domains"]}
    queries_no_domain: int = 0          # queries that failed on ALL domains
    max_pages = cfg.get("max_pages", 1)
    delay     = cfg.get("request_delay_seconds", 2.0)

    for query in cfg["queries"]:
        time.sleep(random.uniform(0.8, 1.2) * delay)
        # Try one random domain per query; only fall through to others on error.
        items, used_domain, errors = try_domains(
            cfg["domains"],
            lambda d, q=query: _scrape_domain(d, q, max_pages, delay),
        )
        for d, err in errors.items():
            # try_domains can return sentinel keys (e.g. "_giveup" when the
            # global block level maxes out) that are NOT domains — skip them,
            # otherwise domain_errors[d] raises KeyError and crashes the monitor.
            if d in domain_errors:
                domain_errors[d].append(err)
        if items is None:
            queries_no_domain += 1
            continue
        domain_usage[used_domain] += 1

        for item in items:
            title_lower = (item.title or "").lower()
            price = item.price or 0

            # Stats sample: keyword/brand match, ignoring price cap.
            if price > 0 and item.id not in seen_for_stats and _keyword_brand_match(title_lower, cfg):
                all_prices.append(price)
                seen_for_stats.add(item.id)

            if item.id in found:
                continue
            favourite_count = item.favourite_count or 0
            if not item_matches(title_lower, price, favourite_count, cfg):
                continue

            entry: dict = {
                "id":              item.id,
                "title":           item.title,
                "price":           price,
                "url":             f"https://www.vinted.it/items/{item.id}",
                "favourite_count": favourite_count,
            }

            # Optional: enrich with discount data if a price DB is available
            if price_db:
                price_new = get_price_new(item.title, price_db)
                if price_new:
                    pct = round((1 - price / price_new) * 100)
                    entry.update({
                        "discount_pct": pct,
                        "discount_str": f"(-{pct}% su ~€{price_new} nuovo)",
                        "label":        discount_label(pct),
                    })
                else:
                    entry.update({
                        "discount_pct": 0,
                        "discount_str": "(modello non in archivio)",
                        "label":        "⚪",
                    })

            found[item.id] = entry

    # End-of-cycle status — now per-domain shows how many queries it served.
    total_queries = len(cfg["queries"])
    for domain in cfg["domains"]:
        used   = domain_usage[domain]
        fails  = len(domain_errors[domain])
        short  = domain.replace("https://www.", "")
        if used == 0 and fails == 0:
            print(f"  · {short}: idle (not picked this cycle)", flush=True)
        elif fails == 0:
            print(f"  ✓ {short}: served {used}/{total_queries} queries", flush=True)
        elif used == 0:
            print(f"  ✗ {short}: tried as fallback, all failed — {domain_errors[domain][-1]}", flush=True)
        else:
            print(f"  ⚠ {short}: served {used} · {fails} fallback fails (last: {domain_errors[domain][-1]})",
                  flush=True)
    if queries_no_domain:
        print(f"  ⚠ {queries_no_domain}/{total_queries} queries failed on ALL domains",
              flush=True)

    return found, all_prices
