"""
Vinted scraping and item filtering.

item_matches()  — checks if an item passes all config filters
search_all()    — runs every query on every domain and returns matching items
"""

import time

from vinted_scraper import VintedScraper

from .storage import get_price_new, discount_label


def item_matches(title_lower: str, price: float, cfg: dict) -> bool:
    """
    Return True if the item passes all filters defined in the config:
      - price is within range
      - none of the exclude keywords appear in the title
      - at least one required keyword appears (if any are set)
      - at least one required brand appears (if any are set)
    """
    min_price = cfg.get("min_price", 0)
    if not (min_price < price <= cfg["max_price"]):
        return False
    if any(e in title_lower for e in cfg.get("keywords_exclude", [])):
        return False
    required = cfg.get("keywords_required", [])
    if required and not any(k in title_lower for k in required):
        return False
    brands = cfg.get("brands_required", [])
    if brands and not any(b in title_lower for b in brands):
        return False
    return True


def _scrape_domain(domain: str, query: str, retries: int = 2) -> list:
    """
    Fetch items from a single Vinted domain for a given query.
    Retries on failure with increasing back-off delays.
    """
    for attempt in range(retries):
        try:
            return VintedScraper(domain).search(
                {"search_text": query, "order": "newest_first", "per_page": 96}
            )
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 + attempt * 5)
            else:
                raise RuntimeError(str(e)) from e
    return []


def search_all(cfg: dict, price_db: dict | None) -> dict:
    """
    Run all configured queries across all configured domains.

    Returns a dict of { item_id: item_data }.
    Prints a ✓ / ✗ status line per domain for real-time feedback in the GUI log.
    If a price_db is provided, each item is enriched with discount information.
    """
    found: dict = {}
    domain_errors: dict[str, list] = {d: [] for d in cfg["domains"]}

    for query in cfg["queries"]:
        for domain in cfg["domains"]:
            time.sleep(2)
            try:
                items = _scrape_domain(domain, query)
                for item in items:
                    if item.id in found:
                        continue
                    title_lower = (item.title or "").lower()
                    price = item.price or 0
                    if not item_matches(title_lower, price, cfg):
                        continue

                    entry: dict = {
                        "id":    item.id,
                        "title": item.title,
                        "price": price,
                        "url":   f"https://www.vinted.it/items/{item.id}",
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

            except Exception as e:
                domain_errors[domain].append(str(e))

    # Print a one-line status per domain (picked up by the GUI log reader)
    for domain, errors in domain_errors.items():
        short = domain.replace("https://www.", "")
        if errors:
            print(f"  ✗ {short}: {errors[-1]}", flush=True)
        else:
            print(f"  ✓ {short}: OK", flush=True)

    return found
