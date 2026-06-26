"""
Persistence helpers.

load_seen / save_seen   — track which item IDs have already been notified
load_price_db           — load reference prices for discount calculation
get_price_new           — look up a new price for a given item title
discount_label          — turn a discount percentage into a human-readable label
load_stats / save_stats — per-monitor stats (total notified, last found time)
"""

import json
import datetime
from pathlib import Path


# ── Seen IDs ──────────────────────────────────────────────────────────────────

def load_seen(path: Path) -> set:
    """Load the set of already-notified item IDs from disk.
    Returns an empty set on missing/corrupt file (parity with other loaders)."""
    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except Exception:
            pass
    return set()


def save_seen(path: Path, seen: set):
    """Persist the set of seen item IDs to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(seen)))


# ── Price DB (optional, used by monitors that track discounts) ────────────────

def load_price_db(path: Path) -> dict:
    """
    Load a price reference database and flatten all categories into one dict.

    Expected JSON shape:
    {
      "category_name": {
        "model key": { "price_new": 499 },
        ...
      },
      ...
    }
    """
    db = json.loads(path.read_text())
    flat: dict = {}
    for category in db.values():
        if isinstance(category, dict):
            flat.update(category)
    return flat


def get_price_new(title: str, db: dict):
    """Return the reference new price for an item, or None if not in DB."""
    title_lower = title.lower()
    for model_key, data in db.items():
        if model_key in title_lower:
            return data.get("price_new")
    return None


def discount_label(pct: int) -> str:
    """Return an emoji label for a given discount percentage."""
    if pct >= 60: return "🔥 AFFARONE"
    if pct >= 50: return "🟠 Ottimo affare"
    if pct >= 40: return "🟡 Buon affare"
    if pct >= 25: return "🟢 Discreto"
    return "⚪ Poco sconto"


# ── Stats ─────────────────────────────────────────────────────────────────────

def load_stats(path: Path) -> dict:
    """Load per-monitor stats from disk."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"total_notified": 0, "last_found_at": None}


def save_stats(path: Path, stats: dict):
    """Persist per-monitor stats to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def bump_stats(path: Path, count: int):
    """Increment total_notified by count and record last_found_at as now."""
    stats = load_stats(path)
    stats["total_notified"] = stats.get("total_notified", 0) + count
    stats["last_found_at"]  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_stats(path, stats)


# ── Notification history ──────────────────────────────────────────────────────

MAX_HISTORY = 500

def load_history(path: Path) -> list:
    """Load the list of past notified items from disk."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return []


def append_history(path: Path, items: list):
    """Append newly notified items to the history file (capped at MAX_HISTORY)."""
    history = load_history(path)
    history.extend(items)
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False))


# ── Price anomaly statistics ──────────────────────────────────────────────────

PRICE_STATS_WINDOW = 500   # rolling window of last N prices
ANOMALIES_KEPT     = 50    # last N flagged anomalies stored


def _empty_price_stats() -> dict:
    return {
        "prices":       [],
        "sample_count": 0,
        "updated_at":   None,
        "anomalies":    [],
    }


def load_price_stats(path: Path) -> dict:
    """Load price statistics for a monitor."""
    if path.exists():
        try:
            data = json.loads(path.read_text())
            # ensure all keys present (forward-compat for older files)
            base = _empty_price_stats()
            base.update(data)
            return base
        except Exception:
            pass
    return _empty_price_stats()


def save_price_stats(path: Path, stats: dict):
    """Persist price statistics to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))


def update_price_stats(path: Path, new_prices: list) -> dict:
    """Append new prices to the rolling window and return the updated stats."""
    stats = load_price_stats(path)
    stats["prices"].extend(new_prices)
    if len(stats["prices"]) > PRICE_STATS_WINDOW:
        stats["prices"] = stats["prices"][-PRICE_STATS_WINDOW:]
    stats["sample_count"] = stats.get("sample_count", 0) + len(new_prices)
    stats["updated_at"]   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_price_stats(path, stats)
    return stats


def compute_median(prices: list):
    """Return the median of a price list, or None if empty."""
    if not prices:
        return None
    s = sorted(prices)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


# ── Tracked items (price snapshots for drop detection) ───────────────────────
#
# Parallel to `seen` (which is just a dedup set), `tracked` stores the last
# observed price for each item ID. On every scan, items still appearing in the
# feed get re-priced; if the new price drops below the baseline by more than
# the configured threshold, we emit a "price drop" event.

def load_tracked(path: Path) -> dict:
    """Load the {item_id: {price, title, url, last_seen_at}} map."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_tracked(path: Path, tracked: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tracked, ensure_ascii=False))


def update_tracked_and_detect_drops(
    path: Path,
    items: dict,
    threshold_pct: float,
    min_abs: float,
) -> list:
    """
    Upsert every scanned item into the tracked store. For items already tracked,
    compare current vs. stored price and emit a drop event when both:
        drop_pct >= threshold_pct  AND  drop_abs >= min_abs

    After emitting, the baseline is updated to the new lower price, so we don't
    re-alert until the price drops further.

    Returns a list of drop event dicts ready for notification / history.
    """
    now     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tracked = load_tracked(path)
    drops:    list = []

    for item_id, item in items.items():
        key       = str(item_id)
        cur_price = item.get("price") or 0
        if cur_price <= 0:
            continue

        prev = tracked.get(key)
        if prev is None:
            # First time we see this item — establish baseline, no drop
            tracked[key] = {
                "price":         cur_price,
                "title":         item.get("title", ""),
                "url":           item.get("url", ""),
                "first_seen_at": now,
                "last_seen_at":  now,
            }
            continue

        prev_price = prev.get("price") or 0
        if prev_price <= 0:
            prev["price"] = cur_price
            prev["last_seen_at"] = now
            continue

        drop_abs = prev_price - cur_price
        drop_pct = (drop_abs / prev_price) * 100 if prev_price else 0

        if threshold_pct and drop_pct >= threshold_pct and drop_abs >= min_abs:
            drops.append({
                "id":              item_id,
                "title":           item.get("title", ""),
                "price":           cur_price,
                "old_price":       prev_price,
                "drop_pct":        round(drop_pct),
                "drop_abs":        round(drop_abs, 2),
                "drop_str":        f"📉 €{int(prev_price)} → €{int(cur_price)} (-{round(drop_pct)}%)",
                "favourite_count": item.get("favourite_count", 0),
                "url":             item.get("url", ""),
                "event_type":      "price_drop",
            })
            # Update baseline to the new lower price so we don't re-alert
            # until the seller drops it further.
            prev["price"] = cur_price

        prev["last_seen_at"] = now
        prev["title"] = item.get("title", prev.get("title", ""))
        prev["url"]   = item.get("url",   prev.get("url",   ""))

    save_tracked(path, tracked)
    return drops


def record_anomaly(path: Path, item: dict, median: float):
    """Save a flagged anomaly to the stats file (last ANOMALIES_KEPT kept)."""
    stats = load_price_stats(path)
    stats["anomalies"].append({
        "price":        item.get("price"),
        "median":       median,
        "discount_pct": round((1 - item["price"] / median) * 100),
        "title":        item.get("title", ""),
        "url":          item.get("url", ""),
        "flagged_at":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    if len(stats["anomalies"]) > ANOMALIES_KEPT:
        stats["anomalies"] = stats["anomalies"][-ANOMALIES_KEPT:]
    save_price_stats(path, stats)
