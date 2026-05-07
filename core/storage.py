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
    """Load the set of already-notified item IDs from disk."""
    if path.exists():
        return set(json.loads(path.read_text()))
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
