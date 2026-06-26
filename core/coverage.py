"""Adaptive scrape-coverage controller.

A second control loop, orthogonal to core.rate_limiter:
  - rate_limiter answers "how fast can I call without getting blocked?"
    (signal = a block);
  - this answers "am I scraping ENOUGH to not miss items?"
    (signal = the overlap between two CONSECUTIVE scrapes).

The signal (user's idea, 2026-05-22): a category feed sorted newest_first should
overlap between consecutive scrapes. If THIS scrape shares no item IDs with the
PREVIOUS one, the whole window rolled over in between — more items were posted
than we fetched, so we silently lost some → fetch more pages next time. Lots of
overlap means we're re-fetching the same items → we can fetch fewer (save calls).

Crucially the comparison is against the previous scrape's IDs persisted on disk,
NOT a fixed time window: it holds across process restarts / downtime. After being
off for a while the next scrape simply sees little overlap and the controller
grows the page count to catch back up.

Bounded by `maximum` pages and by the block signal: if Vinted shows signs of
blocking, shrink regardless — a block beats coverage.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PER_PAGE = 96   # Vinted catalog page size; thresholds are expressed in "pages".

# A non-truncated scrape returns ~60-65 usable items/page (after de-dup). A
# cycle that comes back with far fewer than the page window asked for was cut
# short (rate-limiter granted ~1 page, or a partial block) — its overlap is a
# meaningless small sample, NOT a rollover signal.
EXPECTED_PER_PAGE = 60
THIN_SAMPLE_FRAC = 0.4   # < 40% of the expected window = truncated fetch


def _file(key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return DATA_DIR / f"coverage_{safe}.json"


def _load(key: str) -> dict:
    p = _file(key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save(key: str, data: dict) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _file(key).write_text(json.dumps(data))
    except Exception:
        pass


def state(key: str) -> dict:
    """Read-only snapshot of a source's coverage state (pages, overlap,
    updated_at). Returns {} if there's none yet. Used by the GUI to show the
    current DYNAMIC page count instead of the static config value."""
    return _load(key)


def pages_for(key: str, default: int, maximum: int) -> int:
    """How many pages to fetch THIS cycle — the value `record()` decided last
    cycle (clamped to [1, maximum]); `default` on the first ever run."""
    d = _load(key)
    p = int(d.get("pages", default))
    return max(1, min(maximum, p))


def record(key: str, current_ids, maximum: int,
           under_pressure: bool = False, default: int = 1) -> dict:
    """Compare `current_ids` to the previous scrape, decide next cycle's page
    count, and persist both. Returns a small dict for logging:
        {overlap, prev, pages_next, reason}.
    """
    d = _load(key)
    prev = set(d.get("last_ids", []))
    cur  = list(dict.fromkeys(current_ids))   # de-dup, keep order
    cur_set = set(cur)
    pages = int(d.get("pages", default))
    overlap = len(prev & cur_set)

    # A scrape is a non-signal when it returned NO items (failed/blocked) or far
    # FEWER items than the page window asked for (truncated: rate-limiter gave
    # ~1 page, or a partial block). In both cases `overlap` is meaningless — most
    # damagingly, a truncated scrape has overlap 0 not because items rolled over
    # but because we only sampled a handful. Treating that as "rollover→+2" both
    # bumps pages exactly when we're already constrained AND corrupts the
    # baseline (saving the thin sample as last_ids → cascading false rollovers,
    # seen 2026-05-24). So: do NOT bump, and do NOT overwrite last_ids — keep the
    # last good baseline for the next full scrape to compare against.
    thin = bool(prev) and len(cur) < max(1, int(THIN_SAMPLE_FRAC * pages * EXPECTED_PER_PAGE))
    if not cur or thin:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label = "empty" if not cur else "thin-sample"
        if under_pressure:
            pages = max(1, pages - 1)
            reason = f"{label}+block→shrink"
        else:
            reason = f"{label}→hold"
        d["pages"] = pages
        d["updated_at"] = now
        _save(key, d)
        return {"overlap": overlap, "prev": len(prev),
                "pages_next": pages, "reason": reason}

    if under_pressure:
        pages = max(1, pages - 1)
        reason = "block→shrink"
    elif not prev:
        reason = "baseline"                       # first scrape: nothing to compare
    elif overlap == 0:
        pages = min(maximum, pages + 2)           # full rollover → losing items
        reason = "rollover→+2"
    elif overlap < PER_PAGE:
        pages = min(maximum, pages + 1)           # < 1 page of carry-over → thin
        reason = "thin→+1"
    elif overlap > 3 * PER_PAGE:
        pages = max(1, pages - 1)                 # 3+ pages re-fetched → wasteful
        reason = "over→-1"
    else:
        reason = "hold"

    # Persist only as many IDs as we could plausibly compare next time, so the
    # file can't grow without bound.
    cap = maximum * PER_PAGE + PER_PAGE
    _save(key, {
        "last_ids":   cur[:cap],
        "pages":      pages,
        "overlap":    overlap,
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    return {"overlap": overlap, "prev": len(prev),
            "pages_next": pages, "reason": reason}
