"""Cross-process, file-coordinated rate limiter + block state for Vinted.

Why this exists: each monitor runs as a SEPARATE subprocess, so the old
per-process backoff in vinted_client never coordinated across monitors — N
processes hammered one IP with no shared budget, the #1 reason Vinted blocked us
when several searches + the category monitor ran together. This module moves the
budget and the block state onto disk (one JSON file guarded by an flock) so every
monitor process shares a per-key request PACE and a per-key cooldown.

Lanes (live finding 2026-05-22):
  - the "default" lane is keyed per DOMAIN and is for the catalog/search API,
    which the curl_cffi fingerprint made reliable;
  - the "item" lane is for full /items/{id} HTML pages (the sold-check). DataDome
    rate-limits those by IP — ~15 page loads/min trips it on ALL domains at once,
    regardless of domain or fingerprint. So the item lane is keyed GLOBALLY (one
    bucket for every domain) and paced far more conservatively. Rotating domains
    doesn't help an IP-level block; only going slower (or, Phase 3, proxies) does.

Design notes:
  - GCRA (a leaky-bucket "virtual scheduling" meter) spaces requests ≥ interval
    apart GLOBALLY: each caller reserves the next slot by pushing `next_available`
    forward under the lock, so concurrent processes serialize cleanly.
  - AIMD: a success nudges the rate up (additive), a block halves it
    (multiplicative), each clamped to its lane's [min, max].
  - Retry-After is honoured when the caller passes it.

Degrades safely: if `fcntl` is missing (Windows) we fall back to an in-process
lock only — no cross-process coordination, but nothing crashes.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from pathlib import Path

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:                       # pragma: no cover - Windows
    _HAS_FCNTL = False

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "ratelimit_state.json"
LOCK_FILE  = DATA_DIR / "ratelimit.lock"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ── Lanes: (init, min, max) request rate in req/sec ─────────────────────────
# default: per-domain, catalog/search API (curl_cffi made this reliable).
RATE_INIT = _env_float("VINTED_RATE_INIT", 0.40)   # ~1 every 2.5s
RATE_MIN  = _env_float("VINTED_RATE_MIN",  0.066)  # ~1 every 15s under pressure
RATE_MAX  = _env_float("VINTED_RATE_MAX",  0.83)   # ~1 every 1.2s at best
# item: IP-global, full HTML item pages (sold-check). DataDome blocked at
# ~0.78/s aggregate; MAX is set generously so the AIMD can DISCOVER the real
# sustainable ceiling dynamically (it won't sit at MAX — it converges just under
# the learned safe rate, see below). Lower MAX if you want a harder safety cap.
ITEM_RATE_INIT = _env_float("VINTED_ITEM_RATE_INIT", 0.13)  # ~1 every 7.5s (start gentle)
ITEM_RATE_MIN  = _env_float("VINTED_ITEM_RATE_MIN",  0.04)  # ~1 every 25s floor
ITEM_RATE_MAX  = _env_float("VINTED_ITEM_RATE_MAX",  0.60)  # ceiling for discovery, ~1 every 1.7s

# ip: a GLOBAL bucket every request also passes through, on top of its own lane.
# DataDome rate-limits the IP holistically, so catalog + item requests share it:
# when the scrape bursts it consumes the IP budget and the sold-check slows; when
# the scrape is idle the sold-check gets the whole budget and speeds up; a block
# on EITHER lane backs the IP rate off (and recovery speeds it). The per-lane
# bounds still apply on top, so this only adds coordination, never raises a lane.
IP_KEY        = "ip"
IP_RATE_INIT  = _env_float("VINTED_IP_RATE_INIT", 0.50)   # total IP req/s to start
IP_RATE_MIN   = _env_float("VINTED_IP_RATE_MIN",  0.10)
IP_RATE_MAX   = _env_float("VINTED_IP_RATE_MAX",  1.00)

LANES: dict[str, tuple[float, float, float]] = {
    "default": (RATE_INIT,      RATE_MIN,      RATE_MAX),
    "item":    (ITEM_RATE_INIT, ITEM_RATE_MIN, ITEM_RATE_MAX),
    "ip":      (IP_RATE_INIT,   IP_RATE_MIN,   IP_RATE_MAX),
}

# AIMD with a LEARNED ceiling (TCP-ssthresh style), so throughput tracks how
# much traffic Vinted currently tolerates instead of sitting at a fixed number:
#   - `rate` climbs fast (INC_RAMP) toward `safe_rate`, then is capped there;
#   - every PROBE_EVERY consecutive successes `safe_rate` creeps up by
#     INC_PROBE_STEP (re-probe: capacity may have freed up — fewer monitors, etc.);
#   - on a block `safe_rate` drops to SAFE_BACKOFF× the rate that broke and
#     `rate` is halved (RATE_DEC). Since each block costs a 60s+ cooldown, riding
#     just *under* the learned ceiling (not on it) maximizes useful throughput.
INC_RAMP       = _env_float("VINTED_RATE_RAMP",  0.08)  # fast climb toward safe_rate
INC_PROBE_STEP = _env_float("VINTED_RATE_PROBE", 0.02)  # ceiling re-probe step
PROBE_EVERY    = 50                                     # successes between re-probes
                                                       # (higher = fewer probes = fewer
                                                       #  60s-cooldown blocks)
SAFE_BACKOFF   = 0.7                                    # learned ceiling = 0.7× blocking
                                                       #  rate — stay a margin below it
RATE_DEC = _env_float("VINTED_RATE_DEC", 0.5)      # multiplicative decrease per block
JITTER   = 0.2                                      # ±20% on the interval (anti-metronome)

# Cooldown applied to a key that just blocked, when no Retry-After is given,
# escalating with consecutive blocks. The longer tail (up to 1h) matters for a
# HARD DataDome IP-block: poking a penalised IP every 30 min keeps the block
# alive, so on sustained blocks we back off progressively longer to let it clear.
COOLDOWN_SCHEDULE = [60, 300, 900, 1800, 3600]     # 1m, 5m, 15m, 30m, 1h
COOLDOWN_MAX      = 3600
GLOBAL_MAX_LEVEL  = 8                               # giveup after N "all blocked" cycles

_thread_lock = threading.Lock()


def _blank() -> dict:
    return {"keys": {}, "global_level": 0}


def _entry(state: dict, key: str, lane: str = "default") -> dict:
    """Get-or-create the state entry for `key`, setting its lane authoritatively
    (the caller — acquire/on_success/on_block — always knows the right lane).
    Use `_peek` for read-only checks so they never create an entry under the
    wrong lane (is_blocked runs BEFORE acquire in try_domains)."""
    e = state["keys"].get(key)
    init, _lo, _hi = LANES.get(lane, LANES["default"])
    if e is None:
        e = {"lane": lane, "rate": init}
        state["keys"][key] = e
    e["lane"] = lane                       # authoritative
    e.setdefault("rate",               init)
    e.setdefault("safe_rate",          min(_hi, init * 2))  # learned ceiling; start
                                                            # ~2× init, not at MAX, to
                                                            # avoid a guaranteed overshoot
                                                            # block on a fresh start
    e.setdefault("ok_streak",          0)
    e.setdefault("next_available",     0.0)
    e.setdefault("blocked_until",      0.0)
    e.setdefault("consecutive_blocks", 0)
    return e


def _peek(state: dict, key: str) -> dict | None:
    """Read-only lookup — never creates an entry (so it can't pin the wrong
    lane). Returns None for an untouched key."""
    return state["keys"].get(key)


class _LockedState:
    """Context manager: take the inter-process flock + in-process lock, yield the
    parsed state dict, and atomically write it back on exit."""

    def __enter__(self) -> dict:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _thread_lock.acquire()
        self._fh = open(LOCK_FILE, "a+")
        if _HAS_FCNTL:
            fcntl.flock(self._fh, fcntl.LOCK_EX)
        try:
            self.state = json.loads(STATE_FILE.read_text())
            if not isinstance(self.state, dict) or "keys" not in self.state:
                self.state = _blank()
        except Exception:
            self.state = _blank()
        return self.state

    def __exit__(self, *_exc) -> None:
        try:
            tmp = STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state))
            tmp.replace(STATE_FILE)            # atomic publish
        except Exception:
            pass
        finally:
            if _HAS_FCNTL:
                try:
                    fcntl.flock(self._fh, fcntl.LOCK_UN)
                except Exception:
                    pass
            self._fh.close()
            _thread_lock.release()


def acquire(key: str, lane: str = "default") -> float:
    """Block until this process may send a request on `key`, honouring the GLOBAL
    pace (GCRA). Returns the seconds waited. Callers check `is_blocked` first;
    this only meters pace, it does not enforce cooldowns."""
    with _LockedState() as st:
        e = _entry(st, key, lane)
        now = time.time()
        interval = (1.0 / max(e["rate"], 1e-6)) * random.uniform(1 - JITTER, 1 + JITTER)
        na   = max(e["next_available"], now)
        wait = na - now
        e["next_available"] = na + interval
    if wait > 0:
        time.sleep(wait)
    return wait


def acquire_multi(specs) -> float:
    """Pace a request that must respect SEVERAL buckets at once — e.g. its lane
    AND the global IP bucket. Reserves the next GCRA slot on each (under one
    lock) and sleeps once for the MAX of the individual waits, so the request
    honours every bucket without over-waiting. `specs` = [(key, lane), ...]."""
    with _LockedState() as st:
        now = time.time()
        max_wait = 0.0
        for key, lane in specs:
            e = _entry(st, key, lane)
            interval = (1.0 / max(e["rate"], 1e-6)) * random.uniform(1 - JITTER, 1 + JITTER)
            na = max(e["next_available"], now)
            w  = na - now
            e["next_available"] = na + interval
            if w > max_wait:
                max_wait = w
    if max_wait > 0:
        time.sleep(max_wait)
    return max_wait


def is_blocked(key: str) -> bool:
    with _LockedState() as st:
        e = _peek(st, key)
        return e is not None and time.time() < e.get("blocked_until", 0.0)


def on_success(key: str, lane: str = "default") -> None:
    """Climb `rate` fast toward the learned `safe_rate` (capped there), and every
    PROBE_EVERY consecutive successes nudge `safe_rate` up to re-discover freed
    capacity. Clears the cooldown and resets the global level."""
    with _LockedState() as st:
        e = _entry(st, key, lane)
        _init, lo, hi = LANES.get(e["lane"], LANES["default"])
        safe = min(hi, max(lo, e.get("safe_rate", hi)))
        e["rate"] = min(safe, e["rate"] + INC_RAMP)
        streak = e.get("ok_streak", 0) + 1
        if streak >= PROBE_EVERY:
            streak = 0
            e["safe_rate"] = min(hi, safe + INC_PROBE_STEP)   # re-probe the ceiling
        else:
            e["safe_rate"] = safe
        e["ok_streak"] = streak
        e["blocked_until"] = 0.0
        e["consecutive_blocks"] = 0
        st["global_level"] = 0


def on_block(key: str, retry_after: float | None = None,
             lane: str = "default", set_cooldown: bool = True) -> float:
    """Learn the ceiling (`safe_rate` = SAFE_BACKOFF× the rate that just broke)
    and halve `rate`. When `set_cooldown` is True also set a cooldown (Retry-After
    if given, else the escalating schedule) and return it. The IP coordination
    bucket calls this with set_cooldown=False — it only slows the shared pace, it
    must not park a cooldown (the per-lane key already owns that)."""
    with _LockedState() as st:
        e = _entry(st, key, lane)
        _init, lo, hi = LANES.get(e["lane"], LANES["default"])
        e["safe_rate"] = max(lo, min(e.get("safe_rate", hi), e["rate"] * SAFE_BACKOFF))
        e["rate"] = max(lo, e["rate"] * RATE_DEC)
        e["ok_streak"] = 0
        if not set_cooldown:
            return 0.0
        n = e["consecutive_blocks"]
        e["consecutive_blocks"] = n + 1
        if retry_after and retry_after > 0:
            cd = min(float(retry_after), COOLDOWN_MAX)
        else:
            cd = COOLDOWN_SCHEDULE[min(n, len(COOLDOWN_SCHEDULE) - 1)]
        e["blocked_until"] = time.time() + cd
        return cd


def all_blocked(keys) -> bool:
    now = time.time()
    with _LockedState() as st:
        return all((e := _peek(st, k)) is not None
                   and now < e.get("blocked_until", 0.0) for k in keys)


def cooldown_remaining(keys) -> float:
    """Seconds until the SOONEST of `keys` unblocks (so a caller can sleep exactly
    long enough), capped at COOLDOWN_MAX. 0 if none are blocked."""
    now = time.time()
    with _LockedState() as st:
        until = [(e.get("blocked_until", 0.0) if (e := _peek(st, k)) else 0.0)
                 for k in keys]
    future = [b - now for b in until if b > now]
    return min(min(future), COOLDOWN_MAX) if future else 0.0


def bump_global() -> int:
    with _LockedState() as st:
        st["global_level"] = min(st.get("global_level", 0) + 1, GLOBAL_MAX_LEVEL)
        return st["global_level"]


def global_level() -> int:
    with _LockedState() as st:
        return st.get("global_level", 0)


def snapshot() -> dict:
    """Read-only view of the current state (for logging / a future GUI panel)."""
    with _LockedState() as st:
        return json.loads(json.dumps(st))
