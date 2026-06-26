"""
Shared Vinted scraper helpers concentrating four anti-rate-limit fixes:

  1. Per-domain Accept-Language header injection (vinted_scraper hardcodes
     en-US, which makes vinted.de respond 406 ~24% of the time). Implemented
     via monkey-patch of the package's header builders — they get imported by
     name into the wrapper module, so we patch both the source and the
     re-exported references.

  2. Scraper instance pool keyed by domain. The vanilla code instantiates a
     fresh VintedScraper per query, which means a brand-new cookie fetch on
     every call (3-9s each + extra bot signal). We cache one scraper per
     domain and let VintedWrapper auto-refresh the cookie on 401.

  3. `safe_search` wraps `scraper.search` with exponential backoff on HTTP 429
     so a single rate-limit response doesn't fail an entire scrape.

  4. The caller is expected to iterate domains sequentially (with a small
     pause) instead of firing them all in parallel — that part lives in the
     call sites (discovery / category_monitor) since this module doesn't own
     the loop.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from pathlib import Path

import vinted_scraper.utils._misc as _misc_module
import vinted_scraper._wrapper    as _wrapper_module
from vinted_scraper import VintedScraper

from . import rate_limiter as _rl

# ─── 0. Browser-impersonating transport (curl_cffi) ─────────────────────────
#
# Swap vinted_scraper's httpx.Client for a curl_cffi client that replays a real
# Chrome TLS + HTTP/2 fingerprint. Vinted is fronted by DataDome, which scores
# the JA3/JA4 + HTTP/2 fingerprint before headers — a plain httpx client reads
# as "Python bot" no matter how good the headers are. This is the single
# highest-leverage change against the block. If curl_cffi isn't installed we
# leave the hook untouched and the library keeps using httpx (degraded but
# functional), mirroring the optional-dependency pattern used elsewhere.
try:
    from .curl_transport import (
        make_client as _make_curl_client,
        CURL_CFFI_AVAILABLE,
        IMPERSONATE,
    )
except Exception:  # pragma: no cover - defensive: never break import
    CURL_CFFI_AVAILABLE = False
    IMPERSONATE = None

if CURL_CFFI_AVAILABLE:
    _wrapper_module.CLIENT_FACTORY = _make_curl_client


# ─── Session cookie persistence ─────────────────────────────────────────────
#
# Re-fetching the Vinted session cookie on every `python gui.py` restart is
# what gets us into 403 spirals: the cookie endpoint is the most heavily
# scored by the anti-bot. We persist cookies after a successful fetch and
# reuse them up to COOKIE_TTL_S — the wrapper auto-refreshes on 401 anyway.

BASE_DIR     = Path(__file__).resolve().parent.parent
COOKIE_DIR   = BASE_DIR / "data"
COOKIE_TTL_S = 30 * 60   # 30 minutes — Vinted sessions are usually good ≥1h


def _cookie_file(domain: str) -> Path:
    safe = domain.replace("https://", "").replace("http://", "").replace("/", "_")
    return COOKIE_DIR / f"cookies_{safe}.json"


def _load_disk(domain: str) -> tuple[dict | None, list | None]:
    """Return (access_token_dict | None, jar_list | None) from disk if fresh.

    New format stores the full cookie jar under "jar" so the datadome cookie
    survives process restarts; legacy files stored only "cookies" {name: value}.
    Both are handled."""
    path = _cookie_file(domain)
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text())
        if time.time() - payload.get("saved_at", 0) > COOKIE_TTL_S:
            return None, None
        jar = payload.get("jar")
        if jar:
            tok = {c["name"]: c["value"] for c in jar
                   if c.get("name") == "access_token_web" and c.get("value")}
            return (tok or None), jar
        cookies = payload.get("cookies")
        if isinstance(cookies, dict) and cookies:
            return cookies, None
    except Exception:
        return None, None
    return None, None


def _persist_session(domain: str, scraper) -> None:
    """Persist cookies for `domain`. With the curl_cffi adapter we dump the full
    jar (so datadome is reused by the next process); on the httpx fallback we
    store the access_token dict, as before. Failures are swallowed — persistence
    is an optimization, never required for correctness."""
    client = getattr(scraper, "_client", None)
    try:
        if client is not None and hasattr(client, "dump_cookies"):
            jar = client.dump_cookies()
            if not jar:
                return
            payload = {"jar": jar, "saved_at": time.time()}
        else:
            cookies = getattr(scraper, "session_cookie", None)
            if not cookies:
                return
            payload = {"cookies": cookies, "saved_at": time.time()}
        path = _cookie_file(domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except Exception:
        pass


# ─── 1. Per-domain Accept-Language ──────────────────────────────────────────

LOCALE_BY_DOMAIN: dict[str, str] = {
    "vinted.it":  "it-IT,it;q=0.9,en;q=0.7",
    "vinted.fr":  "fr-FR,fr;q=0.9,en;q=0.7",
    "vinted.de":  "de-DE,de;q=0.9,en;q=0.7",
    "vinted.es":  "es-ES,es;q=0.9,en;q=0.7",
    "vinted.pl":  "pl-PL,pl;q=0.9,en;q=0.7",
    "vinted.com": "en-US,en;q=0.9",
    "vinted.co.uk": "en-GB,en;q=0.9",
}


def _accept_lang_for(base_url: str) -> str:
    for key, lang in LOCALE_BY_DOMAIN.items():
        if key in base_url:
            return lang
    return "en-US,en;q=0.5"


_ORIG_COOKIE = _misc_module.get_cookie_headers
_ORIG_CURL   = _misc_module.get_curl_headers


def _patched_cookie(base_url: str, user_agent: str):
    h = _ORIG_COOKIE(base_url, user_agent)
    h["Accept-Language"] = _accept_lang_for(base_url)
    return h


def _patched_curl(base_url: str, user_agent: str, session_cookies):
    h = _ORIG_CURL(base_url, user_agent, session_cookies)
    h["Accept-Language"] = _accept_lang_for(base_url)
    return h


# Patch the source module …
_misc_module.get_cookie_headers = _patched_cookie
_misc_module.get_curl_headers   = _patched_curl
# … and the re-exported names captured by the wrapper at import time.
_wrapper_module.get_cookie_headers = _patched_cookie
_wrapper_module.get_curl_headers   = _patched_curl


# ─── 2. Scraper pool + shared cross-process rate limiting / block state ──────
#
# Pacing and block state now live in core.rate_limiter — a file-coordinated
# limiter shared by ALL monitor subprocesses (see that module's docstring).
# This module only orchestrates multi-domain fallback in try_domains; the
# limiter owns the per-domain request PACE (GCRA + AIMD) and the per-domain
# `blocked_until` cooldowns plus a global "all blocked" level.
#
# The previous per-process globals (_round_blocked / _global_level) are gone:
# being per-process, they could not coordinate across monitors, which is exactly
# what let several monitors together trip Vinted's anti-bot. A block hit by ANY
# process now backs every process off that domain.

_pool: dict[str, VintedScraper] = {}
_pool_lock = threading.Lock()


BLOCK_STATUSES = ("403", "406", "429", "503")


def _is_block_error(exc: BaseException) -> bool:
    """True if the exception signals a Vinted-side block: either our
    `_BlockedDomain`, or any error whose message mentions a block status code
    (catches the RuntimeError vinted_scraper raises on a non-200)."""
    if exc.__class__.__name__ == "_BlockedDomain":
        return True
    msg = str(exc)
    return any(code in msg for code in BLOCK_STATUSES)


def _retry_after_from_exc(exc: BaseException) -> float | None:
    """Pull a Retry-After (seconds) off a block exception when present.
    `_BlockedDomain` carries it directly (parsed from the response headers);
    other exceptions don't expose headers, so we return None and let the limiter
    fall back to its escalating cooldown schedule."""
    ra = getattr(exc, "retry_after", None)
    if ra is None:
        return None
    try:
        return float(ra)
    except (TypeError, ValueError):
        return None


def get_scraper(domain: str) -> VintedScraper:
    """Return a cached VintedScraper for the given domain (single-shot).

    On first access, instantiates VintedScraper (which fetches a session
    cookie). Cached session cookies on disk are reused when fresh enough,
    sparing us a fresh cookie-fetch that often trips the anti-bot.

    Block errors (403/406/429/503) bubble up to the caller (`try_domains` is
    the orchestrator that does multi-domain fallback + arithmetic timer).
    """
    with _pool_lock:
        sc = _pool.get(domain)
    if sc is not None:
        return sc

    tok, jar = _load_disk(domain)
    try:
        # Passing session_cookie (even just the access token) makes the wrapper
        # skip its GET / cookie-fetch — one fewer DataDome-scored request.
        sc = VintedScraper(domain, session_cookie=tok) if tok else VintedScraper(domain)
    except Exception:
        # Single-shot: let try_domains decide whether to backoff/retry.
        raise
    # Seed the curl_cffi jar from disk so datadome (and the access token) ride
    # along on every request. No-op on the httpx fallback (no load_cookies).
    client = getattr(sc, "_client", None)
    if client is not None and hasattr(client, "load_cookies"):
        if jar:
            client.load_cookies(jar)
        elif tok:
            client.seed(tok)
    _persist_session(domain, sc)
    with _pool_lock:
        _pool[domain] = sc
    return sc


def reset_scraper(domain: str) -> None:
    """Drop the cached scraper for `domain` so the next call rebuilds it from
    scratch (fresh UA + fresh cookie). Call this if a hard error suggests the
    instance is permanently broken (e.g. repeated 403)."""
    with _pool_lock:
        _pool.pop(domain, None)


# ─── 3. Safe search with arithmetic backoff ─────────────────────────────────


# Content marker for an ACTIVE Vinted item: the public item page renders a
# "Buy now" button with data-testid="item-buy-button" only when the listing
# is for sale. Sold / reserved / withdrawn items omit it entirely. We rely
# on the *absence* of the buy button rather than the presence of the
# "sold pill" (`data-testid="item-status"`) because empirical testing
# 2026-05-19 showed the sold pill's byte position varies wildly between
# requests (~100KB on one fetch, ~2.27MB on another — Vinted's React
# hydration reshapes the HTML layout). The buy-button marker, when present,
# is reliable; its position is stable near the end of the page (~98%).
ACTIVE_MARKER = b'data-testid="item-buy-button"'


class _BlockedDomain(RuntimeError):
    """Raised by `_fetch_item_page` so `try_domains` can recognize a block
    without parsing exception text. Carries the HTTP status that signalled the
    block (-1 for transport errors that look like a block) and an optional
    Retry-After (seconds) parsed from the response headers."""
    def __init__(self, status: int, retry_after: float | None = None):
        super().__init__(f"vinted block status {status}")
        self.status = status
        self.retry_after = retry_after


def _parse_retry_after(resp) -> float | None:
    """Best-effort Retry-After (seconds) from a response. DataDome/Vinted send
    it as an integer second-count; the HTTP-date form is ignored (→ None, the
    limiter then uses its escalating cooldown schedule)."""
    try:
        raw = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    except Exception:
        return None
    if not raw:
        return None
    try:
        v = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _fetch_item_page(domain: str, item_id: int):
    """
    Fetch /items/{id} once. Returns the httpx Response on success.

    Raises `_BlockedDomain` on a 403/406/429/503 (or a transport exception
    whose message matches the block-status set), so `try_domains` can
    fall over to the next domain or trigger the global timer. Other
    failures return None (treated as "unknown" by the caller).

    Each request downloads the full ~2.3MB item page. Vinted ignores Range
    headers (verified 2026-05-19) and the marker positions are not stable,
    so we can't stream-abort early. follow_redirects=True is required:
    Vinted ALWAYS 301-redirects /items/{id} → /items/{id}-slug (SEO).
    """
    scraper = get_scraper(domain)
    try:
        resp = scraper._client.get(   # type: ignore[attr-defined]
            f"/items/{item_id}",
            follow_redirects=True,
            timeout=20,
        )
    except Exception as e:
        if _is_block_error(e):
            raise _BlockedDomain(-1) from e
        return None
    if resp.status_code in (403, 406, 429, 503):
        raise _BlockedDomain(resp.status_code, _parse_retry_after(resp))
    # Successful fetch → opportunistically persist the jar (Vinted may have
    # refreshed datadome / the access token on this response).
    _persist_session(domain, scraper)
    return resp


def check_item_status(domain: str, item_id: int) -> str:
    """Single-domain sold/active check (raises `_BlockedDomain` on block —
    let the orchestrator catch it). For automatic multi-domain fallback,
    prefer `check_item_status_any`."""
    resp = _fetch_item_page(domain, item_id)
    if resp is None:
        return "unknown"
    final_path = str(resp.url.path or "")
    # A SOLD item keeps its page (HTTP 200) with a green "Venduto" banner and NO
    # buy button — caught by the 200-without-ACTIVE_MARKER branch below. A 404 /
    # gone page means the listing was REMOVED (seller deleted / withdrew it), which
    # is NOT the same as sold — verified 2026-05-22. Removed items must be excluded
    # from sold stats, so they get their own status.
    if "/items/" not in final_path:
        return "removed"
    sc = resp.status_code
    if sc in (404, 410):
        return "removed"
    if sc != 200:
        return "unknown"
    if ACTIVE_MARKER in resp.content:
        return "active"
    return "sold"


def check_item_status_any(domains: list[str], item_id: int) -> str:
    """Multi-domain `check_item_status` with `try_domains` orchestration:
    rotates through `domains`, falls over to the next on block, and triggers
    the global arithmetic timer once all domains are blocked. Returns
    "unknown" if everything fails and the timer hits its cap."""
    result, _used, _errs = try_domains(
        domains, lambda d: check_item_status(d, item_id),
        key="item_pages", lane="item",
    )
    return result if result is not None else "unknown"


def check_item_exists(domain: str, item_id: int) -> bool | None:
    """Back-compat wrapper around `check_item_status`.
    True=active, False=sold/removed (not live), None=unknown."""
    status = check_item_status(domain, item_id)
    if status == "active":
        return True
    if status in ("sold", "removed"):
        return False
    return None


# Localized "United States" strings as they appear in the embedded user JSON
# on a Vinted item page. The JSON is double-escaped because it lives inside
# an HTML script payload — each quote becomes `\"` in the response body.
# The value follows the locale of the domain that served the page (IT, FR,
# DE, EN). We check all four to be domain-agnostic.
US_COUNTRY_MARKERS = (
    b'\\"country_title_local\\":\\"Stati Uniti\\"',           # IT
    b'\\"country_title_local\\":\\"\\u00c9tats-Unis\\"',      # FR (États-Unis, \u escaped)
    b'\\"country_title_local\\":\\"\xc3\x89tats-Unis\\"',     # FR (raw UTF-8 form)
    b'\\"country_title_local\\":\\"United States\\"',         # EN
    b'\\"country_title_local\\":\\"Vereinigte Staaten\\"',    # DE
)


def is_us_seller(html_bytes: bytes) -> bool:
    """Return True if the embedded user JSON on a Vinted item page marks the
    seller as based in the United States. Looks for `country_title_local`
    set to any localized variant of "United States"."""
    return any(m in html_bytes for m in US_COUNTRY_MARKERS)


# Price marker — use `originalAskingAmount` (always EUR, converted server-side
# for US sellers) instead of the raw `price` block (whose currency varies).
# This is the same value our catalog-feed scrape stores in items.price, so
# comparisons against the DB are apples-to-apples.
#   raw bytes:  \"originalAskingAmount\":{\"amount\":\"104.29\",\"currencyCode\":\"EUR\"}
_PRICE_AMOUNT_RE = re.compile(
    rb'\\"originalAskingAmount\\":\{\\"amount\\":\\"([0-9]+(?:\.[0-9]+)?)\\"',
)


def _extract_price(body: bytes) -> float | None:
    """Parse the current EUR listing price out of the item page payload.
    Returns the converted EUR amount even for non-EUR sellers (matches what
    a catalog-feed scrape stores in items.price)."""
    m = _PRICE_AMOUNT_RE.search(body)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


# Leaf catalog_id of the item, embedded in the page JSON as `"catalog_id":N`.
_CATALOG_ID_RE = re.compile(rb'\\"catalog_id\\":([0-9]+)')


# Like count, embedded in the page JSON as `\"favourite_count\":N` (double-
# escaped, same form as catalog_id). Refreshing it on each sold-check keeps the
# ♥-priority working on fresh data even for items that fell out of the feed.
_FAVOURITES_RE = re.compile(rb'\\"favourite_count\\":([0-9]+)')


def _extract_favourites(body: bytes) -> int | None:
    m = _FAVOURITES_RE.search(body)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


# Item photo URLs in the page. The item's OWN photos share one upload timestamp
# in the path (.../f800/<ts>.webp); recommended-items photos have other
# timestamps — so we keep only the FIRST timestamp's URLs (the main item block,
# rendered first). FREE: parses the page the sold-check already downloaded.
_PHOTO_RE = re.compile(rb'https://images\d+\.vinted\.net/[^\s"\\]+?/(\d+)\.webp\?s=[0-9a-f]+')


def _extract_photo_urls(body: bytes, limit: int = 4) -> str | None:
    by_ts: dict[str, list[str]] = {}
    for m in _PHOTO_RE.finditer(body):
        ts = m.group(1).decode()
        url = m.group(0).decode()
        urls = by_ts.setdefault(ts, [])
        if url not in urls:
            urls.append(url)
    if not by_ts:
        return None
    first_ts = next(iter(by_ts))   # dict keeps insertion order → item's own batch
    return ",".join(by_ts[first_ts][:limit]) or None


def _extract_category(body: bytes) -> tuple[int | None, str | None]:
    """Extract the item's Vinted leaf category from the page HTML:
      - catalog_id (int): the leaf sub-category id (Vinted's own taxonomy)
      - category   (str): the English category name, recovered from the
        breadcrumb URL slug `/catalog/<id>-<english-slug>` (free, accurate —
        no translation needed).
    Returns (None, None) if not found."""
    m = _CATALOG_ID_RE.search(body)
    if not m:
        return None, None
    try:
        cid = int(m.group(1))
    except (TypeError, ValueError):
        return None, None
    slug_re = re.compile(rb'/catalog/' + m.group(1) + rb'-([a-z0-9-]+)')
    sm = slug_re.search(body)
    category = sm.group(1).decode().replace("-", " ") if sm else None
    return cid, category


def check_item_page(domain: str, item_id: int) -> dict:
    """Single-domain rich check (raises `_BlockedDomain` on block).

    Returns a dict with:
        status:        "active" | "sold" | "removed" | "unknown"
        is_us_seller:  bool | None   (None when status="unknown")
        price:         float | None  (EUR price as shown in the catalog feed)
        http_status:   int | None    (final HTTP status code, or None on error)

    "sold" = page still up (HTTP 200) with the "Venduto" banner and no buy
    button; "removed" = 404/gone (seller deleted/withdrew the listing — NOT a
    sale, kept out of sold stats). For multi-domain fallback, use
    `check_item_page_any`.
    """
    out = {"status": "unknown", "is_us_seller": None, "price": None,
           "catalog_id": None, "category": None, "favourites": None,
           "photo_urls": None, "http_status": None}
    resp = _fetch_item_page(domain, item_id)
    if resp is None:
        return out
    out["http_status"] = resp.status_code
    final_path = str(resp.url.path or "")
    if "/items/" not in final_path:
        out["status"] = "removed"
        return out
    sc = resp.status_code
    if sc in (404, 410):
        out["status"] = "removed"
        return out
    if sc != 200:
        return out
    body = resp.content
    out["status"]       = "active" if ACTIVE_MARKER in body else "sold"
    out["is_us_seller"] = is_us_seller(body)
    out["price"]        = _extract_price(body)
    out["favourites"]   = _extract_favourites(body)
    cid, category       = _extract_category(body)
    out["catalog_id"]   = cid
    out["category"]     = category
    out["photo_urls"]   = _extract_photo_urls(body)
    return out


def check_item_page_any(domains: list[str], item_id: int) -> dict:
    """Multi-domain `check_item_page` with `try_domains` orchestration. See
    `check_item_status_any` for the block-handling semantics."""
    result, _used, _errs = try_domains(
        domains, lambda d: check_item_page(d, item_id),
        key="item_pages", lane="item",
    )
    if result is None:
        return {"status": "unknown", "is_us_seller": None,
                "price": None, "http_status": None}
    return result


def try_domains(domains, action, key=None, lane="default"):
    """
    Try `action(domain)` against the configured domains in randomized order,
    coordinated across ALL monitor processes via core.rate_limiter.

    `key` / `lane` select the shared pace + cooldown bucket:
      - key=None (default): per-DOMAIN bucket, `lane="default"` — the catalog/
        search API path. A block cools down only that domain and we fall through
        to the next one.
      - key="<name>", lane="item": a SINGLE IP-global bucket shared by all the
        domains — used for the /items/{id} HTML sold-check, which DataDome
        rate-limits by IP (a block hits every domain at once, so rotating is
        pointless; we just slow down and back off globally).

    Per attempt: skip if the bucket is in cooldown (`is_blocked`), pace via
    `acquire` (GCRA), then on success `on_success` (AIMD up) / on a 403/406/429/
    503 block `on_block` (AIMD down + shared cooldown). When the bucket(s) are all
    in cooldown, sleep until the soonest frees up, bump the global level, and
    retry — giving up at GLOBAL_MAX_LEVEL so a permanent ban can't loop forever.

    Returns (result, used_domain, errors).
    """
    if not domains:
        return None, None, {}

    # Buckets to watch for the "all blocked" sleep: one shared key, or per-domain.
    block_keys = [key] if key else list(domains)
    errors: dict[str, str] = {}
    while True:
        order = list(domains)
        random.shuffle(order)
        attempted = 0
        for d in order:
            bkey = key or d
            if _rl.is_blocked(bkey):
                continue
            attempted += 1
            # Pace against BOTH this lane's bucket and the global IP bucket, so
            # catalog and sold-check coordinate on the one shared IP.
            _rl.acquire_multi([(bkey, lane), (_rl.IP_KEY, "ip")])
            try:
                result = action(d)
                _rl.on_success(bkey, lane=lane)
                _rl.on_success(_rl.IP_KEY, lane="ip")
                return result, d, errors
            except Exception as e:
                if _is_block_error(e):
                    errors[d] = str(e)
                    ra = _retry_after_from_exc(e)
                    cd = _rl.on_block(bkey, ra, lane=lane)
                    # IP bucket: slow the shared pace too, but no cooldown (the
                    # per-lane key already owns the cooldown for this domain).
                    _rl.on_block(_rl.IP_KEY, ra, lane="ip", set_cooldown=False)
                    short = d.replace("https://www.", "")
                    print(f"  ✗ {short}: blocked (cooldown {int(cd)}s), "
                          f"trying next domain", flush=True)
                    # A shared (IP-global) bucket means this block applies to all
                    # domains — no point trying the rest this round.
                    if key:
                        break
                    continue
                # Non-block error: surface immediately, don't burn the round.
                errors[d] = str(e)
                return None, None, errors
        # Reached here = every watched bucket is in a shared cooldown.
        if attempted == 0 and not _rl.all_blocked(block_keys):
            # Empty set, or nothing tried and not all actually blocked.
            return None, None, errors
        wait  = max(1.0, _rl.cooldown_remaining(block_keys))
        level = _rl.bump_global()
        human = f"{int(wait // 60)} min" if wait >= 60 else f"{int(wait)}s"
        print(f"  ⏸  ALL DOMAINS BLOCKED — sleep {human} "
              f"(level {level}/{_rl.GLOBAL_MAX_LEVEL})", flush=True)
        time.sleep(wait)
        errors = {}    # fresh round → fresh error log
        if level >= _rl.GLOBAL_MAX_LEVEL:
            return None, None, {"_giveup": f"global block level {level}"}


def safe_search(domain: str, params: dict) -> list:
    """
    `scraper.search(params)` — single-shot. Block errors bubble up so the
    `try_domains` orchestrator (and its shared rate_limiter) can fall through to
    the next domain or apply the shared cooldown. Cookies are persisted after
    each success to extend their useful life across process restarts.
    """
    scraper = get_scraper(domain)
    result  = scraper.search(params)
    _persist_session(domain, scraper)
    return result
