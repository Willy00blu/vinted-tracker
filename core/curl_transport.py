"""curl_cffi transport for vinted_scraper — browser TLS/HTTP2 impersonation.

Vinted is fronted by DataDome, which scores the TLS (JA3/JA4) and HTTP/2
fingerprint *before* it looks at headers. A stdlib/httpx client is
unmistakably "Python bot" (HTTP/1.1, JA4 `...h1...`, UA `python-httpx`);
curl_cffi replays a real Chrome fingerprint (HTTP/2, browser JA3/JA4 + Akamai
hash). That swap is the single biggest lever against the block — verified on
a JA3 echo: httpx JA4 `t13d1812h1_…` vs curl_cffi `t13d1516h2_…` (real Chrome).

This module exposes the small slice of the `httpx.Client` API that
`vinted_scraper._wrapper` and `core.vinted_client` actually use, so it can be
injected via the `_wrapper.CLIENT_FACTORY` hook with no changes at the call
sites.

Cookies: the curl_cffi `Session` jar is the single source of truth. It captures
every `Set-Cookie` (including `datadome`) and resends automatically, which the
old code threw away (it persisted only `access_token_web`). The library's
per-request `Cookie` header is therefore stripped — mixing a manual Cookie
header with the jar is unreliable (verified). Cross-process persistence is via
`dump_cookies()` / `load_cookies()`.
"""

from __future__ import annotations

import os
from http.cookiejar import Cookie
from urllib.parse import urlsplit

try:
    from curl_cffi import requests as _curl
    CURL_CFFI_AVAILABLE = True
except ImportError:                       # pragma: no cover - optional dep
    _curl = None
    CURL_CFFI_AVAILABLE = False


# Chrome profile curl_cffi impersonates. Override with VINTED_IMPERSONATE if
# Vinted ever starts flagging this exact build (curl_cffi 0.15 ships up to
# chrome146; chrome131 is a very common real-world version).
IMPERSONATE = os.environ.get("VINTED_IMPERSONATE", "chrome131")

# Headers the impersonation profile owns. Letting the library's header builders
# set these would re-create an INCOHERENT fingerprint (a Chrome UA over the
# wrong header set/order), which raises the bot score instead of lowering it.
# We drop them and let curl_cffi emit the correct, browser-ordered set.
#   - Accept-Language is intentionally KEPT (browsers vary it per locale — this
#     is our per-domain it/fr/de patch in vinted_scraper.utils._misc).
#   - Cookie is dropped in favour of the session jar.
#   - Origin is dropped: real browsers don't send it on a plain GET navigation.
_STRIP = {
    "user-agent", "accept", "accept-encoding", "connection",
    "upgrade-insecure-requests", "dnt", "cookie", "origin",
    "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site", "sec-fetch-user",
    "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
}


class _Url(str):
    """`str` subclass that also exposes `.path`, so call sites can use either
    `str(resp.url)` or `resp.url.path` (httpx.URL compatibility — see
    vinted_client._fetch_item_page, which reads `resp.url.path`)."""

    @property
    def path(self) -> str:
        return urlsplit(str(self)).path


class _Resp:
    """Thin proxy over a curl_cffi Response. Delegates everything (status_code,
    content, text, json(), headers, cookies) and only overrides `.url` so it
    carries a `.path`."""

    __slots__ = ("_r",)

    def __init__(self, r):
        self._r = r

    def __getattr__(self, name):
        return getattr(self._r, name)

    @property
    def url(self) -> _Url:
        return _Url(str(self._r.url))


class CurlClient:
    """httpx.Client-compatible facade backed by a curl_cffi impersonating
    Session. Only the methods/attributes used by vinted_scraper and
    core.vinted_client are implemented."""

    def __init__(self, baseurl: str, config: dict | None = None):
        self.base_url = baseurl.rstrip("/")
        netloc = urlsplit(baseurl).netloc
        reg = netloc[4:] if netloc.startswith("www.") else netloc
        # Leading dot → cookie matches www.vinted.it and bare vinted.it alike.
        self._cookie_domain = "." + reg
        # One Session = one cookie jar reused across every request in this
        # process, so datadome set on the cookie-fetch is resent on searches.
        self._s = _curl.Session(impersonate=IMPERSONATE, timeout=20)

    # ── httpx.Client surface ────────────────────────────────────────────────
    def get(self, url, headers=None, params=None, follow_redirects=True,
            timeout=None, **_kw):
        full = url if url.startswith("http") else self.base_url + url
        clean = {k: v for k, v in (headers or {}).items()
                 if k.lower() not in _STRIP}
        r = self._s.get(
            full,
            headers=clean or None,
            params=params,
            allow_redirects=follow_redirects,
            timeout=20 if timeout is None else timeout,
        )
        return _Resp(r)

    @property
    def cookies(self):
        return self._s.cookies

    def close(self):
        try:
            self._s.close()
        except Exception:
            pass

    # ── cross-process cookie persistence ────────────────────────────────────
    def dump_cookies(self) -> list[dict]:
        """Serialize the whole jar (incl. datadome) for disk persistence."""
        return [{"name": c.name, "value": c.value,
                 "domain": c.domain, "path": c.path or "/"}
                for c in self._s.cookies.jar]

    def load_cookies(self, items: list[dict] | None) -> None:
        """Restore a jar previously produced by dump_cookies()."""
        for c in items or []:
            self._set(c["name"], c["value"],
                      c.get("domain") or self._cookie_domain, c.get("path") or "/")

    def seed(self, mapping: dict | None) -> None:
        """Inject name→value cookies (e.g. a cached access_token_web from the
        legacy on-disk format) onto the registrable domain so they're sent even
        though no Set-Cookie populated the jar this session."""
        for name, value in (mapping or {}).items():
            if value:
                self._set(name, value, self._cookie_domain, "/")

    def _set(self, name, value, domain, path):
        self._s.cookies.jar.set_cookie(Cookie(
            0, name, value, None, False,
            domain, True, domain.startswith("."),
            path, True, False, None, False, None, None, {},
        ))


def make_client(baseurl: str, config: dict | None = None) -> CurlClient:
    """Factory injected into vinted_scraper._wrapper.CLIENT_FACTORY."""
    return CurlClient(baseurl, config)
