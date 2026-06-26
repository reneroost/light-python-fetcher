"""
curl_cffi sidecar — FastAPI service providing TLS-impersonating HTTP fetches.

The sidecar's sole responsibility is low-level network execution:
  - Maintaining per-(profile, domain, proxy-exit-identity) sessions that carry
    browser-matching TLS fingerprints, cookie jars, and connection pools.
  - Assembling the small set of context-dependent header overrides
    (Accept-Language, Referer, Sec-Fetch-Site) on top of the profile baseline.
  - Returning raw HTML and upstream HTTP status to the Java layer.

All orchestration (retry logic, challenge detection, proxy session rotation,
strategy selection) lives in the Java news-scraper service.

KEY FIX (session isolation):
  Previously DomainSessionPool used (profile, domain) as the key. Cookies
  minted on residential exit-IP-A were silently replayed on exit-IP-B when the
  Java layer rotated the proxy session, creating a cross-IP cookie replay that
  CDN bot-detection systems flag as an automation tell. The key is now a
  3-tuple (profile, domain, exit_id) where exit_id is derived from the
  DataImpulse sticky-session username, binding each session to the specific
  residential exit that created it.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from enum import Enum
from urllib.parse import urlparse

import uvicorn
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import RequestsError, Timeout
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await _domain_session_pool.close_all()


app = FastAPI(title="curl-cffi-sidecar", version="1.0.0", lifespan=lifespan)

# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum number of concurrent (profile × domain × exit) session entries.
# Each entry holds an open connection pool + TLS state + cookie jar.
MAX_DOMAIN_SESSIONS = 80

DEFAULT_TIMEOUT_SECONDS = 20

# All impersonate profiles accepted by this sidecar.
# Must remain a *superset* of every BrowserProfile.curlCffiName declared in
# the Java CurlCffiFetcher enum. The Java SidecarProfileValidator checks this
# at startup; any profile added to the Java enum must be added here first.
SUPPORTED_IMPERSONATE_PROFILES: frozenset[str] = frozenset({
    "chrome146",
    "chrome145",
    "firefox147",
    "firefox144",
    "safari2601",
    "safari260",
})

FALLBACK_IMPERSONATE_PROFILE = "chrome146"

# ── DTOs ──────────────────────────────────────────────────────────────────────

class CurlCffiFetchRequest(BaseModel):
    """
    Wire-format request from the Java CurlCffiFetcher.

    Fields deliberately limited to what the sidecar actually needs —
    the Java layer owns retry state, backoff, session rotation, and
    challenge detection; these do not appear here.
    """
    url:            str
    proxy_url:      str | None = Field(default=None)
    impersonate:    str        = Field(default=FALLBACK_IMPERSONATE_PROFILE)
    timeout:        int        = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=120)
    # Accept-Language value derived from the proxy exit country on the Java side.
    locale:         str        = Field(default="en-US,en;q=0.9")
    # Semantic referrer hint — resolved to concrete headers in build_contextual_headers.
    # Values must match ReferrerHint.sidecarValue() in the Java enum exactly.
    referrer_hint:  str        = Field(default="direct")


class CurlCffiFetchResponse(BaseModel):
    html:        str | None = None
    final_url:   str | None = None
    status_code: int


# ── ReferrerHint ──────────────────────────────────────────────────────────────

class ReferrerHint(str, Enum):
    """
    Mirror of the Java ReferrerHint enum.

    Member values must stay in sync with ReferrerHint.sidecarValue() in the
    Java codebase — the Java side sends these wire strings and this class
    resolves them to concrete header values.
    """
    GOOGLE_SEARCH = "google_search"
    GOOGLE_NEWS   = "google_news"
    TWITTER       = "twitter"
    DIRECT        = "direct"


# Referrer URL to inject as the Referer header. Empty string → no header.
_REFERRER_URL: dict[ReferrerHint, str] = {
    ReferrerHint.GOOGLE_SEARCH: "https://www.google.com/",
    ReferrerHint.GOOGLE_NEWS:   "https://news.google.com/",
    ReferrerHint.TWITTER:       "https://t.co/",
    ReferrerHint.DIRECT:        "",
}

# Sec-Fetch-Site is coupled to the referrer — they are always derived from the
# same ReferrerHint to guarantee they can never disagree.
_SEC_FETCH_SITE: dict[ReferrerHint, str] = {
    ReferrerHint.GOOGLE_SEARCH: "cross-site",
    ReferrerHint.GOOGLE_NEWS:   "cross-site",
    ReferrerHint.TWITTER:       "cross-site",
    ReferrerHint.DIRECT:        "none",
}


# ── Session pool ──────────────────────────────────────────────────────────────

def _proxy_exit_id(proxy_url: str | None) -> str:
    """
    Derives the sticky-session identity from a DataImpulse proxy URL.

    DataImpulse encodes the session as the HTTP Basic-auth username:
        baseuser__country-XX__session-XXXXXXXX

    Using the full username as the pool key dimension ensures:
      - Cookie jars minted on exit-IP-A are never replayed on exit-IP-B.
      - TLS session tickets (which carry IP context in some CDN implementations)
        are not leaked across proxy rotations.
      - When the Java layer calls ProxySessionManager.generateTargetedUsername()
        with a new sticky session, the sidecar transparently creates a fresh
        session rather than reusing state from the previous exit.

    Fallback: "direct" when no proxy is configured (local / development mode).
    """
    if not proxy_url:
        return "direct"
    parsed = urlparse(proxy_url)
    return parsed.username or "direct"


class DomainSessionPool:
    """
    LRU pool of curl_cffi AsyncSession objects keyed by
    ``(impersonate_profile, domain, proxy_exit_id)``.

    The three-dimensional key is the critical design decision:
      - ``impersonate_profile``: each profile has its own TLS fingerprint;
        sessions must not be shared across profiles.
      - ``domain``: sessions accumulate domain-specific cookies and connection
        state; mixing domains degrades both performance and stealth.
      - ``proxy_exit_id``: a session belongs to a specific residential exit IP.
        Reusing it on a different IP leaks identity signals across IPs.

    Pool capacity is bounded by ``max_entries`` (LRU eviction) to prevent
    unbounded memory growth under many active (profile × domain × session)
    combinations.
    """

    def __init__(self, max_entries: int = MAX_DOMAIN_SESSIONS) -> None:
        self._pool: OrderedDict[tuple[str, str, str], AsyncSession] = OrderedDict()
        self._max  = max_entries
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        profile:   str,
        domain:    str,
        proxy_url: str | None,
    ) -> AsyncSession:
        """
        Returns an existing session for (profile, domain, exit_id) or creates one.

        ``proxy_url`` is the full proxy URL string (not just the exit_id), so
        that the caller does not need to know about exit_id derivation.
        The derivation is encapsulated in :func:`_proxy_exit_id`.
        """
        exit_id = _proxy_exit_id(proxy_url)
        key     = (profile, domain, exit_id)

        async with self._lock:
            if key in self._pool:
                self._pool.move_to_end(key)
                log.debug("Session pool hit: profile=%s domain=%s exit=%s", *key)
                return self._pool[key]

            # LRU eviction: close and remove the least-recently-used entry.
            if len(self._pool) >= self._max:
                evicted_key, evicted_session = self._pool.popitem(last=False)
                log.debug(
                    "Session pool capacity reached (%d); evicting: profile=%s domain=%s exit=%s",
                    self._max, *evicted_key,
                )
                try:
                    await evicted_session.close()
                except Exception as exc:
                    log.warning("Error closing evicted session: %s", exc)

            log.debug("Session pool miss — creating: profile=%s domain=%s exit=%s", *key)
            session = AsyncSession(impersonate=profile)
            self._pool[key] = session
            return session

    async def close_all(self) -> None:
        """Closes all sessions and clears the pool (called on application shutdown)."""
        async with self._lock:
            for session in self._pool.values():
                try:
                    await session.close()
                except Exception as exc:
                    log.warning("Error closing session during shutdown: %s", exc)
            self._pool.clear()
            log.info("Session pool cleared on shutdown.")


# Module-level singleton — one pool shared across all request handlers.
_domain_session_pool = DomainSessionPool()


# ── Header assembly ───────────────────────────────────────────────────────────

def build_contextual_headers(locale: str, hint: ReferrerHint) -> dict[str, str]:
    """
    Assembles the two context-dependent header overrides on top of the
    curl_cffi impersonate profile's baseline.

    WHY ONLY TWO FIELDS:
    The impersonate profile already provides a complete, browser-matched header
    set — User-Agent, sec-ch-ua, Accept, Accept-Encoding, and the underlying
    TLS fingerprint (cipher suites, extensions, ALPN, JA4) — as a single
    atomic bundle. Adding UA or sec-ch-ua from here would create a second,
    lower-priority signal that curl_cffi ignores at the TLS level while the
    Java side believes it was applied.

    The only two values that are legitimately context-dependent (and therefore
    cannot be baked into a static profile) are:

    - ``Accept-Language``: changes per exit country; must match the apparent
      IP geolocation to avoid the language/IP inconsistency signal.
    - ``Referer`` + ``Sec-Fetch-Site``: the Referer URL and its coupled
      Sec-Fetch-Site value form a navigation context that varies per attempt.
      Both are derived from the same ReferrerHint so they cannot disagree.
    """
    headers: dict[str, str] = {
        "Accept-Language": locale,
        "Sec-Fetch-Site":  _SEC_FETCH_SITE[hint],
    }

    referrer_url = _REFERRER_URL[hint]
    if referrer_url:
        headers["Referer"] = referrer_url

    return headers


def _resolve_hint(raw: str) -> ReferrerHint:
    """Parses the referrer_hint wire value, defaulting to DIRECT on unknown input."""
    try:
        return ReferrerHint(raw)
    except ValueError:
        log.warning("Unknown referrer_hint value %r — defaulting to DIRECT.", raw)
        return ReferrerHint.DIRECT


def _resolve_profile(requested: str) -> str:
    """Validates the requested impersonate profile, falling back to the default."""
    if requested in SUPPORTED_IMPERSONATE_PROFILES:
        return requested
    log.warning(
        "Unsupported impersonate profile %r — falling back to %s.",
        requested, FALLBACK_IMPERSONATE_PROFILE,
    )
    return FALLBACK_IMPERSONATE_PROFILE


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/api/v1/fetch", response_model=CurlCffiFetchResponse)
async def execute_fetch(request_dto: CurlCffiFetchRequest) -> CurlCffiFetchResponse:
    """
    Executes a single HTTP GET through curl_cffi with full TLS impersonation.

    Session reuse is scoped to (impersonate, domain, proxy_exit_id) so that
    cookie jars and TLS session tickets are never shared across different proxy
    exit identities. A fresh Java ProxySessionManager sticky session maps to a
    fresh sidecar session automatically, without any coordination signal needed
    from the Java layer.
    """
    active_profile = _resolve_profile(request_dto.impersonate)
    hint           = _resolve_hint(request_dto.referrer_hint)

    # Extract domain for session keying. netloc is preferred; raw URL is the
    # fallback for malformed inputs that still parse as a netloc-less string.
    domain = urlparse(request_dto.url).netloc or request_dto.url

    session = await _domain_session_pool.get_or_create(
        active_profile,
        domain,
        request_dto.proxy_url,  # session key now includes exit identity
    )

    context_headers = build_contextual_headers(request_dto.locale, hint)

    proxies = (
        {"https": request_dto.proxy_url, "http": request_dto.proxy_url}
        if request_dto.proxy_url
        else None
    )

    try:
        response = await session.get(
            request_dto.url,
            headers=context_headers,
            proxies=proxies,
            timeout=request_dto.timeout,
            allow_redirects=True,
        )

        log.info(
            "fetch OK  profile=%-10s status=%d url=%s",
            active_profile, response.status_code, request_dto.url,
        )

        return CurlCffiFetchResponse(
            html=response.text,
            final_url=str(response.url),
            status_code=response.status_code,
        )

    except Timeout:
        # Transient: proxy routing delay or target slow to respond.
        # The Java retry layer handles this — no traceback needed at this level.
        log.warning(
            "fetch TIMEOUT  profile=%-10s url=%s",
            active_profile, request_dto.url,
        )
        return CurlCffiFetchResponse(html=None, final_url=None, status_code=502)

    except RequestsError as exc:
        # Known curl_cffi transport error (connection reset, proxy error, etc.).
        # Still recoverable via Java retry — log at ERROR without traceback.
        log.error(
            "fetch ERR  profile=%-10s url=%s  error=%s",
            active_profile, request_dto.url, exc,
        )
        return CurlCffiFetchResponse(html=None, final_url=None, status_code=502)

    except Exception as exc:
        # Unexpected — not a curl_cffi transport error. Full traceback warranted.
        log.exception(
            "fetch UNEXPECTED  profile=%-10s url=%s  error=%s",
            active_profile, request_dto.url, exc,
        )
        # Return 502 so the Java layer's TargetFetchException path fires and the
        # retry engine handles the failure — do not raise an HTTP exception here
        # because that would bypass the Java retry / circuit-breaker logic.
        return CurlCffiFetchResponse(
            html=None,
            final_url=None,
            status_code=502,
        )


# ── Profiles endpoint ─────────────────────────────────────────────────────────

@app.get("/api/v1/profiles")
async def get_profiles() -> dict[str, list[str]]:
    """
    Returns the set of impersonate profiles this sidecar accepts.
    Called at startup by SidecarProfileValidator to verify the Java
    BrowserProfile enum and the sidecar's supported set are in sync.
    """
    return {"profiles": sorted(SUPPORTED_IMPERSONATE_PROFILES)}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("SIDECAR_HOST", "127.0.0.1")
    port = int(os.getenv("SIDECAR_PORT", "8081"))
    uvicorn.run(app, host=host, port=port, log_level="info")
