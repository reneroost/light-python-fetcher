import asyncio
import logging
import logging.config
import random
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Dict, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, HttpUrl, Field
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError
from charset_normalizer import from_bytes

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "production": {
            "()": "logging.Formatter",
            "fmt": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "uvicorn_clean": {
            "()": "logging.Formatter",
            "fmt": "%(asctime)s [%(levelname)s] uvicorn.system: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "production",
            "stream": "ext://sys.stdout",
        },
        "uvicorn_console": {
            "class": "logging.StreamHandler",
            "formatter": "uvicorn_clean",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "fetcher_service": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.error":   {"handlers": ["uvicorn_console"], "level": "INFO", "propagate": False},
        "uvicorn.access":  {"handlers": ["console"],         "level": "INFO", "propagate": False},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("fetcher_service")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RESPONSE_BYTES: int = 10 * 1024 * 1024  # 10 MB

DEFAULT_IMPERSONATE_PROFILES: list[str] = [
    "chrome146",
    "chrome145",
    "firefox147",
    "firefox144",
    "safari2601",
    "safari260",
]
_KNOWN_PROFILES: frozenset[str] = frozenset(DEFAULT_IMPERSONATE_PROFILES)
MAX_DOMAIN_SESSIONS: int = 200


# ---------------------------------------------------------------------------
# Domain Session Pool
# ---------------------------------------------------------------------------

class DomainSessionPool:
    """
    LRU-bounded pool of ``AsyncSession`` instances keyed by ``(profile, domain)``.
    """

    def __init__(self, max_entries: int = MAX_DOMAIN_SESSIONS) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._pool: OrderedDict[tuple[str, str], AsyncSession] = OrderedDict()
        self._max: int = max_entries
        self._lock: asyncio.Lock = asyncio.Lock()

    async def get_or_create(self, profile: str, domain: str) -> AsyncSession:
        key = (profile, domain)
        async with self._lock:
            if key in self._pool:
                self._pool.move_to_end(key)
                return self._pool[key]

            session = AsyncSession(impersonate=profile)
            self._pool[key] = session
            self._pool.move_to_end(key)

            if len(self._pool) > self._max:
                evicted_key, evicted_session = self._pool.popitem(last=False)
                asyncio.create_task(
                    _close_session_safely(evicted_session),
                    name=f"evict-session-{evicted_key[0]}-{evicted_key[1]}",
                )
                logger.debug(
                    "LRU eviction: closed session for profile=%s domain=%s",
                    evicted_key[0], evicted_key[1],
                )

            return session

    async def close_all(self) -> None:
        async with self._lock:
            for (profile, domain), session in self._pool.items():
                logger.debug("Closing domain session: profile=%s domain=%s", profile, domain)
                await _close_session_safely(session)
            self._pool.clear()

    @property
    def size(self) -> int:
        return len(self._pool)


async def _close_session_safely(session: AsyncSession) -> None:
    try:
        await session.close()
    except Exception as exc:
        logger.error("Session close failed: %s", exc)


def _extract_domain(url: str) -> str:
    return urlparse(url).netloc


_domain_session_pool: DomainSessionPool = DomainSessionPool()


# ---------------------------------------------------------------------------
# RFC 7807 Problem Details Schema
# ---------------------------------------------------------------------------

class ProblemDetailResponse(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

class FetchRequest(BaseModel):
    url: HttpUrl = Field(..., description="Target URL to scrape")
    proxy_url: Optional[str] = Field(None, description="Optional upstream proxy gateway URL")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    impersonate: Optional[str] = Field(
        None, description="Explicit browser profile override matching Java BrowserProfile enum"
    )
    timeout_seconds: int = Field(15, ge=1, le=60, description="Enforced request timeout window")


class FetchResponse(BaseModel):
    status_code: int
    html: str
    final_url: str


# ---------------------------------------------------------------------------
# Custom Domain Exceptions
# ---------------------------------------------------------------------------

class UpstreamFetchError(Exception):
    """Raised exclusively when infrastructure/transport level network operations fail."""
    def __init__(self, detail: str, status_code: int = 502) -> None:
        self.detail = detail
        self.status_code = status_code
        super().__init__(self.detail)


# ---------------------------------------------------------------------------
# Core Business Logic
# ---------------------------------------------------------------------------

async def execute_fetch(request_dto: FetchRequest) -> FetchResponse:
    target_url = str(request_dto.url)
    domain = _extract_domain(target_url)
    proxies = (
        {"http": request_dto.proxy_url, "https": request_dto.proxy_url}
        if request_dto.proxy_url
        else None
    )
    active_impersonate = request_dto.impersonate or random.choice(DEFAULT_IMPERSONATE_PROFILES)
    is_one_off = active_impersonate not in _KNOWN_PROFILES

    logger.info(
        "Executing outbound fetch | Target: %s | Profile: %s | Domain session: %s",
        target_url,
        active_impersonate,
        "one-off" if is_one_off else f"pooled (pool size: {_domain_session_pool.size})",
    )

    if is_one_off:
        logger.warning(
            "Profile '%s' is not a known default — allocating isolated one-off session.",
            active_impersonate,
        )
        session = AsyncSession(impersonate=active_impersonate)
    else:
        session = await _domain_session_pool.get_or_create(active_impersonate, domain)

    try:
        PASSTHROUGH_HEADERS = {"accept-language", "referer"}
        locale_override: Dict[str, str] = {}
        if request_dto.headers:
            locale_override = {
                k: v for k, v in request_dto.headers.items()
                if k.lower() in PASSTHROUGH_HEADERS
            }

        response = await session.get(
            target_url,
            headers=locale_override,
            timeout=request_dto.timeout_seconds,
            allow_redirects=True,
            stream=True,
            proxies=proxies,
        )

        # Critique implementation correction: Upstream error codes are no longer intercepted here.
        # We downstream the content block to give the Spring caller full situational awareness.
        if response.status_code >= 400:
            logger.warning(
                "Upstream peer returned an HTTP error context: %s for %s",
                response.status_code, target_url,
            )

        try:
            content_length = int(response.headers.get("Content-Length", 0))
            if content_length > MAX_RESPONSE_BYTES:
                logger.error(
                    "Upstream Content-Length exceeds limit: %s bytes", content_length
                )
                raise UpstreamFetchError(
                    "Upstream Content-Length exceeds limit", status_code=502
                )
        except ValueError:
            pass

        chunks: list[bytes] = []
        bytes_received = 0
        async for chunk in response.aiter_content(chunk_size=64 * 1024):
            bytes_received += len(chunk)
            if bytes_received > MAX_RESPONSE_BYTES:
                logger.error(
                    "Payload size violation from %s during streaming (exceeded %s bytes)",
                    target_url, MAX_RESPONSE_BYTES,
                )
                raise UpstreamFetchError(
                    "Upstream response size limit exceeded", status_code=502
                )
            chunks.append(chunk)

        content = b"".join(chunks)

        if response.encoding:
            decoded_html = content.decode(response.encoding, errors="replace")
        else:
            detection_result = from_bytes(content).best()
            decoded_html = (
                str(detection_result) if detection_result
                else content.decode("utf-8", errors="replace")
            )

        return FetchResponse(
            status_code=response.status_code,
            html=decoded_html,
            final_url=str(response.url),
        )

    except RequestsError as exc:
        logger.error(
            "Network driver level failure fetching %s: %s", target_url, str(exc)
        )
        raise UpstreamFetchError(
            f"Network transport layer failure: {exc}", status_code=502
        ) from exc
    finally:
        if is_one_off:
            await _close_session_safely(session)


# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Fetcher Engine initialised. Domain session pool capacity: %d entries.",
        MAX_DOMAIN_SESSIONS,
    )
    yield
    logger.info(
        "Draining domain session pool (%d active sessions)...",
        _domain_session_pool.size,
    )
    await _domain_session_pool.close_all()
    logger.info("Fetcher Engine sidecar shutdown complete.")


app = FastAPI(
    title="Light Python Fetcher API",
    version="1.2.0",
    lifespan=lifespan,
    responses={
        500: {"model": ProblemDetailResponse},
        502: {"model": ProblemDetailResponse},
        422: {"model": ProblemDetailResponse},
    },
)


# ---------------------------------------------------------------------------
# RFC 7807 Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(UpstreamFetchError)
async def upstream_fetch_exception_handler(
    request: Request, exc: UpstreamFetchError
) -> JSONResponse:
    problem = ProblemDetailResponse(
        type="https://api.news-aggregator.local/errors/upstream-failure",
        title="Upstream Fetch Failure",
        status=exc.status_code,
        detail=exc.detail,
        instance=str(request.url),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=problem.model_dump(),
        media_type="application/problem+json",
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    error_messages = [
        f"{'.'.join(str(loc) for loc in err['loc'] if loc != 'body')}: {err['msg']}"
        for err in exc.errors()
    ]
    detail_str = (
        "; ".join(error_messages) if error_messages
        else "Invalid request payload layout."
    )
    problem = ProblemDetailResponse(
        type="https://api.news-aggregator.local/errors/validation-error",
        title="Unprocessable Request Entity",
        status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=detail_str,
        instance=str(request.url),
    )
    return JSONResponse(
        status_code=422,
        content=problem.model_dump(),
        media_type="application/problem+json",
    )


@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception caught by global safety net.")
    problem = ProblemDetailResponse(
        type="https://api.news-aggregator.local/errors/internal-error",
        title="Internal Server Error",
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected error occurred processing your request.",
        instance=str(request.url),
    )
    return JSONResponse(
        status_code=500,
        content=problem.model_dump(),
        media_type="application/problem+json",
    )


# ---------------------------------------------------------------------------
# API Routing
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/fetch",
    response_model=FetchResponse,
    summary="Fetch target HTML document via TLS fingerprinting.",
)
async def fetch_page(request: FetchRequest) -> FetchResponse:
    return await execute_fetch(request)


@app.get(
    "/api/v1/profiles",
    response_model=list[str],
    summary="List supported impersonation profiles.",
)
async def list_profiles() -> list[str]:
    return sorted(DEFAULT_IMPERSONATE_PROFILES)


@app.get("/health", status_code=status.HTTP_200_OK, summary="Liveness check hook.")
async def health_check() -> dict:
    return {"status": "healthy", "pool_size": _domain_session_pool.size}
