import logging
import logging.config
import random
from contextlib import asynccontextmanager
from typing import Dict, Optional
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, HttpUrl, Field
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError
from charset_normalizer import from_bytes

# --- Configuration & Logging ---
# Explicitly splitting handlers allows us to rewrite Uvicorn's poorly named internal logger
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
        "uvicorn.error": {"handlers": ["uvicorn_console"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("fetcher_service")

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_IMPERSONATE_PROFILES = ["chrome146", "chrome145", "firefox147", "firefox144", "safari2601", "safari260"]


# --- RFC 7807 Problem Details Schema ---
class ProblemDetailResponse(BaseModel):
    """Strict RFC 7807 compliant error format for machine-readable API error contexts."""
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str


# --- Data Transfer Objects (DTOs) ---
class FetchRequest(BaseModel):
    url: HttpUrl = Field(..., description="Target URL to scrape")
    proxy_url: Optional[str] = Field(None, description="Optional upstream proxy gateway URL")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    impersonate: Optional[str] = Field(None, description="Explicit browser profile override matching Java definition")
    timeout_seconds: int = Field(15, ge=1, le=60, description="Enforced request timeout window")


class FetchResponse(BaseModel):
    status_code: int
    html: str
    final_url: str


# --- Custom Domain Exceptions ---
class UpstreamFetchError(Exception):
    """Raised when the upstream target or the network transport layer fails."""
    def __init__(self, detail: str, status_code: int = 502):
        self.detail = detail
        self.status_code = status_code
        super().__init__(self.detail)


# --- Core Functional Business Logic ---
async def execute_fetch(request_dto: FetchRequest) -> FetchResponse:
    """Core functional orchestrator for executing outbound TLS-spoofed HTTP requests."""
    target_url = str(request_dto.url)
    proxies = {"http": request_dto.proxy_url, "https": request_dto.proxy_url} if request_dto.proxy_url else None
    active_impersonate = request_dto.impersonate or random.choice(DEFAULT_IMPERSONATE_PROFILES)

    logger.info("Executing outbound fetch | Target: %s | Profile: %s", target_url, active_impersonate)

    try:
        async with AsyncSession(impersonate=active_impersonate, proxies=proxies) as session:
            response = await session.get(
                target_url,
                headers=request_dto.headers or {},
                timeout=request_dto.timeout_seconds,
                allow_redirects=True,
            )

            if response.status_code >= 400:
                logger.warning("Upstream peer returned error state %s for %s", response.status_code, target_url)
                raise UpstreamFetchError(f"Upstream returned status code {response.status_code}", status_code=502)

            content = response.content
            if len(content) > MAX_RESPONSE_BYTES:
                logger.error("Payload size violation from %s (%s bytes)", target_url, len(content))
                raise UpstreamFetchError("Upstream response size limit exceeded", status_code=502)

            if response.encoding:
                decoded_html = content.decode(response.encoding, errors="replace")
            else:
                detection_result = from_bytes(content).best()
                decoded_html = str(detection_result) if detection_result else content.decode("utf-8", errors="replace")

            return FetchResponse(
                status_code=response.status_code,
                html=decoded_html,
                final_url=str(response.url)
            )

    except RequestsError as e:
        logger.error("Network driver level failure fetching %s: %s", target_url, str(e))
        raise UpstreamFetchError(f"Network transport layer failure: {str(e)}", status_code=502)


# --- Application Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Fetcher Engine sidecar initialized successfully.")
    yield
    logger.info("Fetcher Engine sidecar shutting down cleanly.")


app = FastAPI(
    title="Light Python Fetcher API",
    version="1.1.0",
    lifespan=lifespan,
    responses={
        500: {"model": ProblemDetailResponse},
        502: {"model": ProblemDetailResponse},
        422: {"model": ProblemDetailResponse}
    }
)


# --- RFC 7807 Exception Handlers ---
@app.exception_handler(UpstreamFetchError)
async def upstream_fetch_exception_handler(request: Request, exc: UpstreamFetchError):
    problem = ProblemDetailResponse(
        type="https://api.news-aggregator.local/errors/upstream-failure",
        title="Upstream Fetch Failure",
        status=exc.status_code,
        detail=exc.detail,
        instance=str(request.url)
    )
    return JSONResponse(status_code=exc.status_code, content=problem.model_dump(), media_type="application/problem+json")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_messages = [
        f"{'.'.join(str(loc) for loc in err['loc'] if loc != 'body')}: {err['msg']}"
        for err in exc.errors()
    ]
    detail_str = "; ".join(error_messages) if error_messages else "Invalid request payload layout."
    
    problem = ProblemDetailResponse(
        type="https://api.news-aggregator.local/errors/validation-error",
        title="Unprocessable Request Entity",
        status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=detail_str,
        instance=str(request.url)
    )
    return JSONResponse(status_code=422, content=problem.model_dump(), media_type="application/problem+json")


@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandleable exception caught by global safety net.")
    problem = ProblemDetailResponse(
        type="https://api.news-aggregator.local/errors/internal-error",
        title="Internal Server Error",
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected error occurred processing your request.",
        instance=str(request.url)
    )
    return JSONResponse(status_code=500, content=problem.model_dump(), media_type="application/problem+json")


# --- API Routing & Controllers ---
@app.post(
    "/api/v1/fetch", 
    response_model=FetchResponse, 
    summary="Fetch target HTML document via TLS fingerprinting."
)
async def fetch_page(request: FetchRequest) -> FetchResponse:
    """Dispatches payload collection jobs directly to the underlying curl_cffi implementation layer."""
    return await execute_fetch(request)


@app.get("/health", status_code=status.HTTP_200_OK, summary="Liveness check hook")
async def health_check() -> dict:
    """Used for orchestration layer routing validations and Caddy monitoring logs."""
    return {"status": "healthy"}
