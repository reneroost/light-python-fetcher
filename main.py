import logging
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError

# Define unified production logging configuration in-code
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "production": {
            "()": "logging.Formatter",
            "fmt": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "production",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "light_fetcher": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# Apply immediately on import so Uvicorn's loggers adapt to your format
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("light_fetcher")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional: logic to run on startup/shutdown
    logger.info("Light Python Fetcher API lifecycle started.")
    yield
    logger.info("Light Python Fetcher API lifecycle terminating.")


app = FastAPI(
    title="Light Python Fetcher API",
    version="1.0.0",
    lifespan=lifespan,
)


class FetchRequest(BaseModel):
    url: HttpUrl
    proxy_url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    impersonate: Optional[str] = "chrome"
    timeout_seconds: Optional[int] = 15


class FetchResponse(BaseModel):
    status_code: int
    html: str
    final_url: str


@app.post("/api/v1/fetch", response_model=FetchResponse)
async def fetch_page(request: FetchRequest) -> FetchResponse:
    target_url = str(request.url)
    proxies = (
        {"http": request.proxy_url, "https": request.proxy_url}
        if request.proxy_url
        else None
    )

    active_impersonate = request.impersonate or "chrome"
    active_timeout = request.timeout_seconds or 15

    logger.info(
        "Initiating fetch for URL: %s | Impersonate: %s",
        target_url,
        active_impersonate,
    )

    try:
        async with AsyncSession(impersonate=active_impersonate, proxies=proxies) as session:
            response = await session.get(
                target_url,
                headers=request.headers or {},
                timeout=active_timeout,
                allow_redirects=True,
            )

            return FetchResponse(
                status_code=response.status_code,
                html=response.text,
                final_url=response.url,
            )

    except RequestsError as e:
        logger.error("CurlCffi failed for %s. Reason: %s", target_url, e)
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {str(e)}")

    except Exception:
        logger.exception("Unexpected error during fetch execution for %s", target_url)
        raise HTTPException(status_code=500, detail="Internal fetcher error")
