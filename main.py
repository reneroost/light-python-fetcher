import logging
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Light Python Fetcher API", version="1.0.0")

class FetchRequest(BaseModel):
    url: HttpUrl
    proxy_url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    impersonate: str = "chrome"
    timeout_seconds: int = 15

class FetchResponse(BaseModel):
    status_code: int
    html: str
    final_url: str

@app.post("/api/v1/fetch", response_model=FetchResponse)
async def fetch_page(request: FetchRequest) -> FetchResponse:
    target_url = str(request.url)
    proxies = {"http": request.proxy_url, "https": request.proxy_url} if request.proxy_url else None
    
    logger.info(f"Initiating fetch for URL: {target_url} | Impersonate: {request.impersonate}")
    
    try:
        async with AsyncSession(impersonate=request.impersonate, proxies=proxies) as session:
            response = await session.get(
                target_url,
                headers=request.headers or {},
                timeout=request.timeout_seconds,
                allow_redirects=True
            )
            
            return FetchResponse(
                status_code=response.status_code,
                html=response.text,
                final_url=response.url
            )
            
    except RequestsError as e:
        logger.error(f"CurlCffi failed for {target_url}. Reason: {e}")
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during fetch for {target_url}. Reason: {e}")
        raise HTTPException(status_code=500, detail="Internal fetcher error")

if __name__ == "__main__":
    import uvicorn
    # Bind to localhost to ensure it's only accessible to the local Java application
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, workers=4)
