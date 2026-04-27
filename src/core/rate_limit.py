from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.configs import server
from src.shared.responses import ApiResponse, ErrorDetail

# Single shared limiter — controllers reach it via app.state.limiter.
limiter = Limiter(key_func=get_remote_address, enabled=server.rate_limit.enabled)


def register_rate_limiter(app: FastAPI) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _handle_rate_limit_exceeded)
    app.add_middleware(SlowAPIMiddleware)


async def _handle_rate_limit_exceeded(_req: Request, exc: RateLimitExceeded):
    error = ErrorDetail(
        title="Too Many Requests",
        code="RATE_LIMITED",
        status=429,
        details=[f"Rate limit exceeded: {exc.detail}"],
    )
    response = ApiResponse.failure(
        error=error,
        message="Too many requests. Please try again later.",
    )
    return JSONResponse(
        status_code=429,
        content=response.model_dump(exclude_none=True, by_alias=True),
    )
