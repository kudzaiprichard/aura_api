import time
import uuid
import logging
from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.middleware.cors import CORSMiddleware
from src.configs import server, extension

CHROME_EXTENSION_ORIGIN_REGEX = r"chrome-extension://[A-Za-z0-9_\-]+"

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = b"x-request-id"


class RequestLoggingMiddleware:
    """
    Raw ASGI middleware for request logging.
    Unlike BaseHTTPMiddleware / @app.middleware("http"),
    this does NOT buffer the response body — so SSE streaming works.

    Also assigns/propagates an X-Request-ID for cross-line correlation.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        method = scope.get("method", "?")
        path = scope.get("path", "?")

        incoming_id = None
        for k, v in scope.get("headers", []):
            if k == REQUEST_ID_HEADER:
                incoming_id = v.decode("latin-1")
                break
        request_id = incoming_id or uuid.uuid4().hex
        scope["request_id"] = request_id

        status_code = None

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER, request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)

        elapsed = round(time.perf_counter() - start, 4)
        if status_code is None:
            logger.error(
                f"{method} {path} — no response sent ({elapsed}s) [rid={request_id}]"
            )
        else:
            logger.info(
                f"{method} {path} — {status_code} ({elapsed}s) [rid={request_id}]"
            )


def register_middleware(app: FastAPI) -> None:
    # Order matters: first added = outermost middleware
    # CORS must be outermost to handle preflight requests
    _add_cors(app)
    # Logging as raw ASGI middleware — does NOT break SSE
    app.add_middleware(RequestLoggingMiddleware)


def _add_cors(app: FastAPI) -> None:
    dashboard_origins = [o for o in server.cors.origins if o]
    extension_origins = [o for o in extension.cors_origins if o]
    # De-dupe while preserving order so the dashboard list stays first.
    seen: set[str] = set()
    origins: list[str] = []
    for o in (*dashboard_origins, *extension_origins):
        if o not in seen:
            seen.add(o)
            origins.append(o)

    allow_credentials = server.cors.allow_credentials
    if allow_credentials and "*" in origins:
        raise RuntimeError(
            "CORS misconfigured: allow_credentials=true cannot be combined with "
            "origins containing '*'. Set CORS_ORIGINS to an explicit origin list "
            "or set CORS_ALLOW_CREDENTIALS=false."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=CHROME_EXTENSION_ORIGIN_REGEX,
        allow_credentials=allow_credentials,
        allow_methods=server.cors.allow_methods,
        allow_headers=server.cors.allow_headers,
    )
