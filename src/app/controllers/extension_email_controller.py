"""Chrome extension `POST /api/v1/emails/analyze` — Gmail message verdict.

Mounted under its own router so the extension surface stays segregated from
the dashboard analysis routes (`/api/v1/analysis/...`) and the extension
auth routes (`/api/v1/auth/extension/...`).
"""
import logging
import time

from fastapi import APIRouter, Depends, Request
from slowapi.util import get_remote_address

from src.configs import server
from src.core.rate_limit import limiter
from src.shared.responses import ApiResponse
from src.app.dependencies import (
    get_extension_email_service,
    require_install,
)
from src.app.dtos.extension import ExtensionAnalyzeRequest
from src.app.helpers.install_token_provider import hash_install_token
from src.app.models.extension_install import ExtensionInstall
from src.app.services.extension_email_service import ExtensionEmailService


router = APIRouter()

logger = logging.getLogger(__name__)


def _install_token_key(request: Request) -> str:
    """Rate-limit `/emails/analyze` per-install rather than per-IP — the
    user's network is unrelated to their fair-share quota (§7).

    SHA-256 hash the bearer token so the limiter key matches the value
    stored on `extension_tokens.token_hash`. Falls back to remote IP when
    the header is absent or malformed so the 401 path still buckets cleanly
    instead of sharing a single key across all anonymous callers.
    """
    auth = request.headers.get("Authorization") or ""
    scheme, _, token = auth.partition(" ")
    token = token.strip()
    if scheme.lower() == "bearer" and token:
        return f"installToken:{hash_install_token(token)}"
    return f"ip:{get_remote_address(request)}"


@router.post("/analyze")
@limiter.limit(
    server.rate_limit.extension_predict, key_func=_install_token_key
)
async def analyze_email(
    request: Request,
    body: ExtensionAnalyzeRequest,
    install: ExtensionInstall = Depends(require_install),
    service: ExtensionEmailService = Depends(get_extension_email_service),
):
    request_id = (
        request.scope.get("request_id") if hasattr(request, "scope") else None
    )
    started = time.perf_counter()
    response = await service.analyze(
        body, install_id=install.id, parent_request_id=request_id
    )
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    # §15 — log install_id, model_version, label, score, latency, request_id.
    # No body, no PII beyond install_id (already pseudonymous).
    logger.info(
        "extension analyze: install_id=%s model_version=%s "
        "predicted_label=%s confidence_score=%.4f latency_ms=%.2f "
        "request_id=%s",
        install.id,
        response.prediction.model_version,
        response.prediction.predicted_label,
        response.prediction.confidence_score,
        latency_ms,
        request_id or "-",
    )
    return ApiResponse.ok(value=response)
