from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs import application, reload_config, sse as sse_config
from src.core.sse import (
    SSEBroker,
    SSEEvent,
    SSEResponse,
    monotonic_event_id,
)
from src.shared.database import get_db_no_transaction
from src.shared.exceptions import NotFoundException
from src.shared.responses import ApiResponse, ErrorDetail
from src.app.dependencies import (
    get_inference_status_service,
    get_sse_broker,
    optional_install,
    require_admin,
)
from src.app.dtos.extension import ExtensionHealthResponse
from src.app.models.extension_install import ExtensionInstall
from src.app.services import InferenceStatusService

router = APIRouter()


_SSE_ECHO_TOPIC_PREFIX = "system.sse-echo."


def _require_echo_enabled() -> None:
    """Echo endpoint is feature-flagged off by default. When disabled it
    behaves as a 404 — never registered as an opaque 403 — so the surface
    is invisible in production unless explicitly turned on."""
    if not sse_config.echo_enabled:
        raise NotFoundException(
            message="The requested resource was not found",
            error_detail=ErrorDetail(
                title="Not Found",
                code="NOT_FOUND",
                status=404,
                details=["The sse-echo smoke endpoint is disabled"],
            ),
        )


@router.get("/health")
async def health():
    return ApiResponse.ok(
        value={
            "status": "ok",
            "name": application.name,
            "version": application.version,
        }
    )


@router.get("/ready")
async def ready(session: AsyncSession = Depends(get_db_no_transaction)):
    await session.execute(text("SELECT 1"))
    return ApiResponse.ok(value={"status": "ready"})


def _resolve_model_version(request: Request) -> str:
    """`model_version` for the extension health probe — never `null`, never
    omitted. Returns the active detector's `version` when present, or the
    literal `"unknown"` so the extension's version-aware cache can treat the
    response uniformly (BACKEND_CONTRACT §5.1)."""
    detector = getattr(request.app.state, "detector", None)
    version = getattr(detector, "version", None) if detector is not None else None
    return version if isinstance(version, str) and version else "unknown"


@router.get("/api/v1/health")
async def extension_health(
    request: Request,
    install: ExtensionInstall | None = Depends(optional_install),
):
    """Extension liveness + model-version probe. Distinct from the existing
    `/health` and `/ready` so neither dashboard route changes shape.

    Auth: optional install token. Absent or valid → 200; present-but-invalid
    → 401 AUTH_FAILED (raised inside `optional_install`).
    """
    return ApiResponse.ok(
        value=ExtensionHealthResponse(
            status="ok",
            name=application.name,
            version=application.version,
            model_version=_resolve_model_version(request),
        )
    )


@router.post("/api/v1/system/reload-config", dependencies=[Depends(require_admin)])
async def reload_application_config():
    reload_config()
    return ApiResponse.ok(value=None, message="Configuration reloaded")


@router.get(
    "/api/v1/system/inference-status",
    dependencies=[Depends(require_admin)],
)
async def inference_status(
    service: InferenceStatusService = Depends(get_inference_status_service),
):
    payload = await service.status()
    return ApiResponse.ok(
        value=payload.model_dump(exclude_none=False, by_alias=True),
    )


# ── SSE smoke endpoint (feature-flagged) ─────────────────────────────
#
# These two routes exist so Phases 8 and 10 have a working reference for
# raw-ASGI streaming; they are NOT part of the public spec and stay off
# unless `SSE_ECHO_ENABLED=true`. Both are admin-only when enabled.

@router.get(
    "/api/v1/system/sse-echo",
    dependencies=[Depends(require_admin)],
)
async def sse_echo_subscribe(
    topic: str = "default",
    broker: SSEBroker = Depends(get_sse_broker),
):
    _require_echo_enabled()
    full_topic = f"{_SSE_ECHO_TOPIC_PREFIX}{topic}"
    return SSEResponse(
        broker=broker,
        topic=full_topic,
        heartbeat_seconds=sse_config.heartbeat_seconds,
    )


@router.post(
    "/api/v1/system/sse-echo",
    dependencies=[Depends(require_admin)],
)
async def sse_echo_publish(
    payload: dict[str, Any] | None = None,
    topic: str = "default",
    broker: SSEBroker = Depends(get_sse_broker),
):
    _require_echo_enabled()
    full_topic = f"{_SSE_ECHO_TOPIC_PREFIX}{topic}"
    delivered = broker.publish(
        full_topic,
        SSEEvent(
            data=payload if payload is not None else {},
            event="echo",
            id=monotonic_event_id(),
        ),
    )
    return ApiResponse.ok(
        value={"topic": full_topic, "delivered": delivered},
        message="Payload published",
    )
