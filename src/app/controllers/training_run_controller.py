from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from src.core.sse import SSEBroker, SSEResponse
from src.shared.database.pagination import PaginationParams, get_pagination
from src.shared.responses import ApiResponse, PaginatedResponse

from src.app.dependencies import (
    get_current_user,
    get_sse_broker,
    get_training_service,
    require_admin,
    require_authenticated,
)
from src.app.dtos.requests import TrainingRunRequest
from src.app.dtos.responses import (
    TrainingRunDetailResponse,
    TrainingRunPreviewResponse,
    TrainingRunResponse,
)
from src.app.models.enums import Role, TrainingRunStatus
from src.app.models.user import User
from src.app.services.training_service import TrainingService


router = APIRouter(dependencies=[Depends(require_authenticated)])


@router.post("", dependencies=[Depends(require_admin)])
async def start_run(
    payload: TrainingRunRequest,
    current_user: User = Depends(get_current_user),
    service: TrainingService = Depends(get_training_service),
):
    run = await service.start_run(payload, actor=current_user)
    return ApiResponse.ok(
        value=TrainingRunDetailResponse.from_run(run),
        message="Training run submitted",
    )


@router.post("/preview", dependencies=[Depends(require_admin)])
async def preview_run(
    payload: TrainingRunRequest,
    service: TrainingService = Depends(get_training_service),
):
    """Side-effect-free preview of the slice a real POST would resolve.
    Drives the create-run form so the gate's refusal never surprises a user
    who had all the inputs they needed to see the outcome ahead of time."""
    preview = await service.preview_run(payload)
    return ApiResponse.ok(
        value=TrainingRunPreviewResponse(**preview),
    )


@router.get("")
async def list_runs(
    pagination: PaginationParams = Depends(get_pagination),
    status: Optional[TrainingRunStatus] = Query(None),
    triggered_by: Optional[UUID] = Query(None, alias="triggeredBy"),
    current_user: User = Depends(get_current_user),
    service: TrainingService = Depends(get_training_service),
):
    runs, total = await service.list_runs(
        page=pagination.page,
        page_size=pagination.page_size,
        status=status,
        triggered_by=triggered_by,
    )
    # IT_ANALYST sees the summary projection; ADMIN sees the detail view with
    # metrics + provenance. Both paginate identically — the envelope shape is
    # stable, only the row body changes per §7.5.
    if current_user.role == Role.ADMIN:
        items = [TrainingRunDetailResponse.from_run(r) for r in runs]
    else:
        items = [TrainingRunResponse.from_run(r) for r in runs]
    return PaginatedResponse.ok(
        value=items,
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


@router.get("/{run_id}")
async def get_run(
    run_id: UUID,
    current_user: User = Depends(get_current_user),
    service: TrainingService = Depends(get_training_service),
):
    run = await service.get_run(run_id)
    if current_user.role == Role.ADMIN:
        return ApiResponse.ok(value=TrainingRunDetailResponse.from_run(run))
    return ApiResponse.ok(value=TrainingRunResponse.from_run(run))


@router.get("/{run_id}/events", dependencies=[Depends(require_admin)])
async def run_events(
    run_id: UUID,
    broker: SSEBroker = Depends(get_sse_broker),
    service: TrainingService = Depends(get_training_service),
):
    # The run must exist; an unknown id never opens a stream that would
    # sit idle forever waiting for an event that can't arrive.
    await service.get_run(run_id)
    return SSEResponse(
        broker=broker,
        topic=service.sse_topic(run_id),
        heartbeat_seconds=service.sse_heartbeat_seconds,
    )


@router.post("/{run_id}/cancel", dependencies=[Depends(require_admin)])
async def cancel_run(
    run_id: UUID,
    service: TrainingService = Depends(get_training_service),
):
    run = await service.cancel(run_id)
    return ApiResponse.ok(
        value=TrainingRunDetailResponse.from_run(run),
        message="Cancellation requested",
    )
