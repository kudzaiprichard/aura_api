from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from src.shared.database.pagination import PaginationParams, get_pagination
from src.shared.responses import ApiResponse, PaginatedResponse

from src.app.dependencies import (
    get_current_user,
    get_escalation_service,
    require_admin,
)
from src.app.dtos.requests import (
    EscalationResolveRequest,
    EscalationReturnRequest,
)
from src.app.dtos.responses import EscalationResponse
from src.app.models.user import User
from src.app.services.escalation_service import EscalationService


router = APIRouter(dependencies=[Depends(require_admin)])


@router.get("")
async def list_escalations(
    pagination: PaginationParams = Depends(get_pagination),
    resolved: Optional[bool] = Query(None),
    service: EscalationService = Depends(get_escalation_service),
):
    escalations, total = await service.list_escalations(
        page=pagination.page,
        page_size=pagination.page_size,
        resolved=resolved,
    )
    return PaginatedResponse.ok(
        value=[EscalationResponse.from_escalation(e) for e in escalations],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


@router.post("/{escalation_id}/resolve")
async def resolve_escalation(
    escalation_id: UUID,
    body: EscalationResolveRequest,
    current_user: User = Depends(get_current_user),
    service: EscalationService = Depends(get_escalation_service),
):
    escalation = await service.resolve(
        escalation_id,
        verdict=body.verdict,
        note=body.note,
        admin=current_user,
    )
    return ApiResponse.ok(
        value=EscalationResponse.from_escalation(escalation),
        message="Escalation resolved",
    )


@router.post("/{escalation_id}/return")
async def return_escalation(
    escalation_id: UUID,
    body: EscalationReturnRequest,
    current_user: User = Depends(get_current_user),
    service: EscalationService = Depends(get_escalation_service),
):
    escalation = await service.return_to_pool(
        escalation_id,
        reason=body.reason,
        admin=current_user,
    )
    return ApiResponse.ok(
        value=EscalationResponse.from_escalation(escalation),
        message="Escalation returned to queue",
    )
