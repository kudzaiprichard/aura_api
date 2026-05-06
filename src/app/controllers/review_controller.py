from uuid import UUID

from fastapi import APIRouter, Depends

from src.shared.database.pagination import PaginationParams, get_pagination
from src.shared.responses import ApiResponse, PaginatedResponse

from src.app.dependencies import (
    get_current_user,
    get_review_service,
    require_admin,
    require_authenticated,
    require_role,
)
from src.app.dtos.requests import (
    AutoReviewBatchRequest,
    ReviewConfirmRequest,
    ReviewDeferRequest,
    ReviewEscalateRequest,
    ReviewReassignRequest,
)
from src.app.dtos.responses import (
    AutoReviewInvocationResponse,
    AutoReviewResponse,
    ReviewItemResponse,
)
from src.app.models.enums import Role
from src.app.models.user import User
from src.app.services.review_service import ReviewService


router = APIRouter(dependencies=[Depends(require_authenticated)])


@router.get("")
async def list_queue(
    pagination: PaginationParams = Depends(get_pagination),
    current_user: User = Depends(get_current_user),
    service: ReviewService = Depends(get_review_service),
):
    items, total = await service.list_queue(
        current_user=current_user,
        page=pagination.page,
        page_size=pagination.page_size,
    )
    return PaginatedResponse.ok(
        value=[ReviewItemResponse.from_item(i) for i in items],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


@router.get("/unassigned")
async def list_unassigned(
    pagination: PaginationParams = Depends(get_pagination),
    service: ReviewService = Depends(get_review_service),
):
    items, total = await service.list_unassigned(
        page=pagination.page, page_size=pagination.page_size,
    )
    return PaginatedResponse.ok(
        value=[ReviewItemResponse.from_item(i) for i in items],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


@router.post("/{item_id}/claim")
async def claim_item(
    item_id: UUID,
    current_user: User = Depends(get_current_user),
    service: ReviewService = Depends(get_review_service),
):
    item = await service.claim(item_id, current_user)
    return ApiResponse.ok(
        value=ReviewItemResponse.from_item(item), message="Item claimed",
    )


@router.post("/{item_id}/release")
async def release_item(
    item_id: UUID,
    current_user: User = Depends(get_current_user),
    service: ReviewService = Depends(get_review_service),
):
    item = await service.release(item_id, current_user)
    return ApiResponse.ok(
        value=ReviewItemResponse.from_item(item), message="Item released",
    )


@router.post("/{item_id}/confirm")
async def confirm_item(
    item_id: UUID,
    body: ReviewConfirmRequest,
    current_user: User = Depends(get_current_user),
    service: ReviewService = Depends(get_review_service),
):
    item = await service.confirm(
        item_id,
        verdict=body.verdict,
        note=body.note,
        agreed_with_auto_review=body.agreed_with_auto_review,
        override_reason=body.override_reason,
        current_user=current_user,
    )
    return ApiResponse.ok(
        value=ReviewItemResponse.from_item(item), message="Item confirmed",
    )


@router.post("/{item_id}/defer")
async def defer_item(
    item_id: UUID,
    body: ReviewDeferRequest,
    current_user: User = Depends(get_current_user),
    service: ReviewService = Depends(get_review_service),
):
    item = await service.defer(
        item_id, note=body.note, current_user=current_user,
    )
    return ApiResponse.ok(
        value=ReviewItemResponse.from_item(item), message="Item deferred",
    )


@router.post(
    "/{item_id}/escalate",
    dependencies=[Depends(require_role(Role.IT_ANALYST))],
)
async def escalate_item(
    item_id: UUID,
    body: ReviewEscalateRequest,
    current_user: User = Depends(get_current_user),
    service: ReviewService = Depends(get_review_service),
):
    item, _escalation = await service.escalate(
        item_id,
        reason=body.reason,
        note=body.note,
        current_user=current_user,
        tentative_label=body.tentative_label,
    )
    return ApiResponse.ok(
        value=ReviewItemResponse.from_item(item), message="Item escalated",
    )


@router.post("/{item_id}/reassign", dependencies=[Depends(require_admin)])
async def reassign_item(
    item_id: UUID,
    body: ReviewReassignRequest,
    current_user: User = Depends(get_current_user),
    service: ReviewService = Depends(get_review_service),
):
    item = await service.reassign(
        item_id, new_user_id=body.user_id, admin=current_user,
    )
    return ApiResponse.ok(
        value=ReviewItemResponse.from_item(item), message="Item reassigned",
    )


# ── auto-review (LLM) endpoints ──

@router.post("/{item_id}/auto-review")
async def trigger_auto_review(
    item_id: UUID,
    current_user: User = Depends(get_current_user),
    service: ReviewService = Depends(get_review_service),
):
    invocation = await service.trigger_auto_review(item_id, current_user)
    return ApiResponse.ok(
        value=AutoReviewResponse.from_invocation(invocation),
        message="Auto-review completed",
    )


@router.post("/auto-review")
async def trigger_auto_review_batch(
    body: AutoReviewBatchRequest,
    current_user: User = Depends(get_current_user),
    service: ReviewService = Depends(get_review_service),
):
    invocations = await service.trigger_auto_review_batch(
        body.review_item_ids, current_user,
    )
    return ApiResponse.ok(
        value=[AutoReviewResponse.from_invocation(i) for i in invocations],
        message="Auto-review batch completed",
    )


@router.get("/{item_id}/auto-reviews")
async def list_auto_reviews(
    item_id: UUID,
    pagination: PaginationParams = Depends(get_pagination),
    current_user: User = Depends(get_current_user),
    service: ReviewService = Depends(get_review_service),
):
    invocations, total = await service.list_auto_reviews_for_item(
        item_id,
        actor=current_user,
        page=pagination.page,
        page_size=pagination.page_size,
    )
    return PaginatedResponse.ok(
        value=[
            AutoReviewInvocationResponse.from_invocation(i)
            for i in invocations
        ],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )
