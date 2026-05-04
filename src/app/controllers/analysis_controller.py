from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Query, Request

from src.configs import inference as inference_config, server
from src.core.rate_limit import limiter
from src.shared.exceptions import BadRequestException
from src.shared.responses import ApiResponse, ErrorDetail, PaginatedResponse
from src.shared.database.pagination import PaginationParams, get_pagination

from src.app.dependencies import (
    get_current_user,
    get_detector,
    get_prediction_service,
    require_authenticated,
)
from src.app.dtos.requests import BatchPredictionRequest, PredictionRequest
from src.app.dtos.responses import (
    PredictionEventSummaryResponse,
    PredictionExplainResponse,
    PredictionResponse,
)
from src.app.helpers.tfidf_topk import (
    compute_top_terms,
    explain_topk,
    explain_topk_enabled,
)
from src.app.models.enums import ConfidenceZone, PredictionSource
from src.app.models.user import User
from src.app.services.prediction_service import PredictionService
from src.shared.inference import PhishingDetector


router = APIRouter(dependencies=[Depends(require_authenticated)])


def _request_id(request: Request) -> str:
    rid = request.scope.get("request_id") if hasattr(request, "scope") else None
    return rid or str(uuid4())


@router.post("/predict")
@limiter.limit(server.rate_limit.predict)
async def predict(
    request: Request,
    body: PredictionRequest,
    current_user: User = Depends(get_current_user),
    service: PredictionService = Depends(get_prediction_service),
):
    event, result = await service.predict(
        sender=body.sender,
        subject=body.subject,
        body=body.body,
        threshold=body.threshold,
        requester=current_user,
        request_id=_request_id(request),
        source=PredictionSource.API,
    )
    return ApiResponse.ok(value=PredictionResponse.from_event(event, result))


@router.post("/predict/batch")
@limiter.limit(server.rate_limit.predict_batch)
async def predict_batch(
    request: Request,
    body: BatchPredictionRequest,
    current_user: User = Depends(get_current_user),
    service: PredictionService = Depends(get_prediction_service),
):
    cap = inference_config.batch.max_emails_per_request
    if len(body.emails) > cap:
        raise BadRequestException(
            message=f"Batch size exceeds the configured cap of {cap} emails",
            error_detail=ErrorDetail(
                title="Validation Failed",
                code="BATCH_TOO_LARGE",
                status=400,
                details=[
                    f"Received {len(body.emails)} emails; "
                    f"inference.batch.max_emails_per_request = {cap}."
                ],
            ),
        )

    request_id = _request_id(request)
    payload = [
        {"sender": e.sender, "subject": e.subject, "body": e.body}
        for e in body.emails
    ]
    outcomes = await service.predict_batch(
        emails=payload,
        threshold=body.threshold,
        requester=current_user,
        request_id=request_id,
        source=PredictionSource.BATCH,
    )
    return ApiResponse.ok(
        value=[
            PredictionResponse.from_event(event, result)
            for event, result in outcomes
        ],
    )


@router.get("/predictions")
async def list_predictions(
    pagination: PaginationParams = Depends(get_pagination),
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    model_version: Optional[str] = Query(None, alias="modelVersion"),
    predicted_label: Optional[int] = Query(None, ge=0, le=1, alias="predictedLabel"),
    confidence_zone: Optional[ConfidenceZone] = Query(None, alias="zone"),
    user_id: Optional[UUID] = Query(None, alias="userId"),
    current_user: User = Depends(get_current_user),
    service: PredictionService = Depends(get_prediction_service),
):
    events, total = await service.list_predictions(
        current_user=current_user,
        page=pagination.page,
        page_size=pagination.page_size,
        date_from=date_from,
        date_to=date_to,
        model_version=model_version,
        predicted_label=predicted_label,
        confidence_zone=confidence_zone,
        user_id=user_id,
    )
    return PaginatedResponse.ok(
        value=[
            PredictionEventSummaryResponse.from_event(
                event,
                redact_body=PredictionService.should_redact_body(event, current_user),
            )
            for event in events
        ],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


@router.get("/predictions/{prediction_event_id}")
async def get_prediction(
    prediction_event_id: UUID,
    current_user: User = Depends(get_current_user),
    service: PredictionService = Depends(get_prediction_service),
):
    event = await service.get_prediction(prediction_event_id)
    return ApiResponse.ok(
        value=PredictionEventSummaryResponse.from_event(
            event,
            redact_body=PredictionService.should_redact_body(event, current_user),
        )
    )


@router.get("/predictions/{prediction_event_id}/explain")
async def explain_prediction(
    prediction_event_id: UUID,
    service: PredictionService = Depends(get_prediction_service),
    detector: PhishingDetector | None = Depends(get_detector),
):
    event = await service.get_prediction(prediction_event_id)
    top_terms: list[dict] | None = None
    if explain_topk_enabled():
        top_terms = compute_top_terms(
            detector=detector,
            subject=event.subject,
            body=event.body,
            k=explain_topk(),
        )
    return ApiResponse.ok(
        value=PredictionExplainResponse.from_event(event, top_terms=top_terms)
    )
