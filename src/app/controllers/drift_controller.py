from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Query

from src.shared.exceptions import BadRequestException
from src.shared.responses import ApiResponse, ErrorDetail

from src.app.dependencies import (
    get_drift_service,
    require_admin,
    require_authenticated,
)
from src.app.dtos.requests import (
    DriftConfirmRequest,
    DriftThresholdUpdateRequest,
)
from src.app.dtos.responses import (
    ConfusionMatrixResponse,
    DriftHistoryResponse,
    DriftSignalResponse,
)
from src.app.services.drift_service import DriftService


router = APIRouter(dependencies=[Depends(require_authenticated)])


_VALID_BUCKETS = {"hour", "day"}


def _resolve_timezone(tz_name: str) -> str:
    try:
        ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise BadRequestException(
            message="Invalid timezone",
            error_detail=ErrorDetail(
                title="Bad Request",
                code="DRIFT_TIMEZONE_INVALID",
                status=400,
                details=[
                    f"{tz_name!r} is not a recognised IANA tz id"
                ],
            ),
        )
    return tz_name


@router.get("/signal")
async def get_signal(
    service: DriftService = Depends(get_drift_service),
):
    signal = service.signal()
    return ApiResponse.ok(value=DriftSignalResponse.from_signal(signal))


@router.get("/confusion-matrix")
async def get_confusion_matrix(
    service: DriftService = Depends(get_drift_service),
):
    matrix = service.confusion_matrix()
    return ApiResponse.ok(value=ConfusionMatrixResponse.from_matrix(matrix))


@router.get("/history")
async def get_history(
    bucket: str = Query("day"),
    date_from: datetime | None = Query(None, alias="from"),
    date_to: datetime | None = Query(None, alias="to"),
    timezone_name: str = Query("UTC", alias="timezone"),
    service: DriftService = Depends(get_drift_service),
):
    if bucket not in _VALID_BUCKETS:
        raise BadRequestException(
            message="Invalid bucket",
            error_detail=ErrorDetail(
                title="Bad Request",
                code="DRIFT_BUCKET_INVALID",
                status=400,
                details=[
                    f"bucket must be one of {sorted(_VALID_BUCKETS)}, "
                    f"got {bucket!r}"
                ],
            ),
        )
    tz = _resolve_timezone(timezone_name)
    rows = await service.history(
        bucket=bucket,
        date_from=date_from,
        date_to=date_to,
        timezone_name=tz,
    )
    return ApiResponse.ok(
        value=DriftHistoryResponse.from_rows(
            rows, bucket=bucket, timezone_name=tz,
        ),
    )


@router.post("/thresholds", dependencies=[Depends(require_admin)])
async def update_thresholds(
    body: DriftThresholdUpdateRequest,
    service: DriftService = Depends(get_drift_service),
):
    signal = service.update_threshold(body.fpr_threshold)
    return ApiResponse.ok(
        value=DriftSignalResponse.from_signal(signal),
        message="Drift threshold updated",
    )


@router.post("/confirm", dependencies=[Depends(require_admin)])
async def manual_confirm(
    body: DriftConfirmRequest,
    service: DriftService = Depends(get_drift_service),
):
    signal = await service.record_manual_confirmation(
        prediction_id=body.prediction_id,
        confirmed_label=body.confirmed_label,
        occurred_at=datetime.now(timezone.utc),
    )
    return ApiResponse.ok(
        value=DriftSignalResponse.from_signal(signal),
        message="Manual confirmation recorded",
    )
