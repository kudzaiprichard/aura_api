from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from src.shared.exceptions import BadRequestException
from src.shared.responses import ApiResponse, ErrorDetail

from src.app.dependencies import (
    get_current_user,
    get_model_admin_service,
    require_admin,
    require_authenticated,
)
from src.app.dtos.requests import (
    ModelActivateRequest,
    ModelPromoteRequest,
    ModelRollbackRequest,
    ModelThresholdRequest,
)
from src.app.dtos.responses import (
    ModelActivationResponse,
    ModelDetailResponse,
    ModelMetricsCompareResponse,
    ModelMetricsResponse,
    ModelSummaryResponse,
    ModelThresholdResponse,
    ModelUploadSummaryResponse,
)
from src.app.models.user import User
from src.app.services.model_admin_service import ModelAdminService


router = APIRouter(dependencies=[Depends(require_authenticated)])


_VALID_BUCKETS = {"hour", "day"}
# Acceptance-criteria stub for §7.6: hard-cap the upload body so a hostile
# client can't exhaust memory before the SHA-256 + registry pipeline runs.
# 100 MiB matches the largest legitimate model artefact we expect to ship.
_UPLOAD_MAX_BYTES = 100 * 1024 * 1024


def _resolve_timezone(tz_name: str) -> str:
    try:
        ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise BadRequestException(
            message="Invalid timezone",
            error_detail=ErrorDetail(
                title="Bad Request",
                code="MODEL_TIMEZONE_INVALID",
                status=400,
                details=[f"{tz_name!r} is not a recognised IANA tz id"],
            ),
        )
    return tz_name


def _require_valid_bucket(bucket: str) -> None:
    if bucket not in _VALID_BUCKETS:
        raise BadRequestException(
            message="Invalid bucket",
            error_detail=ErrorDetail(
                title="Bad Request",
                code="MODEL_BUCKET_INVALID",
                status=400,
                details=[
                    f"bucket must be one of {sorted(_VALID_BUCKETS)}, "
                    f"got {bucket!r}"
                ],
            ),
        )


@router.get("")
async def list_versions(
    service: ModelAdminService = Depends(get_model_admin_service),
):
    rows = await service.list_versions()
    return ApiResponse.ok(
        value=[ModelSummaryResponse.from_row(row) for row in rows],
    )


# `metrics/compare` must be declared before `/{version}` so the router does
# not interpret "compare" as a version segment.
@router.get("/metrics/compare")
async def compare_metrics(
    versions: list[str] = Query(..., alias="versions"),
    bucket: str = Query("day"),
    date_from: datetime | None = Query(None, alias="from"),
    date_to: datetime | None = Query(None, alias="to"),
    timezone_name: str = Query("UTC", alias="timezone"),
    service: ModelAdminService = Depends(get_model_admin_service),
):
    _require_valid_bucket(bucket)
    tz = _resolve_timezone(timezone_name)
    result = await service.compare(
        versions=versions,
        bucket=bucket,
        date_from=date_from,
        date_to=date_to,
        timezone_name=tz,
    )
    return ApiResponse.ok(value=ModelMetricsCompareResponse.from_result(result))


@router.post("/thresholds", dependencies=[Depends(require_admin)])
async def set_thresholds(
    body: ModelThresholdRequest,
    current_user: User = Depends(get_current_user),
    service: ModelAdminService = Depends(get_model_admin_service),
):
    result = await service.set_thresholds(
        version=body.version,
        decision_threshold=body.decision_threshold,
        review_low_threshold=body.review_low_threshold,
        review_high_threshold=body.review_high_threshold,
        actor=current_user,
    )
    return ApiResponse.ok(
        value=ModelThresholdResponse.from_result(result),
        message="Thresholds updated",
    )


@router.post("/upload", dependencies=[Depends(require_admin)])
async def upload_artefact(
    file: UploadFile = File(...),
    expected_sha256: str = Form(..., alias="expectedSha256"),
    source_version: str = Form(..., alias="sourceVersion"),
    signature: Optional[str] = Form(None, alias="signature"),
    current_user: User = Depends(get_current_user),
    service: ModelAdminService = Depends(get_model_admin_service),
):
    payload = await file.read()
    if len(payload) > _UPLOAD_MAX_BYTES:
        raise BadRequestException(
            message="Upload exceeds size cap",
            error_detail=ErrorDetail(
                title="Bad Request",
                code="MODEL_UPLOAD_TOO_LARGE",
                status=400,
                details=[
                    f"received={len(payload)} bytes; "
                    f"max={_UPLOAD_MAX_BYTES} bytes"
                ],
            ),
        )
    result = await service.upload(
        actor=current_user,
        filename=file.filename or "",
        content=payload,
        expected_sha256=expected_sha256,
        source_version=source_version,
        signature=signature,
    )
    return ApiResponse.ok(
        value=ModelUploadSummaryResponse.from_result(result),
        message="Upload accepted",
    )


@router.get("/{version}")
async def get_version(
    version: str,
    service: ModelAdminService = Depends(get_model_admin_service),
):
    detail = await service.get_version(version)
    return ApiResponse.ok(value=ModelDetailResponse.from_detail(detail))


@router.post("/{version}/activate", dependencies=[Depends(require_admin)])
async def activate_version(
    version: str,
    body: Optional[ModelActivateRequest] = None,
    current_user: User = Depends(get_current_user),
    service: ModelAdminService = Depends(get_model_admin_service),
):
    reason = body.reason if body is not None else None
    result = await service.activate(
        version=version, actor=current_user, reason=reason,
    )
    return ApiResponse.ok(
        value=ModelActivationResponse.from_result(result),
        message="Version activated",
    )


@router.post("/{version}/promote", dependencies=[Depends(require_admin)])
async def promote_version(
    version: str,
    body: Optional[ModelPromoteRequest] = None,
    current_user: User = Depends(get_current_user),
    service: ModelAdminService = Depends(get_model_admin_service),
):
    reason = body.reason if body is not None else None
    result = await service.promote(
        version=version, actor=current_user, reason=reason,
    )
    return ApiResponse.ok(
        value=ModelActivationResponse.from_result(result),
        message="Version promoted",
    )


@router.post("/{version}/rollback", dependencies=[Depends(require_admin)])
async def rollback_version(
    version: str,
    body: Optional[ModelRollbackRequest] = None,
    current_user: User = Depends(get_current_user),
    service: ModelAdminService = Depends(get_model_admin_service),
):
    # `version` is the version we are rolling AWAY from; the target rollback
    # version is resolved server-side from the activation history. The path
    # parameter is accepted for routing symmetry with activate/promote and to
    # make the audit log trail unambiguous if the caller hits the wrong row.
    if version != "current":
        # Optional sanity-check: when the caller specifies a concrete version
        # in the path, it must match what the registry currently has active —
        # otherwise the call is racing with another admin and we refuse.
        current_active = service._current_version()
        if current_active is not None and current_active != version:
            raise BadRequestException(
                message="Rollback target mismatch",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="MODEL_ROLLBACK_VERSION_MISMATCH",
                    status=400,
                    details=[
                        f"path version={version!r} but currently active is "
                        f"{current_active!r}; refusing to roll back",
                    ],
                ),
            )

    reason = body.reason if body is not None else None
    result = await service.rollback(actor=current_user, reason=reason)
    return ApiResponse.ok(
        value=ModelActivationResponse.from_result(result),
        message="Rollback complete",
    )


@router.get("/{version}/metrics")
async def get_metrics(
    version: str,
    bucket: str = Query("day"),
    date_from: datetime | None = Query(None, alias="from"),
    date_to: datetime | None = Query(None, alias="to"),
    timezone_name: str = Query("UTC", alias="timezone"),
    service: ModelAdminService = Depends(get_model_admin_service),
):
    _require_valid_bucket(bucket)
    tz = _resolve_timezone(timezone_name)
    result = await service.metrics(
        version=version,
        bucket=bucket,
        date_from=date_from,
        date_to=date_to,
        timezone_name=tz,
    )
    return ApiResponse.ok(value=ModelMetricsResponse.from_result(result))
