from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional

from src.app.models.enums import (
    BalanceStrategy,
    ReviewVerdict,
    Role,
    TrainingBufferSource,
)


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=100)
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UpdateProfileRequest(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    username: Optional[str] = Field(None, min_length=3, max_length=100)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(min_length=8, max_length=128)


class CreateUserRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=100)
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=128)
    role: Role


class UpdateUserRequest(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    role: Optional[Role] = None
    is_active: Optional[bool] = None


class PredictionRequest(BaseModel):
    sender: str = Field(min_length=1)
    subject: str = Field(min_length=0)
    body: str = Field(min_length=1)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class BatchPredictionRequest(BaseModel):
    emails: List[PredictionRequest] = Field(min_length=1)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class ReviewConfirmRequest(BaseModel):
    verdict: ReviewVerdict
    note: Optional[str] = Field(None, max_length=2000)
    # Phase 3 accepts these fields so clients can share one DTO across phases,
    # but rejects non-null values (auto-review isn't wired until Phase 4).
    agreed_with_auto_review: Optional[bool] = Field(
        None, alias="agreedWithAutoReview"
    )
    override_reason: Optional[str] = Field(
        None, alias="overrideReason", max_length=2000
    )

    class Config:
        populate_by_name = True


class ReviewDeferRequest(BaseModel):
    note: Optional[str] = Field(None, max_length=2000)


class ReviewEscalateRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)
    note: Optional[str] = Field(None, max_length=2000)
    # Phase 12 — analyst can record their tentative verdict at escalation
    # time. When the admin-resolved verdict differs, a review_disagreements
    # row is written for audit (gated on review.disagreement_log.enabled).
    tentative_label: Optional[ReviewVerdict] = Field(
        None, alias="tentativeLabel"
    )

    class Config:
        populate_by_name = True


class ReviewReassignRequest(BaseModel):
    user_id: UUID = Field(alias="userId")

    class Config:
        populate_by_name = True


class EscalationResolveRequest(BaseModel):
    verdict: ReviewVerdict
    note: Optional[str] = Field(None, max_length=2000)


class EscalationReturnRequest(BaseModel):
    reason: Optional[str] = Field(None, max_length=2000)


class AutoReviewBatchRequest(BaseModel):
    # Hard cap of 5 enforced server-side too (config.review.auto_review_batch_max)
    # — the 400 message there gives the analyst the actual configured ceiling
    # rather than a Pydantic validation blob.
    review_item_ids: List[UUID] = Field(
        alias="reviewItemIds", min_length=1, max_length=5
    )

    class Config:
        populate_by_name = True


class DriftThresholdUpdateRequest(BaseModel):
    fpr_threshold: float = Field(alias="fprThreshold", ge=0.0, le=1.0)

    class Config:
        populate_by_name = True


class DriftConfirmRequest(BaseModel):
    prediction_id: UUID = Field(alias="predictionId")
    confirmed_label: int = Field(alias="confirmedLabel", ge=0, le=1)

    class Config:
        populate_by_name = True


class TrainingRunFilters(BaseModel):
    """Slice selector for the `training_buffer_items` pull — filters are
    additive (empty = no filter on that dimension). Mirrors §3.5."""

    sources: Optional[List[TrainingBufferSource]] = None
    date_from: Optional[datetime] = Field(default=None, alias="dateFrom")
    date_to: Optional[datetime] = Field(default=None, alias="dateTo")
    categories: Optional[List[str]] = None

    class Config:
        populate_by_name = True


class TrainingRunSelection(BaseModel):
    shuffle: bool = True
    seed: Optional[int] = None
    balance_strategy: BalanceStrategy = Field(
        default=BalanceStrategy.NONE, alias="balanceStrategy"
    )
    max_size: Optional[int] = Field(default=None, alias="maxSize", gt=0)

    class Config:
        populate_by_name = True


class TrainingRunRequest(BaseModel):
    """Body for `POST /api/v1/training/runs` (§3.5). `source_version=None`
    trains against the active version; `auto_promote=true` still goes through
    the §2.7 guard — refused if no holdout is configured."""

    source_version: Optional[str] = Field(
        default=None, alias="sourceVersion", max_length=16
    )
    filters: TrainingRunFilters = Field(default_factory=TrainingRunFilters)
    selection: TrainingRunSelection = Field(default_factory=TrainingRunSelection)
    max_iter_per_call: int = Field(
        default=5, alias="maxIterPerCall", ge=1, le=100
    )
    min_delta_f1: float = Field(default=-0.01, alias="minDeltaF1")
    auto_promote: bool = Field(default=False, alias="autoPromote")

    class Config:
        populate_by_name = True


class ModelActivateRequest(BaseModel):
    """Body for `POST /api/v1/models/{version}/activate` (§7.6). The version
    being activated comes from the path; the body only carries the audit
    reason that gets stamped onto the model_activations row."""

    reason: Optional[str] = Field(None, max_length=500)

    class Config:
        populate_by_name = True


class ModelPromoteRequest(BaseModel):
    """Body for `POST /api/v1/models/{version}/promote` (§7.6). Same audit
    reason; the metrics snapshot promoted alongside the version comes from
    the registry, not the request."""

    reason: Optional[str] = Field(None, max_length=500)

    class Config:
        populate_by_name = True


class ModelRollbackRequest(BaseModel):
    """Body for `POST /api/v1/models/{version}/rollback` (§7.6). The target
    rollback version is resolved server-side from the activation history;
    the body only carries the audit reason."""

    reason: Optional[str] = Field(None, max_length=500)

    class Config:
        populate_by_name = True


class BenchmarkDatasetCreateRequest(BaseModel):
    """Multipart form-field payload alongside the CSV file on
    `POST /api/v1/benchmarks/datasets` (§7.7). `name` is unique per §6.8, so
    a repeat upload with the same name is rejected with 409 before the CSV
    is parsed."""

    name: str = Field(min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=2000)


class BenchmarkRunRequest(BaseModel):
    """Body for `POST /api/v1/benchmarks` (§7.7). Versions are loaded per-run
    via `PhishingDetector.load(version)` — the singleton on app.state is
    never touched, so unregistered or non-active versions can participate."""

    dataset_id: UUID = Field(alias="datasetId")
    versions: List[str] = Field(min_length=1)

    class Config:
        populate_by_name = True


class ModelThresholdRequest(BaseModel):
    """Body for `POST /api/v1/models/thresholds` (§7.6). Live-applies to the
    detector when `version` matches the active version and both review
    bounds are provided; otherwise the row is staged in
    `model_threshold_history` and picked up on the next swap."""

    version: str = Field(max_length=16)
    decision_threshold: float = Field(
        alias="decisionThreshold", ge=0.0, le=1.0
    )
    review_low_threshold: Optional[float] = Field(
        None, alias="reviewLowThreshold", ge=0.0, le=1.0
    )
    review_high_threshold: Optional[float] = Field(
        None, alias="reviewHighThreshold", ge=0.0, le=1.0
    )

    class Config:
        populate_by_name = True
