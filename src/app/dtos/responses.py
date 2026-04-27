from uuid import UUID
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.app.models.auto_review_invocation import AutoReviewInvocation
from src.app.models.benchmark_dataset import BenchmarkDataset
from src.app.models.model_activation import ModelActivation
from src.app.models.model_benchmark import ModelBenchmark
from src.app.models.model_benchmark_version_result import (
    ModelBenchmarkVersionResult,
)
from src.app.models.model_threshold_history import ModelThresholdHistory
from src.app.models.prediction_event import PredictionEvent
from src.app.models.review_escalation import ReviewEscalation
from src.app.models.review_item import ReviewItem
from src.app.models.training_buffer_item import TrainingBufferItem
from src.app.models.training_run import TrainingRun
from src.app.models.user import User
from src.shared.inference import DriftSignal, PredictionResult


class UserResponse(BaseModel):
    id: UUID = Field(alias="id")
    email: str = Field(alias="email")
    username: str = Field(alias="username")
    first_name: str = Field(alias="firstName")
    last_name: str = Field(alias="lastName")
    role: str = Field(alias="role")
    is_active: bool = Field(alias="isActive")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_user(user: User) -> "UserResponse":
        return UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            firstName=user.first_name,
            lastName=user.last_name,
            role=user.role.value,
            isActive=user.is_active,
            createdAt=user.created_at,
            updatedAt=user.updated_at,
        )


class TokenResponse(BaseModel):
    access_token: str = Field(alias="accessToken")
    refresh_token: str = Field(alias="refreshToken")

    class Config:
        populate_by_name = True


class AuthResponse(BaseModel):
    user: UserResponse
    tokens: TokenResponse


class DriftSignalSummary(BaseModel):
    status: str = Field(alias="status")
    false_positive_rate: float = Field(alias="falsePositiveRate")
    total_predictions: int = Field(alias="totalPredictions")
    confirmed_predictions: int = Field(alias="confirmedPredictions")
    threshold: float = Field(alias="threshold")
    message: str = Field(alias="message")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_signal(signal: DriftSignal) -> "DriftSignalSummary":
        return DriftSignalSummary(
            status=signal.status.value,
            falsePositiveRate=signal.false_positive_rate,
            totalPredictions=signal.total_predictions,
            confirmedPredictions=signal.confirmed_predictions,
            threshold=signal.threshold,
            message=signal.message,
        )


# Phase 5 surfaces the same shape under the public name `DriftSignalResponse`
# (per the §7.8 endpoint contract) while keeping `DriftSignalSummary` for the
# embedded use inside `InferenceStatusResponse`.
DriftSignalResponse = DriftSignalSummary


class ConfusionMatrixResponse(BaseModel):
    tp: int = Field(alias="tp")
    tn: int = Field(alias="tn")
    fp: int = Field(alias="fp")
    fn: int = Field(alias="fn")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_matrix(matrix: dict[str, int]) -> "ConfusionMatrixResponse":
        return ConfusionMatrixResponse(
            tp=int(matrix.get("tp", 0)),
            tn=int(matrix.get("tn", 0)),
            fp=int(matrix.get("fp", 0)),
            fn=int(matrix.get("fn", 0)),
        )


class DriftHistoryBucket(BaseModel):
    bucket: datetime = Field(alias="bucket")
    false_positive_rate: float = Field(alias="falsePositiveRate")
    tp: int = Field(alias="tp")
    tn: int = Field(alias="tn")
    fp: int = Field(alias="fp")
    fn: int = Field(alias="fn")
    confirmed: int = Field(alias="confirmed")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_row(row: dict) -> "DriftHistoryBucket":
        return DriftHistoryBucket(
            bucket=row["bucket"],
            falsePositiveRate=float(row["false_positive_rate"]),
            tp=int(row["tp"]),
            tn=int(row["tn"]),
            fp=int(row["fp"]),
            fn=int(row["fn"]),
            confirmed=int(row["confirmed"]),
        )


class DriftHistoryResponse(BaseModel):
    bucket: str = Field(alias="bucket")
    timezone: str = Field(alias="timezone")
    buckets: list[DriftHistoryBucket] = Field(alias="buckets")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_rows(
        rows: list[dict], *, bucket: str, timezone_name: str,
    ) -> "DriftHistoryResponse":
        return DriftHistoryResponse(
            bucket=bucket,
            timezone=timezone_name,
            buckets=[DriftHistoryBucket.from_row(r) for r in rows],
        )


REDACTED_BODY_PLACEHOLDER = "[REDACTED]"


class PredictionResponse(BaseModel):
    """camelCase wrapping of `PredictionResult.to_dict()`.

    Shape mirrors §7.1 response contract for `POST /analysis/predict` and
    carries the persisted audit-row id so the caller can open the detail /
    explain views without a round-trip.
    """

    prediction_event_id: UUID = Field(alias="predictionEventId")
    prediction_id: str | None = Field(default=None, alias="predictionId")
    request_id: str = Field(alias="requestId")
    predicted_label: int = Field(alias="predictedLabel")
    phishing_probability: float = Field(alias="phishingProbability")
    legitimate_probability: float = Field(alias="legitimateProbability")
    raw_phishing_probability: float | None = Field(
        default=None, alias="rawPhishingProbability"
    )
    raw_legitimate_probability: float | None = Field(
        default=None, alias="rawLegitimateProbability"
    )
    calibrated: bool = Field(alias="calibrated")
    threshold: float = Field(alias="threshold")
    model_version: str | None = Field(default=None, alias="modelVersion")
    confidence_zone: str | None = Field(default=None, alias="confidenceZone")
    review_low_threshold: float | None = Field(
        default=None, alias="reviewLowThreshold"
    )
    review_high_threshold: float | None = Field(
        default=None, alias="reviewHighThreshold"
    )
    engineered_features: dict[str, float] = Field(
        default_factory=dict, alias="engineeredFeatures"
    )
    predicted_at: datetime = Field(alias="predictedAt")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_event(
        event: PredictionEvent, result: PredictionResult
    ) -> "PredictionResponse":
        payload = result.to_dict()
        return PredictionResponse(
            predictionEventId=event.id,
            predictionId=payload.get("prediction_id"),
            requestId=event.request_id,
            predictedLabel=int(payload["predicted_label"]),
            phishingProbability=float(payload["phishing_probability"]),
            legitimateProbability=float(payload["legitimate_probability"]),
            rawPhishingProbability=payload.get("raw_phishing_probability"),
            rawLegitimateProbability=payload.get("raw_legitimate_probability"),
            calibrated=bool(payload.get("calibrated", False)),
            threshold=float(payload["threshold"]),
            modelVersion=payload.get("model_version"),
            confidenceZone=payload.get("confidence_zone"),
            reviewLowThreshold=payload.get("review_low_threshold"),
            reviewHighThreshold=payload.get("review_high_threshold"),
            engineeredFeatures=dict(payload.get("engineered_features") or {}),
            predictedAt=event.predicted_at,
        )


class PredictionEventSummaryResponse(BaseModel):
    """List-view projection of a `prediction_events` row.

    Callers that are not an admin and did not own the prediction get the
    body replaced with `REDACTED_BODY_PLACEHOLDER` at construction time.
    """

    id: UUID = Field(alias="id")
    prediction_id: UUID | None = Field(default=None, alias="predictionId")
    request_id: str = Field(alias="requestId")
    requester_id: UUID | None = Field(default=None, alias="requesterId")
    source: str = Field(alias="source")
    model_version: str = Field(alias="modelVersion")
    sender: str = Field(alias="sender")
    subject: str = Field(alias="subject")
    body: str = Field(alias="body")
    predicted_label: int = Field(alias="predictedLabel")
    phishing_probability: float = Field(alias="phishingProbability")
    legitimate_probability: float = Field(alias="legitimateProbability")
    raw_phishing_probability: float | None = Field(
        default=None, alias="rawPhishingProbability"
    )
    raw_legitimate_probability: float | None = Field(
        default=None, alias="rawLegitimateProbability"
    )
    calibrated: bool = Field(alias="calibrated")
    threshold: float = Field(alias="threshold")
    confidence_zone: str | None = Field(default=None, alias="confidenceZone")
    review_low_threshold: float | None = Field(
        default=None, alias="reviewLowThreshold"
    )
    review_high_threshold: float | None = Field(
        default=None, alias="reviewHighThreshold"
    )
    predicted_at: datetime = Field(alias="predictedAt")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_event(
        event: PredictionEvent, *, redact_body: bool = False
    ) -> "PredictionEventSummaryResponse":
        body = REDACTED_BODY_PLACEHOLDER if redact_body else event.body
        return PredictionEventSummaryResponse(
            id=event.id,
            predictionId=event.prediction_id,
            requestId=event.request_id,
            requesterId=event.requester_id,
            source=event.source.value,
            modelVersion=event.model_version,
            sender=event.sender,
            subject=event.subject,
            body=body,
            predictedLabel=event.predicted_label,
            phishingProbability=event.phishing_probability,
            legitimateProbability=event.legitimate_probability,
            rawPhishingProbability=event.raw_phishing_probability,
            rawLegitimateProbability=event.raw_legitimate_probability,
            calibrated=event.calibrated,
            threshold=event.threshold,
            confidenceZone=(
                event.confidence_zone.value if event.confidence_zone is not None else None
            ),
            reviewLowThreshold=event.review_low_threshold,
            reviewHighThreshold=event.review_high_threshold,
            predictedAt=event.predicted_at,
            createdAt=event.created_at,
            updatedAt=event.updated_at,
        )


class TopTermResponse(BaseModel):
    """One row inside `PredictionExplainResponse.topTerms` (§Phase 12).

    `weight` is the signed contribution toward the phishing class (TF-IDF ×
    linear coefficient when the active model exposes `coef_`, else the raw
    TF-IDF). `tfidf` is always the non-negative TF-IDF value so the UI can
    show surface strength separately from direction."""

    term: str = Field(alias="term")
    weight: float = Field(alias="weight")
    source: str = Field(alias="source")
    tfidf: float = Field(alias="tfidf")

    class Config:
        populate_by_name = True


class PredictionExplainResponse(BaseModel):
    """Explain view for `GET /analysis/predictions/{id}/explain`.

    Per §3.11: reads `engineered_features` from the persisted row and surfaces
    the zone context. No re-prediction.
    """

    prediction_event_id: UUID = Field(alias="predictionEventId")
    model_version: str = Field(alias="modelVersion")
    predicted_label: int = Field(alias="predictedLabel")
    phishing_probability: float = Field(alias="phishingProbability")
    legitimate_probability: float = Field(alias="legitimateProbability")
    threshold: float = Field(alias="threshold")
    confidence_zone: str | None = Field(default=None, alias="confidenceZone")
    review_low_threshold: float | None = Field(
        default=None, alias="reviewLowThreshold"
    )
    review_high_threshold: float | None = Field(
        default=None, alias="reviewHighThreshold"
    )
    engineered_features: dict[str, Any] = Field(alias="engineeredFeatures")
    top_terms: list[TopTermResponse] | None = Field(
        default=None, alias="topTerms"
    )

    class Config:
        populate_by_name = True

    @staticmethod
    def from_event(
        event: PredictionEvent,
        *,
        top_terms: list[dict] | None = None,
    ) -> "PredictionExplainResponse":
        projected_terms = (
            [
                TopTermResponse(
                    term=str(t["term"]),
                    weight=float(t["weight"]),
                    source=str(t["source"]),
                    tfidf=float(t["tfidf"]),
                )
                for t in top_terms
            ]
            if top_terms is not None
            else None
        )
        return PredictionExplainResponse(
            predictionEventId=event.id,
            modelVersion=event.model_version,
            predictedLabel=event.predicted_label,
            phishingProbability=event.phishing_probability,
            legitimateProbability=event.legitimate_probability,
            threshold=event.threshold,
            confidenceZone=(
                event.confidence_zone.value if event.confidence_zone is not None else None
            ),
            reviewLowThreshold=event.review_low_threshold,
            reviewHighThreshold=event.review_high_threshold,
            engineeredFeatures=dict(event.engineered_features or {}),
            topTerms=projected_terms,
        )


class ReviewItemResponse(BaseModel):
    """Projection of a `review_items` row with the joined prediction context
    the analyst needs to triage the item (§7.2).

    Non-admins outside their own scope see the body replaced with the
    redaction placeholder, matching the §3.10 privacy rule.
    """

    id: UUID = Field(alias="id")
    prediction_event_id: UUID = Field(alias="predictionEventId")
    status: str = Field(alias="status")
    assigned_to: UUID | None = Field(default=None, alias="assignedTo")
    assigned_at: datetime | None = Field(default=None, alias="assignedAt")
    claimed_at: datetime | None = Field(default=None, alias="claimedAt")
    decided_at: datetime | None = Field(default=None, alias="decidedAt")
    decided_by: UUID | None = Field(default=None, alias="decidedBy")
    verdict: str | None = Field(default=None, alias="verdict")
    reviewer_note: str | None = Field(default=None, alias="reviewerNote")
    auto_review_used: bool = Field(alias="autoReviewUsed")
    auto_review_agreement: str = Field(alias="autoReviewAgreement")
    latest_auto_review_invocation_id: UUID | None = Field(
        default=None, alias="latestAutoReviewInvocationId"
    )
    override_reason: str | None = Field(default=None, alias="overrideReason")

    model_version: str = Field(alias="modelVersion")
    predicted_label: int = Field(alias="predictedLabel")
    phishing_probability: float = Field(alias="phishingProbability")
    legitimate_probability: float = Field(alias="legitimateProbability")
    confidence_zone: str | None = Field(default=None, alias="confidenceZone")
    predicted_at: datetime = Field(alias="predictedAt")

    sender: str = Field(alias="sender")
    subject: str = Field(alias="subject")
    body: str = Field(alias="body")

    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_item(item: ReviewItem, *, redact_body: bool = False) -> "ReviewItemResponse":
        event = item.prediction_event
        body = REDACTED_BODY_PLACEHOLDER if redact_body else event.body
        return ReviewItemResponse(
            id=item.id,
            predictionEventId=item.prediction_event_id,
            status=item.status.value,
            assignedTo=item.assigned_to,
            assignedAt=item.assigned_at,
            claimedAt=item.claimed_at,
            decidedAt=item.decided_at,
            decidedBy=item.decided_by,
            verdict=item.verdict.value if item.verdict is not None else None,
            reviewerNote=item.reviewer_note,
            autoReviewUsed=item.auto_review_used,
            autoReviewAgreement=item.auto_review_agreement.value,
            latestAutoReviewInvocationId=item.latest_auto_review_invocation_id,
            overrideReason=item.override_reason,
            modelVersion=event.model_version,
            predictedLabel=event.predicted_label,
            phishingProbability=event.phishing_probability,
            legitimateProbability=event.legitimate_probability,
            confidenceZone=(
                event.confidence_zone.value
                if event.confidence_zone is not None
                else None
            ),
            predictedAt=event.predicted_at,
            sender=event.sender,
            subject=event.subject,
            body=body,
            createdAt=item.created_at,
            updatedAt=item.updated_at,
        )


class EscalationResponse(BaseModel):
    id: UUID = Field(alias="id")
    review_item_id: UUID = Field(alias="reviewItemId")
    escalated_by: UUID | None = Field(default=None, alias="escalatedBy")
    escalated_to: UUID | None = Field(default=None, alias="escalatedTo")
    reason: str = Field(alias="reason")
    note: str | None = Field(default=None, alias="note")
    resolved_at: datetime | None = Field(default=None, alias="resolvedAt")
    resolved_by: UUID | None = Field(default=None, alias="resolvedBy")
    resolution_verdict: str | None = Field(
        default=None, alias="resolutionVerdict"
    )
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_escalation(
        escalation: ReviewEscalation,
    ) -> "EscalationResponse":
        return EscalationResponse(
            id=escalation.id,
            reviewItemId=escalation.review_item_id,
            escalatedBy=escalation.escalated_by,
            escalatedTo=escalation.escalated_to,
            reason=escalation.reason,
            note=escalation.note,
            resolvedAt=escalation.resolved_at,
            resolvedBy=escalation.resolved_by,
            resolutionVerdict=(
                escalation.resolution_verdict.value
                if escalation.resolution_verdict is not None
                else None
            ),
            createdAt=escalation.created_at,
            updatedAt=escalation.updated_at,
        )


class AutoReviewResponse(BaseModel):
    """Wraps the union returned by `AutoReviewer.review()` (success or
    failure) and surfaces the persisted invocation row id so the caller can
    fetch the history entry directly.

    `outcome` discriminates the two shapes — on FAILURE the `label`,
    `confidence`, `reasoning`, and `rawResponse` fields are null and the
    `userMessage` / `technicalError` pair is populated instead.
    """

    invocation_id: UUID = Field(alias="invocationId")
    review_item_id: UUID = Field(alias="reviewItemId")
    trigger_kind: str = Field(alias="triggerKind")
    batch_group_id: UUID | None = Field(default=None, alias="batchGroupId")
    outcome: str = Field(alias="outcome")
    provider: str = Field(alias="provider")
    model_name: str = Field(alias="modelName")
    duration_ms: int = Field(alias="durationMs")
    label: str | None = Field(default=None, alias="label")
    confidence: str | None = Field(default=None, alias="confidence")
    reasoning: str | None = Field(default=None, alias="reasoning")
    raw_response: dict | None = Field(default=None, alias="rawResponse")
    user_message: str | None = Field(default=None, alias="userMessage")
    technical_error: str | None = Field(default=None, alias="technicalError")
    created_at: datetime = Field(alias="createdAt")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_invocation(
        invocation: AutoReviewInvocation,
    ) -> "AutoReviewResponse":
        return AutoReviewResponse(
            invocationId=invocation.id,
            reviewItemId=invocation.review_item_id,
            triggerKind=invocation.trigger_kind,
            batchGroupId=invocation.batch_group_id,
            outcome=invocation.outcome.value,
            provider=invocation.provider.value,
            modelName=invocation.model_name,
            durationMs=invocation.duration_ms,
            label=invocation.label,
            confidence=invocation.confidence,
            reasoning=invocation.reasoning,
            rawResponse=invocation.raw_payload,
            userMessage=invocation.user_message,
            technicalError=invocation.technical_error,
            createdAt=invocation.created_at,
        )


class AutoReviewInvocationResponse(BaseModel):
    """History view of a single `auto_review_invocations` row (§7.2 GET
    `/review/queue/{id}/auto-reviews`). Same shape as `AutoReviewResponse`
    but kept as a separate type so list-view evolution stays decoupled from
    the trigger-response contract.
    """

    id: UUID = Field(alias="id")
    review_item_id: UUID = Field(alias="reviewItemId")
    triggered_by: UUID | None = Field(default=None, alias="triggeredBy")
    trigger_kind: str = Field(alias="triggerKind")
    batch_group_id: UUID | None = Field(default=None, alias="batchGroupId")
    outcome: str = Field(alias="outcome")
    provider: str = Field(alias="provider")
    model_name: str = Field(alias="modelName")
    duration_ms: int = Field(alias="durationMs")
    label: str | None = Field(default=None, alias="label")
    confidence: str | None = Field(default=None, alias="confidence")
    reasoning: str | None = Field(default=None, alias="reasoning")
    raw_response: dict | None = Field(default=None, alias="rawResponse")
    user_message: str | None = Field(default=None, alias="userMessage")
    technical_error: str | None = Field(default=None, alias="technicalError")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_invocation(
        invocation: AutoReviewInvocation,
    ) -> "AutoReviewInvocationResponse":
        return AutoReviewInvocationResponse(
            id=invocation.id,
            reviewItemId=invocation.review_item_id,
            triggeredBy=invocation.triggered_by,
            triggerKind=invocation.trigger_kind,
            batchGroupId=invocation.batch_group_id,
            outcome=invocation.outcome.value,
            provider=invocation.provider.value,
            modelName=invocation.model_name,
            durationMs=invocation.duration_ms,
            label=invocation.label,
            confidence=invocation.confidence,
            reasoning=invocation.reasoning,
            rawResponse=invocation.raw_payload,
            userMessage=invocation.user_message,
            technicalError=invocation.technical_error,
            createdAt=invocation.created_at,
            updatedAt=invocation.updated_at,
        )


class InferenceStatusResponse(BaseModel):
    detector_loaded: bool = Field(alias="detectorLoaded")
    active_version: str | None = Field(default=None, alias="activeVersion")
    decision_threshold: float | None = Field(default=None, alias="decisionThreshold")
    review_low_threshold: float | None = Field(default=None, alias="reviewLowThreshold")
    review_high_threshold: float | None = Field(default=None, alias="reviewHighThreshold")
    drift_monitor: DriftSignalSummary = Field(alias="driftMonitor")
    auto_reviewer_available: bool = Field(alias="autoReviewerAvailable")
    llm_provider: str | None = Field(default=None, alias="llmProvider")
    llm_model_name: str | None = Field(default=None, alias="llmModelName")
    last_training_run: dict | None = Field(default=None, alias="lastTrainingRun")
    warnings: list[str] = Field(default_factory=list, alias="warnings")

    class Config:
        populate_by_name = True


class BufferStatusResponse(BaseModel):
    """Aggregate view of the training buffer (§7.4 GET /training/buffer/status).

    `classCounts` always carries both binary classes (zero-filled when absent)
    so the UI can render fixed bars; `blockers` lists every reason the buffer
    is locked rather than short-circuiting on the first one."""

    size: int = Field(alias="size")
    class_counts: dict[str, int] = Field(alias="classCounts")
    unlocked: bool = Field(alias="unlocked")
    blockers: list[str] = Field(alias="blockers")
    min_batch_size: int = Field(alias="minBatchSize")
    min_per_class: int = Field(alias="minPerClass")
    require_balance_delta: float = Field(alias="requireBalanceDelta")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_status(status: Any) -> "BufferStatusResponse":
        return BufferStatusResponse(
            size=status.size,
            classCounts={str(k): v for k, v in status.class_counts.items()},
            unlocked=status.unlocked,
            blockers=list(status.blockers),
            minBatchSize=status.min_batch_size,
            minPerClass=status.min_per_class,
            requireBalanceDelta=status.require_balance_delta,
        )


class BufferItemResponse(BaseModel):
    """Projection of a `training_buffer_items` row (§7.4)."""

    id: UUID = Field(alias="id")
    sender: str = Field(alias="sender")
    subject: str = Field(alias="subject")
    body: str = Field(alias="body")
    label: int = Field(alias="label")
    source: str = Field(alias="source")
    source_prediction_event_id: UUID | None = Field(
        default=None, alias="sourcePredictionEventId"
    )
    source_review_item_id: UUID | None = Field(
        default=None, alias="sourceReviewItemId"
    )
    category: str | None = Field(default=None, alias="category")
    content_sha256: str = Field(alias="contentSha256")
    contributed_by: UUID | None = Field(default=None, alias="contributedBy")
    consumed_in_run_ids: list[UUID] = Field(
        default_factory=list, alias="consumedInRunIds"
    )
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_item(item: TrainingBufferItem) -> "BufferItemResponse":
        return BufferItemResponse(
            id=item.id,
            sender=item.sender,
            subject=item.subject,
            body=item.body,
            label=int(item.label),
            source=item.source.value,
            sourcePredictionEventId=item.source_prediction_event_id,
            sourceReviewItemId=item.source_review_item_id,
            category=item.category,
            contentSha256=item.content_sha256,
            contributedBy=item.contributed_by,
            consumedInRunIds=list(item.consumed_in_run_ids or []),
            createdAt=item.created_at,
            updatedAt=item.updated_at,
        )


class TrainingRunResponse(BaseModel):
    """Summary projection of a `training_runs` row (§7.5 list + GET detail).

    The I-role read-only summary uses this same shape — admins see the
    richer `TrainingRunDetailResponse`."""

    id: UUID = Field(alias="id")
    triggered_by: UUID = Field(alias="triggeredBy")
    status: str = Field(alias="status")
    source_version: str = Field(alias="sourceVersion")
    new_version: str | None = Field(default=None, alias="newVersion")
    batch_size: int = Field(alias="batchSize")
    iterations: int = Field(alias="iterations")
    balance_strategy: str = Field(alias="balanceStrategy")
    shuffle: bool = Field(alias="shuffle")
    seed: int | None = Field(default=None, alias="seed")
    min_delta_f1: float = Field(alias="minDeltaF1")
    max_iter_per_call: int = Field(alias="maxIterPerCall")
    promoted: bool = Field(alias="promoted")
    promoted_at: datetime | None = Field(default=None, alias="promotedAt")
    promoted_by: UUID | None = Field(default=None, alias="promotedBy")
    rolled_back_at: datetime | None = Field(
        default=None, alias="rolledBackAt"
    )
    error_message: str | None = Field(default=None, alias="errorMessage")
    started_at: datetime | None = Field(default=None, alias="startedAt")
    finished_at: datetime | None = Field(default=None, alias="finishedAt")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_run(run: TrainingRun) -> "TrainingRunResponse":
        return TrainingRunResponse(
            id=run.id,
            triggeredBy=run.triggered_by,
            status=run.status.value,
            sourceVersion=run.source_version,
            newVersion=run.new_version,
            batchSize=run.batch_size,
            iterations=run.iterations,
            balanceStrategy=run.balance_strategy.value,
            shuffle=run.shuffle,
            seed=run.seed,
            minDeltaF1=run.min_delta_f1,
            maxIterPerCall=run.max_iter_per_call,
            promoted=run.promoted,
            promotedAt=run.promoted_at,
            promotedBy=run.promoted_by,
            rolledBackAt=run.rolled_back_at,
            errorMessage=run.error_message,
            startedAt=run.started_at,
            finishedAt=run.finished_at,
            createdAt=run.created_at,
            updatedAt=run.updated_at,
        )


class TrainingRunDetailResponse(TrainingRunResponse):
    """Admin-only full detail view of a run — adds the metric / provenance
    payloads the list view omits."""

    performance_before: dict[str, Any] | None = Field(
        default=None, alias="performanceBefore"
    )
    performance_after: dict[str, Any] | None = Field(
        default=None, alias="performanceAfter"
    )
    oov_rate_subject: float | None = Field(
        default=None, alias="oovRateSubject"
    )
    oov_rate_body: float | None = Field(default=None, alias="oovRateBody")
    buffer_snapshot_ids: list[UUID] = Field(
        default_factory=list, alias="bufferSnapshotIds"
    )
    class_counts_before_balance: dict[str, int] = Field(
        default_factory=dict, alias="classCountsBeforeBalance"
    )
    class_counts_after_balance: dict[str, int] = Field(
        default_factory=dict, alias="classCountsAfterBalance"
    )

    @staticmethod
    def from_run(run: TrainingRun) -> "TrainingRunDetailResponse":  # type: ignore[override]
        base = TrainingRunResponse.from_run(run)
        return TrainingRunDetailResponse(
            **base.model_dump(by_alias=True),
            performanceBefore=(
                dict(run.performance_before) if run.performance_before else None
            ),
            performanceAfter=(
                dict(run.performance_after) if run.performance_after else None
            ),
            oovRateSubject=run.oov_rate_subject,
            oovRateBody=run.oov_rate_body,
            bufferSnapshotIds=list(run.buffer_snapshot_ids or []),
            classCountsBeforeBalance={
                str(k): int(v)
                for k, v in (run.class_counts_before_balance or {}).items()
            },
            classCountsAfterBalance={
                str(k): int(v)
                for k, v in (run.class_counts_after_balance or {}).items()
            },
        )


class TrainingRunPreviewResponse(BaseModel):
    """Side-effect-free preview of what `POST /training/runs` would do for
    a given filter + selection. Drives the create-run form's live "X samples
    match" indicator so users never submit a run that the gate will reject.

    Always agrees with `_ensure_slice_non_empty` because the service uses the
    same helpers to compute both."""

    matched_total: int = Field(alias="matchedTotal")
    matched_class_0: int = Field(alias="matchedClass0")
    matched_class_1: int = Field(alias="matchedClass1")
    balance_strategy: str = Field(alias="balanceStrategy")
    feasible: bool = Field(alias="feasible")
    # When `feasible=false`, this discriminates the two failure modes so the
    # UI can pick the right remediation hint without parsing the message.
    reason_code: str | None = Field(default=None, alias="reasonCode")
    message: str | None = Field(default=None, alias="message")

    class Config:
        populate_by_name = True


class TrainingRunEvent(BaseModel):
    """Payload of an SSE event emitted from `GET /runs/{id}/events` (§3.5).

    `kind` discriminates the stream: `status` (lifecycle), `iteration`
    (per-iter metrics), `version_registered` (new_version id), `error`."""

    run_id: UUID = Field(alias="runId")
    kind: str = Field(alias="kind")
    status: str | None = Field(default=None, alias="status")
    iteration: int | None = Field(default=None, alias="iteration")
    metrics: dict[str, Any] | None = Field(default=None, alias="metrics")
    oov_rate_subject: float | None = Field(
        default=None, alias="oovRateSubject"
    )
    oov_rate_body: float | None = Field(default=None, alias="oovRateBody")
    new_version: str | None = Field(default=None, alias="newVersion")
    message: str | None = Field(default=None, alias="message")
    emitted_at: datetime = Field(alias="emittedAt")

    class Config:
        populate_by_name = True


class BufferImportRowErrorResponse(BaseModel):
    row_number: int = Field(alias="rowNumber")
    message: str = Field(alias="message")

    class Config:
        populate_by_name = True


class BufferImportSummaryResponse(BaseModel):
    """Outcome of `POST /training/buffer/import` (§7.4).

    `inserted + duplicates + len(errors) == totalRows` for every successful
    request; the file SHA-256 is surfaced so callers can correlate this audit
    record with the server log line."""

    total_rows: int = Field(alias="totalRows")
    inserted: int = Field(alias="inserted")
    duplicates: int = Field(alias="duplicates")
    errors: list[BufferImportRowErrorResponse] = Field(alias="errors")
    file_sha256: str = Field(alias="fileSha256")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_summary(summary: Any) -> "BufferImportSummaryResponse":
        return BufferImportSummaryResponse(
            totalRows=summary.total_rows,
            inserted=summary.inserted,
            duplicates=summary.duplicates,
            errors=[
                BufferImportRowErrorResponse(
                    rowNumber=err.row_number, message=err.message
                )
                for err in summary.errors
            ],
            fileSha256=summary.file_sha256,
        )


# ── Phase 9 — Model version management (§7.6) ──


class ModelSummaryResponse(BaseModel):
    """List-view row for `GET /api/v1/models`. Same shape drives list and
    compare pickers; `metrics` is the registry-recorded snapshot (may be
    empty for versions that were registered before the promotion flow)."""

    version: str = Field(alias="version")
    active: bool = Field(alias="active")
    promoted: bool = Field(alias="promoted")
    source_version: str | None = Field(default=None, alias="sourceVersion")
    metrics: dict[str, float] = Field(default_factory=dict, alias="metrics")
    sha256: str | None = Field(default=None, alias="sha256")
    calibrator_sha256: str | None = Field(
        default=None, alias="calibratorSha256"
    )

    class Config:
        populate_by_name = True

    @staticmethod
    def from_row(row: dict) -> "ModelSummaryResponse":
        return ModelSummaryResponse(
            version=row["version"],
            active=bool(row.get("active", False)),
            promoted=bool(row.get("promoted", False)),
            sourceVersion=row.get("source_version"),
            metrics={
                str(k): float(v) for k, v in (row.get("metrics") or {}).items()
            },
            sha256=row.get("sha256"),
            calibratorSha256=row.get("calibrator_sha256"),
        )


class ModelThresholdSummaryResponse(BaseModel):
    """Currently-effective threshold row for a version (effective_to=null)."""

    decision_threshold: float = Field(alias="decisionThreshold")
    review_low_threshold: float | None = Field(
        default=None, alias="reviewLowThreshold"
    )
    review_high_threshold: float | None = Field(
        default=None, alias="reviewHighThreshold"
    )
    effective_from: datetime = Field(alias="effectiveFrom")
    set_by: UUID | None = Field(default=None, alias="setBy")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_row(row: ModelThresholdHistory) -> "ModelThresholdSummaryResponse":
        return ModelThresholdSummaryResponse(
            decisionThreshold=row.decision_threshold,
            reviewLowThreshold=row.review_low_threshold,
            reviewHighThreshold=row.review_high_threshold,
            effectiveFrom=row.effective_from,
            setBy=row.set_by,
        )


class ModelDetailResponse(ModelSummaryResponse):
    """Detail view for `GET /api/v1/models/{version}` — adds the resolved
    artefact paths and the currently-effective threshold row."""

    paths: dict[str, str | None] = Field(
        default_factory=dict, alias="paths"
    )
    current_thresholds: ModelThresholdSummaryResponse | None = Field(
        default=None, alias="currentThresholds"
    )

    @staticmethod
    def from_detail(detail: dict) -> "ModelDetailResponse":
        base = ModelSummaryResponse.from_row(detail).model_dump(by_alias=True)
        thresholds = detail.get("current_thresholds")
        current = None
        if thresholds is not None:
            current = ModelThresholdSummaryResponse(
                decisionThreshold=thresholds["decisionThreshold"],
                reviewLowThreshold=thresholds.get("reviewLowThreshold"),
                reviewHighThreshold=thresholds.get("reviewHighThreshold"),
                effectiveFrom=datetime.fromisoformat(
                    thresholds["effectiveFrom"]
                ),
                setBy=(
                    UUID(thresholds["setBy"])
                    if thresholds.get("setBy") is not None
                    else None
                ),
            )
        return ModelDetailResponse(
            **base,
            paths=dict(detail.get("paths") or {}),
            currentThresholds=current,
        )


class ModelActivationResponse(BaseModel):
    """Response body for activate/promote/rollback calls (§7.6)."""

    activation_id: UUID = Field(alias="activationId")
    version: str = Field(alias="version")
    previous_version: str | None = Field(
        default=None, alias="previousVersion"
    )
    detector_version: str | None = Field(
        default=None, alias="detectorVersion"
    )
    actor_id: UUID | None = Field(default=None, alias="actorId")
    metrics_snapshot: dict[str, float] | None = Field(
        default=None, alias="metricsSnapshot"
    )

    class Config:
        populate_by_name = True

    @staticmethod
    def from_result(result: dict) -> "ModelActivationResponse":
        return ModelActivationResponse(
            activationId=result["activationId"],
            version=result["version"],
            previousVersion=result.get("previousVersion"),
            detectorVersion=result.get("detectorVersion"),
            actorId=result.get("actorId"),
            metricsSnapshot=(
                {str(k): float(v) for k, v in result["metricsSnapshot"].items()}
                if result.get("metricsSnapshot")
                else None
            ),
        )


class ModelActivationHistoryResponse(BaseModel):
    """Projection of a `model_activations` audit row."""

    id: UUID = Field(alias="id")
    kind: str = Field(alias="kind")
    version: str = Field(alias="version")
    previous_version: str | None = Field(
        default=None, alias="previousVersion"
    )
    actor_id: UUID | None = Field(default=None, alias="actorId")
    reason: str | None = Field(default=None, alias="reason")
    metrics_snapshot: dict[str, Any] | None = Field(
        default=None, alias="metricsSnapshot"
    )
    created_at: datetime = Field(alias="createdAt")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_activation(
        activation: ModelActivation,
    ) -> "ModelActivationHistoryResponse":
        return ModelActivationHistoryResponse(
            id=activation.id,
            kind=activation.kind.value,
            version=activation.version,
            previousVersion=activation.previous_version,
            actorId=activation.actor_id,
            reason=activation.reason,
            metricsSnapshot=(
                dict(activation.metrics_snapshot)
                if activation.metrics_snapshot is not None
                else None
            ),
            createdAt=activation.created_at,
        )


class ModelThresholdResponse(BaseModel):
    """Response body for `POST /api/v1/models/thresholds` (§7.6)."""

    history_id: UUID = Field(alias="historyId")
    version: str = Field(alias="version")
    decision_threshold: float = Field(alias="decisionThreshold")
    review_low_threshold: float | None = Field(
        default=None, alias="reviewLowThreshold"
    )
    review_high_threshold: float | None = Field(
        default=None, alias="reviewHighThreshold"
    )
    effective_from: datetime = Field(alias="effectiveFrom")
    set_by: UUID | None = Field(default=None, alias="setBy")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_result(result: dict) -> "ModelThresholdResponse":
        return ModelThresholdResponse(
            historyId=result["historyId"],
            version=result["version"],
            decisionThreshold=result["decisionThreshold"],
            reviewLowThreshold=result.get("reviewLowThreshold"),
            reviewHighThreshold=result.get("reviewHighThreshold"),
            effectiveFrom=result["effectiveFrom"],
            setBy=result.get("setBy"),
        )


class ModelMetricsBucket(BaseModel):
    """One (version, bucket) row from the bucketed metrics query."""

    bucket: datetime = Field(alias="bucket")
    accuracy: float = Field(alias="accuracy")
    precision: float = Field(alias="precision")
    recall: float = Field(alias="recall")
    f1: float = Field(alias="f1")
    false_positive_rate: float = Field(alias="falsePositiveRate")
    tp: int = Field(alias="tp")
    tn: int = Field(alias="tn")
    fp: int = Field(alias="fp")
    fn: int = Field(alias="fn")
    confirmed: int = Field(alias="confirmed")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_row(row: dict) -> "ModelMetricsBucket":
        return ModelMetricsBucket(
            bucket=row["bucket"],
            accuracy=float(row["accuracy"]),
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            f1=float(row["f1"]),
            falsePositiveRate=float(row["false_positive_rate"]),
            tp=int(row["tp"]),
            tn=int(row["tn"]),
            fp=int(row["fp"]),
            fn=int(row["fn"]),
            confirmed=int(row["confirmed"]),
        )


class ModelMetricsResponse(BaseModel):
    version: str = Field(alias="version")
    bucket: str = Field(alias="bucket")
    timezone: str = Field(alias="timezone")
    buckets: list[ModelMetricsBucket] = Field(alias="buckets")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_result(result: dict) -> "ModelMetricsResponse":
        return ModelMetricsResponse(
            version=result["version"],
            bucket=result["bucket"],
            timezone=result["timezone"],
            buckets=[
                ModelMetricsBucket.from_row(row) for row in result["buckets"]
            ],
        )


class ModelMetricsCompareResponse(BaseModel):
    versions: list[str] = Field(alias="versions")
    bucket: str = Field(alias="bucket")
    timezone: str = Field(alias="timezone")
    series: dict[str, list[ModelMetricsBucket]] = Field(alias="series")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_result(result: dict) -> "ModelMetricsCompareResponse":
        series = {
            version: [ModelMetricsBucket.from_row(row) for row in rows]
            for version, rows in result["series"].items()
        }
        return ModelMetricsCompareResponse(
            versions=list(result["versions"]),
            bucket=result["bucket"],
            timezone=result["timezone"],
            series=series,
        )


class ModelUploadSummaryResponse(BaseModel):
    """Outcome of `POST /api/v1/models/upload` (§7.6 stub)."""

    accepted: bool = Field(alias="accepted")
    filename: str = Field(alias="filename")
    sha256: str = Field(alias="sha256")
    source_version: str = Field(alias="sourceVersion")
    registered: bool = Field(alias="registered")
    message: str = Field(alias="message")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_result(result: dict) -> "ModelUploadSummaryResponse":
        return ModelUploadSummaryResponse(
            accepted=bool(result["accepted"]),
            filename=result["filename"],
            sha256=result["sha256"],
            sourceVersion=result["sourceVersion"],
            registered=bool(result["registered"]),
            message=result["message"],
        )


# ── Phase 10 — Benchmarking (§7.7) ──


class BenchmarkDatasetSummary(BaseModel):
    """List-view row for `GET /api/v1/benchmarks/datasets` (§7.7). Omits the
    `description` to keep the listing payload compact — the detail endpoint
    surfaces it alongside the row-count breakdown."""

    id: UUID = Field(alias="id")
    name: str = Field(alias="name")
    row_count: int = Field(alias="rowCount")
    label_distribution: dict[str, int] = Field(
        default_factory=dict, alias="labelDistribution"
    )
    uploaded_by: UUID | None = Field(default=None, alias="uploadedBy")
    source_csv_sha256: str = Field(alias="sourceCsvSha256")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_dataset(dataset: BenchmarkDataset) -> "BenchmarkDatasetSummary":
        return BenchmarkDatasetSummary(
            id=dataset.id,
            name=dataset.name,
            rowCount=dataset.row_count,
            labelDistribution={
                str(k): int(v)
                for k, v in (dataset.label_distribution or {}).items()
            },
            uploadedBy=dataset.uploaded_by,
            sourceCsvSha256=dataset.source_csv_sha256,
            createdAt=dataset.created_at,
            updatedAt=dataset.updated_at,
        )


class BenchmarkDatasetDetail(BenchmarkDatasetSummary):
    """Detail view for `GET /api/v1/benchmarks/datasets/{id}` — adds the
    free-form description the summary omits."""

    description: str | None = Field(default=None, alias="description")

    @staticmethod
    def from_dataset(dataset: BenchmarkDataset) -> "BenchmarkDatasetDetail":  # type: ignore[override]
        base = BenchmarkDatasetSummary.from_dataset(dataset)
        return BenchmarkDatasetDetail(
            **base.model_dump(by_alias=True),
            description=dataset.description,
        )


class BenchmarkVersionResult(BaseModel):
    """Projection of a `model_benchmark_version_results` row (§6.11). Used
    both inside `BenchmarkDetailResponse` (side-by-side view) and on the SSE
    `version_done` payload so the two shapes align."""

    version: str = Field(alias="version")
    accuracy: float = Field(alias="accuracy")
    precision: float = Field(alias="precision")
    recall: float = Field(alias="recall")
    f1: float = Field(alias="f1")
    roc_auc: float = Field(alias="rocAuc")
    ece: float = Field(alias="ece")
    confusion_matrix: dict[str, int] = Field(
        default_factory=dict, alias="confusionMatrix"
    )
    per_zone_counts: dict[str, int] = Field(
        default_factory=dict, alias="perZoneCounts"
    )
    prediction_ms_p50: float = Field(alias="predictionMsP50")
    prediction_ms_p95: float = Field(alias="predictionMsP95")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_row(
        row: ModelBenchmarkVersionResult,
    ) -> "BenchmarkVersionResult":
        return BenchmarkVersionResult(
            version=row.version,
            accuracy=float(row.accuracy),
            precision=float(row.precision),
            recall=float(row.recall),
            f1=float(row.f1),
            rocAuc=float(row.roc_auc),
            ece=float(row.ece),
            confusionMatrix={
                str(k): int(v)
                for k, v in (row.confusion_matrix or {}).items()
            },
            perZoneCounts={
                str(k): int(v)
                for k, v in (row.per_zone_counts or {}).items()
            },
            predictionMsP50=float(row.prediction_ms_p50),
            predictionMsP95=float(row.prediction_ms_p95),
        )


class BenchmarkSummary(BaseModel):
    """List-view row for `GET /api/v1/benchmarks` (§7.7). The `versions`
    array carries the requested versions — the per-version metrics only
    show up on the detail endpoint."""

    id: UUID = Field(alias="id")
    benchmark_dataset_id: UUID = Field(alias="benchmarkDatasetId")
    triggered_by: UUID | None = Field(default=None, alias="triggeredBy")
    versions: list[str] = Field(default_factory=list, alias="versions")
    status: str = Field(alias="status")
    started_at: datetime | None = Field(default=None, alias="startedAt")
    finished_at: datetime | None = Field(default=None, alias="finishedAt")
    error_message: str | None = Field(default=None, alias="errorMessage")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    class Config:
        populate_by_name = True
        from_attributes = True

    @staticmethod
    def from_benchmark(benchmark: ModelBenchmark) -> "BenchmarkSummary":
        return BenchmarkSummary(
            id=benchmark.id,
            benchmarkDatasetId=benchmark.benchmark_dataset_id,
            triggeredBy=benchmark.triggered_by,
            versions=list(benchmark.versions or []),
            status=benchmark.status.value,
            startedAt=benchmark.started_at,
            finishedAt=benchmark.finished_at,
            errorMessage=benchmark.error_message,
            createdAt=benchmark.created_at,
            updatedAt=benchmark.updated_at,
        )


class BenchmarkDetailResponse(BenchmarkSummary):
    """Detail view for `GET /api/v1/benchmarks/{id}` (§7.7).

    Carries every per-version row plus a `winners` map — one entry per
    metric pointing at the version that scored best. The winner is computed
    at response-build time from the persisted results (never stored on the
    row), so late-arriving versions do not invalidate prior snapshots."""

    results: list[BenchmarkVersionResult] = Field(
        default_factory=list, alias="results"
    )
    # Per-metric winner: {"accuracy": "v1_2", "f1": "v1_1", ...}. Empty when
    # the benchmark has not yet produced any per-version results.
    winners: dict[str, str] = Field(
        default_factory=dict, alias="winners"
    )

    @staticmethod
    def from_detail(
        benchmark: ModelBenchmark,
        results: list[ModelBenchmarkVersionResult],
    ) -> "BenchmarkDetailResponse":
        base = BenchmarkSummary.from_benchmark(benchmark)
        projected = [BenchmarkVersionResult.from_row(r) for r in results]
        return BenchmarkDetailResponse(
            **base.model_dump(by_alias=True),
            results=projected,
            winners=_compute_winners(projected),
        )


# Higher-is-better for accuracy/precision/recall/f1/rocAuc. ECE is a gap
# metric — smaller is better. Latencies are reported but not ranked here
# (they're visible on each per-version row; a "lowest latency" winner would
# muddle the model-quality story the side-by-side table is there to tell).
_BENCHMARK_WINNER_METRICS_MAX: tuple[str, ...] = (
    "accuracy", "precision", "recall", "f1", "rocAuc",
)
_BENCHMARK_WINNER_METRICS_MIN: tuple[str, ...] = ("ece",)


def _compute_winners(
    results: list[BenchmarkVersionResult],
) -> dict[str, str]:
    if not results:
        return {}
    winners: dict[str, str] = {}
    for metric in _BENCHMARK_WINNER_METRICS_MAX:
        best = max(results, key=lambda r: getattr(r, _snake(metric)))
        winners[metric] = best.version
    for metric in _BENCHMARK_WINNER_METRICS_MIN:
        best = min(results, key=lambda r: getattr(r, _snake(metric)))
        winners[metric] = best.version
    return winners


def _snake(metric: str) -> str:
    # Pydantic field names are snake_case; the winner map is keyed by the
    # camelCase alias the rest of the payload uses. `rocAuc` → `roc_auc`.
    return "".join("_" + c.lower() if c.isupper() else c for c in metric)


class BenchmarkEvent(BaseModel):
    """Payload of an SSE event emitted from `GET /benchmarks/{id}/events`
    (§7.7). `kind` discriminates the stream: `status` (lifecycle),
    `version_done` (per-version completion with metrics), `error`."""

    benchmark_id: UUID = Field(alias="benchmarkId")
    kind: str = Field(alias="kind")
    status: str | None = Field(default=None, alias="status")
    version: str | None = Field(default=None, alias="version")
    result: BenchmarkVersionResult | None = Field(default=None, alias="result")
    message: str | None = Field(default=None, alias="message")
    emitted_at: datetime = Field(alias="emittedAt")

    class Config:
        populate_by_name = True


# ── Phase 11 — Dashboards (§9.1 / §9.2) ──


class _UserRef(BaseModel):
    """Compact `{id, username}` reference embedded throughout the dashboard
    payloads. Falls back to `None` on the controller when the underlying
    user id has been SET NULL'd."""

    id: UUID = Field(alias="id")
    username: str = Field(alias="username")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_user(user: User) -> "_UserRef":
        return _UserRef(id=user.id, username=user.username)


class RecentPredictionEntry(BaseModel):
    """One row inside `AdminDashboardResponse.recentPredictions` — the
    last-N feed across all users."""

    prediction_id: UUID = Field(alias="predictionId")
    predicted_at: datetime = Field(alias="predictedAt")
    predicted_label: int = Field(alias="predictedLabel")
    phishing_probability: float = Field(alias="phishingProbability")
    confidence_zone: str | None = Field(default=None, alias="confidenceZone")
    model_version: str = Field(alias="modelVersion")
    requester: _UserRef | None = Field(default=None, alias="requester")

    class Config:
        populate_by_name = True

    @staticmethod
    def from_event(event: PredictionEvent) -> "RecentPredictionEntry":
        return RecentPredictionEntry(
            predictionId=event.id,
            predictedAt=event.predicted_at,
            predictedLabel=int(event.predicted_label),
            phishingProbability=float(event.phishing_probability),
            confidenceZone=(
                event.confidence_zone.value
                if event.confidence_zone is not None
                else None
            ),
            modelVersion=event.model_version,
            requester=(
                _UserRef.from_user(event.requester)
                if event.requester is not None
                else None
            ),
        )


class ReviewQueueHealth(BaseModel):
    pending_count: int = Field(alias="pendingCount")
    unassigned_count: int = Field(alias="unassignedCount")
    in_progress_count: int = Field(alias="inProgressCount")
    average_age_seconds: float = Field(alias="averageAgeSeconds")
    oldest_age_seconds: float = Field(alias="oldestAgeSeconds")
    sla_breach_count: int = Field(alias="slaBreachCount")
    sla_threshold_seconds: int = Field(alias="slaThresholdSeconds")

    class Config:
        populate_by_name = True


class AnalystWorkloadEntry(BaseModel):
    analyst: _UserRef = Field(alias="analyst")
    assigned_count: int = Field(alias="assignedCount")
    completed_today: int = Field(alias="completedToday")
    completed_this_week: int = Field(alias="completedThisWeek")
    completion_rate: float = Field(alias="completionRate")
    average_decision_seconds: float = Field(alias="averageDecisionSeconds")

    class Config:
        populate_by_name = True


class ClassificationBreakdownPoint(BaseModel):
    date: datetime = Field(alias="date")
    phishing: int = Field(alias="phishing")
    legitimate: int = Field(alias="legitimate")
    review: int = Field(alias="review")

    class Config:
        populate_by_name = True


class ClassificationBreakdown(BaseModel):
    bucket: str = Field(alias="bucket")
    date_from: datetime = Field(alias="from")
    date_to: datetime = Field(alias="to")
    series: list[ClassificationBreakdownPoint] = Field(alias="series")

    class Config:
        populate_by_name = True


class ActiveModelSummary(BaseModel):
    version: str | None = Field(default=None, alias="version")
    promoted: bool = Field(alias="promoted")
    activated_at: datetime | None = Field(default=None, alias="activatedAt")
    decision_threshold: float | None = Field(
        default=None, alias="decisionThreshold"
    )
    review_low_threshold: float | None = Field(
        default=None, alias="reviewLowThreshold"
    )
    review_high_threshold: float | None = Field(
        default=None, alias="reviewHighThreshold"
    )
    drift_signal: DriftSignalSummary = Field(alias="driftSignal")

    class Config:
        populate_by_name = True


class TrainingBufferSummary(BaseModel):
    total_count: int = Field(alias="totalCount")
    class_counts: dict[str, int] = Field(alias="classCounts")
    min_batch_size: int = Field(alias="minBatchSize")
    min_per_class: int = Field(alias="minPerClass")
    unlocked: bool = Field(alias="unlocked")
    blockers: list[str] = Field(alias="blockers")
    recent_additions_24h: int = Field(alias="recentAdditions24h")

    class Config:
        populate_by_name = True


class RecentActivityRef(BaseModel):
    type: str = Field(alias="type")
    id: UUID = Field(alias="id")

    class Config:
        populate_by_name = True


class RecentActivityEntry(BaseModel):
    occurred_at: datetime = Field(alias="occurredAt")
    kind: str = Field(alias="kind")
    actor: _UserRef | None = Field(default=None, alias="actor")
    summary: str = Field(alias="summary")
    ref: RecentActivityRef | None = Field(default=None, alias="ref")

    class Config:
        populate_by_name = True


class PredictionVolumePoint(BaseModel):
    t: datetime = Field(alias="t")
    count: int = Field(alias="count")

    class Config:
        populate_by_name = True


class PredictionVolumeSeries(BaseModel):
    bucket: str = Field(alias="bucket")
    date_from: datetime = Field(alias="from")
    date_to: datetime = Field(alias="to")
    series: list[PredictionVolumePoint] = Field(alias="series")

    class Config:
        populate_by_name = True


class AdminDashboardResponse(BaseModel):
    generated_at: datetime = Field(alias="generatedAt")
    timezone: str = Field(alias="timezone")
    recent_predictions: list[RecentPredictionEntry] = Field(
        alias="recentPredictions"
    )
    review_queue_health: ReviewQueueHealth = Field(alias="reviewQueueHealth")
    analyst_workload: list[AnalystWorkloadEntry] = Field(alias="analystWorkload")
    classification_breakdown: ClassificationBreakdown = Field(
        alias="classificationBreakdown"
    )
    active_model: ActiveModelSummary = Field(alias="activeModel")
    training_buffer: TrainingBufferSummary = Field(alias="trainingBuffer")
    recent_activity: list[RecentActivityEntry] = Field(alias="recentActivity")
    prediction_volume: PredictionVolumeSeries = Field(alias="predictionVolume")

    class Config:
        populate_by_name = True


class MyQueueSummary(BaseModel):
    assigned_count: int = Field(alias="assignedCount")
    oldest_age_seconds: float = Field(alias="oldestAgeSeconds")
    average_age_seconds: float = Field(alias="averageAgeSeconds")
    sla_breach_count: int = Field(alias="slaBreachCount")
    sla_threshold_seconds: int = Field(alias="slaThresholdSeconds")

    class Config:
        populate_by_name = True


class PersonalStatsWindow(BaseModel):
    completed: int = Field(alias="completed")
    deferred: int = Field(alias="deferred")
    escalated: int = Field(alias="escalated")
    average_decision_seconds: float = Field(alias="averageDecisionSeconds")

    class Config:
        populate_by_name = True


class PersonalStats(BaseModel):
    today: PersonalStatsWindow = Field(alias="today")
    this_week: PersonalStatsWindow = Field(alias="thisWeek")

    class Config:
        populate_by_name = True


class RecentSubmissionEntry(BaseModel):
    review_item_id: UUID = Field(alias="reviewItemId")
    decided_at: datetime | None = Field(default=None, alias="decidedAt")
    status: str = Field(alias="status")
    verdict: str | None = Field(default=None, alias="verdict")
    auto_review_used: bool = Field(alias="autoReviewUsed")
    agreed_with_auto_review: str = Field(alias="agreedWithAutoReview")
    subject_preview: str = Field(alias="subjectPreview")

    class Config:
        populate_by_name = True


class QueueLabelMix(BaseModel):
    range_days: int = Field(alias="rangeDays")
    phishing: int = Field(alias="phishing")
    legitimate: int = Field(alias="legitimate")
    ratio_phishing: float = Field(alias="ratioPhishing")

    class Config:
        populate_by_name = True


class AutoReviewUsageInvocation(BaseModel):
    invocation_id: UUID = Field(alias="invocationId")
    review_item_id: UUID = Field(alias="reviewItemId")
    created_at: datetime = Field(alias="createdAt")
    trigger_kind: str = Field(alias="triggerKind")
    provider: str = Field(alias="provider")
    model_name: str = Field(alias="modelName")
    outcome: str = Field(alias="outcome")
    label: str | None = Field(default=None, alias="label")
    final_verdict: str | None = Field(default=None, alias="finalVerdict")
    agreed: bool | None = Field(default=None, alias="agreed")

    class Config:
        populate_by_name = True


class AutoReviewUsageSummary(BaseModel):
    range_days: int = Field(alias="rangeDays")
    with_auto_review: int = Field(alias="withAutoReview")
    without_auto_review: int = Field(alias="withoutAutoReview")
    agreed_count: int = Field(alias="agreedCount")
    overridden_count: int = Field(alias="overriddenCount")
    agreement_rate: float = Field(alias="agreementRate")
    invocation_count: int = Field(alias="invocationCount")
    invocation_failure_count: int = Field(alias="invocationFailureCount")
    recent_invocations: list[AutoReviewUsageInvocation] = Field(
        alias="recentInvocations"
    )

    class Config:
        populate_by_name = True


class AnalystDashboardResponse(BaseModel):
    generated_at: datetime = Field(alias="generatedAt")
    timezone: str = Field(alias="timezone")
    analyst: dict[str, Any] = Field(alias="analyst")
    my_queue: MyQueueSummary = Field(alias="myQueue")
    personal_stats: PersonalStats = Field(alias="personalStats")
    recent_submissions: list[RecentSubmissionEntry] = Field(
        alias="recentSubmissions"
    )
    queue_label_mix: QueueLabelMix = Field(alias="queueLabelMix")
    auto_review_usage: AutoReviewUsageSummary = Field(alias="autoReviewUsage")

    class Config:
        populate_by_name = True
