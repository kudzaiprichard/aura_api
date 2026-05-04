import logging
from datetime import datetime, timezone
from typing import Any, Sequence, Tuple
from uuid import UUID

from src.app.models.enums import ConfidenceZone, PredictionSource, Role
from src.app.models.prediction_event import PredictionEvent
from src.app.models.user import User
from src.app.repositories.drift_event_repository import DriftEventRepository
from src.app.repositories.prediction_event_repository import (
    PredictionEventRepository,
)
from src.app.services.review_service import ReviewService
from src.shared.exceptions import (
    BadRequestException,
    NotFoundException,
    ServiceUnavailableException,
)
from src.shared.inference import PhishingDetector, PredictionResult, ValidationError
from src.shared.responses import ErrorDetail


log = logging.getLogger("aura.prediction")


def _coerce_zone(result: PredictionResult) -> ConfidenceZone | None:
    """Map the inference-layer enum to the local SQLAlchemy enum.

    The two enums mirror each other intentionally (see
    src/app/models/enums.py) so that migrations do not reach into
    src.shared.inference. Values are compared by string to stay
    independent of identity.
    """
    if result.confidence_zone is None:
        return None
    return ConfidenceZone(result.confidence_zone.value)


class PredictionService:
    def __init__(
        self,
        prediction_repository: PredictionEventRepository,
        detector: PhishingDetector | None,
        review_service: ReviewService,
        drift_event_repository: DriftEventRepository,
        app_state: Any | None = None,
    ):
        self.prediction_repo = prediction_repository
        self.detector = detector
        self.review_service = review_service
        self.drift_event_repo = drift_event_repository
        # Phase 12 — shadow predictions. The service reads the current
        # shadow detector off the app state at predict time so the swap in
        # ModelAdminService picks up naturally without per-request wiring.
        self.app_state = app_state

    # ── commands ──

    async def predict(
        self,
        *,
        sender: str,
        subject: str,
        body: str,
        threshold: float | None,
        requester: User | None,
        request_id: str,
        source: PredictionSource = PredictionSource.API,
    ) -> Tuple[PredictionEvent, PredictionResult]:
        detector = self._require_detector()
        kwargs: dict = {}
        if threshold is not None:
            kwargs["threshold"] = float(threshold)
        try:
            result = detector.predict(sender, subject, body, **kwargs)
        except ValidationError as exc:
            raise self._validation_error(str(exc)) from exc

        shadow_result = self._shadow_predict(
            sender=sender, subject=subject, body=body, kwargs=kwargs,
        )

        event = await self._persist(
            sender=sender,
            subject=subject,
            body=body,
            result=result,
            requester_id=requester.id if requester is not None else None,
            request_id=request_id,
            source=source,
            shadow_result=shadow_result,
        )
        await self._mirror_drift_prediction(event)
        await self._enqueue_for_review(event, result)
        return event, result

    async def predict_batch(
        self,
        *,
        emails: Sequence[dict],
        threshold: float | None,
        requester: User | None,
        request_id: str,
        source: PredictionSource = PredictionSource.BATCH,
    ) -> list[Tuple[PredictionEvent, PredictionResult]]:
        detector = self._require_detector()
        kwargs: dict = {}
        if threshold is not None:
            kwargs["threshold"] = float(threshold)
        try:
            results = detector.predict_batch(list(emails), **kwargs)
        except ValidationError as exc:
            raise self._validation_error(str(exc)) from exc

        persisted: list[Tuple[PredictionEvent, PredictionResult]] = []
        for email, result in zip(emails, results):
            shadow_result = self._shadow_predict(
                sender=email["sender"],
                subject=email["subject"],
                body=email["body"],
                kwargs=kwargs,
            )
            event = await self._persist(
                sender=email["sender"],
                subject=email["subject"],
                body=email["body"],
                result=result,
                requester_id=requester.id if requester is not None else None,
                request_id=request_id,
                source=source,
                shadow_result=shadow_result,
            )
            await self._mirror_drift_prediction(event)
            await self._enqueue_for_review(event, result)
            persisted.append((event, result))
        return persisted

    # ── queries ──

    async def list_predictions(
        self,
        *,
        current_user: User,
        page: int,
        page_size: int,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        model_version: str | None = None,
        predicted_label: int | None = None,
        confidence_zone: ConfidenceZone | None = None,
        user_id: UUID | None = None,
    ) -> Tuple[Sequence[PredictionEvent], int]:
        scoped_requester = self._scope_requester(current_user, user_id)
        return await self.prediction_repo.paginate_filtered(
            page=page,
            page_size=page_size,
            requester_id=scoped_requester,
            date_from=date_from,
            date_to=date_to,
            model_version=model_version,
            predicted_label=predicted_label,
            confidence_zone=confidence_zone,
        )

    async def get_prediction(
        self, prediction_event_id: UUID
    ) -> PredictionEvent:
        event = await self.prediction_repo.get_by_id(prediction_event_id)
        if event is None:
            raise NotFoundException(
                message="Prediction not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="PREDICTION_NOT_FOUND",
                    status=404,
                    details=[f"No prediction found with id {prediction_event_id}"],
                ),
            )
        return event

    @staticmethod
    def should_redact_body(event: PredictionEvent, current_user: User) -> bool:
        """Non-admins outside their own scope see the body masked (§7.1)."""
        if current_user.role == Role.ADMIN:
            return False
        return event.requester_id != current_user.id

    # ── internals ──

    def _require_detector(self) -> PhishingDetector:
        if self.detector is None:
            raise ServiceUnavailableException(
                message="The prediction service is not available",
                error_detail=ErrorDetail(
                    title="Detector Unavailable",
                    code="DETECTOR_UNAVAILABLE",
                    status=503,
                    details=[
                        "No phishing-detection model is currently loaded. "
                        "Register a model version and reload the service."
                    ],
                ),
            )
        return self.detector

    @staticmethod
    def _validation_error(message: str) -> BadRequestException:
        detail = ErrorDetail(
            title="Validation Failed",
            code="VALIDATION_ERROR",
            status=400,
            details=[message] if message else ["Invalid prediction input"],
        )
        return BadRequestException(message=message or "Invalid prediction input", error_detail=detail)

    @staticmethod
    def _scope_requester(current_user: User, user_id: UUID | None) -> UUID | None:
        # Analysts are pinned to their own rows regardless of the ?userId filter.
        if current_user.role != Role.ADMIN:
            return current_user.id
        return user_id

    async def _persist(
        self,
        *,
        sender: str,
        subject: str,
        body: str,
        result: PredictionResult,
        requester_id: UUID | None,
        request_id: str,
        source: PredictionSource,
        shadow_result: PredictionResult | None = None,
    ) -> PredictionEvent:
        prediction_uuid: UUID | None = None
        if result.prediction_id is not None:
            try:
                prediction_uuid = UUID(result.prediction_id)
            except (TypeError, ValueError):
                prediction_uuid = None

        event = PredictionEvent(
            prediction_id=prediction_uuid,
            request_id=request_id,
            requester_id=requester_id,
            source=source,
            model_version=result.model_version or "",
            sender=sender,
            subject=subject,
            body=body,
            predicted_label=int(result.predicted_label),
            phishing_probability=float(result.phishing_probability),
            legitimate_probability=float(result.legitimate_probability),
            raw_phishing_probability=result.raw_phishing_probability,
            raw_legitimate_probability=result.raw_legitimate_probability,
            calibrated=bool(result.calibrated),
            threshold=float(result.threshold),
            confidence_zone=_coerce_zone(result),
            review_low_threshold=result.review_low_threshold,
            review_high_threshold=result.review_high_threshold,
            engineered_features=dict(result.engineered_features or {}),
            predicted_at=datetime.now(timezone.utc),
            shadow_model_version=(
                shadow_result.model_version if shadow_result is not None else None
            ),
            shadow_predicted_label=(
                int(shadow_result.predicted_label)
                if shadow_result is not None
                else None
            ),
            shadow_phishing_probability=(
                float(shadow_result.phishing_probability)
                if shadow_result is not None
                else None
            ),
            shadow_confidence_zone=(
                _coerce_zone(shadow_result)
                if shadow_result is not None
                else None
            ),
        )
        return await self.prediction_repo.create(event)

    def _shadow_predict(
        self,
        *,
        sender: str,
        subject: str,
        body: str,
        kwargs: dict,
    ) -> PredictionResult | None:
        """Run the captured prior detector against the same inputs.

        Returns None when the shadow slot is empty, expired, or prediction
        fails — shadow verdicts are advisory and must never surface an
        error on the /predict hot path.
        """
        if self.app_state is None:
            return None
        shadow_detector = getattr(self.app_state, "shadow_detector", None)
        if shadow_detector is None:
            return None
        expires_at = getattr(self.app_state, "shadow_expires_at", None)
        if expires_at is not None and datetime.now(timezone.utc) >= expires_at:
            # Lapsed — clear the slot so subsequent predicts short-circuit.
            self.app_state.shadow_detector = None
            self.app_state.shadow_expires_at = None
            return None
        try:
            return shadow_detector.predict(sender, subject, body, **kwargs)
        except Exception:
            log.exception(
                "shadow_detector.predict failed — dropping shadow row"
            )
            return None

    async def _mirror_drift_prediction(self, event: PredictionEvent) -> None:
        """Persist a `drift_events` row alongside the prediction_events row in
        the same transaction so the SQL mirror stays in sync with §6.12.

        Skips silently when the detector did not surface a `prediction_id` —
        the JSONL log keys on the same field, so an event without one cannot
        ever be confirmed and so does not contribute to the FPR signal either.
        """
        if event.prediction_id is None:
            return
        await self.drift_event_repo.record_prediction(
            prediction_id=event.prediction_id,
            predicted_label=event.predicted_label,
            predicted_probability=event.phishing_probability,
            model_version=event.model_version or None,
            occurred_at=event.predicted_at,
        )

    async def _enqueue_for_review(
        self, event: PredictionEvent, result: PredictionResult
    ) -> None:
        """REVIEW-zone predictions land in the analyst queue (§8.8).

        Runs inside the same transaction as `_persist`, so the queue row
        and the prediction row commit together.
        """
        if event.confidence_zone != ConfidenceZone.REVIEW:
            return
        await self.review_service.enqueue(event.id)
        log.debug(
            "review_items enqueued prediction_event_id=%s prediction_id=%s",
            event.id,
            result.prediction_id,
        )
