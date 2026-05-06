import asyncio
import hashlib
import logging
import random
import time
from datetime import datetime, timezone
from typing import Sequence, Tuple
from uuid import UUID, uuid4

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs import review as review_config
from src.app.helpers.auto_review_cache import AutoReviewCache, compute_cache_key
from src.app.helpers.quality_gate import (
    quality_gate_max_oov,
    violates_quality_gate,
)
from src.app.models.auto_review_invocation import AutoReviewInvocation
from src.app.models.enums import (
    AutoReviewAgreement,
    AutoReviewOutcome,
    LLMProviderEnum,
    ReviewItemStatus,
    ReviewVerdict,
    Role,
    TrainingBufferSource,
)
from src.app.models.review_escalation import ReviewEscalation
from src.app.models.review_item import ReviewItem
from src.app.models.training_buffer_item import TrainingBufferItem
from src.app.models.user import User
from src.app.repositories.auto_review_invocation_repository import (
    AutoReviewInvocationRepository,
)
from src.app.repositories.drift_event_repository import DriftEventRepository
from src.app.repositories.prediction_event_repository import (
    PredictionEventRepository,
)
from src.app.repositories.review_escalation_repository import (
    ReviewEscalationRepository,
)
from src.app.repositories.review_item_repository import ReviewItemRepository
from src.app.repositories.training_buffer_repository import (
    TrainingBufferRepository,
)
from src.app.repositories.user_repository import UserRepository
from src.shared.exceptions import (
    AuthorizationException,
    BadRequestException,
    ConflictException,
    NotFoundException,
    ServiceUnavailableException,
)
from src.shared.inference import (
    AutoReviewer,
    AutoReviewFailure,
    AutoReviewSuccess,
    DriftMonitor,
    PhishingDetector,
)
from src.shared.responses import ErrorDetail


log = logging.getLogger("aura.review")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _content_sha256(sender: str, subject: str, body: str) -> str:
    payload = f"{sender}\n{subject}\n{body}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verdict_to_label(verdict: ReviewVerdict) -> int:
    return 1 if verdict == ReviewVerdict.PHISHING else 0


class ReviewService:
    """Owns the review-item lifecycle (§3.3) and the §3.12 feedback loop.

    The transaction is owned by the FastAPI dependency (`get_db`) — repository
    writes only flush; `confirm` registers a post-commit `after_commit` hook
    on the session so the drift-monitor write happens once the row state has
    actually been persisted.
    """

    def __init__(
        self,
        session: AsyncSession,
        review_item_repository: ReviewItemRepository,
        review_escalation_repository: ReviewEscalationRepository,
        training_buffer_repository: TrainingBufferRepository,
        prediction_event_repository: PredictionEventRepository,
        user_repository: UserRepository,
        drift_monitor: DriftMonitor,
        drift_event_repository: DriftEventRepository,
        auto_review_invocation_repository: AutoReviewInvocationRepository | None = None,
        auto_reviewer: AutoReviewer | None = None,
        auto_review_cache: AutoReviewCache | None = None,
        detector: PhishingDetector | None = None,
    ):
        self.session = session
        self.review_item_repo = review_item_repository
        self.escalation_repo = review_escalation_repository
        self.buffer_repo = training_buffer_repository
        self.prediction_repo = prediction_event_repository
        self.user_repo = user_repository
        self.drift_monitor = drift_monitor
        self.drift_event_repo = drift_event_repository
        self.auto_review_repo = auto_review_invocation_repository
        self.auto_reviewer = auto_reviewer
        self.auto_review_cache = auto_review_cache
        # Phase 12 — the training-buffer quality gate uses the active
        # detector's vectorisers to compute the OOV rate. When no detector
        # is loaded, the gate is skipped.
        self.detector = detector

    # ── enqueue (called by PredictionService when zone == REVIEW) ──

    async def enqueue(self, prediction_event_id: UUID) -> ReviewItem:
        """Create a review item for a REVIEW-zone prediction.

        Idempotent: if a review item already exists for this prediction event
        (uniqueness enforced at the DB level too), the existing row is
        returned so re-enqueue is a no-op.
        """
        existing = await self.review_item_repo.get_by_prediction_event_id(
            prediction_event_id
        )
        if existing is not None:
            return existing

        prediction = await self.prediction_repo.get_by_id(prediction_event_id)
        if prediction is None:
            raise NotFoundException(
                message="Prediction not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="PREDICTION_NOT_FOUND",
                    status=404,
                    details=[
                        f"No prediction found with id {prediction_event_id}"
                    ],
                ),
            )

        # Phase 12 — rejection sampler. When the feature is on and the
        # random draw falls outside sample_rate, the item never reaches the
        # analyst queue — it is either auto-confirmed by probability or
        # parked as DEFERRED depending on the policy. When the feature is
        # off (default) or sample_rate >= 1.0, behaviour is unchanged.
        sampled = self._rejection_sampler_includes()
        if not sampled:
            sampler_item = await self._enqueue_sampled_out(prediction)
            if sampler_item is not None:
                return sampler_item

        assignee_id = await self._pick_assignee()
        item = ReviewItem(
            prediction_event_id=prediction.id,
            status=(
                ReviewItemStatus.ASSIGNED
                if assignee_id is not None
                else ReviewItemStatus.UNASSIGNED
            ),
            assigned_to=assignee_id,
            assigned_at=_now() if assignee_id is not None else None,
            auto_review_used=False,
            auto_review_agreement=AutoReviewAgreement.NOT_USED,
        )
        return await self.review_item_repo.create(item)

    @staticmethod
    def _rejection_sampler_includes() -> bool:
        """Return True if this enqueue call survives the rejection sampler.

        Always True when the feature is disabled or sample_rate >= 1.0,
        so the normal enqueue path is unchanged by default.
        """
        cfg = getattr(review_config, "rejection_sampler", None)
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return True
        try:
            rate = float(getattr(cfg, "sample_rate", 1.0))
        except (TypeError, ValueError):
            rate = 1.0
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        return random.random() < rate

    async def _enqueue_sampled_out(
        self, prediction
    ) -> ReviewItem | None:
        """Create the review item for an enqueue that fell outside the sampler.

        Returns None if the policy is unknown — the caller then falls back
        to the normal enqueue path so a mis-configured flag never drops
        work on the floor.
        """
        cfg = getattr(review_config, "rejection_sampler", None)
        if cfg is None:
            return None
        policy = str(getattr(cfg, "policy", "auto_confirm")).strip().lower()
        now = _now()
        if policy == "auto_confirm":
            verdict = (
                ReviewVerdict.PHISHING
                if float(prediction.phishing_probability) >= 0.5
                else ReviewVerdict.LEGITIMATE
            )
            item = ReviewItem(
                prediction_event_id=prediction.id,
                status=ReviewItemStatus.CONFIRMED,
                verdict=verdict,
                decided_at=now,
                decided_by=None,
                reviewer_note="rejection_sampler: auto_confirm (not sampled)",
                auto_review_used=False,
                auto_review_agreement=AutoReviewAgreement.NOT_USED,
            )
            created = await self.review_item_repo.create(item)
            log.info(
                "rejection_sampler auto_confirm prediction_event_id=%s "
                "verdict=%s probability=%.4f",
                prediction.id, verdict.value,
                float(prediction.phishing_probability),
            )
            return created
        if policy == "defer":
            item = ReviewItem(
                prediction_event_id=prediction.id,
                status=ReviewItemStatus.DEFERRED,
                reviewer_note="rejection_sampler: deferred (not sampled)",
                auto_review_used=False,
                auto_review_agreement=AutoReviewAgreement.NOT_USED,
            )
            created = await self.review_item_repo.create(item)
            log.info(
                "rejection_sampler defer prediction_event_id=%s "
                "probability=%.4f",
                prediction.id, float(prediction.phishing_probability),
            )
            return created
        # Unknown policy — fall back to the normal queue path so a typo
        # in config never silently buries work.
        log.warning(
            "rejection_sampler unknown policy=%r — falling back to queue",
            policy,
        )
        return None

    # ── queries ──

    async def list_queue(
        self,
        *,
        current_user: User,
        page: int,
        page_size: int,
    ) -> Tuple[Sequence[ReviewItem], int]:
        """`/queue`: items in flight for the caller (admin sees everything)."""
        assigned_to = (
            None if current_user.role == Role.ADMIN else current_user.id
        )
        return await self.review_item_repo.paginate_filtered(
            page=page,
            page_size=page_size,
            assigned_to=assigned_to,
            statuses=[
                ReviewItemStatus.ASSIGNED,
                ReviewItemStatus.IN_PROGRESS,
                ReviewItemStatus.DEFERRED,
            ],
        )

    async def list_unassigned(
        self,
        *,
        page: int,
        page_size: int,
    ) -> Tuple[Sequence[ReviewItem], int]:
        return await self.review_item_repo.paginate_filtered(
            page=page,
            page_size=page_size,
            status=ReviewItemStatus.UNASSIGNED,
        )

    # ── lifecycle commands ──

    async def claim(self, item_id: UUID, current_user: User) -> ReviewItem:
        item = await self._lock_or_conflict(item_id)
        if item.status not in (
            ReviewItemStatus.UNASSIGNED,
            ReviewItemStatus.ASSIGNED,
        ):
            raise self._terminal_conflict(item.status)
        if (
            item.status == ReviewItemStatus.ASSIGNED
            and item.assigned_to is not None
            and item.assigned_to != current_user.id
            and current_user.role != Role.ADMIN
        ):
            raise ConflictException(
                message="This item is already assigned to another analyst",
                error_detail=ErrorDetail(
                    title="Already Claimed",
                    code="REVIEW_ITEM_ALREADY_CLAIMED",
                    status=409,
                    details=["Item is assigned to another user"],
                ),
            )
        now = _now()
        return await self.review_item_repo.update(
            item,
            {
                "status": ReviewItemStatus.IN_PROGRESS,
                "assigned_to": current_user.id,
                "assigned_at": item.assigned_at or now,
                "claimed_at": now,
            },
        )

    async def release(self, item_id: UUID, current_user: User) -> ReviewItem:
        item = await self._lock_or_conflict(item_id)
        self._require_owner_or_admin(item, current_user)
        if item.status not in (
            ReviewItemStatus.ASSIGNED,
            ReviewItemStatus.IN_PROGRESS,
        ):
            raise BadRequestException(
                message="Only assigned or in-progress items can be released",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="REVIEW_ITEM_NOT_RELEASABLE",
                    status=400,
                    details=[f"Current status: {item.status.value}"],
                ),
            )
        return await self.review_item_repo.update(
            item,
            {
                "status": ReviewItemStatus.UNASSIGNED,
                "assigned_to": None,
                "assigned_at": None,
                "claimed_at": None,
            },
        )

    async def confirm(
        self,
        item_id: UUID,
        *,
        verdict: ReviewVerdict,
        note: str | None,
        agreed_with_auto_review: bool | None,
        override_reason: str | None,
        current_user: User,
    ) -> ReviewItem:
        item = await self._lock_or_conflict(item_id)
        self._require_owner_or_admin(item, current_user)
        self._require_active(item)

        # §8.8 agreement-flag semantics. The DTO carries both fields in every
        # phase so clients share one shape — what changes here is which
        # combinations are accepted depending on whether the LLM was invoked.
        if item.auto_review_used:
            if agreed_with_auto_review is None:
                raise BadRequestException(
                    message=(
                        "agreedWithAutoReview is required when auto-review "
                        "was used"
                    ),
                    error_detail=ErrorDetail(
                        title="Bad Request",
                        code="AUTO_REVIEW_AGREEMENT_REQUIRED",
                        status=400,
                        details=[
                            "Provide agreedWithAutoReview=true|false"
                        ],
                    ),
                )
            if agreed_with_auto_review is False and not override_reason:
                raise BadRequestException(
                    message=(
                        "overrideReason is required when overriding the LLM"
                    ),
                    error_detail=ErrorDetail(
                        title="Bad Request",
                        code="AUTO_REVIEW_OVERRIDE_REASON_REQUIRED",
                        status=400,
                        details=[
                            "Provide overrideReason when "
                            "agreedWithAutoReview=false"
                        ],
                    ),
                )
            agreement = (
                AutoReviewAgreement.AGREED
                if agreed_with_auto_review
                else AutoReviewAgreement.OVERRIDDEN
            )
            persisted_override_reason = (
                override_reason if not agreed_with_auto_review else None
            )
        else:
            if agreed_with_auto_review is not None:
                raise BadRequestException(
                    message=(
                        "agreedWithAutoReview is only valid when auto-review "
                        "was used"
                    ),
                    error_detail=ErrorDetail(
                        title="Bad Request",
                        code="AUTO_REVIEW_AGREEMENT_NOT_APPLICABLE",
                        status=400,
                        details=[
                            "No auto-review invocation exists for this item"
                        ],
                    ),
                )
            if override_reason is not None:
                raise BadRequestException(
                    message=(
                        "overrideReason is only valid when overriding an "
                        "auto-review verdict"
                    ),
                    error_detail=ErrorDetail(
                        title="Bad Request",
                        code="AUTO_REVIEW_OVERRIDE_NOT_APPLICABLE",
                        status=400,
                        details=[
                            "No auto-review invocation exists for this item"
                        ],
                    ),
                )
            agreement = AutoReviewAgreement.NOT_USED
            persisted_override_reason = None

        now = _now()
        await self.review_item_repo.update(
            item,
            {
                "status": ReviewItemStatus.CONFIRMED,
                "verdict": verdict,
                "decided_by": current_user.id,
                "decided_at": now,
                "reviewer_note": note,
                "auto_review_agreement": agreement,
                "override_reason": persisted_override_reason,
            },
        )
        await self._record_buffer_entry(item, verdict, current_user)
        await self._mirror_drift_confirmation(item, verdict, now)
        self._schedule_drift_confirmation(item, verdict)
        return item

    # ── auto-review (LLM) triggers ──

    async def trigger_auto_review(
        self,
        item_id: UUID,
        actor: User,
    ) -> AutoReviewInvocation:
        """Single-item LLM invocation. Loads the owned item, calls the LLM
        through `asyncio.to_thread` (the AutoReviewer uses sync httpx, ~30s
        worst case), persists the invocation row, and stamps
        `auto_review_used=true` plus `latest_auto_review_invocation_id` on
        the item. Idempotent only in the sense that repeat calls add new
        invocation rows — the latest pointer always refers to the newest.
        """
        self._require_auto_reviewer()
        item = await self._lock_or_conflict(item_id)
        self._require_owner_or_admin(item, actor)
        self._require_active(item)
        return await self._invoke_and_persist(
            item=item,
            actor=actor,
            trigger_kind="single",
            batch_group_id=None,
        )

    async def trigger_auto_review_batch(
        self,
        item_ids: Sequence[UUID],
        actor: User,
    ) -> list[AutoReviewInvocation]:
        """Batch trigger. Cap is `review.auto_review_batch_max` (default 5)
        per §8.9 — the DTO already enforces 5 client-side; this re-checks
        server-side so config drift can tighten it without a DTO bump.

        All items in the batch share a generated `batch_group_id` so the
        UI can render the batch as a single unit.
        """
        self._require_auto_reviewer()
        cap = max(1, int(review_config.auto_review_batch_max))
        if len(item_ids) > cap:
            raise BadRequestException(
                message=(
                    f"Auto-review batch is capped at {cap} items per call"
                ),
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="AUTO_REVIEW_BATCH_TOO_LARGE",
                    status=400,
                    details=[
                        f"Received {len(item_ids)} items; max is {cap}"
                    ],
                ),
            )
        if not item_ids:
            raise BadRequestException(
                message="At least one review item id is required",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="AUTO_REVIEW_BATCH_EMPTY",
                    status=400,
                    details=["reviewItemIds must not be empty"],
                ),
            )

        batch_group_id = uuid4()
        invocations: list[AutoReviewInvocation] = []
        for item_id in item_ids:
            item = await self._lock_or_conflict(item_id)
            self._require_owner_or_admin(item, actor)
            self._require_active(item)
            invocation = await self._invoke_and_persist(
                item=item,
                actor=actor,
                trigger_kind="batch",
                batch_group_id=batch_group_id,
            )
            invocations.append(invocation)
        return invocations

    async def list_auto_reviews_for_item(
        self,
        item_id: UUID,
        *,
        actor: User,
        page: int,
        page_size: int,
    ) -> Tuple[Sequence[AutoReviewInvocation], int]:
        """History view. Same ownership rules as the rest of the queue:
        analysts see only their own items, admins see everything. Raises 404
        if the item doesn't exist.
        """
        if self.auto_review_repo is None:
            # Defensive: list-history should be readable even if the
            # provider is disabled (so the UI can still render past runs).
            return [], 0
        item = await self.review_item_repo.get_by_id(item_id)
        if item is None:
            raise NotFoundException(
                message="Review item not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="REVIEW_ITEM_NOT_FOUND",
                    status=404,
                    details=[f"No review item found with id {item_id}"],
                ),
            )
        self._require_owner_or_admin(item, actor)
        return await self.auto_review_repo.paginate_for_item(
            item_id, page=page, page_size=page_size,
        )

    async def _invoke_and_persist(
        self,
        *,
        item: ReviewItem,
        actor: User,
        trigger_kind: str,
        batch_group_id: UUID | None,
    ) -> AutoReviewInvocation:
        assert self.auto_reviewer is not None  # _require_auto_reviewer ran
        assert self.auto_review_repo is not None
        ev = item.prediction_event

        # Phase 12 — cache lookup on the normalised content + model_name.
        # A hit returns the prior AutoReviewSuccess verbatim and records a
        # duration_ms=0 invocation row with a `cached=true` marker in the
        # raw payload; the provider is not called. Only successful verdicts
        # were ever cached, so a hit is always a success.
        cache_key: str | None = None
        cached_response: AutoReviewSuccess | None = None
        if self.auto_review_cache is not None:
            cache_key = compute_cache_key(
                sender=ev.sender,
                subject=ev.subject,
                body=ev.body,
                model_name=self.auto_reviewer.model_name,
            )
            cached_response = self.auto_review_cache.get(cache_key)

        if cached_response is not None:
            response = cached_response
            duration_ms = 0
            cache_hit = True
        else:
            started = time.perf_counter()
            # The AutoReviewer uses sync httpx; offload to a worker thread so we
            # don't block the event loop for the timeout window (~30s default,
            # ~90s with retries on Google).
            response = await asyncio.to_thread(
                self.auto_reviewer.review,
                ev.sender,
                ev.subject,
                ev.body,
                ev.engineered_features or None,
            )
            duration_ms = int((time.perf_counter() - started) * 1000)
            cache_hit = False

        provider_enum = LLMProviderEnum(response.provider.value.upper())
        if isinstance(response, AutoReviewSuccess):
            outcome = AutoReviewOutcome.SUCCESS
            label = response.review_label.value
            confidence = response.confidence
            reasoning = response.reasoning
            raw_payload = dict(response.raw_response or {})
            if cache_hit:
                raw_payload["cached"] = True
            user_message = None
            technical_error = None
            if (
                not cache_hit
                and self.auto_review_cache is not None
                and cache_key is not None
            ):
                self.auto_review_cache.set(cache_key, response)
        else:
            failure: AutoReviewFailure = response
            outcome = AutoReviewOutcome.FAILURE
            label = None
            confidence = None
            reasoning = None
            raw_payload = failure.raw_response
            user_message = failure.user_message
            technical_error = failure.technical_error

        invocation = AutoReviewInvocation(
            review_item_id=item.id,
            triggered_by=actor.id,
            trigger_kind=trigger_kind,
            batch_group_id=batch_group_id,
            provider=provider_enum,
            model_name=response.model_name,
            outcome=outcome,
            label=label,
            confidence=confidence,
            reasoning=reasoning,
            raw_payload=raw_payload,
            user_message=user_message,
            technical_error=technical_error,
            duration_ms=duration_ms,
        )
        invocation = await self.auto_review_repo.create(invocation)

        await self.review_item_repo.update(
            item,
            {
                "auto_review_used": True,
                "latest_auto_review_invocation_id": invocation.id,
            },
        )

        # Per §8.11: log provider, model, duration, outcome — never the body
        # or the API key (the AutoReviewer never logs those itself either).
        log.info(
            "auto_reviewer call provider=%s model=%s duration_ms=%d "
            "outcome=%s cached=%s item_id=%s actor_id=%s trigger=%s",
            provider_enum.value,
            response.model_name,
            duration_ms,
            outcome.value,
            cache_hit,
            item.id,
            actor.id,
            trigger_kind,
        )
        return invocation

    def _require_auto_reviewer(self) -> None:
        if self.auto_reviewer is None or self.auto_review_repo is None:
            raise ServiceUnavailableException(
                message="The AI auto-reviewer is not configured",
                error_detail=ErrorDetail(
                    title="Service Unavailable",
                    code="AUTO_REVIEWER_UNAVAILABLE",
                    status=503,
                    details=[
                        "Set AURA_REVIEW_PROVIDER and AURA_REVIEW_API_KEY "
                        "to enable the auto-reviewer"
                    ],
                ),
            )

    async def defer(
        self,
        item_id: UUID,
        *,
        note: str | None,
        current_user: User,
    ) -> ReviewItem:
        item = await self._lock_or_conflict(item_id)
        self._require_owner_or_admin(item, current_user)
        self._require_active(item)
        return await self.review_item_repo.update(
            item,
            {
                "status": ReviewItemStatus.DEFERRED,
                "reviewer_note": note,
            },
        )

    async def escalate(
        self,
        item_id: UUID,
        *,
        reason: str,
        note: str | None,
        current_user: User,
        tentative_label: ReviewVerdict | None = None,
    ) -> Tuple[ReviewItem, ReviewEscalation]:
        item = await self._lock_or_conflict(item_id)
        self._require_owner_or_admin(item, current_user)
        self._require_active(item)
        await self.review_item_repo.update(
            item,
            {
                "status": ReviewItemStatus.ESCALATED,
            },
        )
        escalation = await self.escalation_repo.create(
            ReviewEscalation(
                review_item_id=item.id,
                escalated_by=current_user.id,
                escalated_to=None,
                reason=reason,
                note=note,
                tentative_label=tentative_label,
            )
        )
        return item, escalation

    async def reassign(
        self,
        item_id: UUID,
        *,
        new_user_id: UUID,
        admin: User,
    ) -> ReviewItem:
        new_assignee = await self.user_repo.get_by_id(new_user_id)
        if new_assignee is None or not new_assignee.is_active:
            raise BadRequestException(
                message="Target user is not an active analyst",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="REVIEW_REASSIGN_INVALID_USER",
                    status=400,
                    details=[
                        f"User {new_user_id} cannot receive review items"
                    ],
                ),
            )
        item = await self._lock_or_conflict(item_id)
        if item.status in (
            ReviewItemStatus.CONFIRMED,
            ReviewItemStatus.ESCALATED,
        ):
            raise self._terminal_conflict(item.status)
        return await self.review_item_repo.update(
            item,
            {
                "status": ReviewItemStatus.ASSIGNED,
                "assigned_to": new_assignee.id,
                "assigned_at": _now(),
                "claimed_at": None,
            },
        )

    # ── feedback-loop helpers (also used by EscalationService.resolve) ──

    async def record_buffer_entry_for_item(
        self,
        item: ReviewItem,
        verdict: ReviewVerdict,
        contributor: User,
    ) -> TrainingBufferItem:
        """Public wrapper so EscalationService can drive the same path
        without reaching into private members.
        """
        return await self._record_buffer_entry(item, verdict, contributor)

    def schedule_drift_confirmation_for_item(
        self, item: ReviewItem, verdict: ReviewVerdict
    ) -> None:
        self._schedule_drift_confirmation(item, verdict)

    async def mirror_drift_confirmation_for_item(
        self, item: ReviewItem, verdict: ReviewVerdict, occurred_at: datetime,
    ) -> None:
        """Public wrapper so EscalationService can persist the drift_events
        mirror row in the same transaction as its review-item update.
        """
        await self._mirror_drift_confirmation(item, verdict, occurred_at)

    # ── internals ──

    async def _lock_or_conflict(self, item_id: UUID) -> ReviewItem:
        item = await self.review_item_repo.claim_for_update(item_id)
        if item is None:
            # Either it doesn't exist or another transaction holds the row
            # under SKIP LOCKED. Disambiguate so callers get the right code.
            existing = await self.review_item_repo.get_by_id(item_id)
            if existing is None:
                raise NotFoundException(
                    message="Review item not found",
                    error_detail=ErrorDetail(
                        title="Not Found",
                        code="REVIEW_ITEM_NOT_FOUND",
                        status=404,
                        details=[f"No review item found with id {item_id}"],
                    ),
                )
            raise ConflictException(
                message="This item is currently locked by another request",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="REVIEW_ITEM_LOCKED",
                    status=409,
                    details=["Try again in a moment"],
                ),
            )
        return item

    @staticmethod
    def _require_owner_or_admin(item: ReviewItem, user: User) -> None:
        if user.role == Role.ADMIN:
            return
        if item.assigned_to != user.id:
            raise AuthorizationException(
                message="You can only act on items assigned to you",
                error_detail=ErrorDetail(
                    title="Access Denied",
                    code="REVIEW_ITEM_NOT_YOURS",
                    status=403,
                    details=["Item is not assigned to you"],
                ),
            )

    @staticmethod
    def _require_active(item: ReviewItem) -> None:
        if item.status in (
            ReviewItemStatus.CONFIRMED,
            ReviewItemStatus.ESCALATED,
        ):
            raise ReviewService._terminal_conflict(item.status)

    @staticmethod
    def _terminal_conflict(status: ReviewItemStatus) -> ConflictException:
        return ConflictException(
            message=f"Item is already in {status.value} state",
            error_detail=ErrorDetail(
                title="Conflict",
                code="REVIEW_ITEM_TERMINAL",
                status=409,
                details=[f"Cannot transition from {status.value}"],
            ),
        )

    async def _pick_assignee(self) -> UUID | None:
        analysts = await self.user_repo.get_all(
            role=Role.IT_ANALYST,
            is_active=True,
        )
        if not analysts:
            return None
        analyst_ids = [a.id for a in analysts]
        counts = await self.review_item_repo.assigned_counts_for_users(
            analyst_ids
        )
        # Stable tie-break by analyst id keeps assignment deterministic in
        # tests while behaving as least-loaded — under steady-state load this
        # converges to round-robin without needing persistent counter state.
        analyst_ids.sort(key=lambda uid: (counts.get(uid, 0), str(uid)))
        return analyst_ids[0]

    async def _record_buffer_entry(
        self,
        item: ReviewItem,
        verdict: ReviewVerdict,
        contributor: User,
    ) -> TrainingBufferItem:
        if await self.buffer_repo.exists_by_source_prediction_event_id(
            item.prediction_event_id
        ):
            raise ConflictException(
                message="This prediction has already fed the training buffer",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="TRAINING_BUFFER_DUPLICATE",
                    status=409,
                    details=[
                        "A buffer row already references this prediction"
                    ],
                ),
            )
        event_row = item.prediction_event

        # Phase 12 — buffer quality gate. Reject items whose combined OOV
        # rate against the active detector's vectorisers exceeds the
        # configured maximum. Skipped when the feature is off or no
        # detector is loaded.
        rejected, rate = violates_quality_gate(
            detector=self.detector,
            subject=event_row.subject,
            body=event_row.body,
        )
        if rejected:
            raise BadRequestException(
                message="Item rejected by the training-buffer quality gate",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="TRAINING_BUFFER_QUALITY_GATE_REJECTED",
                    status=400,
                    details=[
                        f"Combined OOV rate {rate:.3f} exceeds the configured "
                        f"maximum {quality_gate_max_oov():.3f}"
                    ],
                ),
            )

        sha = _content_sha256(
            event_row.sender, event_row.subject, event_row.body
        )
        entry = TrainingBufferItem(
            sender=event_row.sender,
            subject=event_row.subject,
            body=event_row.body,
            label=_verdict_to_label(verdict),
            source=TrainingBufferSource.REVIEW,
            source_prediction_event_id=event_row.id,
            source_review_item_id=item.id,
            content_sha256=sha,
            contributed_by=contributor.id,
            consumed_in_run_ids=[],
        )
        return await self.buffer_repo.create(entry)

    async def _mirror_drift_confirmation(
        self,
        item: ReviewItem,
        verdict: ReviewVerdict,
        occurred_at: datetime,
    ) -> None:
        """Persist a `drift_events` row inside the calling transaction so the
        SQL mirror commits atomically with the review-item status change.
        Skips silently when the prediction event has no `prediction_id` —
        those rows predate the monitor wiring; the JSONL hook (Phase 3)
        already skips them too.
        """
        prediction_id = item.prediction_event.prediction_id
        if prediction_id is None:
            return
        await self.drift_event_repo.record_confirmation(
            prediction_id=prediction_id,
            confirmed_label=_verdict_to_label(verdict),
            occurred_at=occurred_at,
        )

    def _schedule_drift_confirmation(
        self, item: ReviewItem, verdict: ReviewVerdict
    ) -> None:
        """Register an `after_commit` hook on the active session.

        The drift monitor's pending set is keyed by the same UUID string the
        detector emitted at predict time — stored on `prediction_events.
        prediction_id`. If that column is null (e.g. an old row imported from
        before the monitor was wired), we silently skip; the drift mirror
        backfill in Phase 5 covers that gap.
        """
        prediction_id = item.prediction_event.prediction_id
        if prediction_id is None:
            return
        pid_str = str(prediction_id)
        label = _verdict_to_label(verdict)
        monitor = self.drift_monitor

        def _on_commit(_session) -> None:
            try:
                monitor.record_confirmation(pid_str, label)
            except Exception:
                # Post-commit failure is recoverable: the next request that
                # touches the monitor (or the Phase 5 drift_events backfill)
                # replays from the JSONL log. Swallow so the API call still
                # succeeds — the row state is already committed.
                log.exception(
                    "drift_monitor.record_confirmation failed "
                    "prediction_id=%s",
                    pid_str,
                )

        event.listen(
            self.session.sync_session,
            "after_commit",
            _on_commit,
            once=True,
        )
