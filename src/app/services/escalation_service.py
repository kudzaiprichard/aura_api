import logging
from datetime import datetime, timezone
from typing import Sequence, Tuple
from uuid import UUID

from src.configs import review as review_config
from src.app.models.enums import ReviewItemStatus, ReviewVerdict
from src.app.models.review_disagreement import ReviewDisagreement
from src.app.models.review_escalation import ReviewEscalation
from src.app.models.user import User
from src.app.repositories.review_disagreement_repository import (
    ReviewDisagreementRepository,
)
from src.app.repositories.review_escalation_repository import (
    ReviewEscalationRepository,
)
from src.app.repositories.review_item_repository import ReviewItemRepository
from src.app.services.review_service import ReviewService
from src.shared.exceptions import (
    BadRequestException,
    ConflictException,
    NotFoundException,
)
from src.shared.responses import ErrorDetail


log = logging.getLogger("aura.review.escalation")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EscalationService:
    """Admin-side resolution of items pushed up by an analyst (§3.13).

    `resolve` reuses `ReviewService`'s feedback-loop helpers so the same
    transactional contract — buffer insert + status transition in one tx,
    drift confirmation post-commit — applies to admin decisions.
    """

    def __init__(
        self,
        review_item_repository: ReviewItemRepository,
        review_escalation_repository: ReviewEscalationRepository,
        review_service: ReviewService,
        review_disagreement_repository: ReviewDisagreementRepository | None = None,
    ):
        self.review_item_repo = review_item_repository
        self.escalation_repo = review_escalation_repository
        self.review_service = review_service
        # Phase 12 — optional disagreement log. The repo is passed even
        # when the feature flag is off so turning the flag on does not
        # require a service restart.
        self.review_disagreement_repo = review_disagreement_repository

    async def list_escalations(
        self,
        *,
        page: int,
        page_size: int,
        resolved: bool | None = None,
    ) -> Tuple[Sequence[ReviewEscalation], int]:
        return await self.escalation_repo.paginate_filtered(
            page=page,
            page_size=page_size,
            resolved=resolved,
        )

    async def resolve(
        self,
        escalation_id: UUID,
        *,
        verdict: ReviewVerdict,
        note: str | None,
        admin: User,
    ) -> ReviewEscalation:
        escalation = await self._get_open(escalation_id)
        item = await self.review_service._lock_or_conflict(
            escalation.review_item_id
        )
        if item.status != ReviewItemStatus.ESCALATED:
            raise ConflictException(
                message="Underlying review item is no longer escalated",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="ESCALATION_ITEM_STATE_INVALID",
                    status=409,
                    details=[
                        f"Item status is {item.status.value}, expected ESCALATED"
                    ],
                ),
            )

        now = _now()
        await self.review_item_repo.update(
            item,
            {
                "status": ReviewItemStatus.CONFIRMED,
                "verdict": verdict,
                "decided_by": admin.id,
                "decided_at": now,
                "reviewer_note": note,
            },
        )
        resolved = await self.escalation_repo.update(
            escalation,
            {
                "resolved_at": now,
                "resolved_by": admin.id,
                "resolution_verdict": verdict,
                "note": note if note is not None else escalation.note,
            },
        )
        await self.review_service.record_buffer_entry_for_item(
            item, verdict, admin
        )
        await self.review_service.mirror_drift_confirmation_for_item(
            item, verdict, now,
        )
        self.review_service.schedule_drift_confirmation_for_item(item, verdict)
        await self._maybe_log_disagreement(
            escalation=escalation,
            item_id=item.id,
            admin_verdict=verdict,
            admin=admin,
            note=note,
        )
        return resolved

    async def _maybe_log_disagreement(
        self,
        *,
        escalation: ReviewEscalation,
        item_id: UUID,
        admin_verdict: ReviewVerdict,
        admin: User,
        note: str | None,
    ) -> None:
        """Write a review_disagreements row when the analyst's tentative
        verdict differs from the admin resolution and the feature flag is
        on. Silent no-op when the analyst did not record a tentative label.
        """
        cfg = getattr(review_config, "disagreement_log", None)
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return
        if self.review_disagreement_repo is None:
            return
        analyst_verdict = escalation.tentative_label
        if analyst_verdict is None or analyst_verdict == admin_verdict:
            return
        await self.review_disagreement_repo.create(
            ReviewDisagreement(
                review_item_id=item_id,
                review_escalation_id=escalation.id,
                analyst_id=escalation.escalated_by,
                admin_id=admin.id,
                analyst_verdict=analyst_verdict,
                admin_verdict=admin_verdict,
                note=note,
            )
        )
        log.info(
            "review_disagreement logged escalation_id=%s analyst=%s "
            "analyst_verdict=%s admin_verdict=%s admin_id=%s",
            escalation.id, escalation.escalated_by,
            analyst_verdict.value, admin_verdict.value, admin.id,
        )

    async def return_to_pool(
        self,
        escalation_id: UUID,
        *,
        reason: str | None,
        admin: User,
    ) -> ReviewEscalation:
        escalation = await self._get_open(escalation_id)
        item = await self.review_service._lock_or_conflict(
            escalation.review_item_id
        )
        if item.status != ReviewItemStatus.ESCALATED:
            raise BadRequestException(
                message="Underlying review item is no longer escalated",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="ESCALATION_ITEM_STATE_INVALID",
                    status=400,
                    details=[
                        f"Item status is {item.status.value}, expected ESCALATED"
                    ],
                ),
            )

        now = _now()
        await self.review_item_repo.update(
            item,
            {
                "status": ReviewItemStatus.UNASSIGNED,
                "assigned_to": None,
                "assigned_at": None,
                "claimed_at": None,
            },
        )
        return await self.escalation_repo.update(
            escalation,
            {
                "resolved_at": now,
                "resolved_by": admin.id,
                "resolution_verdict": None,
                "note": reason if reason is not None else escalation.note,
            },
        )

    async def _get_open(self, escalation_id: UUID) -> ReviewEscalation:
        escalation = await self.escalation_repo.get_by_id(escalation_id)
        if escalation is None:
            raise NotFoundException(
                message="Escalation not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="ESCALATION_NOT_FOUND",
                    status=404,
                    details=[f"No escalation found with id {escalation_id}"],
                ),
            )
        if escalation.resolved_at is not None:
            raise ConflictException(
                message="Escalation has already been resolved",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="ESCALATION_ALREADY_RESOLVED",
                    status=409,
                    details=["Resolution timestamp already set"],
                ),
            )
        return escalation
