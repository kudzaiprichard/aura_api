import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.models.enums import AutoReviewAgreement, ReviewItemStatus, ReviewVerdict
from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.prediction_event import PredictionEvent
    from src.app.models.user import User


class ReviewItem(BaseModel):
    __tablename__ = "review_items"

    prediction_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prediction_events.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    status: Mapped[ReviewItemStatus] = mapped_column(
        SAEnum(ReviewItemStatus, name="review_item_status_enum"),
        nullable=False,
        default=ReviewItemStatus.UNASSIGNED,
    )
    assigned_to: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    assigned_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    decided_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    verdict: Mapped[ReviewVerdict | None] = mapped_column(
        SAEnum(ReviewVerdict, name="review_verdict_enum"), nullable=True
    )
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Auto-review bookkeeping: columns live from Phase 3 onward so the schema
    # doesn't mutate when Phase 4 activates LLM invocations. Defaults and the
    # nullable FK keep them harmless while auto-review is not yet wired.
    auto_review_used: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    auto_review_agreement: Mapped[AutoReviewAgreement] = mapped_column(
        SAEnum(AutoReviewAgreement, name="auto_review_agreement_enum"),
        nullable=False,
        default=AutoReviewAgreement.NOT_USED,
    )
    latest_auto_review_invocation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    prediction_event: Mapped["PredictionEvent"] = relationship(
        "PredictionEvent", lazy="joined"
    )
    assignee: Mapped["User | None"] = relationship(
        "User", lazy="joined", foreign_keys=[assigned_to]
    )
    decider: Mapped["User | None"] = relationship(
        "User", lazy="joined", foreign_keys=[decided_by]
    )
