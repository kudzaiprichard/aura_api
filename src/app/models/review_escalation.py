import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.models.enums import ReviewVerdict
from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.review_item import ReviewItem
    from src.app.models.user import User


class ReviewEscalation(BaseModel):
    __tablename__ = "review_escalations"

    review_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("review_items.id", ondelete="CASCADE"),
        nullable=False,
    )
    escalated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    escalated_to: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    resolution_verdict: Mapped[ReviewVerdict | None] = mapped_column(
        SAEnum(ReviewVerdict, name="review_verdict_enum"), nullable=True
    )
    # Phase 12 — optional analyst-recorded tentative verdict captured at
    # escalation time. Not required; the disagreement log only fires when
    # a value is present and differs from the admin verdict.
    tentative_label: Mapped[ReviewVerdict | None] = mapped_column(
        SAEnum(ReviewVerdict, name="review_verdict_enum"), nullable=True
    )

    review_item: Mapped["ReviewItem"] = relationship("ReviewItem", lazy="joined")
    escalator: Mapped["User | None"] = relationship(
        "User", lazy="joined", foreign_keys=[escalated_by]
    )
    assignee: Mapped["User | None"] = relationship(
        "User", lazy="joined", foreign_keys=[escalated_to]
    )
    resolver: Mapped["User | None"] = relationship(
        "User", lazy="joined", foreign_keys=[resolved_by]
    )
