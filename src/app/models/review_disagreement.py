"""Phase 12 — analyst/admin disagreement audit log.

A row is written when ``review.disagreement_log.enabled`` and an admin
resolves an escalation whose analyst recorded a ``tentativeLabel`` that
differs from the admin verdict. The table is append-only and carries no
decision semantics — nothing reads it for routing, it exists purely for
audit and offline reviewer-quality analysis.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    Enum as SAEnum,
    ForeignKey,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.models.enums import ReviewVerdict
from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.review_escalation import ReviewEscalation
    from src.app.models.review_item import ReviewItem
    from src.app.models.user import User


class ReviewDisagreement(BaseModel):
    __tablename__ = "review_disagreements"

    review_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("review_items.id", ondelete="CASCADE"),
        nullable=False,
    )
    review_escalation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("review_escalations.id", ondelete="CASCADE"),
        nullable=False,
    )
    analyst_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    admin_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    analyst_verdict: Mapped[ReviewVerdict] = mapped_column(
        SAEnum(ReviewVerdict, name="review_verdict_enum"), nullable=False
    )
    admin_verdict: Mapped[ReviewVerdict] = mapped_column(
        SAEnum(ReviewVerdict, name="review_verdict_enum"), nullable=False
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    review_item: Mapped["ReviewItem"] = relationship(
        "ReviewItem", lazy="joined"
    )
    escalation: Mapped["ReviewEscalation"] = relationship(
        "ReviewEscalation", lazy="joined"
    )
    analyst: Mapped["User | None"] = relationship(
        "User", lazy="joined", foreign_keys=[analyst_id]
    )
    admin: Mapped["User | None"] = relationship(
        "User", lazy="joined", foreign_keys=[admin_id]
    )
