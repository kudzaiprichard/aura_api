import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.models.enums import AutoReviewOutcome, LLMProviderEnum
from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.review_item import ReviewItem
    from src.app.models.user import User


class AutoReviewInvocation(BaseModel):
    __tablename__ = "auto_review_invocations"

    review_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("review_items.id", ondelete="CASCADE"),
        nullable=False,
    )
    triggered_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Free-form on the model side; the migration enforces the SQL CHECK so the
    # value is constrained to {'single', 'batch'} without spending a Postgres
    # enum type on something this narrow.
    trigger_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    batch_group_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    provider: Mapped[LLMProviderEnum] = mapped_column(
        SAEnum(LLMProviderEnum, name="llm_provider_enum"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    outcome: Mapped[AutoReviewOutcome] = mapped_column(
        SAEnum(AutoReviewOutcome, name="auto_review_outcome_enum"),
        nullable=False,
    )
    label: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(8), nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    user_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    technical_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    review_item: Mapped["ReviewItem"] = relationship(
        "ReviewItem", lazy="joined", foreign_keys=[review_item_id]
    )
    actor: Mapped["User | None"] = relationship(
        "User", lazy="joined", foreign_keys=[triggered_by]
    )
