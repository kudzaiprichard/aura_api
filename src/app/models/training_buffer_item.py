import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    Enum as SAEnum,
    ForeignKey,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.models.enums import TrainingBufferSource
from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.prediction_event import PredictionEvent
    from src.app.models.review_item import ReviewItem
    from src.app.models.user import User


class TrainingBufferItem(BaseModel):
    __tablename__ = "training_buffer_items"

    sender: Mapped[str] = mapped_column(Text, nullable=False)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    source: Mapped[TrainingBufferSource] = mapped_column(
        SAEnum(TrainingBufferSource, name="training_buffer_source_enum"),
        nullable=False,
    )
    source_prediction_event_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prediction_events.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_review_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("review_items.id", ondelete="SET NULL"),
        nullable=True,
    )
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    contributed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Populated by training runs in Phase 8. Stored as a UUID array so the
    # full provenance for a training row is available without a join table.
    consumed_in_run_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )

    source_prediction_event: Mapped["PredictionEvent | None"] = relationship(
        "PredictionEvent", lazy="select"
    )
    source_review_item: Mapped["ReviewItem | None"] = relationship(
        "ReviewItem", lazy="select"
    )
    contributor: Mapped["User | None"] = relationship("User", lazy="select")
