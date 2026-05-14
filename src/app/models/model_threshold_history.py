import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.user import User


class ModelThresholdHistory(BaseModel):
    __tablename__ = "model_threshold_history"

    version: Mapped[str] = mapped_column(String(16), nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    review_low_threshold: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    review_high_threshold: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    set_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    effective_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    set_by_user: Mapped["User | None"] = relationship(
        "User", foreign_keys=[set_by], lazy="select"
    )
