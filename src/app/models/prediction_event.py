import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.models.enums import ConfidenceZone, PredictionSource
from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.user import User


class PredictionEvent(BaseModel):
    __tablename__ = "prediction_events"

    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    requester_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    source: Mapped[PredictionSource] = mapped_column(
        SAEnum(PredictionSource, name="prediction_source_enum"), nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(16), nullable=False)
    sender: Mapped[str] = mapped_column(Text, nullable=False)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_label: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    phishing_probability: Mapped[float] = mapped_column(Float, nullable=False)
    legitimate_probability: Mapped[float] = mapped_column(Float, nullable=False)
    raw_phishing_probability: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    raw_legitimate_probability: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    calibrated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_zone: Mapped[ConfidenceZone | None] = mapped_column(
        SAEnum(ConfidenceZone, name="confidence_zone_enum"), nullable=True
    )
    review_low_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    review_high_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    engineered_features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Phase 12 — shadow prediction columns. Populated only when the shadow
    # detector is active (activate() captured the prior version and it has
    # not yet expired). Never used for decisions — never feeds drift,
    # review, or the training buffer.
    shadow_model_version: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )
    shadow_predicted_label: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True
    )
    shadow_phishing_probability: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    shadow_confidence_zone: Mapped[ConfidenceZone | None] = mapped_column(
        SAEnum(ConfidenceZone, name="confidence_zone_enum", create_type=False),
        nullable=True,
    )

    requester: Mapped["User | None"] = relationship("User", lazy="joined")
