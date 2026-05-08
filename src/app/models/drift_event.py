import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, SmallInteger, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.database import BaseModel


# Free-form on the model side; the migration enforces the SQL CHECK so the
# value is constrained to {'prediction', 'confirmation'} without spending a
# Postgres enum type on something this narrow (mirrors the auto_review
# `trigger_kind` pattern).
EVENT_TYPE_PREDICTION = "prediction"
EVENT_TYPE_CONFIRMATION = "confirmation"


class DriftEvent(BaseModel):
    __tablename__ = "drift_events"

    event_type: Mapped[str] = mapped_column(String(16), nullable=False)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    predicted_label: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True
    )
    predicted_probability: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    confirmed_label: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True
    )
    model_version: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
