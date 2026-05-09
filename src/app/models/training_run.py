import uuid
from datetime import datetime
from typing import Any, TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.models.enums import BalanceStrategy, TrainingRunStatus
from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.user import User


class TrainingRun(BaseModel):
    __tablename__ = "training_runs"

    triggered_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[TrainingRunStatus] = mapped_column(
        SAEnum(TrainingRunStatus, name="training_run_status_enum"),
        nullable=False,
    )
    source_version: Mapped[str] = mapped_column(String(16), nullable=False)
    new_version: Mapped[str | None] = mapped_column(String(16), nullable=True)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    balance_strategy: Mapped[BalanceStrategy] = mapped_column(
        SAEnum(BalanceStrategy, name="balance_strategy_enum"),
        nullable=False,
    )
    shuffle: Mapped[bool] = mapped_column(Boolean, nullable=False)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    min_delta_f1: Mapped[float] = mapped_column(Float, nullable=False)
    max_iter_per_call: Mapped[int] = mapped_column(Integer, nullable=False)
    performance_before: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    performance_after: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    oov_rate_subject: Mapped[float | None] = mapped_column(Float, nullable=True)
    oov_rate_body: Mapped[float | None] = mapped_column(Float, nullable=True)
    buffer_snapshot_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    class_counts_before_balance: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    class_counts_after_balance: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    promoted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    promoted_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    rolled_back_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    triggered_by_user: Mapped["User"] = relationship(
        "User", foreign_keys=[triggered_by], lazy="select"
    )
    promoted_by_user: Mapped["User | None"] = relationship(
        "User", foreign_keys=[promoted_by], lazy="select"
    )
