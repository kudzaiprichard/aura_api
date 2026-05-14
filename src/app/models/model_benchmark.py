import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.models.enums import BenchmarkStatus
from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.benchmark_dataset import BenchmarkDataset
    from src.app.models.user import User


class ModelBenchmark(BaseModel):
    __tablename__ = "model_benchmarks"

    benchmark_dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("benchmark_datasets.id", ondelete="RESTRICT"),
        nullable=False,
    )
    triggered_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    versions: Mapped[list[str]] = mapped_column(
        ARRAY(String(length=16)), nullable=False, default=list
    )
    status: Mapped[BenchmarkStatus] = mapped_column(
        SAEnum(BenchmarkStatus, name="benchmark_status_enum"),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    dataset: Mapped["BenchmarkDataset"] = relationship(
        "BenchmarkDataset", foreign_keys=[benchmark_dataset_id], lazy="select"
    )
    triggered_by_user: Mapped["User | None"] = relationship(
        "User", foreign_keys=[triggered_by], lazy="select"
    )
