import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.user import User


class BenchmarkDataset(BaseModel):
    __tablename__ = "benchmark_datasets"

    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # `{"0": N, "1": M}` — kept as string keys so the JSONB round-trips losslessly
    # (JSON object keys are always strings).
    label_distribution: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_csv_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    uploader: Mapped["User | None"] = relationship(
        "User", foreign_keys=[uploaded_by], lazy="select"
    )
