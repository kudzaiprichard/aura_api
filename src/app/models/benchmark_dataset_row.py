import uuid

from sqlalchemy import ForeignKey, SmallInteger, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.database import BaseModel


class BenchmarkDatasetRow(BaseModel):
    __tablename__ = "benchmark_dataset_rows"

    benchmark_dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("benchmark_datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    sender: Mapped[str] = mapped_column(Text, nullable=False)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[int] = mapped_column(SmallInteger, nullable=False)
