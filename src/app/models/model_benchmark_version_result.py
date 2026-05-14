import uuid

from sqlalchemy import Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.database import BaseModel


class ModelBenchmarkVersionResult(BaseModel):
    __tablename__ = "model_benchmark_version_results"
    __table_args__ = (
        UniqueConstraint(
            "model_benchmark_id",
            "version",
            name="uq_model_benchmark_version_results_benchmark_version",
        ),
    )

    model_benchmark_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_benchmarks.id", ondelete="CASCADE"),
        nullable=False,
    )
    version: Mapped[str] = mapped_column(String(16), nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1: Mapped[float] = mapped_column(Float, nullable=False)
    roc_auc: Mapped[float] = mapped_column(Float, nullable=False)
    ece: Mapped[float] = mapped_column(Float, nullable=False)
    # `{tp, tn, fp, fn}` — JSONB so the shape stays human-readable on inspect.
    confusion_matrix: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    # `{NOT_SPAM: n, REVIEW: n, SPAM: n}` — keyed by the detector's
    # ConfidenceZone values so the UI can render the same buckets it uses
    # everywhere else without remapping.
    per_zone_counts: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    prediction_ms_p50: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_ms_p95: Mapped[float] = mapped_column(Float, nullable=False)
