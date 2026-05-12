"""benchmarks: side-by-side dataset + per-version result tables

Revision ID: 0009_benchmarks
Revises: 0008_model_management
Create Date: 2026-04-19 00:00:03.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0009_benchmarks"
down_revision: Union[str, Sequence[str], None] = "0008_model_management"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Phase 10 — Benchmarking (§6.8–6.11). Four tables: curated datasets,
    their rows, per-run headers, and per-version result rows. The version
    array on `model_benchmarks` is materialised (String(16)[]) so the side-
    by-side payload for `GET /benchmarks/{id}` can be assembled from the
    header + its result children without a join back to the registry.
    """
    op.create_table(
        "benchmark_datasets",
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column(
            "label_distribution",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("uploaded_by", sa.UUID(), nullable=True),
        sa.Column("source_csv_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["uploaded_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_benchmark_datasets_name"),
    )
    op.create_index(
        "ix_benchmark_datasets_created_at_desc",
        "benchmark_datasets",
        [sa.text("created_at DESC")],
    )

    op.create_table(
        "benchmark_dataset_rows",
        sa.Column("benchmark_dataset_id", sa.UUID(), nullable=False),
        sa.Column("sender", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("label", sa.SmallInteger(), nullable=False),
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["benchmark_dataset_id"],
            ["benchmark_datasets.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_benchmark_dataset_rows_dataset_id",
        "benchmark_dataset_rows",
        ["benchmark_dataset_id"],
    )

    op.create_table(
        "model_benchmarks",
        sa.Column("benchmark_dataset_id", sa.UUID(), nullable=False),
        sa.Column("triggered_by", sa.UUID(), nullable=True),
        sa.Column(
            "versions",
            postgresql.ARRAY(sa.String(length=16)),
            nullable=False,
            server_default=sa.text("'{}'::varchar[]"),
        ),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "RUNNING",
                "SUCCEEDED",
                "FAILED",
                name="benchmark_status_enum",
            ),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["benchmark_dataset_id"],
            ["benchmark_datasets.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["triggered_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_model_benchmarks_created_at_desc",
        "model_benchmarks",
        [sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_model_benchmarks_dataset_id",
        "model_benchmarks",
        ["benchmark_dataset_id"],
    )
    op.create_index(
        "ix_model_benchmarks_triggered_by_created_at_desc",
        "model_benchmarks",
        ["triggered_by", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_model_benchmarks_status_created_at_desc",
        "model_benchmarks",
        ["status", sa.text("created_at DESC")],
    )

    op.create_table(
        "model_benchmark_version_results",
        sa.Column("model_benchmark_id", sa.UUID(), nullable=False),
        sa.Column("version", sa.String(length=16), nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=False),
        sa.Column("precision", sa.Float(), nullable=False),
        sa.Column("recall", sa.Float(), nullable=False),
        sa.Column("f1", sa.Float(), nullable=False),
        sa.Column("roc_auc", sa.Float(), nullable=False),
        sa.Column("ece", sa.Float(), nullable=False),
        sa.Column(
            "confusion_matrix",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "per_zone_counts",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("prediction_ms_p50", sa.Float(), nullable=False),
        sa.Column("prediction_ms_p95", sa.Float(), nullable=False),
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["model_benchmark_id"],
            ["model_benchmarks.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_benchmark_id",
            "version",
            name="uq_model_benchmark_version_results_benchmark_version",
        ),
    )
    op.create_index(
        "ix_model_benchmark_version_results_benchmark_id",
        "model_benchmark_version_results",
        ["model_benchmark_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_model_benchmark_version_results_benchmark_id",
        table_name="model_benchmark_version_results",
    )
    op.drop_table("model_benchmark_version_results")

    op.drop_index(
        "ix_model_benchmarks_status_created_at_desc",
        table_name="model_benchmarks",
    )
    op.drop_index(
        "ix_model_benchmarks_triggered_by_created_at_desc",
        table_name="model_benchmarks",
    )
    op.drop_index(
        "ix_model_benchmarks_dataset_id", table_name="model_benchmarks"
    )
    op.drop_index(
        "ix_model_benchmarks_created_at_desc",
        table_name="model_benchmarks",
    )
    op.drop_table("model_benchmarks")

    op.drop_index(
        "ix_benchmark_dataset_rows_dataset_id",
        table_name="benchmark_dataset_rows",
    )
    op.drop_table("benchmark_dataset_rows")

    op.drop_index(
        "ix_benchmark_datasets_created_at_desc",
        table_name="benchmark_datasets",
    )
    op.drop_table("benchmark_datasets")

    op.execute("DROP TYPE IF EXISTS benchmark_status_enum")
