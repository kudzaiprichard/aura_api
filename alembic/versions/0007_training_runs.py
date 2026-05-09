"""training_runs: online-learning run history

Revision ID: 0007_training_runs
Revises: 0006_drift_events
Create Date: 2026-04-19 00:00:01.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0007_training_runs"
down_revision: Union[str, Sequence[str], None] = "0006_drift_events"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the training_runs table (§6.6). One row per online-learning run;
    holds the slice, the balance decisions, the pre/post metrics, and the
    registered version so §3.5 end-to-end is auditable from SQL alone.
    """
    op.create_table(
        "training_runs",
        sa.Column("triggered_by", sa.UUID(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "RUNNING",
                "SUCCEEDED",
                "FAILED",
                "ROLLED_BACK",
                name="training_run_status_enum",
            ),
            nullable=False,
        ),
        sa.Column("source_version", sa.String(length=16), nullable=False),
        sa.Column("new_version", sa.String(length=16), nullable=True),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column(
            "iterations",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "balance_strategy",
            sa.Enum(
                "UNDERSAMPLE",
                "OVERSAMPLE",
                "NONE",
                name="balance_strategy_enum",
            ),
            nullable=False,
        ),
        sa.Column("shuffle", sa.Boolean(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("min_delta_f1", sa.Float(), nullable=False),
        sa.Column("max_iter_per_call", sa.Integer(), nullable=False),
        sa.Column("performance_before", postgresql.JSONB(), nullable=True),
        sa.Column("performance_after", postgresql.JSONB(), nullable=True),
        sa.Column("oov_rate_subject", sa.Float(), nullable=True),
        sa.Column("oov_rate_body", sa.Float(), nullable=True),
        sa.Column(
            "buffer_snapshot_ids",
            postgresql.ARRAY(sa.UUID()),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        sa.Column(
            "class_counts_before_balance",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "class_counts_after_balance",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "promoted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "promoted_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("promoted_by", sa.UUID(), nullable=True),
        sa.Column(
            "rolled_back_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
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
            ["triggered_by"], ["users.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["promoted_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_training_runs_created_at_desc",
        "training_runs",
        [sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_training_runs_triggered_by_created_at_desc",
        "training_runs",
        ["triggered_by", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_training_runs_new_version",
        "training_runs",
        ["new_version"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_training_runs_new_version", table_name="training_runs"
    )
    op.drop_index(
        "ix_training_runs_triggered_by_created_at_desc",
        table_name="training_runs",
    )
    op.drop_index(
        "ix_training_runs_created_at_desc", table_name="training_runs"
    )
    op.drop_table("training_runs")
    op.execute("DROP TYPE IF EXISTS balance_strategy_enum")
    op.execute("DROP TYPE IF EXISTS training_run_status_enum")
