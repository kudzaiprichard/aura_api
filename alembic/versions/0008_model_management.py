"""model_management: activation audit + threshold history

Revision ID: 0008_model_management
Revises: 0007_training_runs
Create Date: 2026-04-19 00:00:02.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0008_model_management"
down_revision: Union[str, Sequence[str], None] = "0007_training_runs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Phase 9 — Model version management (§6.7, §6.13). Two append-only-ish
    tables backing activate/promote/rollback audit and zone-threshold history.
    """
    op.create_table(
        "model_activations",
        sa.Column(
            "kind",
            sa.Enum(
                "ACTIVATE",
                "PROMOTE",
                "ROLLBACK",
                name="model_activation_kind_enum",
            ),
            nullable=False,
        ),
        sa.Column("version", sa.String(length=16), nullable=False),
        sa.Column("previous_version", sa.String(length=16), nullable=True),
        sa.Column("actor_id", sa.UUID(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("metrics_snapshot", postgresql.JSONB(), nullable=True),
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
            ["actor_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_model_activations_created_at_desc",
        "model_activations",
        [sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_model_activations_version_created_at_desc",
        "model_activations",
        ["version", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_model_activations_kind_created_at_desc",
        "model_activations",
        ["kind", sa.text("created_at DESC")],
    )

    op.create_table(
        "model_threshold_history",
        sa.Column("version", sa.String(length=16), nullable=False),
        sa.Column("decision_threshold", sa.Float(), nullable=False),
        sa.Column("review_low_threshold", sa.Float(), nullable=True),
        sa.Column("review_high_threshold", sa.Float(), nullable=True),
        sa.Column("set_by", sa.UUID(), nullable=True),
        sa.Column(
            "effective_from", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "effective_to", sa.DateTime(timezone=True), nullable=True
        ),
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
            ["set_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_model_threshold_history_version_effective_from_desc",
        "model_threshold_history",
        ["version", sa.text("effective_from DESC")],
    )
    # Partial unique index enforces the "one current row per version" invariant:
    # only one row per version may have effective_to IS NULL at any time, so the
    # service can safely fetch "the current row" without tie-breaking logic.
    op.create_index(
        "ux_model_threshold_history_current_per_version",
        "model_threshold_history",
        ["version"],
        unique=True,
        postgresql_where=sa.text("effective_to IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "ux_model_threshold_history_current_per_version",
        table_name="model_threshold_history",
    )
    op.drop_index(
        "ix_model_threshold_history_version_effective_from_desc",
        table_name="model_threshold_history",
    )
    op.drop_table("model_threshold_history")

    op.drop_index(
        "ix_model_activations_kind_created_at_desc",
        table_name="model_activations",
    )
    op.drop_index(
        "ix_model_activations_version_created_at_desc",
        table_name="model_activations",
    )
    op.drop_index(
        "ix_model_activations_created_at_desc",
        table_name="model_activations",
    )
    op.drop_table("model_activations")

    op.execute("DROP TYPE IF EXISTS model_activation_kind_enum")
