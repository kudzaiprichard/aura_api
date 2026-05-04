"""auto-review invocations

Revision ID: 0005_auto_review_invocations
Revises: 0004_review_workflow
Create Date: 2026-04-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0005_auto_review_invocations"
down_revision: Union[str, Sequence[str], None] = "0004_review_workflow"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the auto_review_invocations table (§6.15) and back-fill the FK
    on `review_items.latest_auto_review_invocation_id` that Phase 3 left
    dangling because the target table did not exist yet.
    """
    op.create_table(
        "auto_review_invocations",
        sa.Column("review_item_id", sa.UUID(), nullable=False),
        sa.Column("triggered_by", sa.UUID(), nullable=True),
        sa.Column("trigger_kind", sa.String(length=16), nullable=False),
        sa.Column("batch_group_id", sa.UUID(), nullable=True),
        sa.Column(
            "provider",
            sa.Enum("GROQ", "GOOGLE", name="llm_provider_enum"),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column(
            "outcome",
            sa.Enum("SUCCESS", "FAILURE", name="auto_review_outcome_enum"),
            nullable=False,
        ),
        sa.Column("label", sa.Text(), nullable=True),
        sa.Column("confidence", sa.String(length=8), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column(
            "raw_payload",
            sa.dialects.postgresql.JSONB(),
            nullable=True,
        ),
        sa.Column("user_message", sa.Text(), nullable=True),
        sa.Column("technical_error", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
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
            ["review_item_id"], ["review_items.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["triggered_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "trigger_kind IN ('single', 'batch')",
            name="ck_auto_review_invocations_trigger_kind",
        ),
    )
    op.create_index(
        "ix_auto_review_invocations_item_created_at",
        "auto_review_invocations",
        ["review_item_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_auto_review_invocations_triggered_by_created_at",
        "auto_review_invocations",
        ["triggered_by", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_auto_review_invocations_batch_group_id",
        "auto_review_invocations",
        ["batch_group_id"],
        postgresql_where=sa.text("batch_group_id IS NOT NULL"),
    )

    # Phase 3 created `review_items.latest_auto_review_invocation_id` without
    # a FK because this table did not exist yet. Wire it now.
    op.create_foreign_key(
        "fk_review_items_latest_auto_review_invocation_id",
        "review_items",
        "auto_review_invocations",
        ["latest_auto_review_invocation_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_review_items_latest_auto_review_invocation_id",
        "review_items",
        type_="foreignkey",
    )
    op.drop_index(
        "ix_auto_review_invocations_batch_group_id",
        table_name="auto_review_invocations",
    )
    op.drop_index(
        "ix_auto_review_invocations_triggered_by_created_at",
        table_name="auto_review_invocations",
    )
    op.drop_index(
        "ix_auto_review_invocations_item_created_at",
        table_name="auto_review_invocations",
    )
    op.drop_table("auto_review_invocations")
    op.execute("DROP TYPE IF EXISTS auto_review_outcome_enum")
    op.execute("DROP TYPE IF EXISTS llm_provider_enum")
