"""review workflow: review_items, review_escalations, training_buffer_items

Revision ID: 0004_review_workflow
Revises: 0003_prediction_events
Create Date: 2026-04-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0004_review_workflow"
down_revision: Union[str, Sequence[str], None] = "0003_prediction_events"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create Phase 3 tables + enums + indexes.

    Tables are created without the FK to `auto_review_invocations` (Phase 4);
    Phase 4's migration introduces that table and adds the FK then. Columns
    for auto-review bookkeeping are in place from the start so the schema
    does not have to mutate when Phase 4 activates the LLM path.
    """
    op.create_table(
        "review_items",
        sa.Column("prediction_event_id", sa.UUID(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "UNASSIGNED",
                "ASSIGNED",
                "IN_PROGRESS",
                "CONFIRMED",
                "ESCALATED",
                "DEFERRED",
                name="review_item_status_enum",
            ),
            nullable=False,
            server_default="UNASSIGNED",
        ),
        sa.Column("assigned_to", sa.UUID(), nullable=True),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decided_by", sa.UUID(), nullable=True),
        sa.Column(
            "verdict",
            sa.Enum("PHISHING", "LEGITIMATE", name="review_verdict_enum"),
            nullable=True,
        ),
        sa.Column("reviewer_note", sa.Text(), nullable=True),
        sa.Column(
            "auto_review_used",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "auto_review_agreement",
            sa.Enum(
                "NOT_USED",
                "AGREED",
                "OVERRIDDEN",
                name="auto_review_agreement_enum",
            ),
            nullable=False,
            server_default="NOT_USED",
        ),
        sa.Column("latest_auto_review_invocation_id", sa.UUID(), nullable=True),
        sa.Column("override_reason", sa.Text(), nullable=True),
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
            ["prediction_event_id"],
            ["prediction_events.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["assigned_to"], ["users.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["decided_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "prediction_event_id",
            name="uq_review_items_prediction_event_id",
        ),
    )
    op.create_index(
        "ix_review_items_status_assigned_to",
        "review_items",
        ["status", "assigned_to"],
    )
    op.create_index(
        "ix_review_items_assigned_to_claimed_at",
        "review_items",
        ["assigned_to", "claimed_at"],
    )
    op.create_index(
        "ix_review_items_status_created_at",
        "review_items",
        ["status", "created_at"],
    )
    op.create_index(
        "ix_review_items_auto_review_used_agreement",
        "review_items",
        ["auto_review_used", "auto_review_agreement"],
    )

    op.create_table(
        "review_escalations",
        sa.Column("review_item_id", sa.UUID(), nullable=False),
        sa.Column("escalated_by", sa.UUID(), nullable=True),
        sa.Column("escalated_to", sa.UUID(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.UUID(), nullable=True),
        sa.Column(
            "resolution_verdict",
            sa.Enum(
                "PHISHING",
                "LEGITIMATE",
                name="review_verdict_enum",
                create_type=False,
            ),
            nullable=True,
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
            ["review_item_id"], ["review_items.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["escalated_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["escalated_to"], ["users.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["resolved_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_review_escalations_review_item_id",
        "review_escalations",
        ["review_item_id"],
    )
    op.create_index(
        "ix_review_escalations_resolved_at",
        "review_escalations",
        ["resolved_at"],
    )

    op.create_table(
        "training_buffer_items",
        sa.Column("sender", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("label", sa.SmallInteger(), nullable=False),
        sa.Column(
            "source",
            sa.Enum(
                "REVIEW",
                "CSV_IMPORT",
                "GENERATOR",
                "ESCALATION",
                name="training_buffer_source_enum",
            ),
            nullable=False,
        ),
        sa.Column("source_prediction_event_id", sa.UUID(), nullable=True),
        sa.Column("source_review_item_id", sa.UUID(), nullable=True),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("contributed_by", sa.UUID(), nullable=True),
        sa.Column(
            "consumed_in_run_ids",
            sa.dialects.postgresql.ARRAY(sa.UUID()),
            nullable=False,
            server_default=sa.text("ARRAY[]::uuid[]"),
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
            ["source_prediction_event_id"],
            ["prediction_events.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_review_item_id"],
            ["review_items.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["contributed_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_training_buffer_items_label_source",
        "training_buffer_items",
        ["label", "source"],
    )
    op.create_index(
        "ix_training_buffer_items_created_at",
        "training_buffer_items",
        ["created_at"],
    )
    op.create_index(
        "uq_training_buffer_items_content_sha256",
        "training_buffer_items",
        ["content_sha256"],
        unique=True,
    )
    op.create_index(
        "uq_training_buffer_items_source_prediction_event_id",
        "training_buffer_items",
        ["source_prediction_event_id"],
        unique=True,
        postgresql_where=sa.text("source_prediction_event_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_training_buffer_items_source_prediction_event_id",
        table_name="training_buffer_items",
    )
    op.drop_index(
        "uq_training_buffer_items_content_sha256",
        table_name="training_buffer_items",
    )
    op.drop_index(
        "ix_training_buffer_items_created_at",
        table_name="training_buffer_items",
    )
    op.drop_index(
        "ix_training_buffer_items_label_source",
        table_name="training_buffer_items",
    )
    op.drop_table("training_buffer_items")
    op.execute("DROP TYPE IF EXISTS training_buffer_source_enum")

    op.drop_index(
        "ix_review_escalations_resolved_at", table_name="review_escalations"
    )
    op.drop_index(
        "ix_review_escalations_review_item_id",
        table_name="review_escalations",
    )
    op.drop_table("review_escalations")

    op.drop_index(
        "ix_review_items_auto_review_used_agreement",
        table_name="review_items",
    )
    op.drop_index(
        "ix_review_items_status_created_at", table_name="review_items"
    )
    op.drop_index(
        "ix_review_items_assigned_to_claimed_at", table_name="review_items"
    )
    op.drop_index(
        "ix_review_items_status_assigned_to", table_name="review_items"
    )
    op.drop_table("review_items")
    op.execute("DROP TYPE IF EXISTS review_verdict_enum")
    op.execute("DROP TYPE IF EXISTS review_item_status_enum")
    op.execute("DROP TYPE IF EXISTS auto_review_agreement_enum")
