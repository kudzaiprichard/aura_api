"""review disagreements: tentative_label + review_disagreements table

Revision ID: 0011_review_disagreements
Revises: 0010_shadow_predictions
Create Date: 2026-04-19 00:00:05.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0011_review_disagreements"
down_revision: Union[str, Sequence[str], None] = "0010_shadow_predictions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Phase 12 — reviewer agreement log (additive, feature-flagged).

    * `review_escalations.tentative_label` — analyst's own verdict captured
      at escalation time; nullable so existing flows stay unchanged.
    * `review_disagreements` — audit rows written when the admin resolution
      differs from the analyst's tentative verdict and the feature flag is
      on. Both foreign keys cascade on delete so cleaning up an escalation
      cleans up its audit trail.
    """
    op.add_column(
        "review_escalations",
        sa.Column(
            "tentative_label",
            postgresql.ENUM(
                "PHISHING",
                "LEGITIMATE",
                name="review_verdict_enum",
                create_type=False,
            ),
            nullable=True,
        ),
    )

    op.create_table(
        "review_disagreements",
        sa.Column("review_item_id", sa.UUID(), nullable=False),
        sa.Column("review_escalation_id", sa.UUID(), nullable=False),
        sa.Column("analyst_id", sa.UUID(), nullable=True),
        sa.Column("admin_id", sa.UUID(), nullable=True),
        sa.Column(
            "analyst_verdict",
            postgresql.ENUM(
                "PHISHING",
                "LEGITIMATE",
                name="review_verdict_enum",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "admin_verdict",
            postgresql.ENUM(
                "PHISHING",
                "LEGITIMATE",
                name="review_verdict_enum",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("note", sa.Text(), nullable=True),
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
            ["review_escalation_id"],
            ["review_escalations.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["analyst_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["admin_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_review_disagreements_escalation_id",
        "review_disagreements",
        ["review_escalation_id"],
    )
    op.create_index(
        "ix_review_disagreements_created_at_desc",
        "review_disagreements",
        [sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_review_disagreements_created_at_desc",
        table_name="review_disagreements",
    )
    op.drop_index(
        "ix_review_disagreements_escalation_id",
        table_name="review_disagreements",
    )
    op.drop_table("review_disagreements")
    op.drop_column("review_escalations", "tentative_label")
