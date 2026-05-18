"""extension analysis events — per-install audit trail for /emails/analyze

Revision ID: 0013_extension_analysis_events
Revises: 0012_extension_installs_tokens
Create Date: 2026-04-19 00:00:07.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0013_extension_analysis_events"
down_revision: Union[str, Sequence[str], None] = "0012_extension_installs_tokens"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Narrow audit table for the Chrome extension analyse surface.

    Distinct from `prediction_events` (which carries the full dashboard
    audit trail with engineered features, body, drift mirror) — here we keep
    only what the admin activity feed needs (§13). Cascading delete keeps
    the table cheap to clean up when an install is hard-deleted.
    """
    op.create_table(
        "extension_analysis_events",
        sa.Column("install_id", sa.UUID(), nullable=False),
        sa.Column("predicted_label", sa.String(length=16), nullable=False),
        sa.Column("phishing_probability", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
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
            ["install_id"],
            ["extension_installs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_extension_analysis_events_install_id",
        "extension_analysis_events",
        ["install_id"],
    )
    op.create_index(
        "ix_extension_analysis_events_occurred_at",
        "extension_analysis_events",
        ["occurred_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_extension_analysis_events_occurred_at",
        table_name="extension_analysis_events",
    )
    op.drop_index(
        "ix_extension_analysis_events_install_id",
        table_name="extension_analysis_events",
    )
    op.drop_table("extension_analysis_events")
