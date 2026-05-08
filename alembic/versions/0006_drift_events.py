"""drift_events: SQL mirror of the JSONL drift log

Revision ID: 0006_drift_events
Revises: 0005_auto_review_invocations
Create Date: 2026-04-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0006_drift_events"
down_revision: Union[str, Sequence[str], None] = "0005_auto_review_invocations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the drift_events table (§6.12). Mirrors the DriftMonitor JSONL
    log so SQL queries can bucket FPR over time without scanning the file.
    """
    op.create_table(
        "drift_events",
        sa.Column("event_type", sa.String(length=16), nullable=False),
        sa.Column("prediction_id", sa.UUID(), nullable=False),
        sa.Column("predicted_label", sa.SmallInteger(), nullable=True),
        sa.Column("predicted_probability", sa.Float(), nullable=True),
        sa.Column("confirmed_label", sa.SmallInteger(), nullable=True),
        sa.Column("model_version", sa.String(length=16), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "event_type IN ('prediction', 'confirmation')",
            name="ck_drift_events_event_type",
        ),
    )
    op.create_index(
        "ix_drift_events_prediction_id",
        "drift_events",
        ["prediction_id"],
    )
    op.create_index(
        "ix_drift_events_event_type_occurred_at",
        "drift_events",
        ["event_type", "occurred_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_drift_events_event_type_occurred_at", table_name="drift_events"
    )
    op.drop_index(
        "ix_drift_events_prediction_id", table_name="drift_events"
    )
    op.drop_table("drift_events")
