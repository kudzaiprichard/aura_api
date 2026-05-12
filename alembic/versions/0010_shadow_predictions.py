"""shadow predictions: additive shadow_* columns on prediction_events

Revision ID: 0010_shadow_predictions
Revises: 0009_benchmarks
Create Date: 2026-04-19 00:00:04.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0010_shadow_predictions"
down_revision: Union[str, Sequence[str], None] = "0009_benchmarks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Phase 12 — shadow predictions (additive, feature-flagged).

    When `inference.shadow.enabled` is set, each `activate` captures the
    prior detector in-memory and keeps it alive for `days` so `/predict`
    also records the shadow version's verdict on the same row. The shadow
    is never acted on — it never feeds drift, review, or the training
    buffer — it only lands in these columns for offline comparison.

    All columns are nullable so the existing hot path remains unchanged
    when the flag is off.
    """
    op.add_column(
        "prediction_events",
        sa.Column("shadow_model_version", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "prediction_events",
        sa.Column("shadow_predicted_label", sa.SmallInteger(), nullable=True),
    )
    op.add_column(
        "prediction_events",
        sa.Column("shadow_phishing_probability", sa.Float(), nullable=True),
    )
    op.add_column(
        "prediction_events",
        sa.Column(
            "shadow_confidence_zone",
            postgresql.ENUM(
                "SPAM",
                "NOT_SPAM",
                "REVIEW",
                name="confidence_zone_enum",
                create_type=False,
            ),
            nullable=True,
        ),
    )

    # A partial index on shadow_model_version supports the offline
    # comparison queries without bloating the hot-path index footprint.
    op.create_index(
        "ix_prediction_events_shadow_model_version",
        "prediction_events",
        ["shadow_model_version"],
        postgresql_where=sa.text("shadow_model_version IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_prediction_events_shadow_model_version",
        table_name="prediction_events",
    )
    op.drop_column("prediction_events", "shadow_confidence_zone")
    op.drop_column("prediction_events", "shadow_phishing_probability")
    op.drop_column("prediction_events", "shadow_predicted_label")
    op.drop_column("prediction_events", "shadow_model_version")
