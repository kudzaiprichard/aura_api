"""prediction_events: audit-log table for every predict / predict_batch call

Revision ID: 0003_prediction_events
Revises: 0002_token_hash
Create Date: 2026-04-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0003_prediction_events"
down_revision: Union[str, Sequence[str], None] = "0002_token_hash"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create prediction_events and its two enums + indexes."""
    op.create_table(
        "prediction_events",
        sa.Column("prediction_id", sa.UUID(), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("requester_id", sa.UUID(), nullable=True),
        sa.Column(
            "source",
            sa.Enum("API", "BATCH", "BENCHMARK", name="prediction_source_enum"),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(length=16), nullable=False),
        sa.Column("sender", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("predicted_label", sa.SmallInteger(), nullable=False),
        sa.Column("phishing_probability", sa.Float(), nullable=False),
        sa.Column("legitimate_probability", sa.Float(), nullable=False),
        sa.Column("raw_phishing_probability", sa.Float(), nullable=True),
        sa.Column("raw_legitimate_probability", sa.Float(), nullable=True),
        sa.Column("calibrated", sa.Boolean(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column(
            "confidence_zone",
            sa.Enum("SPAM", "NOT_SPAM", "REVIEW", name="confidence_zone_enum"),
            nullable=True,
        ),
        sa.Column("review_low_threshold", sa.Float(), nullable=True),
        sa.Column("review_high_threshold", sa.Float(), nullable=True),
        sa.Column("engineered_features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.ForeignKeyConstraint(["requester_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_prediction_events_predicted_at_desc",
        "prediction_events",
        [sa.text("predicted_at DESC")],
    )
    op.create_index(
        "ix_prediction_events_model_version_predicted_at",
        "prediction_events",
        ["model_version", "predicted_at"],
    )
    op.create_index(
        "ix_prediction_events_requester_predicted_at_desc",
        "prediction_events",
        ["requester_id", sa.text("predicted_at DESC")],
    )
    op.create_index(
        "ix_prediction_events_zone_predicted_at",
        "prediction_events",
        ["confidence_zone", "predicted_at"],
    )
    op.create_index(
        "uq_prediction_events_prediction_id",
        "prediction_events",
        ["prediction_id"],
        unique=True,
        postgresql_where=sa.text("prediction_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_prediction_events_prediction_id", table_name="prediction_events"
    )
    op.drop_index(
        "ix_prediction_events_zone_predicted_at", table_name="prediction_events"
    )
    op.drop_index(
        "ix_prediction_events_requester_predicted_at_desc",
        table_name="prediction_events",
    )
    op.drop_index(
        "ix_prediction_events_model_version_predicted_at",
        table_name="prediction_events",
    )
    op.drop_index(
        "ix_prediction_events_predicted_at_desc", table_name="prediction_events"
    )
    op.drop_table("prediction_events")
    op.execute("DROP TYPE IF EXISTS confidence_zone_enum")
    op.execute("DROP TYPE IF EXISTS prediction_source_enum")
