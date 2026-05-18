"""extension installs and tokens — Chrome extension identity scaffold

Revision ID: 0012_extension_installs_tokens
Revises: 0011_review_disagreements
Create Date: 2026-04-19 00:00:06.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0012_extension_installs_tokens"
down_revision: Union[str, Sequence[str], None] = "0011_review_disagreements"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Chrome extension install identity — additive, isolated from dashboard auth.

    `extension_installs` records one row per Google identity that registered
    the extension. `extension_tokens` holds opaque bearer tokens (SHA-256
    hashed at rest) keyed to those installs; cascading delete keeps cleanup
    cheap when an install is hard-deleted. Neither table touches `users` or
    `tokens` — the dashboard JWT path stays untouched.
    """
    op.create_table(
        "extension_installs",
        sa.Column("google_sub", sa.String(length=64), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(
                "ACTIVE",
                "BLACKLISTED",
                name="extension_install_status_enum",
                create_type=True,
            ),
            nullable=False,
            server_default="ACTIVE",
        ),
        sa.Column("extension_version", sa.String(length=64), nullable=True),
        sa.Column("environment_json", postgresql.JSONB(), nullable=True),
        sa.Column(
            "blacklisted_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("blacklisted_by", sa.UUID(), nullable=True),
        sa.Column("blacklist_reason", sa.String(length=500), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
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
            ["blacklisted_by"], ["users.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("google_sub", name="uq_extension_installs_google_sub"),
    )
    op.create_index(
        "ix_extension_installs_email", "extension_installs", ["email"]
    )
    op.create_index(
        "ix_extension_installs_google_sub",
        "extension_installs",
        ["google_sub"],
    )
    op.create_index(
        "ix_extension_installs_status", "extension_installs", ["status"]
    )

    op.create_table(
        "extension_tokens",
        sa.Column("install_id", sa.UUID(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "is_revoked",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("revoked_reason", sa.String(length=100), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.UniqueConstraint("token_hash", name="uq_extension_tokens_token_hash"),
    )
    op.create_index(
        "ix_extension_tokens_install_id",
        "extension_tokens",
        ["install_id"],
    )
    op.create_index(
        "ix_extension_tokens_token_hash",
        "extension_tokens",
        ["token_hash"],
    )
    op.create_index(
        "ix_extension_tokens_expires_at",
        "extension_tokens",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_extension_tokens_expires_at", table_name="extension_tokens"
    )
    op.drop_index(
        "ix_extension_tokens_token_hash", table_name="extension_tokens"
    )
    op.drop_index(
        "ix_extension_tokens_install_id", table_name="extension_tokens"
    )
    op.drop_table("extension_tokens")

    op.drop_index(
        "ix_extension_installs_status", table_name="extension_installs"
    )
    op.drop_index(
        "ix_extension_installs_google_sub", table_name="extension_installs"
    )
    op.drop_index(
        "ix_extension_installs_email", table_name="extension_installs"
    )
    op.drop_table("extension_installs")

    sa.Enum(name="extension_install_status_enum").drop(
        op.get_bind(), checkfirst=True
    )
