"""tokens.token -> tokens.token_hash (sha256 hex)

Revision ID: 0002_token_hash
Revises: 0001_initial
Create Date: 2026-04-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002_token_hash"
down_revision: Union[str, Sequence[str], None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Replace the raw-token column with a sha256 hash column.

    Existing rows are dropped: previously stored tokens are unrecoverable
    because we never had their hashes. Users must re-authenticate.
    """
    op.execute("DELETE FROM tokens")
    op.drop_index(op.f("ix_tokens_token"), table_name="tokens")
    op.drop_column("tokens", "token")
    op.add_column(
        "tokens",
        sa.Column("token_hash", sa.String(length=64), nullable=False),
    )
    op.create_index(
        op.f("ix_tokens_token_hash"), "tokens", ["token_hash"], unique=True
    )


def downgrade() -> None:
    op.execute("DELETE FROM tokens")
    op.drop_index(op.f("ix_tokens_token_hash"), table_name="tokens")
    op.drop_column("tokens", "token_hash")
    op.add_column(
        "tokens",
        sa.Column("token", sa.Text(), nullable=False),
    )
    op.create_index(op.f("ix_tokens_token"), "tokens", ["token"], unique=True)
