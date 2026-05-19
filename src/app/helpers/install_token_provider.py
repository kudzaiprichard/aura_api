"""Install token provider — opaque random bearer tokens for the Chrome extension.

Distinct from `token_provider.py` (which mints JWT pairs for dashboard auth)
because the extension never carries a JWT and the contract specifies opaque
random strings hashed at rest. Lifetime is sourced from
`security.extension_token_expire_days`.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

from src.configs import security
from src.app.models.extension_token import ExtensionToken
from src.app.repositories.extension_token_repository import (
    ExtensionTokenRepository,
)


# 32 bytes of entropy → 43-char URL-safe string. Plenty of margin against
# online guessing while staying short enough to fit comfortably in headers.
_TOKEN_ENTROPY_BYTES = 32


def hash_install_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _generate_cleartext() -> str:
    return secrets.token_urlsafe(_TOKEN_ENTROPY_BYTES)


async def issue_install_token(
    install_id: UUID,
    token_repo: ExtensionTokenRepository,
) -> tuple[str, ExtensionToken]:
    """Mint a fresh opaque token for `install_id`, persist its SHA-256 hash,
    and return both the cleartext (returned to the client exactly once) and
    the persisted row (so the caller can serialise `expires_at`).
    """
    cleartext = _generate_cleartext()
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=security.extension_token_expire_days
    )
    row = await token_repo.create(
        ExtensionToken(
            install_id=install_id,
            token_hash=hash_install_token(cleartext),
            expires_at=expires_at,
        )
    )
    return cleartext, row


async def rotate_install_token(
    install_id: UUID,
    token_repo: ExtensionTokenRepository,
    *,
    revoke_reason: str = "rotated",
) -> tuple[str, ExtensionToken]:
    """Atomic revoke-old + issue-new for register and renew flows. The single
    transaction owned by `get_db` makes the revoke and the insert visible to
    other readers together — there is no window where the install has zero
    valid tokens (until commit) or two valid tokens (after commit).
    """
    await token_repo.revoke_all_for_install(install_id, reason=revoke_reason)
    return await issue_install_token(install_id, token_repo)
