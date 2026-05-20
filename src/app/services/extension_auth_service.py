"""Business logic for the Chrome extension auth surface (register/renew/logout).

Kept distinct from `AuthService` (dashboard JWT pair) so the two flows never
share state and so the dashboard's invariants stay in one place.
"""
import logging
from typing import Iterable

from src.configs import extension as extension_config
from src.app.helpers.google_oauth import verify_register_identity
from src.app.helpers.install_token_provider import (
    hash_install_token,
    issue_install_token,
    rotate_install_token,
)
from src.app.models.extension_install import ExtensionInstall
from src.app.models.extension_token import ExtensionToken
from src.app.repositories.extension_install_repository import (
    ExtensionInstallRepository,
)
from src.app.repositories.extension_token_repository import (
    ExtensionTokenRepository,
)
from src.shared.exceptions import AuthorizationException
from src.shared.responses import ErrorDetail


logger = logging.getLogger(__name__)


def _domain_of(email: str) -> str:
    _, _, domain = email.partition("@")
    return domain.casefold()


def _normalise(values: Iterable[str]) -> list[str]:
    return [v.strip().casefold() for v in values if v and v.strip()]


def _email_matches(email: str, allowed_emails: Iterable[str]) -> bool:
    target = email.casefold()
    return target in set(_normalise(allowed_emails))


def _domain_matches(email: str, allowed_domains: Iterable[str]) -> bool:
    domain = _domain_of(email)
    if not domain:
        return False
    return domain in set(_normalise(allowed_domains))


def _not_whitelisted() -> AuthorizationException:
    return AuthorizationException(
        message="Your account is not authorised for AURA",
        error_detail=ErrorDetail(
            title="Your account is not authorised for AURA",
            code="NOT_WHITELISTED",
            status=403,
            details=["Email is not on the extension allow-list"],
        ),
    )


def _enforce_allow_block_lists(email: str) -> None:
    """Apply §10 semantics:

    1. Blocklist (domain or email) is checked first — a blocked entry is
       rejected even if it would otherwise pass the allow-list.
    2. If both allow-lists are empty, allow all.
    3. Otherwise, the email must match either the email allow-list or the
       domain allow-list.
    """
    if _email_matches(email, extension_config.blocklist_emails):
        raise _not_whitelisted()
    if _domain_matches(email, extension_config.blocklist_domains):
        raise _not_whitelisted()

    allow_emails = _normalise(extension_config.allowlist_emails)
    allow_domains = _normalise(extension_config.allowlist_domains)
    if not allow_emails and not allow_domains:
        return
    if _email_matches(email, allow_emails) or _domain_matches(
        email, allow_domains
    ):
        return
    raise _not_whitelisted()


def _truncate_sub(sub: str) -> str:
    """For log lines — never log the full Google sub. Keep the first 4 and
    last 2 characters so collisions are still distinguishable in audits."""
    if len(sub) <= 8:
        return sub[:2] + "***"
    return f"{sub[:4]}***{sub[-2:]}"


class ExtensionAuthService:
    def __init__(
        self,
        install_repository: ExtensionInstallRepository,
        token_repository: ExtensionTokenRepository,
    ):
        self.install_repository = install_repository
        self.token_repository = token_repository

    async def register(
        self,
        *,
        google_access_token: str,
        body_email: str,
        body_sub: str,
        environment_payload: dict,
        client_extension_version: str | None,
    ) -> tuple[str, ExtensionToken, ExtensionInstall]:
        """Verify the Google access token, enforce allow/block lists, upsert
        the install row, and issue a fresh opaque install token.

        Returns `(cleartext_token, token_row, install_row)` so the caller can
        serialise both `token` and `expiresAt` without an extra round-trip.
        """
        info = await verify_register_identity(
            access_token=google_access_token,
            expected_sub=body_sub,
            expected_email=body_email,
            configured_audience=(
                extension_config.google_oauth.client_id or None
            ),
        )

        # Use the email Google returned — guards against a client sending a
        # well-formed but unexpected case/whitespace variant.
        canonical_email = info.email
        _enforce_allow_block_lists(canonical_email)

        install = await self.install_repository.get_by_google_sub(info.sub)
        if install is None:
            install = await self.install_repository.create(
                ExtensionInstall(
                    google_sub=info.sub,
                    email=canonical_email,
                    extension_version=client_extension_version,
                    environment_json=environment_payload,
                )
            )
            outcome = "created"
        else:
            await self.install_repository.update(
                install,
                {
                    "email": canonical_email,
                    "extension_version": client_extension_version,
                    "environment_json": environment_payload,
                },
            )
            outcome = "rotated"

        cleartext, token_row = await rotate_install_token(
            install.id,
            self.token_repository,
            revoke_reason="register",
        )

        logger.info(
            "extension register: install_id=%s domain=%s sub=%s outcome=%s",
            install.id,
            _domain_of(canonical_email) or "?",
            _truncate_sub(info.sub),
            outcome,
        )

        return cleartext, token_row, install

    async def renew(
        self, install: ExtensionInstall
    ) -> tuple[str, ExtensionToken]:
        cleartext, token_row = await rotate_install_token(
            install.id,
            self.token_repository,
            revoke_reason="renewed",
        )
        logger.info("extension renew: install_id=%s", install.id)
        return cleartext, token_row

    async def logout(self, raw_token: str) -> bool:
        """Idempotent server-side revocation. A second call with the same
        already-revoked token still resolves to success (the controller
        does not bubble a 401 here so the user never sees a sign-out
        'failure'). Returns ``True`` when the hash is known to us
        (active or already-revoked) and ``False`` when it was *never*
        valid — the controller raises 401 AUTH_FAILED in that case so the
        contract's "idempotent only for previously-valid tokens" rule is
        honoured.
        """
        token_hash = hash_install_token(raw_token)
        row = await self.token_repository.get_by_hash(token_hash)
        if row is None:
            return False
        if not row.is_revoked:
            await self.token_repository.revoke_by_hash(
                token_hash, reason="logout"
            )
        return True
