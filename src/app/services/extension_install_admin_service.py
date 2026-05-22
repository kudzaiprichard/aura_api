"""Admin surface for `/api/v1/extension/installs` (§13).

Kept distinct from `UserManagementService` because the extension install is
a separate identity (Chrome install + Google sub) from a dashboard `User`.
The admin role guard is enforced at the controller layer; this service
trusts that the caller has already passed `require_admin`.
"""
from datetime import datetime, timezone
from typing import List, Sequence, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.exceptions import BadRequestException, NotFoundException
from src.shared.responses import ErrorDetail
from src.app.dtos.extension import (
    ExtensionActivityEntry,
    ExtensionBlacklistResult,
    ExtensionDomainBlacklistResult,
    ExtensionInstallDetail,
    ExtensionInstallSummary,
)
from src.app.models.enums import ExtensionInstallStatus
from src.app.models.extension_install import ExtensionInstall
from src.app.models.prediction_event import PredictionEvent
from src.app.repositories.extension_install_repository import (
    ExtensionInstallRepository,
)
from src.app.repositories.extension_token_repository import (
    ExtensionTokenRepository,
)
from src.app.repositories.prediction_event_repository import (
    PredictionEventRepository,
)
from src.app.services.extension_email_service import label_from_event_fields


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _summary(install: ExtensionInstall) -> ExtensionInstallSummary:
    return ExtensionInstallSummary(
        id=str(install.id),
        email=install.email,
        googleSub=install.google_sub,
        status=install.status.value,
        extensionVersion=install.extension_version,
        lastSeenAt=_iso(install.last_seen_at),
        blacklistedAt=_iso(install.blacklisted_at),
        createdAt=_iso(install.created_at) or "",
    )


def _detail(
    install: ExtensionInstall, *, active_token_count: int
) -> ExtensionInstallDetail:
    return ExtensionInstallDetail(
        id=str(install.id),
        email=install.email,
        googleSub=install.google_sub,
        status=install.status.value,
        extensionVersion=install.extension_version,
        environment=install.environment_json,
        lastSeenAt=_iso(install.last_seen_at),
        blacklistedAt=_iso(install.blacklisted_at),
        blacklistedBy=str(install.blacklisted_by)
        if install.blacklisted_by is not None
        else None,
        blacklistReason=install.blacklist_reason,
        activeTokenCount=active_token_count,
        createdAt=_iso(install.created_at) or "",
        updatedAt=_iso(install.updated_at) or "",
    )


def _activity(event: PredictionEvent) -> ExtensionActivityEntry:
    """Project a `prediction_events` row onto the admin activity wire shape.

    The on-wire `predictedLabel` is the extension's tri-label
    (REVIEW/SPAM/NOT_SPAM), derived from the stored `confidence_zone` and
    binary `predicted_label` — same precedence as the live extension
    response path.
    """
    return ExtensionActivityEntry(
        id=str(event.id),
        occurredAt=_iso(event.predicted_at) or "",
        predictedLabel=label_from_event_fields(
            event.confidence_zone, event.predicted_label
        ),
        phishingProbability=float(event.phishing_probability),
        modelVersion=event.model_version,
    )


class ExtensionInstallAdminService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        install_repository: ExtensionInstallRepository,
        token_repository: ExtensionTokenRepository,
        prediction_event_repository: PredictionEventRepository,
    ):
        self.session = session
        self.install_repo = install_repository
        self.token_repo = token_repository
        self.prediction_event_repo = prediction_event_repository

    # ── List / detail ──

    async def list_installs(
        self,
        *,
        page: int,
        page_size: int,
        email: str | None,
        domain: str | None,
        status: ExtensionInstallStatus | None,
        extension_version: str | None,
        last_seen_after: datetime | None,
        last_seen_before: datetime | None,
    ) -> Tuple[List[ExtensionInstallSummary], int]:
        installs, total = await self.install_repo.paginate_filtered(
            page=page,
            page_size=page_size,
            email=email,
            domain=domain,
            status=status,
            extension_version=extension_version,
            last_seen_after=last_seen_after,
            last_seen_before=last_seen_before,
        )
        return [_summary(i) for i in installs], total

    async def get_install_detail(
        self, install_id: UUID
    ) -> ExtensionInstallDetail:
        install = await self._require_install(install_id)
        active_token_count = await self.token_repo.count_active_for_install(
            install.id
        )
        return _detail(install, active_token_count=active_token_count)

    # ── Blacklist / unblacklist / token revoke ──

    async def blacklist(
        self,
        install_id: UUID,
        *,
        actor_id: UUID,
        reason: str | None,
    ) -> ExtensionBlacklistResult:
        """Atomic: flip status to BLACKLISTED *and* revoke every active
        token in the same transaction. `get_db` owns the surrounding
        transaction so either both writes commit or neither does."""
        install = await self._require_install(install_id)

        if install.status != ExtensionInstallStatus.BLACKLISTED:
            await self.install_repo.blacklist(
                install,
                actor_id=actor_id,
                reason=reason,
                when=datetime.now(timezone.utc),
            )

        revoked = await self.token_repo.revoke_all_for_install(
            install.id, reason=reason or "blacklisted"
        )
        return ExtensionBlacklistResult(
            installId=str(install.id),
            revokedTokenCount=int(revoked),
        )

    async def unblacklist(
        self, install_id: UUID
    ) -> ExtensionInstallDetail:
        """Reverse the BLACKLISTED flag — does NOT reissue tokens. The
        user must re-register through the normal flow (§13)."""
        install = await self._require_install(install_id)
        if install.status == ExtensionInstallStatus.BLACKLISTED:
            await self.install_repo.unblacklist(install)
        active_token_count = await self.token_repo.count_active_for_install(
            install.id
        )
        return _detail(install, active_token_count=active_token_count)

    async def revoke_tokens(
        self, install_id: UUID, *, reason: str | None
    ) -> ExtensionBlacklistResult:
        """Bulk revoke without flipping status. The install can re-register
        normally to obtain a new token."""
        install = await self._require_install(install_id)
        revoked = await self.token_repo.revoke_all_for_install(
            install.id, reason=reason or "admin revoke"
        )
        return ExtensionBlacklistResult(
            installId=str(install.id),
            revokedTokenCount=int(revoked),
        )

    # ── Domain blacklist ──

    async def blacklist_domain(
        self,
        *,
        domain: str,
        actor_id: UUID,
        reason: str | None,
    ) -> ExtensionDomainBlacklistResult:
        """Loop every active install on `domain`, blacklist + revoke under
        the caller's transaction. Throws BAD_REQUEST when `domain` looks
        malformed so we don't accidentally pattern-match the entire table."""
        normalised = (domain or "").strip().lower()
        if not normalised or "@" in normalised or "/" in normalised:
            raise BadRequestException(
                message="Invalid domain",
                error_detail=ErrorDetail(
                    title="Invalid domain",
                    code="BAD_REQUEST",
                    status=400,
                    details=[
                        "Domain must be a bare hostname like example.com"
                    ],
                ),
            )

        installs = await self.install_repo.find_active_by_domain(normalised)
        when = datetime.now(timezone.utc)
        total_revoked = 0
        for install in installs:
            await self.install_repo.blacklist(
                install,
                actor_id=actor_id,
                reason=reason,
                when=when,
            )
            total_revoked += await self.token_repo.revoke_all_for_install(
                install.id, reason=reason or "domain blacklist"
            )

        return ExtensionDomainBlacklistResult(
            domain=normalised,
            blacklistedInstallCount=len(installs),
            revokedTokenCount=int(total_revoked),
        )

    # ── Activity feed ──

    async def list_activity(
        self,
        install_id: UUID,
        *,
        page: int,
        page_size: int,
    ) -> Tuple[List[ExtensionActivityEntry], int]:
        await self._require_install(install_id)
        events, total = await self.prediction_event_repo.paginate_for_install(
            install_id=install_id, page=page, page_size=page_size
        )
        return [_activity(e) for e in events], total

    # ── Internal ──

    async def _require_install(self, install_id: UUID) -> ExtensionInstall:
        install = await self.install_repo.get_by_id(install_id)
        if install is None:
            raise NotFoundException(
                message="Extension install not found",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="INSTALL_NOT_FOUND",
                    status=404,
                    details=[f"No install found with id {install_id}"],
                ),
            )
        return install
