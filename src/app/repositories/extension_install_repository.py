from datetime import datetime
from typing import Sequence, Tuple
from uuid import UUID

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import BaseRepository
from src.app.models.enums import ExtensionInstallStatus
from src.app.models.extension_install import ExtensionInstall


class ExtensionInstallRepository(BaseRepository[ExtensionInstall]):
    def __init__(self, session: AsyncSession):
        super().__init__(ExtensionInstall, session)

    async def get_by_google_sub(
        self, google_sub: str
    ) -> ExtensionInstall | None:
        stmt = select(ExtensionInstall).where(
            ExtensionInstall.google_sub == google_sub
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def touch_last_seen(
        self, install_id: UUID, *, when: datetime
    ) -> None:
        """Bump `last_seen_at` without firing `onupdate=now()` semantics
        elsewhere — the protected-route dependency uses this on every call,
        so a single UPDATE keeps the write path cheap."""
        stmt = (
            update(ExtensionInstall)
            .where(ExtensionInstall.id == install_id)
            .values(last_seen_at=when)
            .execution_options(synchronize_session=False)
        )
        await self.session.execute(stmt)

    async def blacklist(
        self,
        install: ExtensionInstall,
        *,
        actor_id: UUID | None,
        reason: str | None,
        when: datetime,
    ) -> ExtensionInstall:
        install.status = ExtensionInstallStatus.BLACKLISTED
        install.blacklisted_at = when
        install.blacklisted_by = actor_id
        install.blacklist_reason = reason
        await self.session.flush()
        return install

    async def unblacklist(
        self, install: ExtensionInstall
    ) -> ExtensionInstall:
        install.status = ExtensionInstallStatus.ACTIVE
        install.blacklisted_at = None
        install.blacklisted_by = None
        install.blacklist_reason = None
        await self.session.flush()
        return install

    # ── Admin list / domain queries (§13) ──

    def _apply_admin_filters(
        self,
        stmt,
        *,
        email: str | None,
        domain: str | None,
        status: ExtensionInstallStatus | None,
        extension_version: str | None,
        last_seen_after: datetime | None,
        last_seen_before: datetime | None,
    ):
        if email:
            stmt = stmt.where(ExtensionInstall.email.ilike(f"%{email}%"))
        if domain:
            # Match the email domain regardless of case. We anchor on `@`
            # so a query for "acme.com" cannot match "evil-acme.com".
            stmt = stmt.where(
                ExtensionInstall.email.ilike(f"%@{domain}")
            )
        if status is not None:
            stmt = stmt.where(ExtensionInstall.status == status)
        if extension_version:
            stmt = stmt.where(
                ExtensionInstall.extension_version == extension_version
            )
        if last_seen_after is not None:
            stmt = stmt.where(
                ExtensionInstall.last_seen_at >= last_seen_after
            )
        if last_seen_before is not None:
            stmt = stmt.where(
                ExtensionInstall.last_seen_at <= last_seen_before
            )
        return stmt

    async def paginate_filtered(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        email: str | None = None,
        domain: str | None = None,
        status: ExtensionInstallStatus | None = None,
        extension_version: str | None = None,
        last_seen_after: datetime | None = None,
        last_seen_before: datetime | None = None,
    ) -> Tuple[Sequence[ExtensionInstall], int]:
        base = self._apply_admin_filters(
            select(ExtensionInstall),
            email=email,
            domain=domain,
            status=status,
            extension_version=extension_version,
            last_seen_after=last_seen_after,
            last_seen_before=last_seen_before,
        )

        count_stmt = select(func.count()).select_from(base.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = (
            base.order_by(desc(ExtensionInstall.created_at))
            .offset(offset)
            .limit(page_size)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), int(total)

    async def find_active_by_domain(
        self, domain: str
    ) -> Sequence[ExtensionInstall]:
        """Every non-blacklisted install whose email lives on `domain`.

        Used by the bulk-domain blacklist path so we can revoke tokens and
        flip the status row-by-row in the same transaction. We skip
        already-blacklisted installs to keep the operation idempotent.
        """
        stmt = select(ExtensionInstall).where(
            ExtensionInstall.email.ilike(f"%@{domain}"),
            ExtensionInstall.status != ExtensionInstallStatus.BLACKLISTED,
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
