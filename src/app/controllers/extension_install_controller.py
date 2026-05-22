"""Admin surface for `/api/v1/extension/installs` (§13).

Dashboard-facing — every route is gated by the existing `require_admin`
guard. Distinct from the extension's own auth + analyse routers because
this surface speaks the dashboard's camelCase envelope and never accepts
install bearer tokens.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from src.shared.database.pagination import PaginationParams, get_pagination
from src.shared.responses import ApiResponse, PaginatedResponse
from src.app.dependencies import (
    get_current_user,
    get_extension_install_admin_service,
    require_admin,
)
from src.app.dtos.extension import (
    ExtensionActivityEntry,
    ExtensionBlacklistRequest,
    ExtensionBlacklistResult,
    ExtensionDomainBlacklistRequest,
    ExtensionDomainBlacklistResult,
    ExtensionInstallDetail,
    ExtensionInstallSummary,
)
from src.app.models.enums import ExtensionInstallStatus
from src.app.models.user import User
from src.app.services.extension_install_admin_service import (
    ExtensionInstallAdminService,
)


router = APIRouter(dependencies=[Depends(require_admin)])


@router.get("", response_model=PaginatedResponse[ExtensionInstallSummary])
async def list_installs(
    pagination: PaginationParams = Depends(get_pagination),
    email: Optional[str] = Query(None, max_length=255),
    domain: Optional[str] = Query(None, max_length=255),
    status: Optional[ExtensionInstallStatus] = Query(None),
    extension_version: Optional[str] = Query(
        None, alias="extensionVersion", max_length=64
    ),
    last_seen_after: Optional[datetime] = Query(None, alias="lastSeenAfter"),
    last_seen_before: Optional[datetime] = Query(None, alias="lastSeenBefore"),
    service: ExtensionInstallAdminService = Depends(
        get_extension_install_admin_service
    ),
):
    items, total = await service.list_installs(
        page=pagination.page,
        page_size=pagination.page_size,
        email=email,
        domain=domain,
        status=status,
        extension_version=extension_version,
        last_seen_after=last_seen_after,
        last_seen_before=last_seen_before,
    )
    return PaginatedResponse.ok(
        value=items,
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


# Domain-blacklist must be declared *before* `/{install_id}` so FastAPI
# does not try to coerce the literal "domains" into a UUID.
@router.post(
    "/domains/blacklist",
    response_model=ApiResponse[ExtensionDomainBlacklistResult],
)
async def blacklist_domain(
    body: ExtensionDomainBlacklistRequest,
    actor: User = Depends(get_current_user),
    service: ExtensionInstallAdminService = Depends(
        get_extension_install_admin_service
    ),
):
    result = await service.blacklist_domain(
        domain=body.domain,
        actor_id=actor.id,
        reason=body.reason,
    )
    return ApiResponse.ok(value=result, message="Domain blacklisted")


@router.get(
    "/{install_id}", response_model=ApiResponse[ExtensionInstallDetail]
)
async def get_install(
    install_id: UUID,
    service: ExtensionInstallAdminService = Depends(
        get_extension_install_admin_service
    ),
):
    detail = await service.get_install_detail(install_id)
    return ApiResponse.ok(value=detail)


@router.post(
    "/{install_id}/blacklist",
    response_model=ApiResponse[ExtensionBlacklistResult],
)
async def blacklist_install(
    install_id: UUID,
    body: ExtensionBlacklistRequest,
    actor: User = Depends(get_current_user),
    service: ExtensionInstallAdminService = Depends(
        get_extension_install_admin_service
    ),
):
    result = await service.blacklist(
        install_id, actor_id=actor.id, reason=body.reason
    )
    return ApiResponse.ok(value=result, message="Install blacklisted")


@router.post(
    "/{install_id}/unblacklist",
    response_model=ApiResponse[ExtensionInstallDetail],
)
async def unblacklist_install(
    install_id: UUID,
    service: ExtensionInstallAdminService = Depends(
        get_extension_install_admin_service
    ),
):
    detail = await service.unblacklist(install_id)
    return ApiResponse.ok(value=detail, message="Install unblacklisted")


@router.post(
    "/{install_id}/revoke-tokens",
    response_model=ApiResponse[ExtensionBlacklistResult],
)
async def revoke_install_tokens(
    install_id: UUID,
    body: ExtensionBlacklistRequest,
    service: ExtensionInstallAdminService = Depends(
        get_extension_install_admin_service
    ),
):
    result = await service.revoke_tokens(install_id, reason=body.reason)
    return ApiResponse.ok(value=result, message="Install tokens revoked")


@router.get(
    "/{install_id}/activity",
    response_model=PaginatedResponse[ExtensionActivityEntry],
)
async def list_install_activity(
    install_id: UUID,
    pagination: PaginationParams = Depends(get_pagination),
    service: ExtensionInstallAdminService = Depends(
        get_extension_install_admin_service
    ),
):
    items, total = await service.list_activity(
        install_id,
        page=pagination.page,
        page_size=pagination.page_size,
    )
    return PaginatedResponse.ok(
        value=items,
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )
