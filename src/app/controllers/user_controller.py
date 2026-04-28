from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, Query

from src.shared.responses import ApiResponse, PaginatedResponse
from src.shared.database.pagination import PaginationParams, get_pagination
from src.app.models.enums import Role
from src.app.services.user_management_service import UserManagementService
from src.app.dependencies import (
    get_user_management_service,
    require_admin,
)
from src.app.dtos.requests import (
    CreateUserRequest,
    UpdateUserRequest,
    ResetPasswordRequest,
)
from src.app.dtos.responses import UserResponse

router = APIRouter(dependencies=[Depends(require_admin)])


@router.get("", response_model=PaginatedResponse[UserResponse])
async def get_users(
    pagination: PaginationParams = Depends(get_pagination),
    role: Optional[Role] = Query(None),
    is_active: Optional[bool] = Query(None, alias="isActive"),
    service: UserManagementService = Depends(get_user_management_service),
):
    users, total = await service.get_users(
        page=pagination.page,
        page_size=pagination.page_size,
        role=role,
        is_active=is_active,
    )
    return PaginatedResponse.ok(
        value=[UserResponse.from_user(u) for u in users],
        page=pagination.page,
        total=total,
        page_size=pagination.page_size,
    )


@router.get("/{user_id}")
async def get_user(
    user_id: UUID,
    service: UserManagementService = Depends(get_user_management_service),
):
    user = await service.get_user(user_id)
    return ApiResponse.ok(value=UserResponse.from_user(user))


@router.post("", status_code=201)
async def create_user(
    body: CreateUserRequest,
    service: UserManagementService = Depends(get_user_management_service),
):
    user = await service.create_user(
        email=body.email,
        username=body.username,
        first_name=body.first_name,
        last_name=body.last_name,
        password=body.password,
        role=body.role,
    )
    return ApiResponse.ok(value=UserResponse.from_user(user), message="User created")


@router.patch("/{user_id}")
async def update_user(
    user_id: UUID,
    body: UpdateUserRequest,
    service: UserManagementService = Depends(get_user_management_service),
):
    user = await service.update_user(
        user_id=user_id,
        first_name=body.first_name,
        last_name=body.last_name,
        username=body.username,
        role=body.role,
        is_active=body.is_active,
    )
    return ApiResponse.ok(value=UserResponse.from_user(user), message="User updated")


@router.delete("/{user_id}", status_code=200)
async def delete_user(
    user_id: UUID,
    service: UserManagementService = Depends(get_user_management_service),
):
    await service.delete_user(user_id)
    return ApiResponse.ok(value=None, message="User deleted")


@router.post("/{user_id}/deactivate", status_code=200)
async def deactivate_user(
    user_id: UUID,
    service: UserManagementService = Depends(get_user_management_service),
):
    user = await service.set_active_status(user_id, is_active=False)
    return ApiResponse.ok(value=UserResponse.from_user(user), message="User deactivated")


@router.post("/{user_id}/activate", status_code=200)
async def activate_user(
    user_id: UUID,
    service: UserManagementService = Depends(get_user_management_service),
):
    user = await service.set_active_status(user_id, is_active=True)
    return ApiResponse.ok(value=UserResponse.from_user(user), message="User activated")


@router.post("/{user_id}/reset-password", status_code=200)
async def reset_user_password(
    user_id: UUID,
    body: ResetPasswordRequest,
    service: UserManagementService = Depends(get_user_management_service),
):
    user = await service.reset_password(user_id, new_password=body.new_password)
    return ApiResponse.ok(
        value=UserResponse.from_user(user),
        message="Password reset. The user must log in again.",
    )
