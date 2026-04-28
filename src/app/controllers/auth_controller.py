from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials

from src.configs import server
from src.core.rate_limit import limiter
from src.shared.responses import ApiResponse
from src.app.models.user import User
from src.app.services.auth_service import AuthService
from src.app.dependencies import (
    bearer_scheme,
    get_auth_service,
    get_current_user,
)
from src.app.dtos.requests import (
    RegisterRequest,
    LoginRequest,
    RefreshTokenRequest,
    UpdateProfileRequest,
    ChangePasswordRequest,
)
from src.app.dtos.responses import (
    UserResponse,
    TokenResponse,
    AuthResponse,
)

router = APIRouter()


@router.post("/register", status_code=201)
@limiter.limit(server.rate_limit.register)
async def register(
    request: Request,
    body: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    user, tokens = await auth_service.register(
        email=body.email,
        username=body.username,
        first_name=body.first_name,
        last_name=body.last_name,
        password=body.password,
    )
    return ApiResponse.ok(
        value=AuthResponse(
            user=UserResponse.from_user(user),
            tokens=TokenResponse(
                accessToken=tokens["access_token"],
                refreshToken=tokens["refresh_token"],
            ),
        ),
        message="Registration successful",
    )


@router.post("/login")
@limiter.limit(server.rate_limit.login)
async def login(
    request: Request,
    body: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    user, tokens = await auth_service.login(email=body.email, password=body.password)
    return ApiResponse.ok(
        value=AuthResponse(
            user=UserResponse.from_user(user),
            tokens=TokenResponse(
                accessToken=tokens["access_token"],
                refreshToken=tokens["refresh_token"],
            ),
        ),
        message="Login successful",
    )


@router.post("/refresh")
@limiter.limit(server.rate_limit.refresh)
async def refresh_token(
    request: Request,
    body: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    tokens = await auth_service.refresh_token(body.refresh_token)
    return ApiResponse.ok(
        value=TokenResponse(
            accessToken=tokens["access_token"],
            refreshToken=tokens["refresh_token"],
        ),
        message="Tokens refreshed",
    )


@router.post("/logout", status_code=200)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    auth_service: AuthService = Depends(get_auth_service),
):
    await auth_service.logout(credentials.credentials)
    return ApiResponse.ok(value=None, message="Logged out successfully")


@router.get("/me")
async def get_profile(current_user: User = Depends(get_current_user)):
    return ApiResponse.ok(value=UserResponse.from_user(current_user))


@router.patch("/me")
async def update_profile(
    body: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    updated = await auth_service.update_profile(
        user=current_user,
        first_name=body.first_name,
        last_name=body.last_name,
        username=body.username,
    )
    return ApiResponse.ok(value=UserResponse.from_user(updated), message="Profile updated")


@router.patch("/me/password", status_code=200)
async def change_password(
    body: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    await auth_service.change_password(
        user=current_user,
        current_password=body.current_password,
        new_password=body.new_password,
    )
    return ApiResponse.ok(
        value=None,
        message="Password changed. Please log in again.",
    )
