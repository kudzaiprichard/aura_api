"""Chrome extension auth endpoints — register / renew / logout.

Mounted alongside the dashboard `auth_router` under `/api/v1/auth/extension`
so the extension surface is clearly segregated from the dashboard JWT flow.
"""
from fastapi import APIRouter, Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials

from src.configs import server
from src.core.rate_limit import limiter
from src.shared.exceptions import AuthenticationException, ValidationException
from src.shared.responses import ApiResponse, ErrorDetail
from src.app.dependencies import (
    extension_bearer_scheme,
    get_extension_auth_service,
    require_install,
)
from src.app.dtos.extension import (
    ExtensionRegisterRequest,
    ExtensionRegisterResponse,
    ExtensionRenewResponse,
    ExtensionUserEcho,
)
from src.app.models.extension_install import ExtensionInstall
from src.app.services.extension_auth_service import ExtensionAuthService


router = APIRouter()


def _epoch_ms(dt) -> int:
    """Return a UTC datetime as epoch milliseconds (per BACKEND_CONTRACT §4)."""
    return int(dt.timestamp() * 1000)


def _require_google_access_token(header_value: str | None) -> str:
    if not header_value or not header_value.strip():
        raise ValidationException(
            message="X-Google-Access-Token header is required",
            error_detail=ErrorDetail(
                title="Missing Google access token",
                code="VALIDATION_ERROR",
                status=400,
                details=["X-Google-Access-Token header is required"],
            ),
        )
    return header_value.strip()


@router.post("/extension/register")
@limiter.limit(server.rate_limit.extension_register)
async def register_extension(
    request: Request,
    body: ExtensionRegisterRequest,
    x_google_access_token: str | None = Header(
        default=None, alias="X-Google-Access-Token"
    ),
    service: ExtensionAuthService = Depends(get_extension_auth_service),
):
    google_access_token = _require_google_access_token(x_google_access_token)
    environment_payload = body.environment.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    cleartext, token_row, install = await service.register(
        google_access_token=google_access_token,
        body_email=body.email,
        body_sub=body.sub,
        environment_payload=environment_payload,
        client_extension_version=body.environment.extension_version,
    )
    return ApiResponse.ok(
        value=ExtensionRegisterResponse(
            token=cleartext,
            expiresAt=_epoch_ms(token_row.expires_at),
            user=ExtensionUserEcho(email=install.email, sub=install.google_sub),
        ),
        message="Registration successful",
    )


@router.post("/extension/renew")
@limiter.limit(server.rate_limit.extension_renew)
async def renew_extension(
    request: Request,
    install: ExtensionInstall = Depends(require_install),
    service: ExtensionAuthService = Depends(get_extension_auth_service),
):
    cleartext, token_row = await service.renew(install)
    return ApiResponse.ok(
        value=ExtensionRenewResponse(
            token=cleartext,
            expiresAt=_epoch_ms(token_row.expires_at),
        ),
        message="Tokens refreshed",
    )


def _auth_failed_logout(detail: str) -> AuthenticationException:
    return AuthenticationException(
        message="Authentication failed",
        error_detail=ErrorDetail(
            title="Authentication failed",
            code="AUTH_FAILED",
            status=401,
            details=[detail],
        ),
    )


@router.post("/extension/logout")
async def logout_extension(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        extension_bearer_scheme
    ),
    service: ExtensionAuthService = Depends(get_extension_auth_service),
):
    """Idempotent logout — §5.4 of BACKEND_CONTRACT.md.

    A valid or already-revoked bearer token returns 200 so the sign-out UX
    never shows a false error for a token the server has ever issued. A
    syntactically malformed header or a bearer string that the server has
    never seen resolves to 401 AUTH_FAILED — the verify standard's
    "idempotent only applies to previously-valid tokens" rule.
    """
    if credentials is None or not credentials.credentials:
        raise _auth_failed_logout("Install token missing")
    if (credentials.scheme or "").lower() != "bearer":
        raise _auth_failed_logout("Install token missing")
    recognised = await service.logout(credentials.credentials)
    if not recognised:
        raise _auth_failed_logout("Install token was never valid")
    return ApiResponse.ok(value=None, message="Logged out successfully")
