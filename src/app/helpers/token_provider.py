import hashlib
from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt

from src.configs import security
from src.app.models.token import Token
from src.app.models.enums import TokenType
from src.app.repositories.token_repository import TokenRepository
from src.shared.exceptions import AuthenticationException, InternalServerException
from src.shared.responses import ErrorDetail


def _require_secret() -> str:
    if not security.jwt.secret_key:
        raise InternalServerException(
            message="Server is misconfigured",
            error_detail=ErrorDetail(
                title="Misconfigured",
                code="JWT_SECRET_NOT_SET",
                status=500,
                details=["JWT_SECRET_KEY environment variable is not set"],
            ),
        )
    return security.jwt.secret_key


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _encode(payload: dict) -> str:
    return jwt.encode(payload, _require_secret(), algorithm=security.jwt.algorithm)


def _build_payload(
    user_id: UUID,
    token_type: str,
    expires_delta: timedelta,
    role: str | None = None,
) -> dict:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    if role:
        payload["role"] = role
    return payload


async def create_token_pair(user_id: UUID, role: str, token_repo: TokenRepository) -> dict:
    access_delta = timedelta(minutes=security.jwt.access_token_expire_minutes)
    refresh_delta = timedelta(days=security.jwt.refresh_token_expire_days)
    now = datetime.now(timezone.utc)

    access_payload = _build_payload(user_id, "access", access_delta, role=role)
    refresh_payload = _build_payload(user_id, "refresh", refresh_delta)

    access_token = _encode(access_payload)
    refresh_token = _encode(refresh_payload)

    await token_repo.create(Token(
        user_id=user_id,
        token_hash=_hash_token(access_token),
        token_type=TokenType.ACCESS,
        expires_at=now + access_delta,
    ))
    await token_repo.create(Token(
        user_id=user_id,
        token_hash=_hash_token(refresh_token),
        token_type=TokenType.REFRESH,
        expires_at=now + refresh_delta,
    ))

    return {"access_token": access_token, "refresh_token": refresh_token}


def decode_token(token: str, expected_type: str = "access") -> dict:
    try:
        payload = jwt.decode(
            token, _require_secret(), algorithms=[security.jwt.algorithm]
        )
    except jwt.ExpiredSignatureError:
        raise AuthenticationException(
            message="Your session has expired. Please log in again",
            error_detail=ErrorDetail(
                title="Token Expired", code="TOKEN_EXPIRED", status=401,
                details=["Token has expired"],
            ),
        )
    except jwt.InvalidTokenError:
        raise AuthenticationException(
            message="Invalid authentication token",
            error_detail=ErrorDetail(
                title="Invalid Token", code="INVALID_TOKEN", status=401,
                details=["Token is invalid or malformed"],
            ),
        )

    if payload.get("type") != expected_type:
        raise AuthenticationException(
            message="Invalid token type",
            error_detail=ErrorDetail(
                title="Invalid Token Type", code="INVALID_TOKEN_TYPE", status=401,
                details=[f"Expected {expected_type} token"],
            ),
        )

    return payload


async def verify_token(token: str, token_repo: TokenRepository, expected_type: str = "access") -> dict:
    payload = decode_token(token, expected_type)

    if not await token_repo.is_hash_valid(_hash_token(token)):
        raise AuthenticationException(
            message="This token has been revoked",
            error_detail=ErrorDetail(
                title="Token Revoked", code="TOKEN_REVOKED", status=401,
                details=["Token has been revoked"],
            ),
        )

    return payload
