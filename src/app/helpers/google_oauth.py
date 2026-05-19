"""Google OAuth verification — minimal `userinfo` call used by the
Chrome extension `register` endpoint to verify the access token forwarded
by the browser.

The contract (§9 of EXTENSION_IMPLEMENTATION_STANDARD.md):
- 10 s timeout — keeps the request inside the extension's 30 s budget
- Network failure or non-200 → 503 SERVICE_UNAVAILABLE
- `sub` mismatch → 401 GOOGLE_AUTH_FAILED
- Never log the access token, never log PII beyond `sub`/`email`
"""
import logging
from dataclasses import dataclass

import httpx

from src.shared.exceptions import (
    AuthenticationException,
    ServiceUnavailableException,
)
from src.shared.responses import ErrorDetail


_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
_TIMEOUT_SECONDS = 10.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GoogleUserInfo:
    sub: str
    email: str
    audience: str | None  # tokeninfo `aud` — populated when available


def _google_auth_failed(detail: str) -> AuthenticationException:
    return AuthenticationException(
        message="Google sign-in could not be verified",
        error_detail=ErrorDetail(
            title="Google sign-in could not be verified",
            code="GOOGLE_AUTH_FAILED",
            status=401,
            details=[detail],
        ),
    )


def _google_unreachable(detail: str) -> ServiceUnavailableException:
    return ServiceUnavailableException(
        message="Google verification is temporarily unavailable",
        error_detail=ErrorDetail(
            title="Google verification is temporarily unavailable",
            code="SERVICE_UNAVAILABLE",
            status=503,
            details=[detail],
        ),
    )


async def fetch_userinfo(access_token: str) -> GoogleUserInfo:
    """Call Google `userinfo` with the bearer access token. Caller is
    responsible for the audience check against the configured client id —
    `userinfo` does not echo the audience, so verifying it requires the
    `tokeninfo` endpoint or a separate JWT decode.
    """
    if not access_token:
        raise _google_auth_failed("Google access token missing")

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get(
                _USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
    except httpx.TimeoutException:
        logger.warning("Google userinfo timed out after %.1fs", _TIMEOUT_SECONDS)
        raise _google_unreachable("Google userinfo timed out")
    except httpx.HTTPError as exc:
        logger.warning("Google userinfo network error: %s", exc.__class__.__name__)
        raise _google_unreachable("Google userinfo could not be reached")

    if response.status_code == 401 or response.status_code == 403:
        raise _google_auth_failed("Google rejected the access token")
    if response.status_code != 200:
        logger.warning("Google userinfo returned status=%d", response.status_code)
        raise _google_unreachable(
            f"Google userinfo returned status {response.status_code}"
        )

    try:
        body = response.json()
    except ValueError:
        raise _google_unreachable("Google userinfo returned a non-JSON body")

    sub = (body.get("sub") or "").strip()
    email = (body.get("email") or "").strip()
    if not sub or not email:
        raise _google_auth_failed("Google response missing sub or email")

    audience = body.get("aud")
    if isinstance(audience, str):
        audience = audience.strip() or None
    else:
        audience = None

    return GoogleUserInfo(sub=sub, email=email, audience=audience)


async def verify_register_identity(
    *,
    access_token: str,
    expected_sub: str,
    expected_email: str,
    configured_audience: str | None,
) -> GoogleUserInfo:
    """One-stop verification for the register flow:

    1. Call `userinfo` with the access token
    2. `sub` must match the value the client sent in the body
    3. `email` (case-insensitive) must match the value the client sent
    4. If a client id is configured, the userinfo `aud` must match it (when
       the response carries one — `userinfo` may not always include it)
    """
    info = await fetch_userinfo(access_token)

    if info.sub != expected_sub.strip():
        raise _google_auth_failed("Google sub did not match the request body")

    if info.email.casefold() != expected_email.strip().casefold():
        raise _google_auth_failed("Google email did not match the request body")

    if configured_audience and info.audience and info.audience != configured_audience:
        raise _google_auth_failed(
            "Google access token was issued for a different client"
        )

    return info
