"""Phase 12 — signed-uploads helper.

When ``inference.upload.require_signature`` is on, every call to
``POST /api/v1/models/upload`` must carry an ``X-Upload-Signature`` form
field set to ``HMAC-SHA256(upload_hmac_secret, sha256_hex_of_bytes)``.
The endpoint rejects any request that does not verify.

The helper is intentionally small and self-contained so the signature
check can also be reused for any future signed-upload surface (benchmark
datasets, training CSVs) without pulling more infrastructure.
"""

from __future__ import annotations

import hmac
import hashlib
import logging

from src.configs import inference as inference_config
from src.shared.exceptions import BadRequestException, ServiceUnavailableException
from src.shared.responses import ErrorDetail


log = logging.getLogger("aura.upload.signing")


def signature_required() -> bool:
    cfg = getattr(inference_config, "upload", None)
    return bool(cfg is not None and getattr(cfg, "require_signature", False))


def _secret() -> str:
    cfg = getattr(inference_config, "upload", None)
    if cfg is None:
        return ""
    return str(getattr(cfg, "hmac_secret", "") or "")


def compute_signature(*, sha256_hex: str, secret: str | None = None) -> str:
    """HMAC-SHA256 hex digest of ``sha256_hex`` keyed by ``secret``.

    Exposed so admin tooling can produce the signature the server will
    verify. When ``secret`` is None we read ``inference.upload.hmac_secret``.
    """
    key = (secret if secret is not None else _secret()).encode("utf-8")
    return hmac.new(
        key, sha256_hex.lower().encode("utf-8"), hashlib.sha256
    ).hexdigest()


def verify_upload_signature(
    *, sha256_hex: str, provided_signature: str | None,
) -> None:
    """Verify the provided signature against the configured HMAC secret.

    No-ops when the feature flag is off. Raises ``BadRequestException`` on
    missing / mismatched signatures and ``ServiceUnavailableException`` when
    the feature is on but no secret is configured (a configuration error we
    want to flag loudly rather than silently accept everything).
    """
    if not signature_required():
        return
    secret = _secret()
    if not secret:
        raise ServiceUnavailableException(
            message="Upload signature is required but no HMAC secret is configured",
            error_detail=ErrorDetail(
                title="Service Misconfigured",
                code="UPLOAD_SIGNATURE_SECRET_MISSING",
                status=503,
                details=[
                    "Set AURA_UPLOAD_HMAC_SECRET or disable "
                    "AURA_UPLOAD_REQUIRE_SIGNATURE.",
                ],
            ),
        )
    if not provided_signature:
        raise BadRequestException(
            message="Upload signature missing",
            error_detail=ErrorDetail(
                title="Bad Request",
                code="UPLOAD_SIGNATURE_MISSING",
                status=400,
                details=[
                    "X-Upload-Signature (HMAC-SHA256 hex of the artefact "
                    "SHA-256) is required when signed uploads are enabled.",
                ],
            ),
        )
    expected = compute_signature(sha256_hex=sha256_hex, secret=secret)
    if not hmac.compare_digest(expected, provided_signature.lower()):
        log.warning(
            "upload_signature mismatch sha256=%s.. expected=%s.. got=%s..",
            sha256_hex[:12], expected[:12], provided_signature[:12],
        )
        raise BadRequestException(
            message="Upload signature invalid",
            error_detail=ErrorDetail(
                title="Bad Request",
                code="UPLOAD_SIGNATURE_INVALID",
                status=400,
                details=["HMAC-SHA256 signature did not match the artefact."],
            ),
        )
