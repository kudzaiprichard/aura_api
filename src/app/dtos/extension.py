"""DTOs for the Chrome extension surface.

Kept in a dedicated module so the dashboard DTO files stay focused. Wire
shapes mirror `docs/BACKEND_CONTRACT.md` and obey
`docs/EXTENSION_IMPLEMENTATION_STANDARD.md`. Casing rules:

* All extension DTOs use camelCase via `Field(alias=...)` — except the inner
  `prediction` block of `/emails/analyze` and the `model_version` field of
  `/health`, which the contract freezes as snake_case.
* Request DTOs use `extra="ignore"` so unknown fields the extension may add
  in future are silently dropped (per §12 / §5.5.1).
"""
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ── /api/v1/auth/extension/register ─────────────────────────────────────────


class ExtensionEnvironment(BaseModel):
    """Forensic Chrome environment payload — the backend stores it verbatim
    on the install row and never reads it back. Most fields are optional;
    only `extensionVersion` is required so we can correlate behaviour
    against extension releases."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    user_agent: str | None = Field(default=None, alias="userAgent")
    browser: dict[str, str] | None = None
    os: dict[str, str] | None = None
    language: str | None = None
    timezone: str | None = None
    extension_version: str = Field(alias="extensionVersion", min_length=1)


class ExtensionRegisterRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    email: EmailStr
    sub: str = Field(min_length=1)
    environment: ExtensionEnvironment


class ExtensionUserEcho(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    email: str
    sub: str


class ExtensionRegisterResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    token: str
    expires_at: int = Field(alias="expiresAt")
    user: ExtensionUserEcho


# ── /api/v1/auth/extension/renew ────────────────────────────────────────────


class ExtensionRenewResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    token: str
    expires_at: int = Field(alias="expiresAt")


# ── /api/v1/health ──────────────────────────────────────────────────────────


class ExtensionHealthResponse(BaseModel):
    """`model_version` is intentionally snake_case (BACKEND_CONTRACT §5.1)."""

    model_config = ConfigDict(populate_by_name=True)

    status: str
    name: str
    version: str
    model_version: str


# ── /api/v1/emails/analyze ──────────────────────────────────────────────────
# Request DTO — Gmail message shape produced by `utils/emailExtractor.js`.
# `extra="ignore"` lets the extension add fields without breaking us.


class GmailAuthResults(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    dkim: str | None = None
    spf: str | None = None
    dmarc: str | None = None


class GmailHeaders(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    from_: str | None = Field(default=None, alias="from")
    to: str | None = None
    cc: str | None = None
    bcc: str | None = None
    reply_to: str | None = Field(default=None, alias="replyTo")
    return_path: str | None = Field(default=None, alias="returnPath")
    subject: str | None = None
    date: str | None = None
    message_id_header: str | None = Field(
        default=None, alias="messageIdHeader"
    )
    dkim_signature: str | None = Field(default=None, alias="dkimSignature")
    list_unsubscribe: str | None = Field(default=None, alias="listUnsubscribe")
    x_originating_ip: str | None = Field(default=None, alias="xOriginatingIp")
    received: list[str] = Field(default_factory=list)
    auth_results: GmailAuthResults | None = Field(
        default=None, alias="authResults"
    )


class GmailBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    text: str = ""
    html: str = ""


class GmailAttachment(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str | None = None
    mime_type: str | None = Field(default=None, alias="mimeType")
    size: int | None = None


class ExtensionAnalyzeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    message_id: str = Field(alias="messageId", min_length=1, max_length=128)
    thread_id: str | None = Field(default=None, alias="threadId")
    label_ids: list[str] = Field(default_factory=list, alias="labelIds")
    snippet: str | None = None
    headers: GmailHeaders = Field(default_factory=GmailHeaders)
    body: GmailBody = Field(default_factory=GmailBody)
    urls: list[str] = Field(default_factory=list)
    attachments: list[GmailAttachment] = Field(default_factory=list)


class ExtensionEmailIdRef(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str


class ExtensionPrediction(BaseModel):
    """Prediction block — snake_case per BACKEND_CONTRACT §3, by design.

    No aliases declared; the attribute names are emitted as-is under
    `by_alias=True`. Kept separate from `PredictionResponse` (which serves
    the dashboard's camelCase wire) so the surfaces never accidentally fork.
    """

    model_config = ConfigDict(populate_by_name=True)

    predicted_label: str
    confidence_score: float
    phishing_probability: float
    legitimate_probability: float
    threshold_used: float
    should_alert: bool
    message: str | None = None
    email_id: str | None = None
    model_version: str


class ExtensionAnalysisResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    email: ExtensionEmailIdRef
    prediction: ExtensionPrediction


# ── /api/v1/extension/installs (admin surface — Step 6) ─────────────────────
# Defined here to keep all extension wire shapes in one place even though
# only auth + analyze are wired in Step 3.


class ExtensionInstallSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: str
    email: str
    google_sub: str = Field(alias="googleSub")
    status: str
    extension_version: str | None = Field(alias="extensionVersion")
    last_seen_at: str | None = Field(alias="lastSeenAt")
    blacklisted_at: str | None = Field(alias="blacklistedAt")
    created_at: str = Field(alias="createdAt")


class ExtensionInstallDetail(BaseModel):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: str
    email: str
    google_sub: str = Field(alias="googleSub")
    status: str
    extension_version: str | None = Field(alias="extensionVersion")
    environment: dict[str, Any] | None = None
    last_seen_at: str | None = Field(alias="lastSeenAt")
    blacklisted_at: str | None = Field(alias="blacklistedAt")
    blacklisted_by: str | None = Field(alias="blacklistedBy")
    blacklist_reason: str | None = Field(alias="blacklistReason")
    active_token_count: int = Field(alias="activeTokenCount")
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")


class ExtensionBlacklistRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    reason: str | None = Field(default=None, max_length=500)


class ExtensionDomainBlacklistRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    domain: str = Field(min_length=1, max_length=255)
    reason: str | None = Field(default=None, max_length=500)


class ExtensionBlacklistResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    install_id: str = Field(alias="installId")
    revoked_token_count: int = Field(alias="revokedTokenCount")


class ExtensionDomainBlacklistResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    domain: str
    blacklisted_install_count: int = Field(alias="blacklistedInstallCount")
    revoked_token_count: int = Field(alias="revokedTokenCount")


class ExtensionActivityEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: str
    occurred_at: str = Field(alias="occurredAt")
    predicted_label: str | None = Field(alias="predictedLabel")
    phishing_probability: float | None = Field(alias="phishingProbability")
    model_version: str | None = Field(alias="modelVersion")
