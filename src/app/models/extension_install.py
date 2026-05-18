import uuid
from datetime import datetime
from typing import List, TYPE_CHECKING

from sqlalchemy import String, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database import BaseModel
from src.app.models.enums import ExtensionInstallStatus

if TYPE_CHECKING:
    from src.app.models.extension_token import ExtensionToken


class ExtensionInstall(BaseModel):
    __tablename__ = "extension_installs"

    google_sub: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[ExtensionInstallStatus] = mapped_column(
        SAEnum(ExtensionInstallStatus, name="extension_install_status_enum"),
        nullable=False,
        default=ExtensionInstallStatus.ACTIVE,
    )
    extension_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    environment_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    blacklisted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    blacklisted_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    blacklist_reason: Mapped[str | None] = mapped_column(
        String(500), nullable=True
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    tokens: Mapped[List["ExtensionToken"]] = relationship(
        "ExtensionToken",
        back_populates="install",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
