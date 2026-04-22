import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import String, Boolean, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database import BaseModel
from src.app.models.enums import TokenType

if TYPE_CHECKING:
    from src.app.models.user import User


class Token(BaseModel):
    __tablename__ = "tokens"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # SHA-256 hex digest of the raw JWT — never store the JWT itself.
    token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    token_type: Mapped[TokenType] = mapped_column(
        SAEnum(TokenType, name="token_type_enum"), nullable=False
    )
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="tokens")

    @property
    def is_valid(self) -> bool:
        return not self.is_revoked and self.expires_at > datetime.now(timezone.utc)

    def revoke(self) -> None:
        self.is_revoked = True
