import uuid
from typing import Any, TYPE_CHECKING

from sqlalchemy import Enum as SAEnum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.models.enums import ModelActivationKind
from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.user import User


class ModelActivation(BaseModel):
    __tablename__ = "model_activations"

    kind: Mapped[ModelActivationKind] = mapped_column(
        SAEnum(ModelActivationKind, name="model_activation_kind_enum"),
        nullable=False,
    )
    version: Mapped[str] = mapped_column(String(16), nullable=False)
    previous_version: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )
    actor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    actor: Mapped["User | None"] = relationship(
        "User", foreign_keys=[actor_id], lazy="select"
    )
