from typing import List, TYPE_CHECKING

from sqlalchemy import String, Boolean, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.models.enums import Role
from src.shared.database import BaseModel

if TYPE_CHECKING:
    from src.app.models.token import Token


class User(BaseModel):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    username: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[Role] = mapped_column(
        SAEnum(Role, name="role_enum"), nullable=False, default=Role.IT_ANALYST
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    tokens: Mapped[List["Token"]] = relationship(
        "Token",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
