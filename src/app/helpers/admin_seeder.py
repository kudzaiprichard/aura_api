import logging

from src.shared.database import async_session
from src.app.models.user import User
from src.app.models.enums import Role
from src.app.repositories.user_repository import UserRepository
from src.app.helpers.password_hasher import hash_password
from src.configs import security

logger = logging.getLogger(__name__)


async def seed_admin() -> None:
    async with async_session() as session:
        async with session.begin():
            repo = UserRepository(session)

            if await repo.exists(role=Role.ADMIN):
                logger.info("Admin user already exists — skipping seed")
                return

            if not security.admin.password:
                logger.warning(
                    "ADMIN_PASSWORD not set — skipping admin seed. "
                    "Set ADMIN_PASSWORD and restart to seed the default admin."
                )
                return

            if await repo.exists(email=security.admin.email):
                logger.warning(
                    "Cannot seed admin: email %s is already taken by a non-admin user",
                    security.admin.email,
                )
                return

            if await repo.exists(username=security.admin.username):
                logger.warning(
                    "Cannot seed admin: username %s is already taken by a non-admin user",
                    security.admin.username,
                )
                return

            admin = User(
                email=security.admin.email,
                username=security.admin.username,
                first_name=security.admin.first_name,
                last_name=security.admin.last_name,
                password_hash=hash_password(security.admin.password),
                role=Role.ADMIN,
            )

            await repo.create(admin)
            logger.info("Default admin user created (%s)", security.admin.email)
