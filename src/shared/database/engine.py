from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from src.configs import database

engine = create_async_engine(
    database.url,
    echo=database.echo,
    pool_size=database.pool_size,
    max_overflow=database.max_overflow,
    pool_timeout=database.pool_timeout,
    pool_pre_ping=database.pool_pre_ping,
    pool_recycle=database.pool_recycle,
)

async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
