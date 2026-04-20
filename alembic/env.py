import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from src.configs import database
from src.shared.database.base_model import Base

# Import your models here so Alembic can detect them
from src.app.models.user import User  # noqa: F401
from src.app.models.token import Token  # noqa: F401
from src.app.models.prediction_event import PredictionEvent  # noqa: F401
from src.app.models.review_item import ReviewItem  # noqa: F401
from src.app.models.review_escalation import ReviewEscalation  # noqa: F401
from src.app.models.training_buffer_item import TrainingBufferItem  # noqa: F401
from src.app.models.auto_review_invocation import AutoReviewInvocation  # noqa: F401
from src.app.models.drift_event import DriftEvent  # noqa: F401
from src.app.models.training_run import TrainingRun  # noqa: F401
from src.app.models.model_activation import ModelActivation  # noqa: F401
from src.app.models.model_threshold_history import ModelThresholdHistory  # noqa: F401
from src.app.models.benchmark_dataset import BenchmarkDataset  # noqa: F401
from src.app.models.benchmark_dataset_row import BenchmarkDatasetRow  # noqa: F401
from src.app.models.model_benchmark import ModelBenchmark  # noqa: F401
from src.app.models.model_benchmark_version_result import ModelBenchmarkVersionResult  # noqa: F401
from src.app.models.review_disagreement import ReviewDisagreement  # noqa: F401
from src.app.models.extension_install import ExtensionInstall  # noqa: F401
from src.app.models.extension_token import ExtensionToken  # noqa: F401


config = context.config

# Set the database URL from our config system
config.set_main_option("sqlalchemy.url", database.url)

# Only configure logging when Alembic is run directly from the CLI.
# When called programmatically via alembic.command (e.g. from lifespan.py),
# configure_logger=False is passed through config.attributes so we skip
# fileConfig entirely and leave the application's logging setup untouched.
if config.attributes.get("configure_logger", True):
    if config.config_file_name is not None:
        fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in offline mode (generates SQL without DB connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in online mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()