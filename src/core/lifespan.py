import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from alembic.config import Config as AlembicConfig
from alembic import command as alembic_command

# Silence Alembic's verbose plugin-setup and migration INFO lines before
# anything else imports alembic so the lines never reach the root logger.
logging.getLogger("alembic").setLevel(logging.WARNING)

from fastapi import FastAPI
from src.shared.database import async_session
from src.shared.database.engine import engine
from src.configs import (
    inference as inference_config,
    logging as log_config,
    sse as sse_config,
)
from src.app.models.drift_event import (
    DriftEvent,
    EVENT_TYPE_CONFIRMATION,
    EVENT_TYPE_PREDICTION,
)
from src.app.helpers.auto_review_cache import AutoReviewCache
from src.app.repositories.drift_event_repository import DriftEventRepository
from src.app.services import (
    seed_admin,
    start_install_token_cleanup,
    start_token_cleanup,
)
from src.core.background_jobs import BackgroundJobRunner
from src.core.sse import SSEBroker
from src.shared.inference import (
    AutoReviewer,
    DriftMonitor,
    LLMProvider,
    PhishingDetector,
)


logger = logging.getLogger(__name__)


def _run_migrations() -> None:
    alembic_cfg = AlembicConfig("alembic.ini")
    alembic_cfg.attributes["configure_logger"] = False
    alembic_command.upgrade(alembic_cfg, "head")


def _setup_logging() -> None:
    log_dir = os.path.dirname(log_config.file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_level = getattr(logging, log_config.level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format=log_config.format,
        handlers=[
            logging.FileHandler(log_config.file_path),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _build_drift_monitor() -> DriftMonitor:
    log_path = inference_config.drift.log_path
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return DriftMonitor(
        log_path=log_path,
        fpr_threshold=inference_config.drift.fpr_threshold,
    )


def _build_detector(drift_monitor: DriftMonitor) -> PhishingDetector | None:
    models_dir = inference_config.models_dir
    try:
        loaded = PhishingDetector.load_production(models_root=models_dir)
    except FileNotFoundError as e:
        logger.warning(
            "Detector not loaded — no model versions under %s (%s). "
            "Prediction endpoints will return 503 until a version is registered.",
            models_dir, e,
        )
        return None

    return PhishingDetector(
        loaded.model,
        loaded.subject_vectorizer,
        loaded.body_vectorizer,
        version=loaded.version,
        calibrator=loaded.calibrator,
        review_low_threshold=inference_config.review.low_threshold,
        review_high_threshold=inference_config.review.high_threshold,
        drift_monitor=drift_monitor,
    )


def _parse_iso(ts: str | None) -> datetime:
    if not ts:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def _replay_jsonl(log_path: Path) -> Iterator[DriftEvent]:
    if not log_path.exists():
        return
    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = record.get("type")
            pid_raw = record.get("prediction_id")
            if not pid_raw:
                continue
            try:
                from uuid import UUID
                pid = UUID(str(pid_raw))
            except (TypeError, ValueError):
                continue
            occurred_at = _parse_iso(record.get("timestamp"))
            if kind == "prediction":
                try:
                    predicted_label = int(record["predicted_label"])
                except (KeyError, TypeError, ValueError):
                    continue
                yield DriftEvent(
                    event_type=EVENT_TYPE_PREDICTION,
                    prediction_id=pid,
                    predicted_label=predicted_label,
                    predicted_probability=float(
                        record.get("predicted_probability") or 0.0
                    ),
                    confirmed_label=None,
                    model_version=record.get("model_version") or None,
                    occurred_at=occurred_at,
                )
            elif kind == "confirmation":
                try:
                    confirmed_label = int(record["confirmed_label"])
                except (KeyError, TypeError, ValueError):
                    continue
                yield DriftEvent(
                    event_type=EVENT_TYPE_CONFIRMATION,
                    prediction_id=pid,
                    predicted_label=None,
                    predicted_probability=None,
                    confirmed_label=confirmed_label,
                    model_version=None,
                    occurred_at=occurred_at,
                )


def _count_jsonl(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    count = 0
    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") in ("prediction", "confirmation"):
                count += 1
    return count


async def _reconcile_drift_mirror(drift_monitor: DriftMonitor) -> None:
    log_path = drift_monitor.log_path
    expected = _count_jsonl(log_path)

    async with async_session() as session:
        async with session.begin():
            repo = DriftEventRepository(session)
            actual = await repo.count_total()

            if expected == actual:
                logger.info(
                    "Drift mirror reconciled — JSONL=%d SQL=%d (in sync)",
                    expected, actual,
                )
                return

            logger.warning(
                "Drift mirror divergence detected — JSONL=%d SQL=%d. "
                "Rebuilding SQL mirror from JSONL.",
                expected, actual,
            )
            await repo.truncate()
            inserted = await repo.bulk_insert(_replay_jsonl(log_path))
            logger.info(
                "Drift mirror rebuilt — inserted %d row(s) from %s",
                inserted, log_path,
            )


def _build_auto_reviewer() -> AutoReviewer | None:
    cfg = inference_config.auto_reviewer
    provider_name = (cfg.provider or "").strip().lower()
    api_key = (cfg.api_key or "").strip()
    if not provider_name or not api_key:
        logger.info(
            "Auto-reviewer disabled (provider=%r, api_key set=%s)",
            provider_name, bool(api_key),
        )
        return None

    try:
        provider = LLMProvider(provider_name)
    except ValueError:
        logger.warning(
            "Auto-reviewer disabled — unknown provider %r (expected one of %s)",
            provider_name, [p.value for p in LLMProvider],
        )
        return None

    return AutoReviewer(
        provider=provider,
        api_key=api_key,
        model_name=(cfg.model_name or None) or None,
        timeout_seconds=cfg.timeout_seconds,
        max_retries=cfg.max_retries,
    )


def _build_auto_review_cache() -> AutoReviewCache | None:
    cfg = inference_config.auto_reviewer
    if not bool(getattr(cfg, "cache_enabled", False)):
        return None
    return AutoReviewCache(
        max_size=int(getattr(cfg, "cache_max_size", 256)),
        ttl_seconds=int(getattr(cfg, "cache_ttl_seconds", 600)),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    _setup_logging()
    logger.info("Starting up — logging configured, DB pool initialised")

    # Run any pending migrations before anything touches the database.
    # Alembic is synchronous so we offload to a thread to avoid blocking
    # the event loop.
    logger.info("Running database migrations...")
    await asyncio.to_thread(_run_migrations)
    logger.info("Database migrations complete")

    await seed_admin()

    drift_monitor = _build_drift_monitor()
    detector = _build_detector(drift_monitor)
    auto_reviewer = _build_auto_reviewer()
    auto_review_cache = (
        _build_auto_review_cache() if auto_reviewer is not None else None
    )

    app.state.detector = detector
    app.state.drift_monitor = drift_monitor
    app.state.auto_reviewer = auto_reviewer
    app.state.auto_review_cache = auto_review_cache
    app.state.shadow_detector = None
    app.state.shadow_expires_at = None
    app.state.detector_lock = asyncio.Lock()
    app.state.training_lock = asyncio.Lock()
    app.state.background_jobs = BackgroundJobRunner()
    app.state.sse_broker = SSEBroker(
        subscriber_queue_max=sse_config.subscriber_queue_max,
        replay_window_size=sse_config.replay_window_size,
    )

    if detector is not None:
        logger.info("Detector loaded — version=%s", detector.version)
    logger.info(
        "Drift monitor loaded — log=%s fpr_threshold=%.3f",
        drift_monitor.log_path, drift_monitor.fpr_threshold,
    )
    logger.info(
        "Auto-reviewer %s",
        "ready" if auto_reviewer is not None else "disabled",
    )
    if auto_review_cache is not None:
        logger.info(
            "Auto-reviewer cache enabled — max_size=%d ttl_seconds=%d",
            auto_review_cache.max_size,
            auto_review_cache.ttl_seconds,
        )

    try:
        await _reconcile_drift_mirror(drift_monitor)
    except Exception:
        logger.exception("Drift mirror reconciliation failed; continuing")

    token_cleanup_task = asyncio.create_task(start_token_cleanup())
    install_token_cleanup_task = asyncio.create_task(
        start_install_token_cleanup()
    )

    yield

    # ── Shutdown ──
    for task in (token_cleanup_task, install_token_cleanup_task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    await app.state.background_jobs.stop()
    await app.state.sse_broker.stop()

    await engine.dispose()
    logger.info("Shutting down — DB pool disposed")