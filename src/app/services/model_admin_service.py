"""Model-version administration (§3.8, §3.16, Phase 9).

Wraps `ModelRegistry` / `PhishingDetector` so activate / promote / rollback /
threshold-update / metrics live behind the FastAPI layer. The detector
singleton held on `app.state.detector` is swapped atomically under
`app.state.detector_lock` — the lock is held only around the registry write +
fresh-detector construction + pointer swap, so in-flight requests finish
against the pre-swap detector and the next request picks up the new one.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs import inference as inference_config
from src.app.helpers.pickle_sandbox import (
    sandbox_enabled as pickle_sandbox_enabled,
    sandboxed_joblib_load,
)
from src.app.helpers.upload_signing import (
    signature_required,
    verify_upload_signature,
)
from src.app.models.enums import ModelActivationKind, Role
from src.app.models.training_run import TrainingRun
from src.app.models.user import User
from src.app.repositories.model_activation_repository import (
    ModelActivationRepository,
)
from src.app.repositories.model_threshold_history_repository import (
    ModelThresholdHistoryRepository,
)
from src.app.repositories.prediction_event_repository import (
    PredictionEventRepository,
)
from src.shared.exceptions import (
    BadRequestException,
    ConflictException,
    NotFoundException,
    ServiceUnavailableException,
)
from src.shared.inference import DriftMonitor, PhishingDetector
from src.shared.inference.registry import ModelRegistry
from src.shared.responses import ErrorDetail


log = logging.getLogger("aura.models.admin")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelAdminService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        activation_repository: ModelActivationRepository,
        threshold_history_repository: ModelThresholdHistoryRepository,
        prediction_event_repository: PredictionEventRepository,
        detector_lock: asyncio.Lock,
        drift_monitor: DriftMonitor,
        app_state: Any,
    ):
        self.session = session
        self.activation_repo = activation_repository
        self.threshold_repo = threshold_history_repository
        self.prediction_repo = prediction_event_repository
        self.detector_lock = detector_lock
        self.drift_monitor = drift_monitor
        self.app_state = app_state

    # ── helpers ──

    def _registry(self) -> ModelRegistry:
        return ModelRegistry(inference_config.models_dir)

    def _current_detector(self) -> PhishingDetector | None:
        return getattr(self.app_state, "detector", None)

    def _current_version(self) -> str | None:
        detector = self._current_detector()
        if detector is not None and detector.version is not None:
            return detector.version
        return self._registry().active_version()

    def _version_entry(
        self, registry: ModelRegistry, version: str
    ) -> dict[str, Any]:
        meta = registry._read_registry_metadata()
        return dict(meta.get("versions", {}).get(version, {}))

    def _require_known_version(
        self, registry: ModelRegistry, version: str
    ) -> None:
        if version not in registry.list_versions():
            raise NotFoundException(
                message="Unknown model version",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="MODEL_VERSION_UNKNOWN",
                    status=404,
                    details=[
                        f"version={version!r} is not registered; "
                        f"known versions: {registry.list_versions()}"
                    ],
                ),
            )

    # ── queries ──

    async def list_versions(self) -> list[dict[str, Any]]:
        registry = self._registry()
        active = registry.active_version()
        meta = registry._read_registry_metadata()
        versions_meta = meta.get("versions", {})
        out: list[dict[str, Any]] = []
        for version in registry.list_versions():
            entry = versions_meta.get(version, {})
            out.append(
                {
                    "version": version,
                    "active": version == active,
                    "promoted": bool(entry.get("promoted", False)),
                    "metrics": dict(entry.get("metrics") or {}),
                    "sha256": entry.get("sha256"),
                    "source_version": entry.get("source_version"),
                    "calibrator_sha256": entry.get("calibrator_sha256"),
                }
            )
        return out

    async def get_version(self, version: str) -> dict[str, Any]:
        registry = self._registry()
        self._require_known_version(registry, version)
        entry = self._version_entry(registry, version)
        try:
            paths = registry.paths_for(version)
        except FileNotFoundError as exc:
            raise ServiceUnavailableException(
                message="Version artefacts missing",
                error_detail=ErrorDetail(
                    title="Service Unavailable",
                    code="MODEL_ARTEFACT_MISSING",
                    status=503,
                    details=[str(exc)],
                ),
            ) from exc

        current_thresholds = await self.threshold_repo.current_for_version(
            version
        )
        return {
            "version": version,
            "active": registry.active_version() == version,
            "promoted": bool(entry.get("promoted", False)),
            "metrics": dict(entry.get("metrics") or {}),
            "sha256": entry.get("sha256"),
            "calibrator_sha256": entry.get("calibrator_sha256"),
            "source_version": entry.get("source_version"),
            "paths": {
                key: (str(path) if path is not None else None)
                for key, path in paths.items()
            },
            "current_thresholds": (
                {
                    "decisionThreshold": current_thresholds.decision_threshold,
                    "reviewLowThreshold": current_thresholds.review_low_threshold,
                    "reviewHighThreshold": current_thresholds.review_high_threshold,
                    "effectiveFrom": current_thresholds.effective_from.isoformat(),
                    "setBy": (
                        str(current_thresholds.set_by)
                        if current_thresholds.set_by is not None
                        else None
                    ),
                }
                if current_thresholds is not None
                else None
            ),
        }

    # ── commands ──

    async def activate(
        self, *, version: str, actor: User, reason: str | None = None
    ) -> dict[str, Any]:
        registry = self._registry()
        self._require_known_version(registry, version)

        active = registry.active_version()
        if active == version and self._current_detector() is not None:
            raise ConflictException(
                message="Version is already active",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="MODEL_VERSION_ALREADY_ACTIVE",
                    status=409,
                    details=[f"version={version} is already active"],
                ),
            )

        previous_version = self._current_version()

        try:
            new_detector = await self._swap_detector(
                registry=registry, version=version
            )
        except ValueError as exc:
            # registry.set_active raises ValueError on integrity mismatch /
            # unknown version — surface as 409 so the caller learns the swap
            # was refused without touching the singleton.
            raise ConflictException(
                message="Model integrity check failed",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="MODEL_INTEGRITY_MISMATCH",
                    status=409,
                    details=[str(exc)],
                ),
            ) from exc
        except FileNotFoundError as exc:
            raise ServiceUnavailableException(
                message="Version artefacts missing",
                error_detail=ErrorDetail(
                    title="Service Unavailable",
                    code="MODEL_ARTEFACT_MISSING",
                    status=503,
                    details=[str(exc)],
                ),
            ) from exc

        record = await self.activation_repo.record(
            kind=ModelActivationKind.ACTIVATE,
            version=version,
            previous_version=previous_version,
            actor_id=actor.id,
            reason=reason,
            metrics_snapshot=None,
        )
        log.info(
            "model_activate version=%s previous=%s actor_id=%s detector_version=%s",
            version, previous_version, actor.id, new_detector.version,
        )
        return {
            "activationId": record.id,
            "version": version,
            "previousVersion": previous_version,
            "detectorVersion": new_detector.version,
            "actorId": actor.id,
        }

    async def promote(
        self,
        *,
        version: str,
        actor: User,
        reason: str | None = None,
    ) -> dict[str, Any]:
        registry = self._registry()
        self._require_known_version(registry, version)

        entry = self._version_entry(registry, version)
        metrics = dict(entry.get("metrics") or {})
        if not metrics:
            raise ConflictException(
                message="Cannot promote without recorded metrics",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="MODEL_PROMOTE_NO_METRICS",
                    status=409,
                    details=[
                        f"version={version} has no metrics in the registry; "
                        "promotion requires a metrics snapshot (§2.7)",
                    ],
                ),
            )

        previous_active = registry.active_version()

        # Promotion flips registry.active_version too; swap the detector under
        # the same lock so /predict sees the promoted version immediately.
        try:
            new_detector = await self._swap_detector(
                registry=registry, version=version, promote_metrics=metrics
            )
        except ValueError as exc:
            raise ConflictException(
                message="Model integrity check failed",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="MODEL_INTEGRITY_MISMATCH",
                    status=409,
                    details=[str(exc)],
                ),
            ) from exc
        except FileNotFoundError as exc:
            raise ServiceUnavailableException(
                message="Version artefacts missing",
                error_detail=ErrorDetail(
                    title="Service Unavailable",
                    code="MODEL_ARTEFACT_MISSING",
                    status=503,
                    details=[str(exc)],
                ),
            ) from exc

        # Stamp the originating training_runs row (if any) as promoted.
        await self.session.execute(
            update(TrainingRun)
            .where(TrainingRun.new_version == version)
            .where(TrainingRun.promoted.is_(False))
            .values(promoted=True, promoted_at=_now(), promoted_by=actor.id)
        )

        record = await self.activation_repo.record(
            kind=ModelActivationKind.PROMOTE,
            version=version,
            previous_version=previous_active,
            actor_id=actor.id,
            reason=reason,
            metrics_snapshot=metrics,
        )
        log.info(
            "model_promote version=%s previous=%s actor_id=%s",
            version, previous_active, actor.id,
        )
        return {
            "activationId": record.id,
            "version": version,
            "previousVersion": previous_active,
            "detectorVersion": new_detector.version,
            "metricsSnapshot": metrics,
            "actorId": actor.id,
        }

    async def rollback(
        self, *, actor: User, reason: str | None = None
    ) -> dict[str, Any]:
        registry = self._registry()
        current = registry.active_version()
        if current is None:
            raise ConflictException(
                message="No active version to roll back",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="MODEL_ROLLBACK_NO_ACTIVE",
                    status=409,
                    details=["The registry has no active version set"],
                ),
            )

        prior = await self.activation_repo.latest_activate_or_promote(
            exclude_version=current
        )
        if prior is None:
            raise ConflictException(
                message="No prior version to roll back to",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="MODEL_ROLLBACK_NO_PRIOR",
                    status=409,
                    details=[
                        "No prior ACTIVATE/PROMOTE entry exists for another "
                        "version; cannot determine the rollback target",
                    ],
                ),
            )

        target_version = prior.version
        self._require_known_version(registry, target_version)

        try:
            new_detector = await self._swap_detector(
                registry=registry, version=target_version
            )
        except ValueError as exc:
            raise ConflictException(
                message="Model integrity check failed",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="MODEL_INTEGRITY_MISMATCH",
                    status=409,
                    details=[str(exc)],
                ),
            ) from exc
        except FileNotFoundError as exc:
            raise ServiceUnavailableException(
                message="Version artefacts missing",
                error_detail=ErrorDetail(
                    title="Service Unavailable",
                    code="MODEL_ARTEFACT_MISSING",
                    status=503,
                    details=[str(exc)],
                ),
            ) from exc

        # §3.16: stamp training_runs whose new_version is no longer active.
        await self.session.execute(
            update(TrainingRun)
            .where(TrainingRun.new_version == current)
            .where(TrainingRun.rolled_back_at.is_(None))
            .values(rolled_back_at=_now())
        )

        record = await self.activation_repo.record(
            kind=ModelActivationKind.ROLLBACK,
            version=target_version,
            previous_version=current,
            actor_id=actor.id,
            reason=reason,
            metrics_snapshot=None,
        )
        log.info(
            "model_rollback from=%s to=%s actor_id=%s",
            current, target_version, actor.id,
        )
        return {
            "activationId": record.id,
            "version": target_version,
            "previousVersion": current,
            "detectorVersion": new_detector.version,
            "actorId": actor.id,
        }

    async def set_thresholds(
        self,
        *,
        version: str,
        decision_threshold: float,
        review_low_threshold: float | None,
        review_high_threshold: float | None,
        actor: User,
    ) -> dict[str, Any]:
        if actor.role != Role.ADMIN:
            # Defence-in-depth; the router already enforces require_admin.
            raise BadRequestException(
                message="Only admins can set thresholds",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="MODEL_THRESHOLD_FORBIDDEN",
                    status=400,
                    details=["actor role must be ADMIN"],
                ),
            )
        if not 0.0 <= decision_threshold <= 1.0:
            raise BadRequestException(
                message="decisionThreshold out of range",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="MODEL_THRESHOLD_OUT_OF_RANGE",
                    status=400,
                    details=[
                        f"decisionThreshold={decision_threshold!r} is not in [0, 1]"
                    ],
                ),
            )
        if (review_low_threshold is None) ^ (review_high_threshold is None):
            raise BadRequestException(
                message="Review thresholds must be provided together",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="MODEL_REVIEW_THRESHOLDS_PAIRED",
                    status=400,
                    details=[
                        "reviewLowThreshold and reviewHighThreshold must both "
                        "be null or both be numbers",
                    ],
                ),
            )
        if review_low_threshold is not None and review_high_threshold is not None:
            for name, value in (
                ("reviewLowThreshold", review_low_threshold),
                ("reviewHighThreshold", review_high_threshold),
            ):
                if not 0.0 <= value <= 1.0:
                    raise BadRequestException(
                        message=f"{name} out of range",
                        error_detail=ErrorDetail(
                            title="Bad Request",
                            code="MODEL_REVIEW_THRESHOLD_OUT_OF_RANGE",
                            status=400,
                            details=[f"{name}={value!r} is not in [0, 1]"],
                        ),
                    )
            if review_low_threshold >= review_high_threshold:
                raise BadRequestException(
                    message="reviewLowThreshold must be below reviewHighThreshold",
                    error_detail=ErrorDetail(
                        title="Bad Request",
                        code="MODEL_REVIEW_THRESHOLDS_INVERTED",
                        status=400,
                        details=[
                            f"low={review_low_threshold!r} >= high="
                            f"{review_high_threshold!r}",
                        ],
                    ),
                )

        registry = self._registry()
        self._require_known_version(registry, version)

        now = _now()
        await self.threshold_repo.close_current(version=version, closed_at=now)
        row = await self.threshold_repo.open(
            version=version,
            decision_threshold=float(decision_threshold),
            review_low_threshold=(
                float(review_low_threshold)
                if review_low_threshold is not None
                else None
            ),
            review_high_threshold=(
                float(review_high_threshold)
                if review_high_threshold is not None
                else None
            ),
            set_by=actor.id,
            effective_from=now,
        )

        # Live-apply to the running detector when the thresholds target the
        # active version — callers that set thresholds on a non-active version
        # only affect history (the swap code re-reads history on load in a
        # future phase; for now we only mutate the live detector here).
        detector = self._current_detector()
        if (
            detector is not None
            and detector.version == version
            and review_low_threshold is not None
            and review_high_threshold is not None
        ):
            async with self.detector_lock:
                detector.review_low_threshold = float(review_low_threshold)
                detector.review_high_threshold = float(review_high_threshold)

        log.info(
            "model_threshold_update version=%s decision=%.4f review=[%s, %s] "
            "actor_id=%s",
            version, decision_threshold,
            review_low_threshold, review_high_threshold, actor.id,
        )
        return {
            "historyId": row.id,
            "version": version,
            "decisionThreshold": row.decision_threshold,
            "reviewLowThreshold": row.review_low_threshold,
            "reviewHighThreshold": row.review_high_threshold,
            "effectiveFrom": row.effective_from,
            "setBy": row.set_by,
        }

    async def metrics(
        self,
        *,
        version: str,
        bucket: str,
        date_from: datetime | None,
        date_to: datetime | None,
        timezone_name: str,
    ) -> dict[str, Any]:
        registry = self._registry()
        self._require_known_version(registry, version)
        rows = await self.prediction_repo.bucketed_metrics_for_versions(
            versions=[version],
            bucket=bucket,
            date_from=date_from,
            date_to=date_to,
            timezone_name=timezone_name,
        )
        return {
            "version": version,
            "bucket": bucket,
            "timezone": timezone_name,
            "buckets": rows,
        }

    async def compare(
        self,
        *,
        versions: Sequence[str],
        bucket: str,
        date_from: datetime | None,
        date_to: datetime | None,
        timezone_name: str,
    ) -> dict[str, Any]:
        registry = self._registry()
        if not versions:
            raise BadRequestException(
                message="versions is required",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="MODEL_COMPARE_NO_VERSIONS",
                    status=400,
                    details=["Pass at least one version to compare"],
                ),
            )
        known = set(registry.list_versions())
        unknown = [v for v in versions if v not in known]
        if unknown:
            raise NotFoundException(
                message="Unknown model version(s)",
                error_detail=ErrorDetail(
                    title="Not Found",
                    code="MODEL_VERSION_UNKNOWN",
                    status=404,
                    details=[
                        f"Not registered: {unknown}; known: {sorted(known)}"
                    ],
                ),
            )
        rows = await self.prediction_repo.bucketed_metrics_for_versions(
            versions=list(versions),
            bucket=bucket,
            date_from=date_from,
            date_to=date_to,
            timezone_name=timezone_name,
        )
        grouped: dict[str, list[dict[str, Any]]] = {v: [] for v in versions}
        for row in rows:
            grouped.setdefault(row["version"], []).append(row)
        return {
            "versions": list(versions),
            "bucket": bucket,
            "timezone": timezone_name,
            "series": grouped,
        }

    async def upload(
        self,
        *,
        actor: User,
        filename: str,
        content: bytes,
        expected_sha256: str,
        source_version: str,
        metrics: dict[str, float] | None = None,
        signature: str | None = None,
    ) -> dict[str, Any]:
        """Phase 9 §7.6 stub: validate + SHA-256 verify + register.

        Phase 12 hardening layers on two optional checks before the stub
        returns: HMAC-SHA256 signature verification (``inference.upload.
        require_signature``) and an unpickle smoke test in a short-lived
        subprocess (``inference.upload.pickle_sandbox_enabled``). Both are
        feature-flagged off by default so existing flows are unaffected.
        """
        if not filename:
            raise BadRequestException(
                message="filename is required",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="MODEL_UPLOAD_NO_FILENAME",
                    status=400,
                    details=["Upload must carry a filename"],
                ),
            )
        actual_sha = hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha256.lower():
            raise ConflictException(
                message="SHA-256 mismatch on uploaded artefact",
                error_detail=ErrorDetail(
                    title="Conflict",
                    code="MODEL_UPLOAD_SHA_MISMATCH",
                    status=409,
                    details=[
                        f"expected={expected_sha256[:12]}.., "
                        f"actual={actual_sha[:12]}..",
                    ],
                ),
            )

        # Phase 12 — HMAC signature verification (no-op when the feature
        # flag is off). Runs before any further processing so a forged or
        # missing signature short-circuits the rest of the pipeline.
        verify_upload_signature(
            sha256_hex=actual_sha, provided_signature=signature,
        )

        registry = self._registry()
        if source_version not in registry.list_versions():
            raise BadRequestException(
                message="Unknown source_version",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="MODEL_UPLOAD_UNKNOWN_SOURCE",
                    status=400,
                    details=[
                        f"source_version={source_version!r}; known: "
                        f"{registry.list_versions()}"
                    ],
                ),
            )

        sandbox_summary: str | None = None
        if pickle_sandbox_enabled():
            result = await sandboxed_joblib_load(content)
            if not result.ok:
                detail = (
                    result.message
                    or (
                        "Sandbox timed out"
                        if result.timed_out
                        else "Sandbox exited non-zero"
                    )
                )
                log.warning(
                    "model_upload sandbox rejected filename=%s sha256=%s.. "
                    "timed_out=%s rc=%s message=%s",
                    filename, actual_sha[:12], result.timed_out,
                    result.returncode, detail,
                )
                raise BadRequestException(
                    message="Uploaded artefact failed pickle sandbox",
                    error_detail=ErrorDetail(
                        title="Bad Request",
                        code="MODEL_UPLOAD_SANDBOX_REJECTED",
                        status=400,
                        details=[detail],
                    ),
                )
            sandbox_summary = (
                f"sandbox_ok type={result.type_name} module={result.module}"
            )

        # Full pipeline deferred per spec. Returning an accepted stub here
        # keeps the contract honest: the caller sees that the SHA matched
        # and that nothing was registered yet.
        log.info(
            "model_upload verified filename=%s sha256=%s.. actor_id=%s "
            "signed=%s sandboxed=%s (registry-register deferred — stub "
            "pipeline per §7.6)",
            filename, actual_sha[:12], actor.id,
            signature_required(), pickle_sandbox_enabled(),
        )
        message = (
            "Artefact SHA-256 verified; full pipeline (unpickle + "
            "register) deferred per §7.6 stub acceptance."
        )
        if sandbox_summary:
            message = f"{message} {sandbox_summary}."
        return {
            "accepted": True,
            "filename": filename,
            "sha256": actual_sha,
            "sourceVersion": source_version,
            "registered": False,
            "message": message,
        }

    # ── internals ──

    async def _swap_detector(
        self,
        *,
        registry: ModelRegistry,
        version: str,
        promote_metrics: dict[str, float] | None = None,
    ) -> PhishingDetector:
        """Atomically swap `app.state.detector` to `version`.

        Mirrors `_build_detector` in lifespan — the new detector is
        constructed through the full constructor so the review thresholds
        and the drift monitor get re-attached (load_production does not
        take them).

        When `promote_metrics` is provided, `registry.promote` is used
        instead of `set_active` so the registry flags `promoted=true` +
        records the metrics snapshot atomically with the pointer flip.
        """
        async with self.detector_lock:
            if promote_metrics is not None:
                registry.promote(version, promote_metrics)
            else:
                registry.set_active(version, verify_integrity=True)

            loaded = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: PhishingDetector.load_production(
                    models_root=inference_config.models_dir
                ),
            )
            new_detector = PhishingDetector(
                loaded.model,
                loaded.subject_vectorizer,
                loaded.body_vectorizer,
                version=loaded.version,
                calibrator=loaded.calibrator,
                review_low_threshold=inference_config.review.low_threshold,
                review_high_threshold=inference_config.review.high_threshold,
                drift_monitor=self.drift_monitor,
            )

            # Phase 12 — shadow predictions. Capture the prior detector and
            # give it a TTL before swapping to the new one. /predict picks
            # this up and records a shadow verdict alongside the live one.
            # Failures here must never block the swap — log and continue.
            self._maybe_capture_shadow(
                prior=getattr(self.app_state, "detector", None),
                new_version=loaded.version,
            )

            self.app_state.detector = new_detector
            return new_detector

    def _maybe_capture_shadow(
        self,
        *,
        prior: PhishingDetector | None,
        new_version: str | None,
    ) -> None:
        shadow_cfg = getattr(inference_config, "shadow", None)
        if shadow_cfg is None or not bool(
            getattr(shadow_cfg, "enabled", False)
        ):
            self.app_state.shadow_detector = None
            self.app_state.shadow_expires_at = None
            return
        if prior is None:
            self.app_state.shadow_detector = None
            self.app_state.shadow_expires_at = None
            return
        if prior.version is not None and prior.version == new_version:
            # No-op swap — nothing to shadow.
            return
        try:
            days = int(getattr(shadow_cfg, "days", 7))
        except (TypeError, ValueError):
            days = 7
        if days <= 0:
            self.app_state.shadow_detector = None
            self.app_state.shadow_expires_at = None
            return
        expires_at = _now() + timedelta(days=days)
        self.app_state.shadow_detector = prior
        self.app_state.shadow_expires_at = expires_at
        log.info(
            "shadow_detector captured version=%s expires_at=%s",
            prior.version, expires_at.isoformat(),
        )

    async def recent_activations(
        self, *, version: str, page: int = 1, page_size: int = 20
    ):
        return await self.activation_repo.paginate_for_version(
            version=version, page=page, page_size=page_size
        )
