"""Dashboard aggregation (§9.1 / §9.2, Phase 11).

Read-only roll-ups over every table built across Phases 2–10. No new tables,
no mutations, no write locks. The service composes a handful of small repo
queries per endpoint rather than one monolithic join so slow lanes can be
replaced with materialised views later (§3.15) without reshaping the
response DTO.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.configs import inference as inference_config
from src.app.dtos.responses import (
    ActiveModelSummary,
    AdminDashboardResponse,
    AnalystDashboardResponse,
    AnalystWorkloadEntry,
    AutoReviewUsageInvocation,
    AutoReviewUsageSummary,
    ClassificationBreakdown,
    ClassificationBreakdownPoint,
    DriftSignalSummary,
    MyQueueSummary,
    PersonalStats,
    PersonalStatsWindow,
    PredictionVolumePoint,
    PredictionVolumeSeries,
    QueueLabelMix,
    RecentActivityEntry,
    RecentActivityRef,
    RecentPredictionEntry,
    RecentSubmissionEntry,
    ReviewQueueHealth,
    TrainingBufferSummary,
    _UserRef,
)
from src.app.models.auto_review_invocation import AutoReviewInvocation
from src.app.models.enums import (
    AutoReviewAgreement,
    ModelActivationKind,
    ReviewItemStatus,
    TrainingRunStatus,
)
from src.app.models.model_activation import ModelActivation
from src.app.models.review_item import ReviewItem
from src.app.models.token import Token
from src.app.models.training_run import TrainingRun
from src.app.models.user import User
from src.app.repositories.auto_review_invocation_repository import (
    AutoReviewInvocationRepository,
)
from src.app.repositories.model_activation_repository import (
    ModelActivationRepository,
)
from src.app.repositories.prediction_event_repository import (
    PredictionEventRepository,
)
from src.app.repositories.review_item_repository import ReviewItemRepository
from src.app.repositories.token_repository import TokenRepository
from src.app.repositories.training_buffer_repository import (
    TrainingBufferRepository,
)
from src.app.repositories.training_run_repository import TrainingRunRepository
from src.app.repositories.user_repository import UserRepository
from src.app.services.training_buffer_service import (
    BufferStatus,
    TrainingBufferService,
)
from src.shared.exceptions import BadRequestException
from src.shared.inference import DriftMonitor, PhishingDetector
from src.shared.inference.registry import ModelRegistry
from src.shared.responses import ErrorDetail


_SUBJECT_PREVIEW_CHARS = 120
# The `recentActivity` feed spans several tables; pulling a fixed lookback
# window keeps per-source queries cheap while still covering the usual
# deployment cadence. 7 days matches the default `predictionVolume` range.
_RECENT_ACTIVITY_LOOKBACK_DAYS = 7


@dataclass
class AdminDashboardQuery:
    recent_limit: int = 20
    activity_limit: int = 20
    breakdown_days: int = 14
    volume_bucket: str = "day"
    volume_range_hours: int = 168
    sla_seconds: int | None = None
    timezone: str = "UTC"


@dataclass
class AnalystDashboardQuery:
    recent_limit: int = 20
    invocation_limit: int = 10
    range_days: int = 30
    sla_seconds: int | None = None
    timezone: str = "UTC"


def _resolve_timezone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise BadRequestException(
            message="Invalid timezone",
            error_detail=ErrorDetail(
                title="Bad Request",
                code="DASHBOARD_TIMEZONE_INVALID",
                status=400,
                details=[f"{tz_name!r} is not a recognised IANA tz id"],
            ),
        ) from exc


def _today_start_utc(tz: ZoneInfo) -> datetime:
    now_local = datetime.now(tz)
    local_midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return local_midnight.astimezone(timezone.utc)


def _week_start_utc(tz: ZoneInfo) -> datetime:
    """Monday 00:00 in the caller's tz, returned in UTC.

    Matches the ISO week definition so the `thisWeek` bucket lines up with
    most dashboards users already see in other tools."""
    now_local = datetime.now(tz)
    midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    monday_local = midnight - timedelta(days=midnight.weekday())
    return monday_local.astimezone(timezone.utc)


def _safe_preview(text: str | None, *, limit: int = _SUBJECT_PREVIEW_CHARS) -> str:
    if not text:
        return ""
    trimmed = text.strip()
    if len(trimmed) <= limit:
        return trimmed
    return trimmed[:limit]


class DashboardService:
    """Composes dashboard responses from the per-domain repositories.

    All writes elsewhere flush-only and commit at the request boundary; this
    service performs strictly read-only work, so there is no transaction
    ownership story here beyond the session injected by `get_db`.
    """

    def __init__(
        self,
        *,
        prediction_event_repository: PredictionEventRepository,
        review_item_repository: ReviewItemRepository,
        auto_review_invocation_repository: AutoReviewInvocationRepository,
        training_buffer_repository: TrainingBufferRepository,
        training_buffer_service: TrainingBufferService,
        training_run_repository: TrainingRunRepository,
        model_activation_repository: ModelActivationRepository,
        token_repository: TokenRepository,
        user_repository: UserRepository,
        detector: PhishingDetector | None,
        drift_monitor: DriftMonitor,
        models_dir: str,
    ):
        self.prediction_repo = prediction_event_repository
        self.review_repo = review_item_repository
        self.auto_review_repo = auto_review_invocation_repository
        self.buffer_repo = training_buffer_repository
        self.buffer_service = training_buffer_service
        self.training_run_repo = training_run_repository
        self.activation_repo = model_activation_repository
        self.token_repo = token_repository
        self.user_repo = user_repository
        self.detector = detector
        self.drift_monitor = drift_monitor
        self.models_dir = models_dir

    # ── admin dashboard (§9.1) ──────────────────────────────────────────

    async def admin_snapshot(
        self, query: AdminDashboardQuery
    ) -> AdminDashboardResponse:
        tz = _resolve_timezone(query.timezone)
        sla_seconds = self._effective_sla(query.sla_seconds)
        now_utc = datetime.now(timezone.utc)

        recent_predictions = await self._recent_predictions(query.recent_limit)
        review_queue_health = await self._review_queue_health(sla_seconds)
        analyst_workload = await self._analyst_workload(tz)
        classification_breakdown = await self._classification_breakdown(
            tz, breakdown_days=query.breakdown_days, now_utc=now_utc
        )
        active_model = await self._active_model_summary()
        training_buffer = await self._training_buffer_summary(now_utc)
        recent_activity = await self._recent_activity(
            limit=query.activity_limit, now_utc=now_utc
        )
        prediction_volume = await self._prediction_volume(
            tz,
            bucket=query.volume_bucket,
            range_hours=query.volume_range_hours,
            now_utc=now_utc,
        )

        return AdminDashboardResponse(
            generatedAt=now_utc,
            timezone=query.timezone,
            recentPredictions=recent_predictions,
            reviewQueueHealth=review_queue_health,
            analystWorkload=analyst_workload,
            classificationBreakdown=classification_breakdown,
            activeModel=active_model,
            trainingBuffer=training_buffer,
            recentActivity=recent_activity,
            predictionVolume=prediction_volume,
        )

    # ── analyst dashboard (§9.2) ────────────────────────────────────────

    async def analyst_snapshot(
        self, *, caller: User, query: AnalystDashboardQuery
    ) -> AnalystDashboardResponse:
        tz = _resolve_timezone(query.timezone)
        sla_seconds = self._effective_sla(query.sla_seconds)
        now_utc = datetime.now(timezone.utc)
        range_days = max(0, query.range_days)
        since = now_utc - timedelta(days=range_days)

        my_queue_raw = await self.review_repo.my_queue_stats(
            user_id=caller.id, sla_seconds=sla_seconds
        )
        my_queue = MyQueueSummary(
            assignedCount=my_queue_raw["assigned_count"],
            averageAgeSeconds=my_queue_raw["average_age_seconds"],
            oldestAgeSeconds=my_queue_raw["oldest_age_seconds"],
            slaBreachCount=my_queue_raw["sla_breach_count"],
            slaThresholdSeconds=sla_seconds,
        )

        today_start = _today_start_utc(tz)
        week_start = _week_start_utc(tz)
        today_stats = await self._personal_window(caller.id, today_start)
        week_stats = await self._personal_window(caller.id, week_start)
        personal_stats = PersonalStats(today=today_stats, thisWeek=week_stats)

        recent_submissions = await self._recent_submissions(
            caller.id, query.recent_limit
        )
        queue_label_mix = await self._queue_label_mix(caller.id, since, range_days)
        auto_review_usage = await self._auto_review_usage(
            caller=caller,
            since=since,
            range_days=range_days,
            invocation_limit=query.invocation_limit,
        )

        return AnalystDashboardResponse(
            generatedAt=now_utc,
            timezone=query.timezone,
            analyst={
                "id": str(caller.id),
                "username": caller.username,
                "role": caller.role.value,
            },
            myQueue=my_queue,
            personalStats=personal_stats,
            recentSubmissions=recent_submissions,
            queueLabelMix=queue_label_mix,
            autoReviewUsage=auto_review_usage,
        )

    # ── admin helpers ────────────────────────────────────────────────────

    async def _recent_predictions(
        self, limit: int
    ) -> list[RecentPredictionEntry]:
        events = await self.prediction_repo.recent_events(limit=max(0, limit))
        return [RecentPredictionEntry.from_event(e) for e in events]

    async def _review_queue_health(self, sla_seconds: int) -> ReviewQueueHealth:
        stats = await self.review_repo.pending_queue_stats(
            sla_seconds=sla_seconds
        )
        return ReviewQueueHealth(
            pendingCount=stats["pending_count"],
            unassignedCount=stats["unassigned_count"],
            inProgressCount=stats["in_progress_count"],
            averageAgeSeconds=stats["average_age_seconds"],
            oldestAgeSeconds=stats["oldest_age_seconds"],
            slaBreachCount=stats["sla_breach_count"],
            slaThresholdSeconds=sla_seconds,
        )

    async def _analyst_workload(
        self, tz: ZoneInfo
    ) -> list[AnalystWorkloadEntry]:
        analysts = await self.user_repo.list_active_analysts()
        if not analysts:
            return []
        user_ids = [u.id for u in analysts]

        today_start = _today_start_utc(tz)
        week_start = _week_start_utc(tz)

        assigned = await self.review_repo.assigned_counts_for_users(user_ids)
        completed_today = await self.review_repo.analyst_decided_counts(
            user_ids=user_ids, since=today_start
        )
        completed_week = await self.review_repo.analyst_decided_counts(
            user_ids=user_ids, since=week_start
        )
        avg_decision = await self.review_repo.analyst_avg_decision_seconds(
            user_ids=user_ids, since=week_start
        )

        out: list[AnalystWorkloadEntry] = []
        for user in analysts:
            done_week = completed_week.get(user.id, 0)
            still_assigned = assigned.get(user.id, 0)
            denom = done_week + still_assigned
            completion_rate = (done_week / denom) if denom > 0 else 0.0
            out.append(
                AnalystWorkloadEntry(
                    analyst=_UserRef.from_user(user),
                    assignedCount=still_assigned,
                    completedToday=completed_today.get(user.id, 0),
                    completedThisWeek=done_week,
                    completionRate=completion_rate,
                    averageDecisionSeconds=avg_decision.get(user.id, 0.0),
                )
            )
        return out

    async def _classification_breakdown(
        self,
        tz: ZoneInfo,
        *,
        breakdown_days: int,
        now_utc: datetime,
    ) -> ClassificationBreakdown:
        days = max(1, breakdown_days)
        date_from = now_utc - timedelta(days=days)
        rows = await self.prediction_repo.classification_breakdown(
            date_from=date_from,
            date_to=now_utc,
            timezone_name=str(tz),
        )
        series = [
            ClassificationBreakdownPoint(
                date=row["bucket"],
                phishing=row["phishing"],
                legitimate=row["legitimate"],
                review=row["review"],
            )
            for row in rows
        ]
        return ClassificationBreakdown(
            bucket="day",
            **{"from": date_from, "to": now_utc},  # type: ignore[arg-type]
            series=series,
        )

    async def _active_model_summary(self) -> ActiveModelSummary:
        detector = self.detector
        drift = DriftSignalSummary.from_signal(self.drift_monitor.drift_signal())
        if detector is None:
            return ActiveModelSummary(
                version=None,
                promoted=False,
                activatedAt=None,
                decisionThreshold=None,
                reviewLowThreshold=None,
                reviewHighThreshold=None,
                driftSignal=drift,
            )

        version = detector.version
        promoted = False
        activated_at: datetime | None = None
        if version is not None:
            meta = ModelRegistry(self.models_dir)._read_registry_metadata()
            entry = meta.get("versions", {}).get(version, {}) or {}
            promoted = bool(entry.get("promoted", False))
            latest = await self.activation_repo.latest_for_version(
                version=version
            )
            if latest is not None:
                activated_at = latest.created_at
        return ActiveModelSummary(
            version=version,
            promoted=promoted,
            activatedAt=activated_at,
            decisionThreshold=inference_config.decision_threshold,
            reviewLowThreshold=detector.review_low_threshold,
            reviewHighThreshold=detector.review_high_threshold,
            driftSignal=drift,
        )

    async def _training_buffer_summary(
        self, now_utc: datetime
    ) -> TrainingBufferSummary:
        status: BufferStatus = await self.buffer_service.status()
        recent_additions = await self.buffer_repo.count_since(
            now_utc - timedelta(hours=24)
        )
        return TrainingBufferSummary(
            totalCount=status.size,
            classCounts={
                "legitimate": status.class_counts.get(0, 0),
                "phishing": status.class_counts.get(1, 0),
            },
            minBatchSize=status.min_batch_size,
            minPerClass=status.min_per_class,
            unlocked=status.unlocked,
            blockers=list(status.blockers),
            recentAdditions24h=recent_additions,
        )

    async def _recent_activity(
        self, *, limit: int, now_utc: datetime
    ) -> list[RecentActivityEntry]:
        limit = max(0, limit)
        if limit == 0:
            return []
        since = now_utc - timedelta(days=_RECENT_ACTIVITY_LOOKBACK_DAYS)

        activations = await self.activation_repo.recent_since(
            since=since, limit=limit
        )
        training_runs = await self.training_run_repo.recent_runs_touched_since(
            since=since, limit=limit
        )
        logins = await self.token_repo.recent_logins_since(
            since=since, limit=limit
        )

        events: list[RecentActivityEntry] = []
        for row in activations:
            events.append(self._project_activation(row))
        for run in training_runs:
            events.extend(self._project_training_run(run, since=since))
        for token in logins:
            events.append(self._project_login(token))

        events.sort(key=lambda e: e.occurred_at, reverse=True)
        return events[:limit]

    async def _prediction_volume(
        self,
        tz: ZoneInfo,
        *,
        bucket: str,
        range_hours: int,
        now_utc: datetime,
    ) -> PredictionVolumeSeries:
        if bucket not in {"hour", "day"}:
            raise BadRequestException(
                message="Invalid bucket",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="DASHBOARD_BUCKET_INVALID",
                    status=400,
                    details=[
                        f"volumeBucket must be one of ['day', 'hour'], "
                        f"got {bucket!r}"
                    ],
                ),
            )
        hours = max(1, range_hours)
        date_from = now_utc - timedelta(hours=hours)
        rows = await self.prediction_repo.volume_buckets(
            bucket=bucket,
            date_from=date_from,
            date_to=now_utc,
            timezone_name=str(tz),
        )
        series = [
            PredictionVolumePoint(t=row["t"], count=row["count"]) for row in rows
        ]
        return PredictionVolumeSeries(
            bucket=bucket,
            **{"from": date_from, "to": now_utc},  # type: ignore[arg-type]
            series=series,
        )

    # ── activity projectors ────────────────────────────────────────────

    @staticmethod
    def _project_activation(row: ModelActivation) -> RecentActivityEntry:
        kind_map = {
            ModelActivationKind.PROMOTE: "PROMOTE",
            ModelActivationKind.ACTIVATE: "ACTIVATE",
            ModelActivationKind.ROLLBACK: "ROLLBACK",
        }
        kind = kind_map[row.kind]
        if row.kind is ModelActivationKind.PROMOTE:
            summary = f"Promoted {row.version}"
            f1 = None
            if isinstance(row.metrics_snapshot, dict):
                f1 = row.metrics_snapshot.get("f1")
            if isinstance(f1, (int, float)):
                summary = f"Promoted {row.version} (F1 {float(f1):.3f})"
        elif row.kind is ModelActivationKind.ACTIVATE:
            summary = f"Activated {row.version}"
        else:
            summary = (
                f"Rolled back to {row.version}"
                if row.previous_version is None
                else f"Rolled back from {row.previous_version} to {row.version}"
            )
        actor_ref: _UserRef | None = None
        if row.actor is not None:
            actor_ref = _UserRef.from_user(row.actor)
        return RecentActivityEntry(
            occurredAt=row.created_at,
            kind=kind,
            actor=actor_ref,
            summary=summary,
            ref=RecentActivityRef(type="model", id=row.id),
        )

    @staticmethod
    def _project_training_run(
        run: TrainingRun, *, since: datetime
    ) -> list[RecentActivityEntry]:
        """A single run contributes up to three events (started, succeeded /
        failed) — we emit whichever lands in the window so rapid iterations
        show their distinct phases on the feed."""
        actor_ref: _UserRef | None = None
        if getattr(run, "triggered_by_user", None) is not None:
            actor_ref = _UserRef.from_user(run.triggered_by_user)
        ref = RecentActivityRef(type="training_run", id=run.id)
        out: list[RecentActivityEntry] = []
        if run.started_at is not None and run.started_at >= since:
            out.append(
                RecentActivityEntry(
                    occurredAt=run.started_at,
                    kind="TRAINING_RUN_STARTED",
                    actor=actor_ref,
                    summary=f"Training run {run.id} started",
                    ref=ref,
                )
            )
        if run.finished_at is not None and run.finished_at >= since:
            if run.status is TrainingRunStatus.SUCCEEDED:
                summary = (
                    f"Training run {run.id} succeeded"
                    + (
                        f" → {run.new_version}"
                        if run.new_version is not None
                        else ""
                    )
                )
                out.append(
                    RecentActivityEntry(
                        occurredAt=run.finished_at,
                        kind="TRAINING_RUN_SUCCEEDED",
                        actor=actor_ref,
                        summary=summary,
                        ref=ref,
                    )
                )
            elif run.status is TrainingRunStatus.FAILED:
                detail = (
                    f": {run.error_message}"
                    if run.error_message
                    else ""
                )
                out.append(
                    RecentActivityEntry(
                        occurredAt=run.finished_at,
                        kind="TRAINING_RUN_FAILED",
                        actor=actor_ref,
                        summary=f"Training run {run.id} failed{detail}",
                        ref=ref,
                    )
                )
        return out

    @staticmethod
    def _project_login(token: Token) -> RecentActivityEntry:
        user = token.user
        actor_ref = _UserRef.from_user(user) if user is not None else None
        summary = (
            f"{user.username} logged in"
            if user is not None
            else "User logged in"
        )
        ref = (
            RecentActivityRef(type="user", id=user.id)
            if user is not None
            else None
        )
        return RecentActivityEntry(
            occurredAt=token.created_at,
            kind="LOGIN",
            actor=actor_ref,
            summary=summary,
            ref=ref,
        )

    # ── analyst helpers ─────────────────────────────────────────────────

    async def _personal_window(
        self, user_id: UUID, since: datetime
    ) -> PersonalStatsWindow:
        raw = await self.review_repo.personal_decision_stats(
            user_id=user_id, since=since
        )
        return PersonalStatsWindow(
            completed=raw["completed"],
            deferred=raw["deferred"],
            escalated=raw["escalated"],
            averageDecisionSeconds=raw["average_decision_seconds"],
        )

    async def _recent_submissions(
        self, user_id: UUID, limit: int
    ) -> list[RecentSubmissionEntry]:
        rows = await self.review_repo.recent_decisions_for_user(
            user_id=user_id, limit=max(0, limit)
        )
        return [self._project_submission(row) for row in rows]

    @staticmethod
    def _project_submission(row: ReviewItem) -> RecentSubmissionEntry:
        prediction = row.prediction_event
        subject_preview = _safe_preview(
            getattr(prediction, "subject", None)
        )
        return RecentSubmissionEntry(
            reviewItemId=row.id,
            decidedAt=row.decided_at,
            status=row.status.value,
            verdict=row.verdict.value if row.verdict is not None else None,
            autoReviewUsed=bool(row.auto_review_used),
            agreedWithAutoReview=row.auto_review_agreement.value,
            subjectPreview=subject_preview,
        )

    async def _queue_label_mix(
        self, user_id: UUID, since: datetime, range_days: int
    ) -> QueueLabelMix:
        counts = await self.review_repo.label_mix_for_user(
            user_id=user_id, since=since
        )
        phishing = counts.get("phishing", 0)
        legitimate = counts.get("legitimate", 0)
        total = phishing + legitimate
        ratio = (phishing / total) if total > 0 else 0.0
        return QueueLabelMix(
            rangeDays=range_days,
            phishing=phishing,
            legitimate=legitimate,
            ratioPhishing=ratio,
        )

    async def _auto_review_usage(
        self,
        *,
        caller: User,
        since: datetime,
        range_days: int,
        invocation_limit: int,
    ) -> AutoReviewUsageSummary:
        agreement = await self.review_repo.auto_review_agreement_counts_for_user(
            user_id=caller.id, since=since
        )
        invocation_stats = await self.auto_review_repo.stats_for_actor_since(
            triggered_by=caller.id, since=since
        )
        recent = await self.auto_review_repo.recent_for_actor_since(
            triggered_by=caller.id,
            since=since,
            limit=max(0, invocation_limit),
        )

        agreed = agreement["agreed_count"]
        overridden = agreement["overridden_count"]
        denom = agreed + overridden
        agreement_rate = (agreed / denom) if denom > 0 else 0.0

        return AutoReviewUsageSummary(
            rangeDays=range_days,
            withAutoReview=agreement["with_auto_review"],
            withoutAutoReview=agreement["without_auto_review"],
            agreedCount=agreed,
            overriddenCount=overridden,
            agreementRate=agreement_rate,
            invocationCount=invocation_stats["invocation_count"],
            invocationFailureCount=invocation_stats[
                "invocation_failure_count"
            ],
            recentInvocations=[
                self._project_invocation(inv) for inv in recent
            ],
        )

    @staticmethod
    def _project_invocation(
        inv: AutoReviewInvocation,
    ) -> AutoReviewUsageInvocation:
        review_item = inv.review_item
        final_verdict: str | None = None
        agreed: bool | None = None
        if review_item is not None:
            final_verdict = (
                review_item.verdict.value
                if review_item.verdict is not None
                else None
            )
            agreement = review_item.auto_review_agreement
            if review_item.status is ReviewItemStatus.CONFIRMED:
                if agreement is AutoReviewAgreement.AGREED:
                    agreed = True
                elif agreement is AutoReviewAgreement.OVERRIDDEN:
                    agreed = False
        return AutoReviewUsageInvocation(
            invocationId=inv.id,
            reviewItemId=inv.review_item_id,
            createdAt=inv.created_at,
            triggerKind=inv.trigger_kind,
            provider=inv.provider.value,
            modelName=inv.model_name,
            outcome=inv.outcome.value,
            label=inv.label,
            finalVerdict=final_verdict,
            agreed=agreed,
        )

    # ── misc ────────────────────────────────────────────────────────────

    @staticmethod
    def _effective_sla(override: int | None) -> int:
        from src.configs import review as review_config

        if override is None:
            return int(review_config.sla_seconds)
        if override <= 0:
            raise BadRequestException(
                message="Invalid SLA override",
                error_detail=ErrorDetail(
                    title="Bad Request",
                    code="DASHBOARD_SLA_INVALID",
                    status=400,
                    details=[
                        f"slaSeconds must be a positive integer, got {override!r}"
                    ],
                ),
            )
        return int(override)
