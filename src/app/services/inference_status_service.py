from src.configs import inference as inference_config
from src.shared.inference import AutoReviewer, DriftMonitor, PhishingDetector
from src.app.dtos import DriftSignalSummary, InferenceStatusResponse
from src.app.models.training_run import TrainingRun
from src.app.repositories.training_run_repository import TrainingRunRepository


class InferenceStatusService:
    def __init__(
        self,
        detector: PhishingDetector | None,
        drift_monitor: DriftMonitor,
        auto_reviewer: AutoReviewer | None,
        training_run_repository: TrainingRunRepository,
    ):
        self.detector = detector
        self.drift_monitor = drift_monitor
        self.auto_reviewer = auto_reviewer
        self.training_run_repository = training_run_repository

    async def status(self) -> InferenceStatusResponse:
        warnings: list[str] = []

        if self.detector is None:
            warnings.append(
                "Detector is not loaded — prediction endpoints will be unavailable "
                "until a model version is registered."
            )

        if self.auto_reviewer is None:
            warnings.append(
                "Auto-reviewer is not configured — LLM decision support is disabled."
            )

        drift_signal = self.drift_monitor.drift_signal()
        if drift_signal.false_positive_rate > self.drift_monitor.fpr_threshold:
            warnings.append(drift_signal.message)

        latest_run = await self.training_run_repository.latest()

        return InferenceStatusResponse(
            detectorLoaded=self.detector is not None,
            activeVersion=self.detector.version if self.detector is not None else None,
            decisionThreshold=inference_config.decision_threshold,
            reviewLowThreshold=(
                self.detector.review_low_threshold if self.detector is not None else None
            ),
            reviewHighThreshold=(
                self.detector.review_high_threshold if self.detector is not None else None
            ),
            driftMonitor=DriftSignalSummary.from_signal(drift_signal),
            autoReviewerAvailable=self.auto_reviewer is not None,
            llmProvider=(
                self.auto_reviewer.provider.value if self.auto_reviewer is not None else None
            ),
            llmModelName=(
                self.auto_reviewer.model_name if self.auto_reviewer is not None else None
            ),
            lastTrainingRun=_summarize_training_run(latest_run),
            warnings=warnings,
        )


def _summarize_training_run(run: TrainingRun | None) -> dict | None:
    if run is None:
        return None
    return {
        "id": str(run.id),
        "status": run.status.value,
        "newVersion": run.new_version,
        "promoted": run.promoted,
        "createdAt": run.created_at.isoformat() if run.created_at else None,
        "startedAt": run.started_at.isoformat() if run.started_at else None,
        "finishedAt": run.finished_at.isoformat() if run.finished_at else None,
    }
