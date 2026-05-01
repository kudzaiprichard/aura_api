"""OnlineLearner — partial_fit wrapper with locking and holdout validation."""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import portalocker

from src.configs import training as training_config
from src.shared.inference.preprocessing import build_feature_matrix
from src.shared.inference.registry import ModelRegistry, default_models_root
from src.shared.inference.schema import OnlineLearningResult, ValidationError
from src.shared.inference.validation import validate_training_batch

log = logging.getLogger('aura.inference')


# NOTEBOOK_CONTRACT §3.2 — training uses a single `mlp.fit(X_train, y_train)`
# call with no partial_fit. Online learning is an inference-module extension;
# we cap iterations per call to prevent runaway partial_fit loops. The cap
# is tunable via TRAINING_MAX_ITER_PER_CALL.
DEFAULT_MAX_ITER_PER_CALL = training_config.max_iter_per_call
DEFAULT_OOV_WARN_THRESHOLD = 0.30


class OnlineLearner:
    """Concurrency model: per-instance in-memory state; cross-process writes
    are coordinated via a portalocker file lock on `{models_root}/.lock`.
    The class is NOT thread-safe for shared instances; use one per thread.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        *,
        models_root: Path | str | None = None,
        holdout_set: tuple[pd.DataFrame, np.ndarray] | None = None,
        oov_warn_threshold: float = DEFAULT_OOV_WARN_THRESHOLD,
    ):
        if registry is None:
            root = Path(models_root).resolve() if models_root else default_models_root()
            registry = ModelRegistry(root)
        self.registry = registry
        self.holdout_set = holdout_set
        self.oov_warn_threshold = oov_warn_threshold

    # -- paths ------------------------------------------------------------
    @property
    def _lock_path(self) -> Path:
        return self.registry.models_root / '.lock'

    # -- performance on holdout -------------------------------------------
    def _holdout_metrics(self, model, subject_vec, body_vec) -> dict[str, float]:
        if self.holdout_set is None:
            return {}
        df, y = self.holdout_set
        emails = [
            {'sender': r.get('sender', ''),
             'subject': r.get('subject', ''),
             'body': r.get('body', '')}
            for _, r in df.iterrows()
        ]
        X = build_feature_matrix(emails, subject_vec, body_vec)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y_pred = model.predict(X)
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
        )
        return {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1': float(f1_score(y, y_pred, zero_division=0)),
        }

    # -- oov --------------------------------------------------------------
    @staticmethod
    def _oov_rate(texts: list[str], vectorizer) -> float:
        if not texts:
            return 0.0
        analyse = vectorizer.build_analyzer()
        vocab = vectorizer.vocabulary_
        total = 0
        oov = 0
        for t in texts:
            tokens = analyse(t)
            total += len(tokens)
            oov += sum(1 for tok in tokens if tok not in vocab)
        return 0.0 if total == 0 else oov / total

    # -- core -------------------------------------------------------------
    def partial_fit_batch(
        self,
        emails: list[dict],
        *,
        source_version: str | None = None,
        max_iter_per_call: int = DEFAULT_MAX_ITER_PER_CALL,
    ) -> OnlineLearningResult:
        validate_training_batch(emails, min_per_class=1)
        # NOTEBOOK_CONTRACT §3.2 — cap partial_fit iterations per call.
        if max_iter_per_call < 1:
            raise ValidationError('max_iter_per_call must be >= 1')

        source = source_version or self.registry.active_version() or self.registry.latest_version()
        if source is None:
            raise FileNotFoundError('No source version available to fine-tune')

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.touch(exist_ok=True)
        with portalocker.Lock(str(self._lock_path), mode='a', timeout=60):
            paths = self.registry.paths_for(source)
            model = joblib.load(paths['model'])
            subject_vec = joblib.load(paths['subject_vectorizer'])
            body_vec = joblib.load(paths['body_vectorizer'])

            before = self._holdout_metrics(model, subject_vec, body_vec)

            subjects = [e['subject'] for e in emails]
            bodies = [e['body'] for e in emails]
            oov_subject = self._oov_rate(subjects, subject_vec)
            oov_body = self._oov_rate(bodies, body_vec)
            if oov_subject > self.oov_warn_threshold:
                log.warning('subject OOV rate %.3f exceeds threshold %.3f',
                            oov_subject, self.oov_warn_threshold)
            if oov_body > self.oov_warn_threshold:
                log.warning('body OOV rate %.3f exceeds threshold %.3f',
                            oov_body, self.oov_warn_threshold)

            X = build_feature_matrix(emails, subject_vec, body_vec)
            y = np.array([e['label'] for e in emails], dtype=np.int64)

            updated = copy.deepcopy(model)
            if hasattr(updated, 'early_stopping'):
                updated.early_stopping = False
            iterations = 0
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for _ in range(max_iter_per_call):
                    updated.partial_fit(X, y, classes=[0, 1])
                    iterations += 1

            after = self._holdout_metrics(updated, subject_vec, body_vec)

            new_version = self.registry.register_new_version(
                updated, source_version=source, metrics=after,
            )

        return OnlineLearningResult(
            new_version=new_version,
            source_version=source,
            batch_size=len(emails),
            iterations=iterations,
            performance_before=before,
            performance_after=after,
            oov_rate_subject=oov_subject,
            oov_rate_body=oov_body,
            promoted=False,
        )

    def promote(self, version: str, *, min_delta_f1: float = -0.01) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.touch(exist_ok=True)
        with portalocker.Lock(str(self._lock_path), mode='a', timeout=60):
            meta = self.registry._read_registry_metadata()
            versions = meta.get('versions', {})
            entry = versions.get(version)
            if entry is None:
                raise ValueError(f'Unknown version: {version!r}')
            after = entry.get('metrics', {})
            source = entry.get('source_version')
            before: dict[str, float] = {}
            if source:
                before = versions.get(source, {}).get('metrics', {})
            if before and after and 'f1' in before and 'f1' in after:
                delta = after['f1'] - before['f1']
                if delta < min_delta_f1:
                    raise ValidationError(
                        f'Refusing to promote {version}: f1 delta {delta:.4f} '
                        f'< min_delta_f1 {min_delta_f1:.4f}'
                    )
            self.registry.promote(version, metrics=after)
