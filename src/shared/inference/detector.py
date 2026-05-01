"""PhishingDetector — the public prediction entry point."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import joblib
import numpy as np

from src.shared.inference.preprocessing import (
    build_feature_matrix,
    build_feature_row,
    extract_engineered_features,
)
from src.shared.inference.registry import ModelRegistry, default_models_root

if TYPE_CHECKING:
    from src.shared.inference.drift_monitor import DriftMonitor
from src.shared.inference.schema import (
    BODY_TFIDF_DIM,
    ConfidenceZone,
    ENGINEERED_FEATURE_ORDER,
    PredictionResult,
    SUBJECT_TFIDF_DIM,
    TOTAL_FEATURES,
    ValidationError,
)
from src.shared.inference.validation import (
    validate_email_inputs,
    validate_review_thresholds,
    validate_threshold,
)

log = logging.getLogger('aura.inference')


def classify_zone(
    probability: float,
    low_threshold: float,
    high_threshold: float,
) -> ConfidenceZone:
    # Boundary convention: prob < low → NOT_SPAM;
    #                     low <= prob < high → REVIEW;
    #                     prob >= high → SPAM.
    # This matches the existing `int(phish >= threshold)` rule for predicted_label,
    # so exact boundaries bucket upward (0.3 is REVIEW, 0.8 is SPAM).
    if probability < low_threshold:
        return ConfidenceZone.NOT_SPAM
    if probability >= high_threshold:
        return ConfidenceZone.SPAM
    return ConfidenceZone.REVIEW


class PhishingDetector:
    def __init__(
        self,
        model,
        subject_vectorizer,
        body_vectorizer,
        *,
        version: str | None = None,
        calibrator=None,
        review_low_threshold: float | None = None,
        review_high_threshold: float | None = None,
        drift_monitor: 'DriftMonitor | None' = None,
    ):
        subj_dim = self._vectorizer_output_dim(subject_vectorizer)
        body_dim = self._vectorizer_output_dim(body_vectorizer)
        if subj_dim != SUBJECT_TFIDF_DIM:
            raise ValueError(
                f'subject vectoriser output dim {subj_dim} != expected '
                f'{SUBJECT_TFIDF_DIM}'
            )
        if body_dim != BODY_TFIDF_DIM:
            raise ValueError(
                f'body vectoriser output dim {body_dim} != expected '
                f'{BODY_TFIDF_DIM}'
            )
        self._check_model_shape(model)
        if (review_low_threshold is None) ^ (review_high_threshold is None):
            raise ValidationError(
                'review_low_threshold and review_high_threshold must be '
                'provided together (both None or both set).'
            )
        if review_low_threshold is not None and review_high_threshold is not None:
            validate_review_thresholds(review_low_threshold, review_high_threshold)
        self.model = model
        self.subject_vectorizer = subject_vectorizer
        self.body_vectorizer = body_vectorizer
        self.version = version
        self.calibrator = calibrator
        self.review_low_threshold = (
            float(review_low_threshold) if review_low_threshold is not None else None
        )
        self.review_high_threshold = (
            float(review_high_threshold) if review_high_threshold is not None else None
        )
        self.drift_monitor = drift_monitor

    # -- constructors -----------------------------------------------------
    @classmethod
    def load_production(cls, models_root: Path | str | None = None) -> 'PhishingDetector':
        root = Path(models_root).resolve() if models_root else default_models_root()
        registry = ModelRegistry(root)
        version = registry.active_version() or registry.latest_version()
        if version is None:
            raise FileNotFoundError(f'No versions found under {root}')
        return cls._load_version(registry, version)

    @classmethod
    def load(cls, version: str, models_root: Path | str | None = None) -> 'PhishingDetector':
        root = Path(models_root).resolve() if models_root else default_models_root()
        registry = ModelRegistry(root)
        return cls._load_version(registry, version)

    @classmethod
    def from_paths(
        cls,
        model_path: Path | str,
        subject_vectorizer_path: Path | str,
        body_vectorizer_path: Path | str,
        *,
        calibrator_path: Path | str | None = None,
        review_low_threshold: float | None = None,
        review_high_threshold: float | None = None,
        drift_monitor: 'DriftMonitor | None' = None,
    ) -> 'PhishingDetector':
        model = joblib.load(Path(model_path))
        subj = joblib.load(Path(subject_vectorizer_path))
        body = joblib.load(Path(body_vectorizer_path))
        calibrator = joblib.load(Path(calibrator_path)) if calibrator_path else None
        return cls(
            model, subj, body,
            calibrator=calibrator,
            review_low_threshold=review_low_threshold,
            review_high_threshold=review_high_threshold,
            drift_monitor=drift_monitor,
        )

    @classmethod
    def _load_version(cls, registry: ModelRegistry, version: str) -> 'PhishingDetector':
        paths = registry.paths_for(version)
        log.info('loading model version=%s from %s', version, paths['model'])
        model = joblib.load(paths['model'])
        subj = joblib.load(paths['subject_vectorizer'])
        body = joblib.load(paths['body_vectorizer'])
        calibrator_path = paths.get('calibrator')
        calibrator = joblib.load(calibrator_path) if calibrator_path is not None else None
        return cls(model, subj, body, version=version, calibrator=calibrator)

    # -- inspection -------------------------------------------------------
    @staticmethod
    def _vectorizer_output_dim(vec) -> int:
        # sklearn TfidfVectorizer exposes vocabulary_ once fitted
        if hasattr(vec, 'get_feature_names_out'):
            return len(vec.get_feature_names_out())
        if hasattr(vec, 'vocabulary_'):
            return len(vec.vocabulary_)
        raise TypeError('unsupported vectoriser object; missing vocabulary_')

    @staticmethod
    def _check_model_shape(model) -> None:
        n_features = getattr(model, 'n_features_in_', None)
        if n_features is not None and n_features != TOTAL_FEATURES:
            raise ValueError(
                f'model.n_features_in_ = {n_features}, expected {TOTAL_FEATURES}'
            )

    # -- calibration ------------------------------------------------------
    def _apply_calibrator(self, raw_prob: float) -> float:
        if self.calibrator is None:
            return float(raw_prob)
        # Prob-in/prob-out calibrators have one of two APIs:
        #   - netcal HistogramBinning: .transform(1D probs)
        #   - sklearn IsotonicRegression, betacal BetaCalibration: .predict(1D probs)
        # Classifier-style calibrators (CalibratedClassifierCV) need the original
        # feature matrix and are not compatible with this scalar interface.
        if hasattr(self.calibrator, 'predict_proba'):
            raise TypeError(
                f'Unsupported calibrator type {type(self.calibrator).__name__}: '
                f'classifier-style calibrators require the original feature '
                f'matrix, not a scalar probability.'
            )
        arr = np.asarray([raw_prob], dtype=np.float64)
        transform = getattr(self.calibrator, 'transform', None)
        out = transform(arr) if transform is not None else self.calibrator.predict(arr)
        calibrated = float(np.asarray(out).ravel()[0])
        # Clip to [0, 1] so the legitimate_probability = 1 - phish invariant stays sane
        # even if a calibrator (e.g. extrapolating isotonic) produces a value just
        # outside the range.
        if calibrated < 0.0:
            return 0.0
        if calibrated > 1.0:
            return 1.0
        return calibrated

    # -- zone bucketing ---------------------------------------------------
    def _zone_for(
        self, probability: float
    ) -> tuple[ConfidenceZone | None, float | None, float | None]:
        if self.review_low_threshold is None or self.review_high_threshold is None:
            return None, None, None
        zone = classify_zone(
            probability, self.review_low_threshold, self.review_high_threshold
        )
        return zone, self.review_low_threshold, self.review_high_threshold

    # -- prediction -------------------------------------------------------
    def predict(
        self,
        sender: str,
        subject: str,
        body: str,
        *,
        threshold: float = 0.75,
    ) -> PredictionResult:
        validate_email_inputs(sender, subject, body)
        validate_threshold(threshold)
        X = build_feature_row(sender, subject, body, self.subject_vectorizer, self.body_vectorizer)
        probs = self.model.predict_proba(X)[0]
        raw_phish = float(probs[1])
        raw_legit = float(probs[0])
        calibrated = self.calibrator is not None
        phish = self._apply_calibrator(raw_phish) if calibrated else raw_phish
        legit = (1.0 - phish) if calibrated else raw_legit
        engineered_vec = extract_engineered_features(sender, subject, body)
        engineered = {
            name: float(engineered_vec[i])
            for i, name in enumerate(ENGINEERED_FEATURE_ORDER)
        }
        label = int(phish >= threshold)
        zone, low_thr, high_thr = self._zone_for(phish)
        prediction_id = str(uuid.uuid4())
        if self.drift_monitor is not None:
            self.drift_monitor.record_prediction(
                prediction_id=prediction_id,
                predicted_label=label,
                predicted_probability=phish,
                model_version=self.version or '',
            )
        return PredictionResult(
            predicted_label=label,
            phishing_probability=phish,
            legitimate_probability=legit,
            threshold=float(threshold),
            model_version=self.version,
            engineered_features=engineered,
            raw_phishing_probability=raw_phish,
            raw_legitimate_probability=raw_legit,
            calibrated=calibrated,
            confidence_zone=zone,
            review_low_threshold=low_thr,
            review_high_threshold=high_thr,
            prediction_id=prediction_id,
        )

    def predict_safe(
        self,
        sender: str,
        subject: str,
        body: str,
        *,
        threshold: float = 0.75,
    ) -> dict:
        try:
            return self.predict(
                sender, subject, body, threshold=threshold
            ).to_dict()
        except ValidationError as e:
            return {'error': 'validation_error', 'message': str(e)}
        except Exception as e:  # noqa: BLE001
            log.exception('predict failed')
            return {'error': 'internal_error', 'message': str(e)}

    def predict_batch(
        self,
        emails: Iterable[dict],
        *,
        threshold: float = 0.75,
    ) -> list[PredictionResult]:
        validate_threshold(threshold)
        emails_list = list(emails)
        for i, e in enumerate(emails_list):
            if not isinstance(e, dict):
                raise ValidationError(f'email[{i}] must be a dict')
            for field in ('sender', 'subject', 'body'):
                if field not in e:
                    raise ValidationError(f'email[{i}]: missing field {field!r}')
                if not isinstance(e[field], str):
                    raise ValidationError(
                        f'email[{i}]: {field!r} must be str'
                    )
        X = build_feature_matrix(emails_list, self.subject_vectorizer, self.body_vectorizer)
        probs = self.model.predict_proba(X)
        calibrated = self.calibrator is not None
        results: list[PredictionResult] = []
        for i, email in enumerate(emails_list):
            raw_phish = float(probs[i, 1])
            raw_legit = float(probs[i, 0])
            phish = self._apply_calibrator(raw_phish) if calibrated else raw_phish
            legit = (1.0 - phish) if calibrated else raw_legit
            label = int(phish >= threshold)
            engineered_vec = extract_engineered_features(
                email['sender'], email['subject'], email['body']
            )
            engineered = {
                name: float(engineered_vec[j])
                for j, name in enumerate(ENGINEERED_FEATURE_ORDER)
            }
            zone, low_thr, high_thr = self._zone_for(phish)
            prediction_id = str(uuid.uuid4())
            if self.drift_monitor is not None:
                self.drift_monitor.record_prediction(
                    prediction_id=prediction_id,
                    predicted_label=label,
                    predicted_probability=phish,
                    model_version=self.version or '',
                )
            results.append(PredictionResult(
                predicted_label=label,
                phishing_probability=phish,
                legitimate_probability=legit,
                threshold=float(threshold),
                model_version=self.version,
                engineered_features=engineered,
                raw_phishing_probability=raw_phish,
                raw_legitimate_probability=raw_legit,
                calibrated=calibrated,
                confidence_zone=zone,
                review_low_threshold=low_thr,
                review_high_threshold=high_thr,
                prediction_id=prediction_id,
            ))
        return results
