"""AURA inference package — public API."""
from .schema import (
    PredictionResult,
    OnlineLearningResult,
    ValidationError,
    ConfidenceZone,
    AutoReviewSuccess,
    AutoReviewFailure,
    AutoReviewResponse,
    LLMProvider,
    ReviewLabel,
)

from .drift_monitor import DriftMonitor, DriftSignal, DriftStatus
from .online_learner import OnlineLearner
from .registry import ModelRegistry
from .auto_reviewer import AutoReviewer
from .detector import PhishingDetector


__all__ = [
    'PhishingDetector',
    'OnlineLearner',
    'ModelRegistry',
    'PredictionResult',
    'OnlineLearningResult',
    'ValidationError',
    'ConfidenceZone',
    'DriftMonitor',
    'DriftSignal',
    'DriftStatus',
    'AutoReviewer',
    'AutoReviewSuccess',
    'AutoReviewFailure',
    'AutoReviewResponse',
    'LLMProvider',
    'ReviewLabel',
]