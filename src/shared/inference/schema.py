"""Data contracts for the AURA inference package.

Every constant in this file is a verbatim port from NOTEBOOK_CONTRACT.md.
Do not edit without updating the contract and the parity fixture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Union


# NOTEBOOK_CONTRACT §2.1 — 15 URL patterns (feature notebook cell 5).
# These are the patterns used by `count_urls` (body_url_count) and by
# `normalize_text_for_tfidf`. The 16th bare-domain pattern used only by the
# body cleaner in the data-cleaning notebook is intentionally excluded here;
# the inference module does not reimplement that cleaner.
URL_PATTERNS: tuple[str, ...] = (
    r'https?://[^\s<>\"\'\)]+',
    r'ftp://[^\s<>\"\'\)]+',
    r'ftps://[^\s<>\"\'\)]+',
    r'sftp://[^\s<>\"\'\)]+',
    r'www\.[^\s<>\"\'\)]+',
    r'file://[^\s<>\"\'\)]+',
    r'ssh://[^\s<>\"\'\)]+',
    r'telnet://[^\s<>\"\'\)]+',
    r'git://[^\s<>\"\'\)]+',
    r'svn://[^\s<>\"\'\)]+',
    r'mailto:[^\s<>\"\'\)]+',
    r'news:[^\s<>\"\'\)]+',
    r'nntp://[^\s<>\"\'\)]+',
    r'irc://[^\s<>\"\'\)]+',
    r'webcal://[^\s<>\"\'\)]+',
)


# NOTEBOOK_CONTRACT §2.4 — `final_selected_features` from cell 36, in order.
# Do not reorder: these are column indices 7000..7014 of the training matrix.
ENGINEERED_FEATURE_ORDER: tuple[str, ...] = (
    'body_word_count',
    'body_exclamation_count',
    'email_local_length',
    'name_email_consistency',
    'body_url_density',
    'body_url_count',
    'body_entropy',
    'email_digit_ratio',
    'domain_entropy',
    'domain_length',
    'subject_entropy',
    'body_avg_word_length',
    'sender_name_exists',
    'subject_exclamation_count',
    'domain_vowel_consonant_ratio',
)


# NOTEBOOK_CONTRACT §2.5 + §2.4.
SUBJECT_TFIDF_DIM: int = 2000
BODY_TFIDF_DIM: int = 5000
ENGINEERED_DIM: int = len(ENGINEERED_FEATURE_ORDER)
TOTAL_FEATURES: int = SUBJECT_TFIDF_DIM + BODY_TFIDF_DIM + ENGINEERED_DIM  # 7015


class ValidationError(Exception):
    """Raised when caller-provided inputs fail validation."""


class ConfidenceZone(str, Enum):
    SPAM = 'SPAM'
    NOT_SPAM = 'NOT_SPAM'
    REVIEW = 'REVIEW'


@dataclass
class PredictionResult:
    predicted_label: int
    phishing_probability: float
    legitimate_probability: float
    threshold: float
    model_version: str | None = None
    engineered_features: dict[str, float] = field(default_factory=dict)
    raw_phishing_probability: float | None = None
    raw_legitimate_probability: float | None = None
    calibrated: bool = False
    confidence_zone: ConfidenceZone | None = None
    review_low_threshold: float | None = None
    review_high_threshold: float | None = None
    prediction_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Force the enum to its string value — asdict() preserves the enum
        # instance, which repr()s as "ConfidenceZone.REVIEW" rather than "REVIEW".
        if self.confidence_zone is not None:
            d['confidence_zone'] = self.confidence_zone.value
        return d


# Re-export drift DTOs so callers can import them from inference.schema
# (the canonical location for all inference data contracts).
from src.shared.inference.drift_monitor import DriftSignal, DriftStatus  # noqa: E402, F401


class LLMProvider(str, Enum):
    GROQ = 'groq'
    GOOGLE = 'google'


class ReviewLabel(str, Enum):
    PHISHING = 'PHISHING'
    LEGITIMATE = 'LEGITIMATE'
    UNCERTAIN = 'UNCERTAIN'

@dataclass
class AutoReviewSuccess:
    """LLM produced a well-formed verdict."""
    review_label: ReviewLabel
    reasoning: str
    confidence: str
    provider: LLMProvider
    model_name: str
    raw_response: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['review_label'] = self.review_label.value
        d['provider'] = self.provider.value
        d['outcome'] = 'success'
        return d


@dataclass
class AutoReviewFailure:
    """LLM call failed — no verdict available.

    `user_message` is a short, human-readable summary safe to show to an
    end user. `technical_error` retains the raw upstream message (status
    codes, JSON fragments) for logs and debugging.
    """
    user_message: str
    technical_error: str
    provider: LLMProvider
    model_name: str
    raw_response: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['provider'] = self.provider.value
        d['outcome'] = 'failure'
        return d


# Sealed union: the only two things `review()` ever returns.
AutoReviewResponse = Union[AutoReviewSuccess, AutoReviewFailure]


@dataclass
class OnlineLearningResult:
    new_version: str
    source_version: str
    batch_size: int
    iterations: int
    performance_before: dict[str, float]
    performance_after: dict[str, float]
    oov_rate_subject: float
    oov_rate_body: float
    promoted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)