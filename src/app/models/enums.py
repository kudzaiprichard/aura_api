import enum


class Role(str, enum.Enum):
    ADMIN = "ADMIN"
    IT_ANALYST = "IT_ANALYST"


class TokenType(str, enum.Enum):
    ACCESS = "ACCESS"
    REFRESH = "REFRESH"


class PredictionSource(str, enum.Enum):
    API = "API"
    BATCH = "BATCH"
    BENCHMARK = "BENCHMARK"


# Mirrors src.shared.inference.schema.ConfidenceZone. Persisted locally so
# the SQLAlchemy layer is independent of the inference package and migrations
# do not reach across the src.shared boundary.
class ConfidenceZone(str, enum.Enum):
    SPAM = "SPAM"
    NOT_SPAM = "NOT_SPAM"
    REVIEW = "REVIEW"


class ReviewItemStatus(str, enum.Enum):
    UNASSIGNED = "UNASSIGNED"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    CONFIRMED = "CONFIRMED"
    ESCALATED = "ESCALATED"
    DEFERRED = "DEFERRED"


class ReviewVerdict(str, enum.Enum):
    PHISHING = "PHISHING"
    LEGITIMATE = "LEGITIMATE"


class TrainingBufferSource(str, enum.Enum):
    REVIEW = "REVIEW"
    CSV_IMPORT = "CSV_IMPORT"
    GENERATOR = "GENERATOR"
    ESCALATION = "ESCALATION"


# Phase 4 will flip AGREED / OVERRIDDEN from within the review-confirm path;
# Phase 3 only writes NOT_USED, but the column carries all three values from
# the start so the schema and migration are stable across phases.
class AutoReviewAgreement(str, enum.Enum):
    NOT_USED = "NOT_USED"
    AGREED = "AGREED"
    OVERRIDDEN = "OVERRIDDEN"


class AutoReviewOutcome(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


# Mirrors src.shared.inference.schema.LLMProvider but persisted upper-case so
# the SQLAlchemy enum matches the rest of the project's UPPER_SNAKE convention
# (the inference package keeps lower-case for direct LLM SDK compatibility).
class LLMProviderEnum(str, enum.Enum):
    GROQ = "GROQ"
    GOOGLE = "GOOGLE"


class TrainingRunStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


class BalanceStrategy(str, enum.Enum):
    UNDERSAMPLE = "UNDERSAMPLE"
    OVERSAMPLE = "OVERSAMPLE"
    NONE = "NONE"


class ModelActivationKind(str, enum.Enum):
    ACTIVATE = "ACTIVATE"
    PROMOTE = "PROMOTE"
    ROLLBACK = "ROLLBACK"


class BenchmarkStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class ExtensionInstallStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    BLACKLISTED = "BLACKLISTED"
