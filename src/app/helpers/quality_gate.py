"""Phase 12 — training-buffer quality gate (OOV rate).

Items with a combined (subject + body) out-of-vocabulary rate above the
configured threshold are rejected at enqueue and on CSV import. The gate
uses the active detector's TF-IDF vectorisers; when no detector is loaded,
or when the gate is disabled, no rejection happens.

The OOV computation mirrors ``OnlineLearner._oov_rate`` in the inference
package — we re-implement it here rather than importing so the
``src/shared/inference`` tree stays untouched (see CLAUDE.md).
"""

from __future__ import annotations

from typing import Any

from src.configs import training as training_config


def _vocab_oov(text: str, vectorizer: Any) -> tuple[int, int]:
    """Return ``(token_count, oov_count)`` for ``text`` against ``vectorizer``.

    Returns ``(0, 0)`` when the vectoriser does not expose a usable analyser
    or vocabulary — the caller treats that as a skipped gate, not a failure.
    """
    try:
        analyse = vectorizer.build_analyzer()
        vocab = vectorizer.vocabulary_
    except Exception:
        return 0, 0
    try:
        tokens = analyse(text or "")
    except Exception:
        return 0, 0
    if not tokens:
        return 0, 0
    oov = sum(1 for tok in tokens if tok not in vocab)
    return len(tokens), oov


def compute_combined_oov_rate(
    *, detector: Any, subject: str, body: str
) -> float | None:
    """Return the combined OOV rate for (subject, body), or ``None`` when
    the gate cannot run (no detector, missing vectorisers, empty inputs).
    """
    if detector is None:
        return None
    subject_vec = getattr(detector, "subject_vectorizer", None)
    body_vec = getattr(detector, "body_vectorizer", None)
    if subject_vec is None or body_vec is None:
        return None
    s_total, s_oov = _vocab_oov(subject, subject_vec)
    b_total, b_oov = _vocab_oov(body, body_vec)
    total = s_total + b_total
    if total == 0:
        return None
    return (s_oov + b_oov) / total


def quality_gate_enabled() -> bool:
    cfg = getattr(training_config, "quality_gate", None)
    return bool(cfg is not None and getattr(cfg, "enabled", False))


def quality_gate_max_oov() -> float:
    cfg = getattr(training_config, "quality_gate", None)
    if cfg is None:
        return 1.0
    try:
        return float(getattr(cfg, "max_oov_rate", 1.0))
    except (TypeError, ValueError):
        return 1.0


def violates_quality_gate(
    *, detector: Any, subject: str, body: str
) -> tuple[bool, float | None]:
    """Return ``(rejected, rate)``.

    ``rejected`` is True when the gate is enabled, the detector exposes
    both vectorisers, and the measured OOV rate exceeds the threshold.
    ``rate`` is the measured rate (or ``None`` when the gate was skipped).
    """
    if not quality_gate_enabled():
        return False, None
    rate = compute_combined_oov_rate(
        detector=detector, subject=subject, body=body
    )
    if rate is None:
        return False, None
    return rate > quality_gate_max_oov(), rate
