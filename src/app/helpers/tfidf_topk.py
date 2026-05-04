"""Phase 12 — top-k TF-IDF term highlights for the explain endpoint.

When the ``inference.explain.tfidf_topk_enabled`` flag is on, the explain
response surfaces the top-k terms driving the TF-IDF portion of the feature
vector. Terms are ranked by ``|tfidf * weight|`` when the active classifier
exposes linear coefficients; otherwise we fall back to raw TF-IDF magnitude
so the feature still works for non-linear detectors.

The helper is entirely read-only against the detector passed in — no state
is mutated and no re-prediction is performed. The persisted row's
label/probability remain authoritative.
"""

from __future__ import annotations

from typing import Any

from src.configs import inference as inference_config


def _normalise_for_tfidf(vectorizer: Any, text: str) -> str:
    """Return ``text`` normalised the same way the training pipeline normalises
    before calling ``transform``. sklearn TfidfVectorizer already applies its
    own analyser inside ``transform``, so we pass the raw text through and let
    the vectoriser do the normalisation."""
    return text or ""


def _feature_names(vectorizer: Any) -> list[str]:
    if hasattr(vectorizer, "get_feature_names_out"):
        return [str(n) for n in vectorizer.get_feature_names_out()]
    if hasattr(vectorizer, "vocabulary_"):
        vocab = vectorizer.vocabulary_
        return [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    return []


def _linear_coef(model: Any) -> Any | None:
    """Return a 1-D coefficient vector when ``model`` is a linear classifier.

    Returns None for non-linear models (MLP, RF, etc.) — the caller falls
    back to raw TF-IDF magnitude in that case.
    """
    coef = getattr(model, "coef_", None)
    if coef is None:
        return None
    try:
        import numpy as np

        arr = np.asarray(coef)
    except Exception:
        return None
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 1:
        return arr
    return None


def explain_topk_enabled() -> bool:
    cfg = getattr(inference_config, "explain", None)
    return bool(cfg is not None and getattr(cfg, "tfidf_topk_enabled", False))


def explain_topk() -> int:
    cfg = getattr(inference_config, "explain", None)
    if cfg is None:
        return 10
    try:
        k = int(getattr(cfg, "tfidf_topk", 10))
    except (TypeError, ValueError):
        return 10
    return max(1, k)


def compute_top_terms(
    *, detector: Any, subject: str, body: str, k: int,
) -> list[dict]:
    """Return up to ``k`` top-weighted TF-IDF terms across subject and body.

    Each entry is ``{term, weight, source, tfidf}``:
      * ``weight`` — signed contribution toward the phishing class
        (``tfidf * coef`` when a linear coefficient is available, else the
        raw TF-IDF value).
      * ``source`` — ``"subject"`` or ``"body"``.
      * ``tfidf`` — the raw TF-IDF value (non-negative) so callers can show
        the term's surface strength independently of the signed weight.

    Empty inputs return ``[]``. Errors from the underlying vectorisers or
    classifier are swallowed — the endpoint always responds even when the
    detector is mid-swap.
    """
    if detector is None or k <= 0:
        return []
    subject_vec = getattr(detector, "subject_vectorizer", None)
    body_vec = getattr(detector, "body_vectorizer", None)
    if subject_vec is None or body_vec is None:
        return []

    try:
        import numpy as np

        subj_row = subject_vec.transform([_normalise_for_tfidf(subject_vec, subject)])
        body_row = body_vec.transform([_normalise_for_tfidf(body_vec, body)])
    except Exception:
        return []

    subj_names = _feature_names(subject_vec)
    body_names = _feature_names(body_vec)
    if not subj_names and not body_names:
        return []

    coef = _linear_coef(getattr(detector, "model", None))
    subj_dim = len(subj_names)
    body_dim = len(body_names)

    entries: list[tuple[float, float, str, str]] = []

    try:
        subj_coo = subj_row.tocoo()
        for col, val in zip(subj_coo.col.tolist(), subj_coo.data.tolist()):
            if col >= subj_dim:
                continue
            tfidf = float(val)
            weight = (
                tfidf * float(coef[col])
                if coef is not None and col < len(coef)
                else tfidf
            )
            entries.append((weight, tfidf, subj_names[col], "subject"))
    except Exception:
        pass

    try:
        body_coo = body_row.tocoo()
        for col, val in zip(body_coo.col.tolist(), body_coo.data.tolist()):
            if col >= body_dim:
                continue
            tfidf = float(val)
            weight = (
                tfidf * float(coef[subj_dim + col])
                if coef is not None and (subj_dim + col) < len(coef)
                else tfidf
            )
            entries.append((weight, tfidf, body_names[col], "body"))
    except Exception:
        pass

    if not entries:
        return []

    entries.sort(key=lambda e: abs(e[0]), reverse=True)
    top = entries[:k]
    return [
        {"term": term, "weight": weight, "source": source, "tfidf": tfidf}
        for weight, tfidf, term, source in top
    ]
