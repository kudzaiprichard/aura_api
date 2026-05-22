"""Chrome extension `/emails/analyze` business logic.

Projects the Gmail DTO to the core prediction flow. Extension-sourced
predictions persist to `prediction_events` with `requester_id=NULL` (an
install is not a dashboard user) and
`request_id = f"ext:{install_id}:{parent_rid}"` — the `ext:` prefix is
how the admin activity feed scopes `prediction_events` down to a single
install without adding an install_id column.

REVIEW-zone verdicts are enqueued onto the analyst queue by
`PredictionService._enqueue_for_review`, the same path the dashboard
`/analysis/predict` endpoint uses; drift and training-buffer signals
follow from that flow.
"""
import logging
import re
from html.parser import HTMLParser
from uuid import UUID, uuid4

from src.configs import inference as inference_config
from src.shared.exceptions import (
    BadRequestException,
    ServiceUnavailableException,
)
from src.shared.inference import ConfidenceZone, PredictionResult
from src.shared.responses import ErrorDetail
from src.app.dtos.extension import (
    ExtensionAnalysisResponse,
    ExtensionAnalyzeRequest,
    ExtensionEmailIdRef,
    ExtensionPrediction,
)
from src.app.models.enums import PredictionSource
from src.app.services.prediction_service import PredictionService


logger = logging.getLogger(__name__)


_HTML_SKIP_TAGS = frozenset(("script", "style", "noscript", "head"))
_WHITESPACE_RE = re.compile(r"\s+")


class _HTMLTextExtractor(HTMLParser):
    """Stdlib-only HTML → text projector. Drops `<script>`/`<style>` blocks
    and collapses runs of whitespace so the detector's TF-IDF vectoriser
    sees a flat string. Sufficient for a feature-extraction projection —
    not a full HTML renderer."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs) -> None:
        if tag.lower() in _HTML_SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag) -> None:
        if tag.lower() in _HTML_SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data) -> None:
        if self._skip_depth == 0 and data:
            self._chunks.append(data)

    def get_text(self) -> str:
        return _WHITESPACE_RE.sub(" ", " ".join(self._chunks)).strip()


def _strip_html(html: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 — stdlib parser raises on malformed input
        return ""
    return parser.get_text()


def _project_body(text: str | None, html: str | None) -> str:
    """§12: prefer `body.text`; fall back to HTML-stripped `body.html`."""
    candidate = (text or "").strip()
    if candidate:
        return candidate
    return _strip_html(html or "")


def extension_label(result: PredictionResult) -> str:
    """REVIEW takes precedence over the binary label per §12 / §5.5.2.

    A REVIEW-zone result can also have predicted_label=1 when the decision
    threshold falls inside the review band, but the contract pins REVIEW
    first so the extension renders the "Be careful" popover instead of a
    hard SPAM verdict.
    """
    if result.confidence_zone == ConfidenceZone.REVIEW:
        return "REVIEW"
    return "SPAM" if int(result.predicted_label) == 1 else "NOT_SPAM"


def label_from_event_fields(
    confidence_zone: ConfidenceZone | None,
    predicted_label: int,
) -> str:
    """Same precedence as `extension_label` but keyed off the persisted
    prediction_events columns — used by the admin activity feed which
    reads rows rather than live `PredictionResult`s.
    """
    if confidence_zone == ConfidenceZone.REVIEW:
        return "REVIEW"
    return "SPAM" if int(predicted_label) == 1 else "NOT_SPAM"


def _confidence_for_label(label: str, result: PredictionResult) -> float:
    """Per §5.5.2: confidence_score is the probability of the *displayed*
    label, not always the phishing probability. For REVIEW (no clear winner)
    surface the dominant side so the UI bar is never below 0.5."""
    if label == "SPAM":
        return float(result.phishing_probability)
    if label == "NOT_SPAM":
        return float(result.legitimate_probability)
    return max(
        float(result.phishing_probability),
        float(result.legitimate_probability),
    )


_EXT_PREFIX = "ext:"
# prediction_events.request_id is VARCHAR(64). The install-scope prefix
# `ext:{install_id.hex}:` is exactly 37 chars (4 + 32 + 1), leaving 27
# chars for the trace-correlation suffix. UUIDs are stored in hex form
# (no dashes) so the full install id still fits losslessly; the parent
# request id is truncated because it is only used for log correlation,
# not for lookup.
_EXT_SUFFIX_MAX = 27


def tagged_request_id(install_id: UUID, parent_request_id: str | None) -> str:
    """Build the `ext:{install_id.hex}:{rid}` request_id used to scope
    extension-sourced prediction_events rows to a single install.

    Total length is capped at 64 characters so the value fits in
    `prediction_events.request_id` (VARCHAR(64)). The install id is
    preserved in full as a left-anchored prefix so repository filters
    can match by `LIKE 'ext:{install_id.hex}:%'`.
    """
    suffix = parent_request_id or uuid4().hex
    return f"{_EXT_PREFIX}{install_id.hex}:{suffix[:_EXT_SUFFIX_MAX]}"


def install_request_id_pattern(install_id: UUID) -> str:
    """Shared LIKE pattern for filtering `prediction_events.request_id`
    by install — mirrors the format `tagged_request_id` produces.
    """
    return f"{_EXT_PREFIX}{install_id.hex}:%"


class ExtensionEmailService:
    def __init__(self, prediction_service: PredictionService):
        self.prediction_service = prediction_service

    async def analyze(
        self,
        request: ExtensionAnalyzeRequest,
        *,
        install_id: UUID,
        parent_request_id: str | None = None,
    ) -> ExtensionAnalysisResponse:
        sender = (request.headers.from_ or "").strip()
        subject = (request.headers.subject or "").strip()
        body = _project_body(request.body.text, request.body.html)

        max_bytes = int(inference_config.emails.max_body_bytes)
        if max_bytes > 0 and len(body.encode("utf-8")) > max_bytes:
            raise BadRequestException(
                message="Email too large to analyse",
                error_detail=ErrorDetail(
                    title="Email too large to analyse",
                    code="BAD_REQUEST",
                    status=400,
                    details=[
                        f"Projected body exceeds the {max_bytes}-byte limit"
                    ],
                ),
            )

        request_id = tagged_request_id(install_id, parent_request_id)

        try:
            _event, result = await self.prediction_service.predict(
                sender=sender,
                subject=subject,
                body=body,
                threshold=None,
                requester=None,
                request_id=request_id,
                source=PredictionSource.API,
            )
        except BadRequestException as exc:
            # PredictionService frames detector ValidationError with
            # title="Validation Failed"; the extension contract pins
            # "Email could not be analysed" for the same 400.
            detail = exc.error_detail
            if detail is not None and detail.code == "VALIDATION_ERROR":
                message = exc.message or "Email could not be analysed"
                raise BadRequestException(
                    message=message,
                    error_detail=ErrorDetail(
                        title="Email could not be analysed",
                        code="VALIDATION_ERROR",
                        status=400,
                        details=detail.details or [message],
                    ),
                ) from exc
            raise
        except ServiceUnavailableException as exc:
            # Normalise the detector-unavailable contract the extension pins
            # (code=SERVICE_UNAVAILABLE, not DETECTOR_UNAVAILABLE).
            detail = exc.error_detail
            if detail is not None and detail.code == "DETECTOR_UNAVAILABLE":
                raise ServiceUnavailableException(
                    message="The detection service is not available",
                    error_detail=ErrorDetail(
                        title="Detection service unavailable",
                        code="SERVICE_UNAVAILABLE",
                        status=503,
                        details=[
                            "No phishing-detection model is currently loaded."
                        ],
                    ),
                ) from exc
            raise

        label = extension_label(result)
        phishing_prob = float(result.phishing_probability)
        alert_threshold = float(inference_config.alert_threshold)
        should_alert = label == "SPAM" and phishing_prob >= alert_threshold

        prediction = ExtensionPrediction(
            predicted_label=label,
            confidence_score=_confidence_for_label(label, result),
            phishing_probability=phishing_prob,
            legitimate_probability=float(result.legitimate_probability),
            threshold_used=float(result.threshold),
            should_alert=should_alert,
            email_id=request.message_id,
            model_version=result.model_version or "unknown",
        )
        return ExtensionAnalysisResponse(
            email=ExtensionEmailIdRef(id=request.message_id),
            prediction=prediction,
        )
