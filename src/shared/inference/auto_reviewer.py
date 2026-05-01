"""AutoReviewer — LLM-backed adjudicator for REVIEW-zone predictions.

Two providers are supported via direct HTTP (no SDKs):
- Groq (OpenAI-compatible `/v1/chat/completions`)
- Google AI Studio (Generative Language `generateContent`)

`review()` returns an `AutoReviewResponse`, which is the sealed union of
`AutoReviewSuccess` (the LLM produced a valid verdict) and `AutoReviewFailure`
(the call failed for any reason). `review()` never raises — all errors,
timeouts, and malformed responses are captured as `AutoReviewFailure`.

Typical caller:

    response = reviewer.review(sender, subject, body)
    if isinstance(response, AutoReviewSuccess):
        # every field guaranteed populated
        print(response.review_label.value, response.reasoning)
    else:
        # short message safe to show a user; full detail in technical_error
        print(response.user_message)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx

from src.shared.inference.schema import (
    AutoReviewFailure,
    AutoReviewResponse,
    AutoReviewSuccess,
    ConfidenceZone,
    LLMProvider,
    ReviewLabel,
)

if TYPE_CHECKING:
    from src.shared.inference.schema import PredictionResult

log = logging.getLogger('aura.inference')


_GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
_GROQ_DEFAULT_MODEL = 'llama-3.3-70b-versatile'
_GOOGLE_URL_TMPL = (
    'https://generativelanguage.googleapis.com/v1beta/models/'
    '{model}:generateContent'
)
_GOOGLE_DEFAULT_MODEL = 'gemini-3-flash-preview'

# Upstream errors (Google especially) return verbose JSON bodies; 200 chars
# truncates the useful part mid-sentence. 1000 is comfortable for diagnostics
# while keeping logs bounded.
_ERROR_BODY_SLICE = 1000


class _ProviderError(Exception):
    """Internal marker — caught by `review()` and surfaced as an
    `AutoReviewFailure`. Never raised to callers.

    Carries an optional `user_message` with a short, non-technical summary
    suitable for display. If not set, a generic message is used.
    """
    user_message: str | None = None
    retryable: bool = False


class AutoReviewer:
    def __init__(
        self,
        provider: LLMProvider,
        api_key: str,
        *,
        model_name: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
    ):
        if not isinstance(provider, LLMProvider):
            raise TypeError(
                f'provider must be LLMProvider, got {type(provider).__name__}'
            )
        if not isinstance(api_key, str) or not api_key:
            raise ValueError('api_key must be a non-empty string')
        if timeout_seconds <= 0:
            raise ValueError(f'timeout_seconds must be > 0, got {timeout_seconds!r}')
        if max_retries < 0:
            raise ValueError(f'max_retries must be >= 0, got {max_retries!r}')
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name or self._default_model(provider)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self._http_client = http_client

    @staticmethod
    def _default_model(provider: LLMProvider) -> str:
        if provider == LLMProvider.GROQ:
            return _GROQ_DEFAULT_MODEL
        if provider == LLMProvider.GOOGLE:
            return _GOOGLE_DEFAULT_MODEL
        raise ValueError(f'Unknown provider: {provider!r}')

    # -- public API ------------------------------------------------------
    def review(
        self,
        sender: str,
        subject: str,
        body: str,
        engineered_features: dict | None = None,
    ) -> AutoReviewResponse:
        prompt = self._build_prompt(sender, subject, body, engineered_features)
        try:
            parsed = self._call_provider_with_retries(prompt)
        except _ProviderError as e:
            log.warning('auto_reviewer: provider call failed: %s', e)
            return self._failure(e)
        # Validate the parsed dict has the required fields.
        try:
            label = parsed['label']
            confidence = parsed['confidence']
            reasoning = parsed['reasoning']
        except (KeyError, TypeError) as e:
            return AutoReviewFailure(
                user_message=(
                    'The AI reviewer returned an incomplete response. '
                    'Please try again.'
                ),
                technical_error=f'LLM response missing required field: {e}',
                provider=self.provider,
                model_name=self.model_name,
                raw_response=parsed if isinstance(parsed, dict) else None,
            )
        try:
            review_label = ReviewLabel(str(label).upper())
        except ValueError:
            return AutoReviewFailure(
                user_message=(
                    'The AI reviewer returned an unexpected verdict. '
                    'Please try again.'
                ),
                technical_error=f'LLM returned unrecognised label: {label!r}',
                provider=self.provider,
                model_name=self.model_name,
                raw_response=parsed,
            )
        return AutoReviewSuccess(
            review_label=review_label,
            reasoning=str(reasoning),
            confidence=str(confidence),
            provider=self.provider,
            model_name=self.model_name,
            raw_response=parsed,
        )

    def review_if_uncertain(
        self,
        prediction_result: 'PredictionResult',
        sender: str,
        subject: str,
        body: str,
    ) -> AutoReviewResponse | None:
        # Short-circuit: only REVIEW-zone predictions are worth the LLM call.
        # PredictionResult does not carry the raw email text (that would double
        # the in-memory footprint for every prediction), so the caller passes
        # the same (sender, subject, body) they used for predict().
        if prediction_result.confidence_zone != ConfidenceZone.REVIEW:
            return None
        return self.review(
            sender, subject, body,
            engineered_features=prediction_result.engineered_features,
        )

    # -- prompt ----------------------------------------------------------
    def _build_prompt(
        self,
        sender: str,
        subject: str,
        body: str,
        engineered_features: dict | None,
    ) -> str:
        lines: list[str] = [
            'You are a security analyst reviewing an email that an ML model '
            'flagged as uncertain between phishing and legitimate. Make a '
            'best-effort judgment based on the evidence below.',
            '',
            'Respond with ONLY a JSON object — no preamble, no markdown fences, '
            'no text outside the JSON. The schema is:',
            '{',
            '  "label": "PHISHING" | "LEGITIMATE" | "UNCERTAIN",',
            '  "confidence": "high" | "medium" | "low",',
            '  "reasoning": "<one or two sentences>"',
            '}',
            '',
            '--- Email ---',
            f'From: {sender}',
            f'Subject: {subject}',
            'Body:',
            body,
        ]
        if engineered_features:
            signals: list[tuple[str, Any]] = []
            for key in (
                'body_url_count',
                'body_url_density',
                'name_email_consistency',
                'domain_entropy',
            ):
                if key in engineered_features:
                    signals.append((key, engineered_features[key]))
            if signals:
                lines.append('')
                lines.append('--- Supporting signals (not primary evidence) ---')
                for k, v in signals:
                    lines.append(f'{k}: {v}')
        return '\n'.join(lines)

    # -- provider dispatch ----------------------------------------------
    def _call_provider_with_retries(self, prompt: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == LLMProvider.GROQ:
                    return self._call_groq(prompt)
                if self.provider == LLMProvider.GOOGLE:
                    return self._call_google(prompt)
                raise _ProviderError(f'Unknown provider: {self.provider!r}')
            except _ProviderError as e:
                last_err = e
                # Retry on transient errors (timeout, 5xx). Client errors
                # (4xx) are deterministic — don't retry.
                if not getattr(e, 'retryable', False):
                    raise
                log.info(
                    'auto_reviewer: transient error on attempt %d/%d: %s',
                    attempt + 1, self.max_retries + 1, e,
                )
        assert last_err is not None
        raise last_err

    def _client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        return httpx.Client(timeout=self.timeout_seconds)

    def _owns_client(self) -> bool:
        return self._http_client is None

    # -- provider calls --------------------------------------------------
    def _call_groq(self, prompt: str) -> dict:
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.0,
            'response_format': {'type': 'json_object'},
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        response = self._post(_GROQ_URL, json=payload, headers=headers)
        try:
            body = response.json()
        except ValueError as e:
            raise _ProviderError(f'Groq: response body is not JSON: {e}') from e
        try:
            content = body['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as e:
            raise _ProviderError(
                f'Groq: unexpected response shape: {e}'
            ) from e
        return self._parse_llm_content(content)

    def _call_google(self, prompt: str) -> dict:
        url = _GOOGLE_URL_TMPL.format(model=self.model_name)
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.0,
                'responseMimeType': 'application/json',
            },
        }
        params = {'key': self.api_key}
        response = self._post(url, json=payload, params=params)
        try:
            body = response.json()
        except ValueError as e:
            raise _ProviderError(f'Google: response body is not JSON: {e}') from e
        try:
            content = body['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError, TypeError) as e:
            raise _ProviderError(
                f'Google: unexpected response shape: {e}'
            ) from e
        return self._parse_llm_content(content)

    # -- low-level HTTP --------------------------------------------------
    def _post(
        self,
        url: str,
        *,
        json: dict,
        headers: dict | None = None,
        params: dict | None = None,
    ) -> httpx.Response:
        client = self._client()
        try:
            try:
                response = client.post(
                    url,
                    json=json,
                    headers=headers,
                    params=params,
                    timeout=self.timeout_seconds,
                )
            except httpx.TimeoutException as e:
                err = _ProviderError(f'timeout after {self.timeout_seconds}s')
                err.retryable = True  # type: ignore[attr-defined]
                err.user_message = (  # type: ignore[attr-defined]
                    'The AI reviewer did not respond in time. '
                    'Please try again.'
                )
                raise err from e
            except httpx.HTTPError as e:
                err = _ProviderError(f'HTTP transport error: {e}')
                err.retryable = True  # type: ignore[attr-defined]
                err.user_message = (  # type: ignore[attr-defined]
                    'Could not reach the AI reviewer. '
                    'Please check your connection and try again.'
                )
                raise err from e
            status = response.status_code
            if status >= 500:
                err = _ProviderError(
                    f'provider returned {status}: {response.text[:_ERROR_BODY_SLICE]}'
                )
                err.retryable = True  # type: ignore[attr-defined]
                err.user_message = (  # type: ignore[attr-defined]
                    'The AI reviewer is temporarily unavailable. '
                    'Please try again shortly.'
                )
                raise err
            if status >= 400:
                err = _ProviderError(
                    f'provider returned {status}: {response.text[:_ERROR_BODY_SLICE]}'
                )
                err.user_message = (  # type: ignore[attr-defined]
                    self._user_message_for_client_error(status)
                )
                raise err
            return response
        finally:
            if self._owns_client():
                client.close()

    @staticmethod
    def _user_message_for_client_error(status: int) -> str:
        if status in (401, 403):
            return (
                'The AI reviewer rejected the request '
                '(authentication or permission problem).'
            )
        if status == 429:
            return (
                'The AI reviewer is rate-limited or over quota. '
                'Please try again later.'
            )
        return (
            'The AI reviewer could not process the request. '
            'Please try again.'
        )

    @staticmethod
    def _parse_llm_content(content: Any) -> dict:
        if not isinstance(content, str):
            raise _ProviderError(
                f'LLM content is not a string: {type(content).__name__}'
            )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise _ProviderError(
                f'LLM content is not valid JSON: {e}'
            ) from e
        if not isinstance(parsed, dict):
            raise _ProviderError(
                f'LLM content parsed to {type(parsed).__name__}, expected object'
            )
        return parsed

    def _failure(self, e: _ProviderError) -> AutoReviewFailure:
        user_message = getattr(e, 'user_message', None) or (
            'The AI reviewer is unavailable right now. Please try again.'
        )
        return AutoReviewFailure(
            user_message=user_message,
            technical_error=str(e),
            provider=self.provider,
            model_name=self.model_name,
            raw_response=None,
        )