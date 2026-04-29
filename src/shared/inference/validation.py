"""Input validators. Every function raises ValidationError on failure."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from src.shared.inference.schema import ValidationError


def _assert_text(name: str, value: Any) -> str:
    if value is None:
        raise ValidationError(f'{name}: must not be None')
    if not isinstance(value, str):
        raise ValidationError(f'{name}: must be a string, got {type(value).__name__}')
    return value


def validate_email_inputs(sender: Any, subject: Any, body: Any) -> None:
    sender_s = _assert_text('sender', sender)
    subject_s = _assert_text('subject', subject)
    body_s = _assert_text('body', body)
    if sender_s.strip() == '' and subject_s.strip() == '' and body_s.strip() == '':
        raise ValidationError('sender, subject, and body are all empty')


def validate_threshold(threshold: Any) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValidationError(
            f'threshold must be a float in [0, 1], got {type(threshold).__name__}'
        )
    if threshold < 0.0 or threshold > 1.0:
        raise ValidationError(
            f'threshold must be in [0, 1], got {threshold!r}'
        )


def validate_review_thresholds(low: Any, high: Any) -> None:
    for name, v in (('review_low_threshold', low), ('review_high_threshold', high)):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValidationError(
                f'{name} must be a float in [0, 1], got {type(v).__name__}'
            )
        if v < 0.0 or v > 1.0:
            raise ValidationError(
                f'{name} must be in [0, 1], got {v!r}'
            )
    if low >= high:
        raise ValidationError(
            f'review_low_threshold ({low!r}) must be strictly less than '
            f'review_high_threshold ({high!r})'
        )


def validate_training_batch(
    emails: Iterable[dict],
    *,
    min_per_class: int = 1,
) -> None:
    emails_list = list(emails)
    if len(emails_list) == 0:
        raise ValidationError('training batch is empty')
    labels: list[int] = []
    for i, e in enumerate(emails_list):
        if not isinstance(e, dict):
            raise ValidationError(
                f'email[{i}]: expected dict, got {type(e).__name__}'
            )
        for field in ('sender', 'subject', 'body', 'label'):
            if field not in e:
                raise ValidationError(f'email[{i}]: missing field {field!r}')
        label = e['label']
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValidationError(
                f'email[{i}]: label must be int 0 or 1, got {label!r}'
            )
        if label not in (0, 1):
            raise ValidationError(
                f'email[{i}]: label must be 0 or 1, got {label}'
            )
        for field in ('sender', 'subject', 'body'):
            if not isinstance(e[field], str):
                raise ValidationError(
                    f'email[{i}]: field {field!r} must be str, got '
                    f'{type(e[field]).__name__}'
                )
        labels.append(label)
    counts = Counter(labels)
    for cls in (0, 1):
        if counts[cls] < min_per_class:
            raise ValidationError(
                f'training batch is unbalanced: class {cls} has '
                f'{counts[cls]} samples, need at least {min_per_class}'
            )
