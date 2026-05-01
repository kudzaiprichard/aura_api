"""DriftMonitor — append-only JSONL bookkeeping of predictions vs. confirmations.

Maintains a running 2x2 confusion matrix and surfaces a typed `DriftSignal`
when the false-positive rate crosses a configured threshold. State is
reconstructed from the JSONL log on construction, so the monitor survives
process restarts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import portalocker

log = logging.getLogger('aura.inference')

_RECORD_PREDICTION = 'prediction'
_RECORD_CONFIRMATION = 'confirmation'


class DriftStatus(str, Enum):
    OK = 'OK'
    WARNING = 'WARNING'


@dataclass
class DriftSignal:
    status: DriftStatus
    false_positive_rate: float
    total_predictions: int
    confirmed_predictions: int
    threshold: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        return d


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DriftMonitor:
    """Concurrency model: append-only writes are serialised via a
    portalocker exclusive lock on the log file itself. In-memory state is
    per-instance and not thread-safe; use one instance per thread/process.
    """

    def __init__(
        self,
        log_path: str | Path,
        *,
        fpr_threshold: float = 0.10,
    ):
        if not isinstance(fpr_threshold, (int, float)) or isinstance(fpr_threshold, bool):
            raise TypeError(
                f'fpr_threshold must be a number, got {type(fpr_threshold).__name__}'
            )
        if fpr_threshold < 0.0 or fpr_threshold > 1.0:
            raise ValueError(f'fpr_threshold must be in [0, 1], got {fpr_threshold!r}')
        self.log_path = Path(log_path)
        self.fpr_threshold = float(fpr_threshold)
        # In-memory state: id → (predicted_label) for unconfirmed predictions.
        self._pending: dict[str, int] = {}
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0
        self._total_predictions = 0
        self._confirmed_predictions = 0
        self._confirmed_ids: set[str] = set()
        self._replay_from_disk()

    # -- replay ----------------------------------------------------------
    def _replay_from_disk(self) -> None:
        if not self.log_path.exists():
            return
        # Use a portalocker Lock (not a plain open) so that replay blocks while
        # another process holds the exclusive write lock. Without this, Windows
        # would raise PermissionError mid-replay when a concurrent writer is
        # active on the same log.
        with portalocker.Lock(
            str(self.log_path),
            mode='r',
            encoding='utf-8',
            timeout=60,
        ) as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(
                        'drift_monitor: skipping malformed line %d in %s',
                        lineno, self.log_path,
                    )
                    continue
                kind = record.get('type')
                if kind == _RECORD_PREDICTION:
                    self._apply_prediction_in_memory(
                        record['prediction_id'], int(record['predicted_label'])
                    )
                elif kind == _RECORD_CONFIRMATION:
                    pid = record['prediction_id']
                    confirmed = int(record['confirmed_label'])
                    self._apply_confirmation_in_memory(pid, confirmed, replay=True)
                else:
                    log.warning(
                        'drift_monitor: unknown record type %r at line %d',
                        kind, lineno,
                    )

    # -- in-memory state transitions -------------------------------------
    def _apply_prediction_in_memory(self, prediction_id: str, predicted_label: int) -> None:
        self._pending[prediction_id] = predicted_label
        self._total_predictions += 1

    def _apply_confirmation_in_memory(
        self, prediction_id: str, confirmed_label: int, *, replay: bool,
    ) -> None:
        predicted = self._pending.pop(prediction_id, None)
        if predicted is None:
            if replay:
                # Log predates the confirmation or is corrupt — skip rather
                # than raise, so replay never hard-fails on a partial file.
                log.warning(
                    'drift_monitor: confirmation for unknown prediction_id=%s during replay',
                    prediction_id,
                )
                return
            raise ValueError(
                f'Unknown prediction_id={prediction_id!r}: no matching prediction '
                f'has been recorded on this monitor.'
            )
        if prediction_id in self._confirmed_ids:
            # Duplicate confirmation on the same id — ignore so replay is idempotent.
            if replay:
                return
            raise ValueError(
                f'prediction_id={prediction_id!r} has already been confirmed.'
            )
        self._confirmed_ids.add(prediction_id)
        self._confirmed_predictions += 1
        if predicted == 1 and confirmed_label == 1:
            self._tp += 1
        elif predicted == 0 and confirmed_label == 0:
            self._tn += 1
        elif predicted == 1 and confirmed_label == 0:
            self._fp += 1
        elif predicted == 0 and confirmed_label == 1:
            self._fn += 1
        else:
            raise ValueError(
                f'predicted_label and confirmed_label must be 0 or 1, got '
                f'predicted={predicted!r}, confirmed={confirmed_label!r}'
            )

    # -- disk writes -----------------------------------------------------
    def _append_record(self, record: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, ensure_ascii=False) + '\n'
        # Exclusive lock on the log file itself — the file is both the lock
        # and the data store. `mode='a'` ensures the write lands at EOF even
        # when multiple writers are active.
        with portalocker.Lock(
            str(self.log_path),
            mode='a',
            encoding='utf-8',
            timeout=60,
        ) as f:
            f.write(payload)
            f.flush()

    # -- public API ------------------------------------------------------
    def record_prediction(
        self,
        prediction_id: str,
        predicted_label: int,
        predicted_probability: float,
        model_version: str,
        timestamp: str | None = None,
    ) -> None:
        if not isinstance(prediction_id, str) or not prediction_id:
            raise ValueError('prediction_id must be a non-empty string')
        if predicted_label not in (0, 1):
            raise ValueError(f'predicted_label must be 0 or 1, got {predicted_label!r}')
        record = {
            'type': _RECORD_PREDICTION,
            'prediction_id': prediction_id,
            'predicted_label': int(predicted_label),
            'predicted_probability': float(predicted_probability),
            'model_version': model_version,
            'timestamp': timestamp or _utcnow_iso(),
        }
        self._append_record(record)
        self._apply_prediction_in_memory(prediction_id, int(predicted_label))

    def record_confirmation(
        self,
        prediction_id: str,
        confirmed_label: int,
        timestamp: str | None = None,
    ) -> None:
        if not isinstance(prediction_id, str) or not prediction_id:
            raise ValueError('prediction_id must be a non-empty string')
        if confirmed_label not in (0, 1):
            raise ValueError(f'confirmed_label must be 0 or 1, got {confirmed_label!r}')
        # Apply first so we fail BEFORE writing on unknown ids.
        self._apply_confirmation_in_memory(
            prediction_id, int(confirmed_label), replay=False
        )
        record = {
            'type': _RECORD_CONFIRMATION,
            'prediction_id': prediction_id,
            'confirmed_label': int(confirmed_label),
            'timestamp': timestamp or _utcnow_iso(),
        }
        self._append_record(record)

    def confusion_matrix(self) -> dict[str, int]:
        return {
            'tp': self._tp,
            'tn': self._tn,
            'fp': self._fp,
            'fn': self._fn,
        }

    def false_positive_rate(self) -> float:
        denom = self._fp + self._tn
        if denom == 0:
            return 0.0
        return self._fp / denom

    def drift_signal(self) -> DriftSignal:
        fpr = self.false_positive_rate()
        exceeded = fpr > self.fpr_threshold
        status = DriftStatus.WARNING if exceeded else DriftStatus.OK
        if exceeded:
            message = (
                f'FPR {fpr:.4f} exceeds threshold {self.fpr_threshold:.4f} '
                f'— model may need retraining'
            )
        else:
            message = (
                f'FPR {fpr:.4f} within threshold {self.fpr_threshold:.4f}'
            )
        return DriftSignal(
            status=status,
            false_positive_rate=fpr,
            total_predictions=self._total_predictions,
            confirmed_predictions=self._confirmed_predictions,
            threshold=self.fpr_threshold,
            message=message,
        )
