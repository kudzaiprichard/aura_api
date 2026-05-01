"""Model registry.

Resolves model and vectoriser paths for a versioned layout:

    <models_root>/
        pipeline_components/
            subject_vectorizer.pkl
            body_vectorizer.pkl
        v<major>_<minor>/production/
            phishing_detector_mlp_classifier.pkl
            model_metadata.json
        model_metadata.json        # registry-level: active_version, versions

Addresses NOTEBOOK_CONTRACT §4.4 and the gaps from INFERENCE_GAPS.md §6.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

log = logging.getLogger('aura.inference')

_VERSION_RE = re.compile(r'^v(\d+)_(\d+)$')
_MODEL_FILE = 'phishing_detector_mlp_classifier.pkl'
_SUBJECT_VEC_FILE = 'subject_vectorizer.pkl'
_BODY_VEC_FILE = 'body_vectorizer.pkl'
_CALIBRATOR_FILE = 'calibrator.pkl'
_REGISTRY_METADATA = 'model_metadata.json'
_VERSION_METADATA = 'model_metadata.json'


def _parse_version(name: str) -> tuple[int, int] | None:
    m = _VERSION_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', dir=path.parent, delete=False, suffix='.tmp'
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


@dataclass(frozen=True)
class VersionPaths:
    model: Path
    subject_vectorizer: Path
    body_vectorizer: Path
    calibrator: Path | None = None

    def as_dict(self) -> dict[str, Path | None]:
        return {
            'model': self.model,
            'subject_vectorizer': self.subject_vectorizer,
            'body_vectorizer': self.body_vectorizer,
            'calibrator': self.calibrator,
        }


class ModelRegistry:
    def __init__(self, models_root: Path | str):
        self.models_root = Path(models_root).resolve()
        if not self.models_root.exists():
            raise FileNotFoundError(f'models_root does not exist: {self.models_root}')

    # -- discovery --------------------------------------------------------
    def list_versions(self) -> list[str]:
        versions: list[tuple[tuple[int, int], str]] = []
        for item in self.models_root.iterdir():
            if not item.is_dir():
                continue
            parsed = _parse_version(item.name)
            if parsed is None:
                continue
            model_file = item / 'production' / _MODEL_FILE
            if model_file.exists():
                versions.append((parsed, item.name))
        versions.sort(key=lambda x: x[0])
        return [name for _, name in versions]

    def active_version(self) -> str | None:
        meta = self._read_registry_metadata()
        return meta.get('active_version')

    def latest_version(self) -> str | None:
        versions = self.list_versions()
        return versions[-1] if versions else None

    # -- paths ------------------------------------------------------------
    def paths_for(self, version: str) -> dict[str, Path | None]:
        if _parse_version(version) is None:
            raise ValueError(f'Invalid version string: {version!r}')
        version_dir = self.models_root / version / 'production'
        model_path = version_dir / _MODEL_FILE
        subject_vec_path = self.models_root / 'pipeline_components' / _SUBJECT_VEC_FILE
        body_vec_path = self.models_root / 'pipeline_components' / _BODY_VEC_FILE
        for p in (model_path, subject_vec_path, body_vec_path):
            if not p.exists():
                raise FileNotFoundError(f'Missing artefact: {p}')
        calibrator_path = self.models_root / 'pipeline_components' / _CALIBRATOR_FILE
        calibrator_resolved = calibrator_path.resolve() if calibrator_path.exists() else None
        return VersionPaths(
            model=model_path.resolve(),
            subject_vectorizer=subject_vec_path.resolve(),
            body_vectorizer=body_vec_path.resolve(),
            calibrator=calibrator_resolved,
        ).as_dict()

    # -- metadata I/O -----------------------------------------------------
    def _registry_metadata_path(self) -> Path:
        return self.models_root / _REGISTRY_METADATA

    def _read_registry_metadata(self) -> dict[str, Any]:
        path = self._registry_metadata_path()
        if not path.exists():
            return {'active_version': None, 'versions': {}}
        return json.loads(path.read_text(encoding='utf-8'))

    def _write_registry_metadata(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self._registry_metadata_path(), payload)

    # -- active-version management ---------------------------------------
    def set_active(self, version: str, *, verify_integrity: bool = True) -> None:
        if version not in self.list_versions():
            raise ValueError(f'Unknown version: {version!r}')
        if verify_integrity:
            self._verify_integrity(version)
        meta = self._read_registry_metadata()
        meta['active_version'] = version
        self._write_registry_metadata(meta)
        log.info('active_version set to %s', version)

    def promote(self, version: str, metrics: dict[str, float]) -> None:
        if version not in self.list_versions():
            raise ValueError(f'Unknown version: {version!r}')
        meta = self._read_registry_metadata()
        versions = meta.setdefault('versions', {})
        entry = versions.setdefault(version, {})
        entry['metrics'] = dict(metrics)
        entry['promoted'] = True
        meta['active_version'] = version
        self._write_registry_metadata(meta)
        log.info('promoted version=%s metrics=%s', version, metrics)

    # -- registration -----------------------------------------------------
    def register_new_version(
        self,
        model,
        source_version: str,
        metrics: dict[str, float] | None = None,
        *,
        calibrator_path: Path | str | None = None,
    ) -> str:
        if _parse_version(source_version) is None:
            raise ValueError(f'Invalid source_version: {source_version!r}')
        new_version = self._next_version(source_version)
        version_dir = self.models_root / new_version / 'production'
        version_dir.mkdir(parents=True, exist_ok=False)
        model_path = version_dir / _MODEL_FILE
        joblib.dump(model, model_path)
        sha = _sha256(model_path)
        calibrator_sha: str | None = None
        if calibrator_path is not None:
            # The calibrator is a shared pipeline component, so it lives at
            # models/pipeline_components/calibrator.pkl rather than inside the
            # per-version directory (same convention as the vectorizers).
            src = Path(calibrator_path)
            if not src.exists():
                raise FileNotFoundError(f'calibrator_path does not exist: {src}')
            dest_dir = self.models_root / 'pipeline_components'
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / _CALIBRATOR_FILE
            shutil.copyfile(src, dest)
            calibrator_sha = _sha256(dest)
        version_meta_path = version_dir / _VERSION_METADATA
        version_meta: dict[str, Any] = {
            'source_version': source_version,
            'metrics': metrics or {},
            'sha256': sha,
        }
        if calibrator_sha is not None:
            version_meta['calibrator_sha256'] = calibrator_sha
        _atomic_write_json(version_meta_path, version_meta)
        meta = self._read_registry_metadata()
        versions = meta.setdefault('versions', {})
        versions[new_version] = {
            'source_version': source_version,
            'sha256': sha,
            'metrics': metrics or {},
            'promoted': False,
        }
        if calibrator_sha is not None:
            versions[new_version]['calibrator_sha256'] = calibrator_sha
        self._write_registry_metadata(meta)
        log.info('registered new_version=%s (source=%s)', new_version, source_version)
        return new_version

    def _next_version(self, source_version: str) -> str:
        parsed = _parse_version(source_version)
        if parsed is None:
            raise ValueError(f'Invalid source_version: {source_version!r}')
        major, _minor = parsed
        all_parsed = [
            _parse_version(v) for v in self.list_versions()
        ]
        max_minor_for_major = max(
            (p[1] for p in all_parsed if p is not None and p[0] == major),
            default=-1,
        )
        return f'v{major}_{max_minor_for_major + 1}'

    def _verify_integrity(self, version: str) -> None:
        paths = self.paths_for(version)
        meta = self._read_registry_metadata()
        entry = meta.get('versions', {}).get(version, {})
        recorded = entry.get('sha256')
        if recorded is not None:
            actual = _sha256(paths['model'])
            if actual != recorded:
                raise ValueError(
                    f'Integrity check failed for {version}: '
                    f'sha256 mismatch (expected {recorded[:12]}.., got {actual[:12]}..).'
                )
        recorded_calibrator = entry.get('calibrator_sha256')
        calibrator_path = paths.get('calibrator')
        if recorded_calibrator is not None and calibrator_path is not None:
            actual_calibrator = _sha256(calibrator_path)
            if actual_calibrator != recorded_calibrator:
                raise ValueError(
                    f'Integrity check failed for {version} calibrator: '
                    f'sha256 mismatch (expected {recorded_calibrator[:12]}.., '
                    f'got {actual_calibrator[:12]}..).'
                )


def default_models_root() -> Path:
    env = os.environ.get('AURA_MODELS_DIR')
    if env:
        return Path(env).resolve()
    candidate = Path.cwd() / 'models'
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        'Could not locate models directory. Set AURA_MODELS_DIR or run from a '
        'directory that contains a `models/` folder.'
    )
