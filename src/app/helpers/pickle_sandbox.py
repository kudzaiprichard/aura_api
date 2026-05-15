"""Phase 12 — pickle sandbox.

When ``inference.upload.pickle_sandbox_enabled`` is on, every uploaded
model artefact is unpickled inside a short-lived Python subprocess with a
wall-clock timeout. A malicious payload that executes arbitrary code on
unpickle is therefore contained to the child process — the API process
state stays untouched.

The sandbox is intentionally minimal: spawn ``python -c "<loader>"``
against a temp-file path, capture stdout/stderr, enforce the timeout.
The loader runs ``joblib.load`` (the same call the production pipeline
uses) and prints a one-line JSON summary so the caller can log what
showed up. We do *not* try to validate the object's structure here —
that's the job of the full upload pipeline (vectoriser shape check,
smoke predict) once it lands.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.configs import inference as inference_config


log = logging.getLogger("aura.upload.pickle_sandbox")


_LOADER_SCRIPT = r"""
import json, sys, traceback
try:
    import joblib
    obj = joblib.load(sys.argv[1])
    summary = {
        "ok": True,
        "type": type(obj).__name__,
        "module": type(obj).__module__,
    }
    sys.stdout.write(json.dumps(summary))
except Exception as e:
    sys.stdout.write(json.dumps({
        "ok": False,
        "type": type(e).__name__,
        "module": type(e).__module__,
        "message": str(e),
        "traceback": traceback.format_exc(limit=5),
    }))
    sys.exit(1)
"""


@dataclass
class SandboxResult:
    ok: bool
    type_name: str | None
    module: str | None
    message: str | None
    timed_out: bool
    returncode: int | None


def sandbox_enabled() -> bool:
    cfg = getattr(inference_config, "upload", None)
    return bool(cfg is not None and getattr(cfg, "pickle_sandbox_enabled", False))


def _sandbox_timeout() -> float:
    cfg = getattr(inference_config, "upload", None)
    if cfg is None:
        return 15.0
    try:
        return float(getattr(cfg, "pickle_sandbox_timeout_seconds", 15.0))
    except (TypeError, ValueError):
        return 15.0


async def sandboxed_joblib_load(content: bytes) -> SandboxResult:
    """Run ``joblib.load`` in a subprocess against ``content`` with a timeout.

    The bytes are written to a named temp file that is deleted once the
    subprocess exits. The child process inherits only the environment — we
    do not pipe stdin and never eval the content in-process.
    """
    timeout = _sandbox_timeout()
    with tempfile.NamedTemporaryFile(
        suffix=".joblib", delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(content)
        handle.flush()

    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _LOADER_SCRIPT, str(tmp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            log.warning(
                "pickle_sandbox timeout after %.1fs path=%s",
                timeout, tmp_path,
            )
            return SandboxResult(
                ok=False,
                type_name=None,
                module=None,
                message=f"Sandbox timed out after {timeout:.1f}s",
                timed_out=True,
                returncode=None,
            )

        rc = process.returncode
        payload_text = stdout.decode("utf-8", errors="replace").strip()
        try:
            payload = json.loads(payload_text) if payload_text else {}
        except json.JSONDecodeError:
            payload = {"ok": False, "message": payload_text}

        ok = bool(rc == 0 and payload.get("ok"))
        if not ok:
            err = stderr.decode("utf-8", errors="replace").strip()
            log.warning(
                "pickle_sandbox load failed rc=%s payload=%s stderr=%s",
                rc, payload, err[:500],
            )
        return SandboxResult(
            ok=ok,
            type_name=payload.get("type"),
            module=payload.get("module"),
            message=payload.get("message"),
            timed_out=False,
            returncode=rc,
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            log.debug("pickle_sandbox temp unlink failed path=%s", tmp_path)
