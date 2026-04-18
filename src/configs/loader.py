import os
import re
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


# ──────────────────────────────────────────────
# Type casting
# ──────────────────────────────────────────────

_TYPE_CASTERS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": lambda v: str(v).lower() in ("true", "1", "yes", "on"),
    "list": lambda v: [
        item.strip() for item in str(v).split(",") if item.strip()
    ],
}


def _cast(value: Any, type_hint: str, key_path: str) -> Any:
    """Cast a resolved value to the declared type."""
    caster = _TYPE_CASTERS.get(type_hint)
    if caster is None:
        raise ValueError(f"{key_path}: unknown type '{type_hint}'")
    try:
        return caster(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{key_path}: expected {type_hint}, got '{value}' — {e}")


# ──────────────────────────────────────────────
# Env var resolution
# ──────────────────────────────────────────────

_ENV_PATTERN = re.compile(r"^\$\{([^}]+)\}$")


def _resolve_value(raw: str) -> tuple[Optional[str], bool]:
    """
    Resolve a ${VAR} or ${VAR:default} placeholder.

    Returns:
        (resolved_value, was_from_env)
    """
    match = _ENV_PATTERN.match(raw)
    if not match:
        return raw, False

    expr = match.group(1)

    if ":" in expr:
        var_name, default = expr.split(":", 1)
        value = os.environ.get(var_name, default)
        return value, var_name in os.environ
    else:
        value = os.environ.get(expr)
        if value is None:
            return None, False
        return value, True


# ──────────────────────────────────────────────
# Pipe format parsing
# ──────────────────────────────────────────────

def _parse_pipe(raw: str) -> tuple[str, str, bool]:
    """
    Parse a pipe-delimited config value.

    Format: "${VAR:default} | type" or "${VAR} | type | required"

    Returns:
        (value_part, type_hint, is_required)
    """
    parts = [p.strip() for p in raw.rsplit("|", raw.count("|"))]

    if len(parts) == 3:
        return parts[0], parts[1], parts[2].lower() == "required"
    elif len(parts) == 2:
        return parts[0], parts[1], False
    else:
        return raw, "str", False


def _is_leaf(node: Any) -> bool:
    """Check if a YAML node is a pipe-formatted leaf string."""
    return isinstance(node, str) and "|" in node


def _is_section(node: Any) -> bool:
    """Check if a YAML node is a nested section."""
    return isinstance(node, dict)


# ──────────────────────────────────────────────
# Core loader
# ──────────────────────────────────────────────

def _process_node(
    node: Any,
    path: str,
    errors: List[str],
) -> Any:
    """Recursively process a YAML node into resolved, typed values."""
    if _is_section(node):
        ns = SimpleNamespace()
        for key, child in node.items():
            child_path = f"{path}.{key}" if path else key
            python_key = key.replace("-", "_").replace(" ", "_")
            setattr(ns, python_key, _process_node(child, child_path, errors))
        return ns

    if _is_leaf(node):
        value_part, type_hint, required = _parse_pipe(node)
        resolved, from_env = _resolve_value(value_part)

        if resolved is None:
            if required:
                errors.append(f"{path}: required but not set")
            return None

        try:
            return _cast(resolved, type_hint, path)
        except ValueError as e:
            errors.append(str(e))
            return None

    # Plain value (no pipe) — return as-is
    return node


def load_config(
    config_path: Optional[str] = None,
    env_path: Optional[str] = None,
) -> Dict[str, SimpleNamespace]:
    """
    Load YAML config, resolve env vars, validate types.

    Args:
        config_path: Path to application.yaml (defaults to ./application.yaml)
        env_path: Path to .env file (defaults to project root .env)

    Returns:
        Dict mapping top-level section names to SimpleNamespace objects.

    Raises:
        FileNotFoundError: If application.yaml is missing.
        ConfigError: If any required values are missing or types are invalid.
    """
    configs_dir = Path(__file__).resolve().parent
    if config_path is None:
        config_path = configs_dir / "application.yaml"
    else:
        config_path = Path(config_path)

    if env_path is None:
        env_path = configs_dir.parent.parent / ".env"
    else:
        env_path = Path(env_path)

    # Load .env (silently skip if missing — env vars may come from system)
    if env_path.exists():
        load_dotenv(env_path)

    # Load YAML
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f) or {}

    # Process all sections
    errors: List[str] = []
    sections: Dict[str, SimpleNamespace] = {}

    for section_name, section_data in raw_config.items():
        sections[section_name] = _process_node(
            section_data, section_name, errors
        )

    # Fail fast
    if errors:
        raise ConfigError(errors)

    return sections
