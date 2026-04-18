import sys
import logging
from src.configs.loader import load_config, ConfigError
from src.configs.generate import generate_stub

logger = logging.getLogger("configs")
# Stderr-only handler for the brief window between config import and lifespan startup.
# `_setup_logging()` in src/core/lifespan.py later replaces all handlers via force=True
# so config-bootstrap messages reach stderr but are not duplicated into the log file.
logging.basicConfig(level=logging.INFO)


def _boot_config():
    """Load config with clean error reporting."""
    logger.info("Loading system configs...")

    try:
        sections = load_config()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ConfigError as e:
        logger.error("Found %d configuration error(s):", len(e.errors))
        for err in e.errors:
            logger.error("  %s", err)
        logger.error("Fix the above in your .env or application.yaml and restart.")
        sys.exit(1)

    for name in sections:
        logger.info("Loaded section: %s", name)

    logger.info("Config loaded successfully (%d sections)", len(sections))

    # Only regenerate the IDE stub in development. In production / read-only
    # filesystems / containers, writing into the source tree is undesirable.
    application_section = sections.get("application")
    environment = getattr(application_section, "environment", "development")
    if environment == "development":
        try:
            generate_stub()
        except Exception as e:
            logger.warning("Stub generation failed: %s", e)

    return sections


_sections = _boot_config()

for _name, _ns in _sections.items():
    globals()[_name] = _ns


def reload_config() -> None:
    """Reload configuration from YAML and regenerate stub."""
    global _sections
    _sections = _boot_config()
    for name, ns in _sections.items():
        globals()[name] = ns
