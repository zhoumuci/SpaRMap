"""Initialize logging once at the public entry point."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import zy_log
except ModuleNotFoundError as exc:
    if exc.name != "zy_log":
        raise
    zy_log = None


def get_logger(name: str) -> logging.Logger:
    return zy_log.get_logger(name) if zy_log is not None else logging.getLogger(name)


def setup_logging(outdir: Path) -> Path:
    path = outdir / "sparmap.log"
    if zy_log is not None:
        return zy_log.setup_logging(path, level="INFO", show_startup=False)

    logger = logging.getLogger("sparmap")
    for handler in list(logger.handlers):
        if getattr(handler, "_sparmap_handler", False):
            logger.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        handler._sparmap_handler = True
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return path


def flush_logging() -> None:
    if zy_log is not None:
        zy_log.flush_logging()
    else:
        for handler in logging.getLogger("sparmap").handlers:
            handler.flush()


def shutdown_logging() -> None:
    if zy_log is not None:
        zy_log.shutdown_logging()
    else:
        logger = logging.getLogger("sparmap")
        for handler in list(logger.handlers):
            if getattr(handler, "_sparmap_handler", False):
                logger.removeHandler(handler)
                handler.close()
