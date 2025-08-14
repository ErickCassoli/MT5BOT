import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import load_config

_initialized = False


def _setup_root_logger():
    global _initialized
    if _initialized:
        return
    base = Path(__file__).resolve().parents[1]
    cfg_path = base / "config.yaml"
    try:
        cfg = load_config(str(cfg_path))
        level_name = getattr(cfg, "log_level", "INFO")
    except Exception:
        level_name = "INFO"
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logger = logging.getLogger("fxbot")
    logger.setLevel(level)
    logs_dir = base / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(logs_dir / "fxbot.log", maxBytes=1_000_000, backupCount=5)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    _initialized = True


def get_logger(name: str = "fxbot") -> logging.Logger:
    _setup_root_logger()
    return logging.getLogger(name)


# logger padrão do pacote
logger = get_logger("fxbot")
