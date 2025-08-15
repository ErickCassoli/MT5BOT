import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

_LOGGER_NAME = "fxbot"
_CONFIGURED = False


def _configure_root() -> logging.Logger:
    """Configura o logger principal 'fxbot' apenas uma vez (console + arquivo rotativo)."""
    global _CONFIGURED
    root = logging.getLogger(_LOGGER_NAME)
    if _CONFIGURED:
        return root

    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Arquivo com rotação (5MB, 5 backups)
    file_handler = RotatingFileHandler(
        log_dir / "fxbot.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Nível default via env (run_live pode sobrescrever com cfg.log_level)
    env_level = os.getenv("FXBOT_LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, env_level, logging.INFO))
    root.propagate = False

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger filho do logger principal do projeto."""
    return _configure_root().getChild(name)


def set_level(level: str | int) -> None:
    """Atualiza dinamicamente o nível do logger raiz."""
    root = _configure_root()
    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), logging.INFO)
    else:
        lvl = int(level)
    root.setLevel(lvl)
