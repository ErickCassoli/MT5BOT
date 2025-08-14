import logging
from pathlib import Path


def _configure_root() -> logging.Logger:
    """Configure o logger principal 'fxbot' apenas uma vez."""
    root = logging.getLogger("fxbot")
    if not root.handlers:
        log_dir = Path(__file__).resolve().parents[1] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "fxbot.log", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)
        root.setLevel(logging.INFO)
    return root


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger filho do logger principal do projeto."""
    return _configure_root().getChild(name)
