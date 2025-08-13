from pathlib import Path
from datetime import datetime
import json

class JSONSummary:
    def __init__(self, out_dir: Path | None = None):
        base = Path(__file__).resolve().parents[1]  # .../fxbot
        self.out_dir = out_dir or (base / "logs")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = None

    def start(self, session_start: datetime):
        ts = session_start.strftime("%Y%m%d_%H%M%S")
        self.path = self.out_dir / f"session_{ts}.json"

    def write(self, payload: dict):
        if self.path is None:
            # fallback: cria com timestamp atual
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.path = self.out_dir / f"session_{ts}.json"
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
