from pathlib import Path
import csv
from datetime import datetime
from typing import Optional

class CSVLogger:
    def __init__(self, logs_dir: Optional[Path] = None):
        base = Path(__file__).resolve().parents[1]   # .../fxbot
        self.logs_dir = logs_dir or (base / "logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = None
        self._writer = None

    def start(self, session_start: datetime, baseline_equity: float):
        ts = session_start.strftime("%Y%m%d_%H%M%S")
        self.path = self.logs_dir / f"session_{ts}.csv"
        newfile = not self.path.exists()
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        # inclui coluna de estratégia para identificar qual classe gerou o log
        self._writer = csv.DictWriter(self._fh, fieldnames=[
            "ts","event","strategy","symbol","side","volume","price","sl","tp","retcode","comment","atr",
            "dist_up","dist_low","near_thr","adx_h1","confidence","ticket","extra"
        ])
        if newfile: self._writer.writeheader()
        self.log_event("session_start", extra=f"baseline={baseline_equity}")

    def _write(self, row: dict):
        row["ts"] = datetime.utcnow().isoformat()
        self._writer.writerow(row); self._fh.flush()

    def log_event(self, event:str, **kwargs): self._write({"event":event, **kwargs})
    def log_signal(self, symbol, side, atr, conf, dist_up=None, dist_low=None, near_thr=None,
                   adx_h1=None, strategy: str | None = None, extra=""):
        """Registra um sinal gerado pela estratégia."""
        self._write({
            "event": "signal", "strategy": strategy, "symbol": symbol, "side": side,
            "atr": atr, "confidence": conf, "dist_up": dist_up, "dist_low": dist_low,
            "near_thr": near_thr, "adx_h1": adx_h1, "extra": extra
        })

    def log_order(self, symbol, side, volume, price, sl, tp, retcode, comment="",
                  ticket=None, strategy: str | None = None):
        """Registra o resultado de uma ordem enviada ao broker."""
        self._write({
            "event": "order", "strategy": strategy, "symbol": symbol, "side": side,
            "volume": volume, "price": price, "sl": sl, "tp": tp,
            "retcode": retcode, "comment": comment, "ticket": ticket
        })
    def log_partial(self, symbol, ticket, volume): 
        self._write({"event":"partial_close","symbol":symbol,"ticket":ticket,"volume":volume})
    def log_sltp(self, symbol, ticket, sl, tp):
        self._write({"event":"sltp_update","symbol":symbol,"ticket":ticket,"sl":sl,"tp":tp})
    def log_summary(self, text: str, strategy: str | None = None):
        """Resumo final da sessão com identificação da estratégia."""
        self._write({"event": "summary", "strategy": strategy, "extra": text})
