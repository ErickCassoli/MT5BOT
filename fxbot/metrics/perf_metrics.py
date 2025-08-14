from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import csv
import json
from typing import Any, Dict, List, Optional

try:
    from prometheus_client import CollectorRegistry, Counter, generate_latest  # type: ignore
except Exception:  # pragma: no cover - prometheus optional
    CollectorRegistry = Counter = generate_latest = None  # type: ignore


@dataclass
class StrategyMetrics:
    name: str
    signals: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)

    def _append(self, bucket: List[Dict[str, Any]], data: Dict[str, Any]):
        row = {"ts": datetime.utcnow().isoformat()}
        row.update(data)
        bucket.append(row)

    def log_signal(self, **data):
        self._append(self.signals, data)

    def log_order(self, **data):
        self._append(self.orders, data)

    def log_result(self, **data):
        self._append(self.results, data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "signals": self.signals,
            "orders": self.orders,
            "results": self.results,
        }

    def export_json(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def export_csv(self, out_dir: Path | str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.signals:
            self._write_csv(out / f"{self.name}_signals.csv", self.signals)
        if self.orders:
            self._write_csv(out / f"{self.name}_orders.csv", self.orders)
        if self.results:
            self._write_csv(out / f"{self.name}_results.csv", self.results)

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]]):
        fields: List[str] = sorted({k for r in rows for k in r.keys()})
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


class PerformanceMetrics:
    """Coletor de métricas por estratégia."""

    def __init__(self, enable_prometheus: bool = False):
        self._strategies: Dict[str, StrategyMetrics] = {}
        self._prom = None
        if enable_prometheus and CollectorRegistry:
            reg = CollectorRegistry()
            self._prom = {
                "registry": reg,
                "signals": Counter("fxbot_signals_total", "Sinais gerados", ["strategy"], registry=reg),
                "orders": Counter("fxbot_orders_total", "Ordens enviadas", ["strategy"], registry=reg),
                "results": Counter("fxbot_results_total", "Resumos registrados", ["strategy"], registry=reg),
            }

    def _get(self, name: str) -> StrategyMetrics:
        if name not in self._strategies:
            self._strategies[name] = StrategyMetrics(name)
        return self._strategies[name]

    def log_signal(self, strategy: str, **data):
        self._get(strategy).log_signal(**data)
        if self._prom:
            self._prom["signals"].labels(strategy=strategy).inc()

    def log_order(self, strategy: str, **data):
        self._get(strategy).log_order(**data)
        if self._prom:
            self._prom["orders"].labels(strategy=strategy).inc()

    def log_result(self, strategy: str, **data):
        self._get(strategy).log_result(**data)
        if self._prom:
            self._prom["results"].labels(strategy=strategy).inc()

    def to_dict(self) -> Dict[str, Any]:
        return {name: sm.to_dict() for name, sm in self._strategies.items()}

    def export_json(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def export_csv(self, out_dir: Path | str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for sm in self._strategies.values():
            sm.export_csv(out)

    # ---- prometheus ----
    def prometheus_registry(self):
        return self._prom["registry"] if self._prom else None

    def prometheus_metrics(self) -> Optional[bytes]:
        if self._prom and generate_latest:
            return generate_latest(self._prom["registry"])
        return None
