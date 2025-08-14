# fxbot/run_live.py (apenas as linhas do main mudam)
from pathlib import Path
from core.config import load_config
from core.utils import import_from_path
from adapters.broker import Broker
from exec.execution import Executor
from logs.csv_logger import CSVLogger
from metrics.perf_metrics import PerformanceMetrics
from datetime import datetime, timezone

PKG_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    cfg = load_config(str(PKG_DIR / "config.yaml"))
    BrokerCls = import_from_path(cfg.broker.class_path)
    broker: Broker = BrokerCls(cfg.broker.params)
    broker.initialize()

    ml = None
    if cfg.ml_model:
        try:
            MLCls = import_from_path(cfg.ml_model["class_path"])
            ml = MLCls(**cfg.ml_model.get("params", {}))
        except FileNotFoundError as e:
            print(f"[ML] {e}\n[ML] Fallback para DummyModel.")
            from .ml.dummy_model import DummyModel
            ml = DummyModel()
    if ml:
        print(
            f"[ML] loaded={cfg.ml_model['class_path']} thr={getattr(ml,'threshold',None)}")

    Strat = import_from_path(cfg.strategy.class_path)
    strategy = Strat(**cfg.strategy.params)

    logger = CSVLogger()
    metrics = PerformanceMetrics()
    ex = Executor(cfg, broker, strategy, ml_model=ml, logger=logger, metrics=metrics)
    ex.start_session(datetime.now(timezone.utc), broker.account_equity())
    print("Live engine up.")

    import time
    try:
        while True:
            for s in cfg.symbols:
                try:
                    ex.step_symbol(s)
                except Exception as e:
                    print(f"[{s}] step error: {e}")
            try:
                ex.manage_open_positions()
                ex.maybe_summary_once()   # <<< chama o resumo quando der o horário
            except Exception as e:
                print("manage/summary error:", e)
            time.sleep(5)
    except KeyboardInterrupt:
        print("Interrupção manual, encerrando...")
    finally:
        logs_dir = PKG_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        metrics.export_json(logs_dir / f"metrics_{ts}.json")
