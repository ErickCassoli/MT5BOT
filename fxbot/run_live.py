# fxbot/run_live.py (apenas as linhas do main mudam)
from pathlib import Path
from core.config import load_config
from core.utils import import_from_path
from adapters.broker import Broker
from exec.execution import Executor
from logs.csv_logger import CSVLogger
from datetime import datetime, timezone
from core.logging import get_logger

PKG_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    log = get_logger(__name__)
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
            log.warning(f"[ML] {e}\n[ML] Fallback para DummyModel.")
            from .ml.dummy_model import DummyModel
            ml = DummyModel()
    if ml:
        log.info(
            f"[ML] loaded={cfg.ml_model['class_path']} thr={getattr(ml,'threshold',None)}")

    Strat = import_from_path(cfg.strategy.class_path)
    strategy = Strat(**cfg.strategy.params)

    csv_logger = CSVLogger()
    ex = Executor(cfg, broker, strategy, ml_model=ml, logger=csv_logger)
    ex.start_session(datetime.now(timezone.utc), broker.account_equity())
    log.info("Live engine up.")

    import time
    while True:
        for s in cfg.symbols:
            try:
                ex.step_symbol(s)
            except Exception as e:
                log.error(f"[{s}] step error: {e}")
        try:
            ex.manage_open_positions()
            ex.maybe_summary_once()   # <<< chama o resumo quando der o horário
        except Exception as e:
            log.error("manage/summary error: %s", e)
        time.sleep(5)
