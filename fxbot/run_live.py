from pathlib import Path
from datetime import datetime, timezone
import argparse
import logging
import time

from core.config import load_config
from core.utils import import_from_path
from adapters.broker import Broker
from exec.execution import Executor
from core.logging import get_logger

PKG_DIR = Path(__file__).resolve().parent


def main() -> None:
    # CLI: permite escolher o arquivo de config
    parser = argparse.ArgumentParser(description="MT5BOT live runner")
    parser.add_argument(
        "-c",
        "--config",
        default=str(PKG_DIR / "config.yaml"),
        help="Caminho para o arquivo de configuração YAML (default: %(default)s)",
    )
    args = parser.parse_args()

    log = get_logger(__name__)

    # Carrega config
    cfg = load_config(args.config)

    # Ajusta nível do logger conforme config
    try:
        level = getattr(logging, cfg.log_level.upper(), logging.INFO)
        logging.getLogger("fxbot").setLevel(level)
    except Exception:
        log.exception("Falha ao ajustar nível de log; mantendo padrão INFO")

    # Broker
    BrokerCls = import_from_path(cfg.broker.class_path)
    broker: Broker = BrokerCls(cfg.broker.params)
    broker.initialize()

    # Modelo de ML (opcional)
    ml = None
    if cfg.ml_model:
        try:
            MLCls = import_from_path(cfg.ml_model["class_path"])
            ml = MLCls(**cfg.ml_model.get("params", {}))
        except FileNotFoundError as e:
            log.warning(f"[ML] {e}\n[ML] Fallback para DummyModel.")
            from .ml.dummy_model import DummyModel  # type: ignore[import-not-found]
            ml = DummyModel()
        except Exception:
            log.exception("[ML] Erro ao carregar modelo; seguindo sem ML")
            ml = None

    if ml:
        log.info(f"[ML] loaded={cfg.ml_model['class_path']} thr={getattr(ml, 'threshold', None)}")

    # Estratégia (injeta ml_model se existir)
    Strat = import_from_path(cfg.strategy.class_path)
    strategy = Strat(**cfg.strategy.params, ml_model=ml)

    # Executor
    ex = Executor(cfg, broker, strategy, ml_model=ml)
    ex.start_session(datetime.now(timezone.utc), broker.account_equity())
    log.info("Live engine up.")

    try:
        while True:
            for s in cfg.symbols:
                try:
                    ex.step_symbol(s)
                except Exception:
                    # inclui traceback para facilitar debug
                    log.exception(f"[{s}] erro no step")

            try:
                ex.manage_open_positions()
                ex.maybe_summary_once()
            except Exception:
                # não referenciar 's' aqui (fora do escopo do loop de símbolos)
                log.exception("Erro em manage_open_positions / maybe_summary_once")

            # Se o resumo já foi emitido (fim da sessão), encerra graciosamente
            if getattr(ex, "_summary_done", False):
                log.info("Sessão finalizada. Encerrando broker e saindo.")
                try:
                    shutdown = getattr(broker, "shutdown", None)
                    if callable(shutdown):
                        shutdown()
                except Exception:
                    logging.getLogger("fxbot").exception("Erro ao fechar broker")
                break

            time.sleep(5)
    except KeyboardInterrupt:
        log.info("Interrompido pelo usuário (Ctrl+C). Encerrando...")
        try:
            # Gera resumo antes de encerrar
            ex.maybe_summary_once(force=True)
        except Exception:
            log.exception("Erro ao gerar resumo final")
        try:
            shutdown = getattr(broker, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            logging.getLogger("fxbot").exception("Erro ao fechar broker no shutdown")


if __name__ == "__main__":
    main()
