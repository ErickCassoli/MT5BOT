from pydantic import BaseModel
from typing import List, Optional, Dict
import yaml

class SessionConfig(BaseModel):
    hours: int = 8
    start_hour: int = 8
    end_hour: int = 22
    profit_target_pct: float = 1.8
    loss_limit_pct: float = 3.0
    max_concurrent_trades: int = 3
    continue_after_target: bool = False
    allow_pyramiding: bool = False
    max_stack_per_symbol: int = 1      # 1 = não empilha (estado atual)
    min_stack_increase_r: float = 0.5  # só empilha se lucro >= 0.5R
    pyramiding_risk_scale: float = 0.5 # risco da nova entrada = 50% do base

class RiskConfig(BaseModel):
    risk_per_trade_pct: float = 0.6
    atr_mult_sl: float = 1.6
    atr_mult_tp: float = 3.2
    atr_trail_mult: float = 1.1
    partial_at_1r: float = 0.5
    risk_per_trade_pct_after_target: Optional[float] = None

class SpreadConfig(BaseModel):
    hard_cap_points: int = 25
    max_atr_ratio: float = 0.28

class StrategyConfig(BaseModel):
    class_path: str = "fxbot.strategies.donchian_breakout.DonchianBreakout"
    params: Dict = {}

class BrokerConfig(BaseModel):
    class_path: str = "fxbot.adapters.mt5.MT5Broker"
    params: Dict = {}

class AppConfig(BaseModel):
    symbols: List[str] = ["EURUSD","GBPUSD","USDJPY","USDCHF"]
    timeframe_exec: str = "M5"
    timeframe_regime: str = "H1"
    cooldown_minutes: int = 8
    magic: int = 5120812
    deviation_points: int = 20
    log_every_bar: bool = True
    strategy: StrategyConfig = StrategyConfig()
    broker: BrokerConfig = BrokerConfig()
    session: SessionConfig = SessionConfig()
    risk: RiskConfig = RiskConfig()
    spread: SpreadConfig = SpreadConfig()
    ml_model: Optional[Dict] = None

def load_config(path="config.yaml") -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
