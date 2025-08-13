from abc import ABC, abstractmethod
from typing import Optional, Dict
import pandas as pd
from core.types import Signal


class Strategy(ABC):
    def __init__(self, params: Dict = None, ml_model=None):
        self.params = params or {}
        self.ml = ml_model

    @abstractmethod
    def generate_signal(self, symbol: str, df_exec: pd.DataFrame, df_regime: pd.DataFrame) -> Optional[Signal]:
        ...
