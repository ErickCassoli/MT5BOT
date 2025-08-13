from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict
from datetime import datetime

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Bar:
    time: datetime
    o: float; h: float; l: float; c: float; v: int

@dataclass
class Signal:
    symbol: str
    side: Side
    confidence: float
    atr: float
    meta: Dict = None

@dataclass
class OrderRequest:
    symbol: str
    side: Side
    volume: float
    price: float
    sl: float
    tp: float
    comment: str
    magic: int

@dataclass
class PositionView:
    ticket: int
    symbol: str
    side: Side
    volume: float
    price_open: float
    sl: float
    tp: float
    magic: int
