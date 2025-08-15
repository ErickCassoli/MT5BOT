import math
from typing import Optional

from adapters.broker import Broker
from core.types import OrderRequest, Side


def _decimals_from_step(step: float) -> int:
    """
    Estima quantas casas decimais devem ser usadas para arredondar um valor
    quantizado por `step` (ex.: 0.01 -> 2, 0.001 -> 3, 1.0 -> 0).
    """
    if step >= 1 or step <= 0:
        return 0
    s = f"{step:.10f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


class RiskManager:
    """
    Responsável por sanitizar SL/TP conforme restrições do símbolo e
    dimensionar o lote pelo risco desejado.
    """

    def __init__(self, broker: Broker, cfg):
        self.broker = broker
        self.cfg = cfg

    def _sanitize_sltp(self, symbol: str, side: Side, entry: float, sl: float, tp: float):
        """
        Garante que SL/TP respeitem:
        - relação correta com o preço de entrada (SL < entry no BUY; SL > entry no SELL; TP inverso)
        - distância mínima em points (max(spread*1.2, trade_stops_level+1, 10))
        - arredondamento ao número de dígitos do símbolo
        """
        pt = self.broker.get_point(symbol)
        digits = self.broker.get_digits(symbol)

        # distância mínima em points: spread*1.2 vs stop_level da corretora vs 10 (fallback),
        # adicionando +1 pto como margem para evitar erros 10016 (invalid stops).
        info = self.broker.symbol_info(symbol)
        stop_level = int(getattr(info, "trade_stops_level", 0) or 0)
        spread_pts = int(self.broker.get_spread_points(symbol) or 0)
        min_pts = max(int(spread_pts * 1.2), stop_level + 1, 10)
        min_dist = min_pts * pt

        if side == Side.BUY:
            # forçar relações corretas
            sl = min(sl, entry - pt)
            tp = max(tp, entry + pt)
            # aplicar distância mínima
            if (entry - sl) < min_dist:
                sl = entry - min_dist
            if (tp - entry) < min_dist:
                tp = entry + min_dist
        else:
            sl = max(sl, entry + pt)
            tp = min(tp, entry - pt)
            if (sl - entry) < min_dist:
                sl = entry + min_dist
            if (entry - tp) < min_dist:
                tp = entry - min_dist

        return round(entry, digits), round(sl, digits), round(tp, digits)

    def lot_by_risk(self, symbol: str, stop_distance_price: float, risk_pct: float) -> float:
        """
        Calcula o lote para que a perda ao atingir o SL ≈ equity * (risk_pct/100).
        Respeita volume_min/max e quantização por volume_step do símbolo.
        """
        info = self.broker.symbol_info(symbol)
        if not info or stop_distance_price <= 0:
            return 0.0

        tick_value = float(info.trade_tick_value or 0.0)
        tick_size = float(info.trade_tick_size or 0.0)
        if tick_value <= 0.0 or tick_size <= 0.0:
            return 0.0

        loss_per_lot = (stop_distance_price / tick_size) * tick_value  # $ por 1.0 lote se bater SL
        equity = float(self.broker.account_equity())
        risk_money = equity * (float(risk_pct) / 100.0)

        lots_raw = risk_money / max(loss_per_lot, 1e-9)

        step = float(info.volume_step or 0.01)
        vol_min = float(info.volume_min or 0.01)
        vol_max = float(info.volume_max or 100.0)

        # quantizar para baixo no múltiplo de step
        lots_q = math.floor(lots_raw / step) * step
        lots_q = min(max(vol_min, lots_q), vol_max)

        # arredondar de acordo com o step (em vez de fixar 2 casas)
        dec = _decimals_from_step(step)
        return round(lots_q, dec)

    def build_order(
        self,
        symbol: str,
        side: Side,
        atr_value: float,
        confidence: float,
        magic: int,
        risk_pct: Optional[float] = None,
    ) -> OrderRequest:
        """
        Monta a ordem com entry/SL/TP baseados em múltiplos de ATR e
        volume dimensionado pelo risco desejado.
        """
        t = self.broker.symbol_info_tick(symbol)
        if not t:
            raise RuntimeError(f"symbol_info_tick falhou para {symbol}")

        entry = t.ask if side == Side.BUY else t.bid

        sl = (
            entry - self.cfg.atr_mult_sl * atr_value
            if side == Side.BUY
            else entry + self.cfg.atr_mult_sl * atr_value
        )
        tp = (
            entry + self.cfg.atr_mult_tp * atr_value
            if side == Side.BUY
            else entry - self.cfg.atr_mult_tp * atr_value
        )

        entry, sl, tp = self._sanitize_sltp(symbol, side, entry, sl, tp)

        rp = float(risk_pct) if risk_pct is not None else float(self.cfg.risk_per_trade_pct)
        lots = self.lot_by_risk(symbol, abs(entry - sl), rp)

        return OrderRequest(
            symbol=symbol,
            side=side,
            volume=lots,
            price=entry,
            sl=sl,
            tp=tp,
            comment=f"py-modular conf={confidence:.2f}",
            magic=magic,
        )
