from typing import Optional
from fxbot.core.types import OrderRequest, Side
from fxbot.adapters.broker import Broker


class RiskManager:
    def __init__(self, broker: Broker, cfg):
        self.broker = broker
        self.cfg = cfg

    def _sanitize_sltp(self, symbol, side: Side, entry: float, sl: float, tp: float):
        pt = self.broker.get_point(symbol)
        digits = self.broker.get_digits(symbol)
        min_pts = max(int(self.broker.get_spread_points(symbol)*1.2), 10)
        min_dist = min_pts * pt
        if side == Side.BUY:
            sl = min(sl, entry-pt)
            tp = max(tp, entry+pt)
            if (entry-sl) < min_dist:
                sl = entry-min_dist
            if (tp-entry) < min_dist:
                tp = entry+min_dist
        else:
            sl = max(sl, entry+pt)
            tp = min(tp, entry-pt)
            if (sl-entry) < min_dist:
                sl = entry+min_dist
            if (entry-tp) < min_dist:
                tp = entry-min_dist
        return round(entry, digits), round(sl, digits), round(tp, digits)

    def lot_by_risk(self, symbol, stop_distance_price: float, risk_pct: float):
        import MetaTrader5 as mt5
        info = mt5.symbol_info(symbol)
        if not info or stop_distance_price <= 0:
            return 0.0
        tick_value = info.trade_tick_value or 0.0
        tick_size = info.trade_tick_size or 0.0
        if tick_value <= 0 or tick_size <= 0:
            return 0.0
        loss_per_lot = (stop_distance_price / tick_size) * tick_value
        risk_money = self.broker.account_equity() * (risk_pct/100.0)
        lots = risk_money / max(loss_per_lot, 1e-9)
        step = info.volume_step or 0.01
        lots = (lots // step) * step
        lots = max(info.volume_min or 0.01, min(
            info.volume_max or 100.0, lots))
        return round(lots, 2)

    def build_order(self, symbol: str, side: Side, atr_value: float, confidence: float, magic: int,
                    risk_pct: Optional[float] = None) -> OrderRequest:
        from MetaTrader5 import symbol_info_tick
        t = symbol_info_tick(symbol)
        entry = t.ask if side == Side.BUY else t.bid
        sl = entry - self.cfg.atr_mult_sl * \
            atr_value if side == Side.BUY else entry + self.cfg.atr_mult_sl*atr_value
        tp = entry + self.cfg.atr_mult_tp * \
            atr_value if side == Side.BUY else entry - self.cfg.atr_mult_tp*atr_value
        entry, sl, tp = self._sanitize_sltp(symbol, side, entry, sl, tp)
        rp = (risk_pct if risk_pct is not None else self.cfg.risk_per_trade_pct)
        lots = self.lot_by_risk(symbol, abs(entry-sl), rp)
        return OrderRequest(symbol=symbol, side=side, volume=lots, price=entry, sl=sl, tp=tp,
                            comment=f"py-modular conf={confidence:.2f}", magic=magic)
