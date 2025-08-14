import MetaTrader5 as mt5
import pandas as pd
from typing import List, Optional
from core.types import OrderRequest, PositionView, Side
from adapters.broker import Broker

_TF = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1
}


class MT5Broker(Broker):
    def __init__(self, params=None):
        self.params = params or {}

    def initialize(self):
        if not mt5.initialize():
            raise SystemExit(f"MT5 init failed: {mt5.last_error()}")

    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, _TF[str(timeframe)], 0, count)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'open': 'o', 'high': 'h', 'low': 'l',
                  'close': 'c', 'tick_volume': 'v'}, inplace=True)
        return df

    def get_spread_points(self, symbol: str) -> int:
        info = mt5.symbol_info(symbol)
        return int(info.spread)

    def get_point(self, symbol: str) -> float:
        return mt5.symbol_info(symbol).point

    def get_digits(self, symbol: str) -> int:
        return mt5.symbol_info(symbol).digits

    def place_order(self, req: OrderRequest):
        t = mt5.symbol_info_tick(req.symbol)
        price = t.ask if req.side == Side.BUY else t.bid
        mreq = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": req.symbol,
            "volume": req.volume,
            "type": mt5.ORDER_TYPE_BUY if req.side == Side.BUY else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": req.sl, "tp": req.tp,
            "deviation": 20,
            "magic": req.magic,
            "comment": req.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK
        }
        return mt5.order_send(mreq)

    def symbol_info(self, symbol: str):
        """Retorna informações completas do símbolo."""
        return mt5.symbol_info(symbol)

    def symbol_info_tick(self, symbol: str):
        """Obtém o último tick do símbolo."""
        return mt5.symbol_info_tick(symbol)

    def positions(self, symbol: Optional[str] = None) -> List[PositionView]:
        poss = mt5.positions_get(symbol=symbol)
        out = []
        if poss:
            for p in poss:
                out.append(PositionView(
                    ticket=p.ticket, symbol=p.symbol,
                    side=Side.BUY if p.type == mt5.POSITION_TYPE_BUY else Side.SELL,
                    volume=p.volume, price_open=p.price_open,
                    sl=p.sl, tp=p.tp, magic=p.magic
                ))
        return out

    def modify_sltp(self, ticket: int, sl: float, tp: float):
        return mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": sl, "tp": tp})

    def close_position(self, ticket: int, volume: float):
        pos = [x for x in self.positions() if x.ticket == ticket][0]
        t = mt5.symbol_info_tick(pos.symbol)
        otype = mt5.ORDER_TYPE_SELL if pos.side == Side.BUY else mt5.ORDER_TYPE_BUY
        price = t.bid if pos.side == Side.BUY else t.ask
        return mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": volume,
                               "type": otype, "price": price, "deviation": 20, "position": ticket, "comment": "close"})

    def account_equity(self) -> float:
        return mt5.account_info().equity

    # --- histórico para resumo ---
    def history_deals_df(self, start_dt, end_dt):
        s = start_dt.replace(tzinfo=None)
        e = end_dt.replace(tzinfo=None)
        deals = mt5.history_deals_get(s, e)
        recs = []
        if deals:
            for d in deals:
                recs.append({
                    "time": pd.to_datetime(d.time, unit="s"),
                    "symbol": d.symbol,
                    "position_id": getattr(d, "position_id", getattr(d, "position", 0)),
                    "profit": float(d.profit),
                    "entry": int(d.entry),
                    "type": int(d.type),
                    "price": float(d.price)
                })
        return pd.DataFrame(recs)
