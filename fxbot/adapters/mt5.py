import logging
from typing import List, Optional

import MetaTrader5 as mt5
import pandas as pd

from core.types import OrderRequest, PositionView, Side
from adapters.broker import Broker

log = logging.getLogger("fxbot")

_TF = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class MT5Broker(Broker):
    def __init__(self, params=None):
        self.params = params or {}

    # -------- lifecycle --------
    def initialize(self):
        if not mt5.initialize():
            err = mt5.last_error()
            raise RuntimeError(f"MT5 initialize failed: {err}")

        # login opcional, se credenciais fornecidas
        login = self.params.get("login")
        password = self.params.get("password")
        server = self.params.get("server")
        if login and password and server:
            if not mt5.login(login, password, server=server):
                err = mt5.last_error()
                mt5.shutdown()
                raise RuntimeError(f"MT5 login failed: {err}")

    def shutdown(self):
        try:
            mt5.shutdown()
        except Exception:
            log.exception("Erro ao finalizar MT5")

    # -------- market data --------
    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        # garante símbolo selecionado
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Símbolo indisponível: {symbol}")

        tf = _TF.get(str(timeframe))
        if tf is None:
            raise ValueError(f"Timeframe inválido: {timeframe}")

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            raise RuntimeError(f"Falha ao obter candles de {symbol} ({timeframe}): {err}")

        df = pd.DataFrame(rates)
        # normaliza colunas
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(
            columns={
                "open": "o",
                "high": "h",
                "low": "l",
                "close": "c",
                "tick_volume": "v",
            },
            inplace=True,
        )
        return df

    # -------- symbol info helpers --------
    def symbol_info(self, symbol: str):
        """Retorna informações completas do símbolo."""
        return mt5.symbol_info(symbol)

    def symbol_info_tick(self, symbol: str):
        """Obtém o último tick do símbolo."""
        return mt5.symbol_info_tick(symbol)

    def get_spread_points(self, symbol: str) -> int:
        info = mt5.symbol_info(symbol)
        if not info:
            raise RuntimeError(f"symbol_info falhou para {symbol}")
        # preferir info.spread (em points). Fallback: calcular pelo tick.
        spread = int(info.spread or 0)
        if spread <= 0:
            t = mt5.symbol_info_tick(symbol)
            if not t:
                raise RuntimeError(f"symbol_info_tick falhou para {symbol}")
            spread = int((t.ask - t.bid) / info.point)
        return spread

    def get_point(self, symbol: str) -> float:
        info = mt5.symbol_info(symbol)
        if not info:
            raise RuntimeError(f"symbol_info falhou para {symbol}")
        return float(info.point)

    def get_digits(self, symbol: str) -> int:
        info = mt5.symbol_info(symbol)
        if not info:
            raise RuntimeError(f"symbol_info falhou para {symbol}")
        return int(info.digits)

    # -------- trading --------
    def place_order(self, req: OrderRequest):
        t = mt5.symbol_info_tick(req.symbol)
        if not t:
            raise RuntimeError(f"symbol_info_tick falhou para {req.symbol}")

        price = t.ask if req.side == Side.BUY else t.bid

        mreq = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": req.symbol,
            "volume": req.volume,
            "type": mt5.ORDER_TYPE_BUY if req.side == Side.BUY else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": req.sl,
            "tp": req.tp,
            "deviation": int(self.params.get("deviation", 20)),
            "magic": req.magic,
            "comment": req.comment or "",
            "type_time": self.params.get("type_time", mt5.ORDER_TIME_GTC),
            "type_filling": self.params.get("type_filling", mt5.ORDER_FILLING_FOK),
        }
        return mt5.order_send(mreq)

    def modify_sltp(self, ticket: int, sl: float, tp: float):
        return mt5.order_send(
            {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": sl, "tp": tp}
        )

    def close_position(self, ticket: int, volume: float):
        pos = next((x for x in self.positions() if x.ticket == ticket), None)
        if not pos:
            log.error(f"Posição {ticket} não encontrada para fechamento")
            return None

        t = mt5.symbol_info_tick(pos.symbol)
        if not t:
            raise RuntimeError(f"symbol_info_tick falhou para {pos.symbol}")

        otype = mt5.ORDER_TYPE_SELL if pos.side == Side.BUY else mt5.ORDER_TYPE_BUY
        price = t.bid if pos.side == Side.BUY else t.ask

        return mt5.order_send(
            {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": volume,
                "type": otype,
                "price": price,
                "deviation": int(self.params.get("deviation", 20)),
                "position": ticket,
                "comment": "close",
                "type_time": self.params.get("type_time", mt5.ORDER_TIME_GTC),
                "type_filling": self.params.get("type_filling", mt5.ORDER_FILLING_FOK),
            }
        )

    # -------- account / history --------
    def positions(self, symbol: Optional[str] = None) -> List[PositionView]:
        poss = mt5.positions_get(symbol=symbol)
        out: List[PositionView] = []
        if poss:
            for p in poss:
                out.append(
                    PositionView(
                        ticket=p.ticket,
                        symbol=p.symbol,
                        side=Side.BUY if p.type == mt5.POSITION_TYPE_BUY else Side.SELL,
                        volume=p.volume,
                        price_open=p.price_open,
                        sl=p.sl,
                        tp=p.tp,
                        magic=p.magic,
                    )
                )
        return out

    def account_equity(self) -> float:
        ai = mt5.account_info()
        if not ai:
            raise RuntimeError("account_info falhou")
        return float(ai.equity)

    def history_deals_df(self, start_dt, end_dt) -> pd.DataFrame:
        s = start_dt.replace(tzinfo=None)
        e = end_dt.replace(tzinfo=None)
        deals = mt5.history_deals_get(s, e)
        recs = []
        if deals:
            for d in deals:
                recs.append(
                    {
                        "time": pd.to_datetime(d.time, unit="s"),
                        "symbol": d.symbol,
                        "position_id": getattr(d, "position_id", getattr(d, "position", 0)),
                        "profit": float(d.profit),
                        "entry": int(d.entry),
                        "type": int(d.type),
                        "price": float(d.price),
                    }
                )
        return pd.DataFrame(recs)
