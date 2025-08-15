from datetime import datetime, timedelta, timezone
import math

from core.types import Side
from core.utils import atr, donchian, adx, ema
from adapters.broker import Broker
from logs.json_summary import JSONSummary
from core.logging import get_logger


def spread_grace(df_exec, df_regime, atr_now, spread_price, params) -> bool:
    """Aplica a lógica de 'graça' no spread em tendência forte e preço próximo a Donchian."""
    try:
        win = params.get("donchian", 14)
        c0 = float(df_exec["c"].iloc[-1])
        adx_h1 = float(adx(df_regime["h"], df_regime["l"], df_regime["c"], 14).iloc[-1])
        strong_trend = adx_h1 >= max(14.0, params.get("adx_thr", 14) + 2.0)

        near_ratio = params.get("near_by_atr_ratio", params.get("near_vwap_by_atr", 0.30))
        near_thr = float(near_ratio * atr_now)

        dist_ok = False
        try:
            up, lo = donchian(df_exec["h"], df_exec["l"], win)
            dist_up = max(0.0, float(up.iloc[-1] - c0))
            dist_low = max(0.0, float(c0 - lo.iloc[-1]))
            dist_ok = (dist_up <= max(near_thr, 0.2 * atr_now)) or (dist_low <= max(near_thr, 0.2 * atr_now))
        except Exception:
            dist_ok = False

        return strong_trend and dist_ok and (spread_price <= 0.95 * max(atr_now, 1e-9))
    except Exception:
        return False


log = get_logger(__name__)


def _decimals_from_step(step: float) -> int:
    """Casas decimais necessárias para quantizar pelo step (ex.: 0.01 -> 2)."""
    if step >= 1 or step <= 0:
        return 0
    s = f"{step:.10f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


class Executor:
    def __init__(self, cfg, broker: Broker, strategy, ml_model=None):
        self.cfg = cfg
        self.broker = broker
        self.strategy = strategy
        self.ml_model = ml_model
        self.session_start = None
        self.session_end = None
        self.baseline = None
        self.last_signal_ts = {s: None for s in cfg.symbols}
        self.verbose = bool(getattr(cfg, "log_every_bar", True))
        self._summary_done = False
        self.json_summary = JSONSummary()

    # -------- sessão / equity ----------
    def start_session(self, now_utc: datetime, baseline_equity: float):
        self.session_start = now_utc
        self.session_end = now_utc + timedelta(hours=self.cfg.session.hours)
        self.baseline = baseline_equity
        if self.verbose:
            log.info(
                f"[SESSION] start={self.session_start} end={self.session_end} baseline={self.baseline}"
            )
        self.json_summary.start(self.session_start)

    def equity_gain_pct(self):
        eq = self.broker.account_equity()
        return max(0.0, (eq - self.baseline) / self.baseline * 100) if self.baseline else 0.0

    def equity_dd_pct(self):
        eq = self.broker.account_equity()
        return max(0.0, (self.baseline - eq) / self.baseline * 100) if self.baseline else 0.0

    # -------- regras globais ----------
    def _can_trade_now(self):
        now = datetime.now(timezone.utc)
        if now >= self.session_end:
            return False, "session finished"
        if not (self.cfg.session.start_hour <= now.hour < self.cfg.session.end_hour):
            return False, f"outside window {self.cfg.session.start_hour}-{self.cfg.session.end_hour}h"
        if (self.equity_gain_pct() >= self.cfg.session.profit_target_pct) and (not self.cfg.session.continue_after_target):
            return False, f"profit target hit {self.equity_gain_pct():.2f}%"
        if self.equity_dd_pct() >= self.cfg.session.loss_limit_pct:
            return False, f"loss limit hit {self.equity_dd_pct():.2f}%"

        # Limite de concorrência apenas para posições com o nosso magic
        open_mine = [p for p in self.broker.positions() if p.magic == self.cfg.magic]
        if len(open_mine) >= self.cfg.session.max_concurrent_trades:
            return False, "max concurrent trades reached"
        return True, ""

    def current_risk_pct(self) -> float:
        base = self.cfg.risk.risk_per_trade_pct
        if self.cfg.session.continue_after_target and self.equity_gain_pct() >= self.cfg.session.profit_target_pct:
            after = self.cfg.risk.risk_per_trade_pct_after_target
            return float(after if (after is not None) else max(0.1, base * 0.5))
        return float(base)

    # -------- utilidades internas ----------
    def _fetch_data(self, symbol: str):
        """Busca dados de mercado e calcula o ATR atual."""
        df_e = self.broker.get_rates(symbol, self.cfg.timeframe_exec, 600)
        df_r = self.broker.get_rates(symbol, self.cfg.timeframe_regime, 600)
        atr_now = float(atr(df_e["h"], df_e["l"], df_e["c"], 14).iloc[-1])
        return df_e, df_r, atr_now

    def _check_spread(self, symbol: str, df_e, df_r, atr_now) -> bool:
        """Aplica filtros de spread e a possível 'graça'."""
        spread_pts = self.broker.get_spread_points(symbol)
        spread_price = spread_pts * self.broker.get_point(symbol)
        hard_cap = self.cfg.spread.hard_cap_points
        dyn_cap = self.cfg.spread.max_atr_ratio * max(atr_now, 1e-9)
        dyn_ok = spread_price <= dyn_cap
        abs_ok = spread_pts <= hard_cap

        grace_ok = False
        if not dyn_ok and abs_ok:
            grace_ok = spread_grace(df_e, df_r, atr_now, spread_price, self.cfg.strategy.params)

        if self.verbose:
            log.debug(
                f"[{symbol}] spread={spread_pts}pts | cap={hard_cap} | spr_price={spread_price:.6f} | "
                f"atr={atr_now:.6f} | dyn_ok={dyn_ok} | grace_ok={grace_ok}"
            )

        if not ((dyn_ok or grace_ok) and abs_ok):
            if self.verbose:
                log.info(f"[{symbol}] skip: spread filter")
            return False
        return True

    def _prepare_signal(self, symbol: str, df_e, df_r, atr_now):
        """Gera e valida o sinal. Retorna (sinal, risco, nome_estratégia) ou None."""
        now = datetime.now(timezone.utc)
        last = self.last_signal_ts.get(symbol)
        if last and (now - last).total_seconds() < self.cfg.cooldown_minutes * 60:
            if self.verbose:
                log.info(f"[{symbol}] skip: cooldown")
            return None

        sig = self.strategy.generate_signal(symbol, df_e, df_r)

        thr = 0.5
        if self.ml_model is not None:
            thr = float(getattr(self.ml_model, "threshold", thr))

        if sig is None:
            if self.verbose:
                log.debug(f"[{symbol}] sem sinal da estratégia")
            return None

        if sig.confidence < thr:
            if self.verbose:
                log.info(f"[{symbol}] skip: filtrado pelo ML p={sig.confidence:.3f} < thr={thr:.3f}")
            return None

        open_mine = [p for p in self.broker.positions(symbol) if p.magic == self.cfg.magic]
        risk_now = self.current_risk_pct()

        if open_mine:
            if not getattr(self.cfg.session, "allow_pyramiding", False):
                if self.verbose:
                    log.info(f"[{symbol}] skip: já existe posição")
                return None

            if len(open_mine) >= getattr(self.cfg.session, "max_stack_per_symbol", 1):
                if self.verbose:
                    log.info(f"[{symbol}] skip: máximo empilhamento")
                return None

            pos = open_mine[0]
            t = self.broker.symbol_info_tick(symbol)
            pos_is_buy = (getattr(pos, "side", None) == Side.BUY) or (
                getattr(pos, "side", None) and getattr(pos, "side").name == "BUY"
            )
            side_match = (pos_is_buy and sig.side == Side.BUY) or ((not pos_is_buy) and sig.side == Side.SELL)
            price_now = t.bid if pos_is_buy else t.ask
            profit = (price_now - pos.price_open) if pos_is_buy else (pos.price_open - price_now)
            r_val = (
                abs(pos.price_open - (pos.sl or pos.price_open))
                if getattr(pos, "sl", 0) > 0
                else sig.atr * self.cfg.risk.atr_mult_sl
            )

            if (profit < self.cfg.session.min_stack_increase_r * r_val) or (not side_match):
                if self.verbose:
                    log.info(f"[{symbol}] skip: pirâmide inválida")
                return None

            risk_now *= float(self.cfg.session.pyramiding_risk_scale)

        strat_name = (
            sig.meta.get("strategy") if sig.meta and "strategy" in sig.meta else self.strategy.__class__.__name__
        )
        # Log do sinal: apenas informações essenciais
        log.info(
            f"[{symbol}] sinal {sig.side.value} atr={sig.atr:.6f} conf={sig.confidence:.3f} "
            f"strategy={strat_name} params={self.cfg.strategy.params}"
        )

        return sig, risk_now, strat_name

    def _place_order(self, symbol: str, sig, risk_now: float, strat_name: str):
        """Envia a ordem ao broker e registra o resultado."""
        from risk.risk_manager import RiskManager

        rm = RiskManager(self.broker, self.cfg.risk)
        if self.verbose and self.cfg.session.continue_after_target and self.equity_gain_pct() >= self.cfg.session.profit_target_pct:
            log.info(f"[{symbol}] post-target mode: risk_per_trade={risk_now:.2f}%")

        req = rm.build_order(symbol, sig.side, sig.atr, sig.confidence, self.cfg.magic, risk_pct=risk_now)
        r = self.broker.place_order(req)
        retcode = getattr(r, "retcode", None)
        comment = getattr(r, "comment", "")
        ticket = getattr(r, "order", None)

        log.info(
            f"[{symbol}] ordem {sig.side.value} vol={req.volume} price={req.price} "
            f"sl={req.sl} tp={req.tp} retcode={retcode} ticket={ticket} "
            f"strategy={strat_name} params={self.cfg.strategy.params}"
        )

        if retcode is not None:
            self.last_signal_ts[symbol] = datetime.now(timezone.utc)
        else:
            log.error(f"[{symbol}] erro ao enviar ordem: {comment}")

    # -------- por símbolo ----------
    def step_symbol(self, symbol: str):
        ok, reason = self._can_trade_now()
        if not ok:
            if self.verbose:
                log.info(f"[{symbol}] skip: {reason}")
            return

        df_e, df_r, atr_now = self._fetch_data(symbol)

        if not self._check_spread(symbol, df_e, df_r, atr_now):
            return

        result = self._prepare_signal(symbol, df_e, df_r, atr_now)
        if result is None:
            return
        sig, risk_now, strat_name = result

        self._place_order(symbol, sig, risk_now, strat_name)

    # -------- gestão das posições ----------
    def manage_open_positions(self):
        for p in self.broker.positions():
            if p.magic != self.cfg.magic:
                continue

            df = self.broker.get_rates(p.symbol, self.cfg.timeframe_exec, 200)
            a = float(atr(df["h"], df["l"], df["c"], 14).iloc[-1])

            # R por posição
            r_val = abs(p.price_open - p.sl) if getattr(p, "sl", 0) > 0 else self.cfg.risk.atr_mult_sl * a

            t = self.broker.symbol_info_tick(p.symbol)
            is_buy = (p.side == Side.BUY)
            price_now = t.bid if is_buy else t.ask
            profit = (price_now - p.price_open) if is_buy else (p.price_open - price_now)

            # ---- Parcial em 1R (respeitando volume_min/step do símbolo) ----
            info = self.broker.symbol_info(p.symbol)
            step = float(getattr(info, "volume_step", 0.01) or 0.01)
            vol_min = float(getattr(info, "volume_min", 0.01) or 0.01)
            dec = _decimals_from_step(step)

            if profit >= r_val and p.volume > vol_min:
                vol_target = p.volume * float(self.cfg.risk.partial_at_1r)
                vol_q = math.floor(vol_target / step) * step
                vol_q = max(vol_min, min(vol_q, p.volume))  # não ultrapassar o volume da posição
                vol_q = round(vol_q, dec)

                if vol_q >= vol_min and vol_q <= p.volume - step + 1e-12:
                    self.broker.close_position(p.ticket, vol_q)
                    log.info(
                        f"[{p.symbol}] partial_close ticket={p.ticket} volume={vol_q}"
                    )

            # ---- Trailing + Break-even ----
            new_sl = p.sl
            if profit >= r_val:
                be = p.price_open
                trail = float(self.cfg.risk.atr_trail_mult) * a
                new_sl = max(be, price_now - trail) if is_buy else min(be, price_now + trail)

                # atualiza SL apenas se mover pelo menos 2 pontos
                pt = self.broker.get_point(p.symbol)
                if abs((new_sl or 0) - (p.sl or 0)) > pt * 2:
                    self.broker.modify_sltp(p.ticket, new_sl, p.tp)
                    log.info(
                        f"[{p.symbol}] sltp_update ticket={p.ticket} sl={new_sl} tp={p.tp}"
                    )

    # -------- resumo ----------
    def maybe_summary_once(self):
        now = datetime.now(timezone.utc)
        if self._summary_done or now < self.session_end:
            return
        try:
            df = self.broker.history_deals_df(self.session_start, now)
            realized = float(df["profit"].sum()) if df is not None and not df.empty else 0.0
            wins = losses = 0
            pf = None
            if df is not None and not df.empty:
                agg = df.groupby("position_id")["profit"].sum()
                wins = int((agg > 0).sum())
                losses = int((agg < 0).sum())
                pos_profit = float(agg[agg > 0].sum()) if (agg > 0).any() else 0.0
                pos_loss = float(agg[agg < 0].sum()) if (agg < 0).any() else 0.0
                pf = (pos_profit / abs(pos_loss)) if pos_loss < 0 else None

            eq_now = self.broker.account_equity()
            gain_pct = self.equity_gain_pct()
            closed = wins + losses

            text = (
                f"baseline={self.baseline:.2f} | equity_now={eq_now:.2f} | gain_pct={gain_pct:.2f}% | "
                f"realized_pnl={realized:.2f} | closed_trades={closed} | wins={wins} | losses={losses} | "
                f"winrate={(100*wins/closed):.1f}% " + (f"| PF={pf:.2f}" if pf is not None else "")
            ) if closed > 0 else (
                f"baseline={self.baseline:.2f} | equity_now={eq_now:.2f} | gain_pct={gain_pct:.2f}% | realized_pnl={realized:.2f} | closed_trades=0"
            )
            log.info("\n=== SESSION SUMMARY ===")
            log.info(text)

            payload = {
                "started_at": self.session_start.isoformat(),
                "ended_at": now.isoformat(),
                "baseline": float(self.baseline),
                "equity_now": float(eq_now),
                "gain_pct": float(gain_pct),
                "realized_pnl": float(realized),
                "closed_trades": int(closed),
                "wins": int(wins),
                "losses": int(losses),
                "winrate_pct": float(100 * wins / closed) if closed > 0 else None,
                "profit_factor": float(pf) if pf is not None else None,
                "symbols": list(self.cfg.symbols),
                "strategy": {
                    "params": self.cfg.strategy.params
                },
                "risk": {
                    "risk_per_trade_pct": self.cfg.risk.risk_per_trade_pct,
                    "risk_per_trade_pct_after_target": self.cfg.risk.risk_per_trade_pct_after_target,
                    "atr_mult_sl": self.cfg.risk.atr_mult_sl,
                    "atr_mult_tp": self.cfg.risk.atr_mult_tp,
                    "atr_trail_mult": self.cfg.risk.atr_trail_mult,
                    "partial_at_1r": self.cfg.risk.partial_at_1r
                },
                "session": {
                    "hours": self.cfg.session.hours,
                    "start_hour": self.cfg.session.start_hour,
                    "end_hour": self.cfg.session.end_hour,
                    "profit_target_pct": self.cfg.session.profit_target_pct,
                    "loss_limit_pct": self.cfg.session.loss_limit_pct,
                    "continue_after_target": self.cfg.session.continue_after_target,
                    "max_concurrent_trades": self.cfg.session.max_concurrent_trades
                }
            }
            self.json_summary.write(payload, strategy_class_path=self.cfg.strategy.class_path)

        except Exception as e:
            log.error("Summary error: %s", e)
        finally:
            self._summary_done = True
