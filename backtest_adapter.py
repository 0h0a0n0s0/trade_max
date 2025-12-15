"""
backtest_adapter.py

Backtest Adapter: 使用與實盤完全相同的 GridStrategy 邏輯進行回測。

此模組放在專案根目錄，供 backtest/optimize_params_parallel.py 匯入：
- from backtest_adapter import BacktestAdapter
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import logging
import pandas as pd

from strategy_usdttwd_grid_refactored import GridStrategy

getcontext().prec = 28
log = logging.getLogger("BacktestAdapter")


@dataclass
class SimulatedOrder:
    order_id: str
    side: str          # "buy" or "sell"
    price: Decimal
    qty: Decimal
    layer_idx: Optional[int] = None
    tag: str = "grid"
    status: str = "open"   # "open" | "filled" | "cancelled"
    filled_qty: Decimal = Decimal("0")


class BacktestAdapter:
    """
    回測適配器：
    - 使用與實盤相同的 GridStrategy 參數與網格結構
    - 模擬掛單、成交與資金變化
    - 輸出與實盤一致格式的績效統計（roi_pct, max_drawdown_pct, total_pnl, total_trades）
    """

    def __init__(
        self,
        strategy: GridStrategy,
        init_usdt: Decimal,
        init_twd: Decimal,
        fee_rate: Decimal = Decimal("0.0004"),
        verbose: bool = False,
    ) -> None:
        self.strategy = strategy
        self.verbose = verbose

        # 資金狀態
        self.usdt_balance: Decimal = init_usdt
        self.twd_balance: Decimal = init_twd
        self.available_usdt: Decimal = init_usdt
        self.available_twd: Decimal = init_twd

        # 訂單
        self.active_orders: Dict[str, SimulatedOrder] = {}
        self._order_seq: int = 0

        # 交易與權益歷史
        self.trade_log: List[Dict[str, Any]] = []
        self.equity_history: List[Decimal] = []

        # 其它狀態
        self.current_time: Optional[datetime] = None
        self.current_price: Decimal = Decimal("0")
        self.fee_rate: Decimal = fee_rate
        self.min_order_value_twd: Decimal = Decimal(
            str(strategy.params.get("min_order_value_twd", "300.0"))
        )

        # 黑天鵝保護
        self.is_halted: bool = False
        self.halt_reason: Optional[str] = None

        log.info(
            "BacktestAdapter initialized: init_usdt=%s, init_twd=%s",
            init_usdt,
            init_twd,
        )

    # ------------------------------------------------------------------ #
    # 基礎工具
    # ------------------------------------------------------------------ #
    def _gen_order_id(self, tag: str = "grid") -> str:
        self._order_seq += 1
        return f"{tag}_{self._order_seq:06d}"

    def _total_equity(self, price: Decimal) -> Decimal:
        return self.twd_balance + self.usdt_balance * price

    # ------------------------------------------------------------------ #
    # 掛單 / 成交模擬
    # ------------------------------------------------------------------ #
    def _place_order(
        self,
        side: str,
        price: Decimal,
        qty: Decimal,
        layer_idx: Optional[int],
        tag: str,
    ) -> Optional[str]:
        """
        簡化版掛單邏輯，不與實盤 API 完全一致，但保留：
        - 最小下單金額檢查
        - 可用餘額檢查
        """
        value_twd = price * qty
        if value_twd < self.min_order_value_twd:
            return None

        if side == "buy":
            if self.available_twd < value_twd:
                return None
            self.available_twd -= value_twd
        else:
            if self.available_usdt < qty:
                return None
            self.available_usdt -= qty

        oid = self._gen_order_id(tag)
        self.active_orders[oid] = SimulatedOrder(
            order_id=oid,
            side=side,
            price=price,
            qty=qty,
            layer_idx=layer_idx,
            tag=tag,
        )
        return oid

    def _check_fills(
        self, high: Decimal, low: Decimal, bar_index: int
    ) -> None:
        """
        根據本根 K 線 high/low 檢查是否成交。
        成交後即刻更新資金與下替補單（標準網格行為）。
        """
        for oid, order in list(self.active_orders.items()):
            if order.status != "open":
                continue

            hit = low <= order.price <= high
            if not hit:
                continue

            # 完全成交
            order.status = "filled"
            order.filled_qty = order.qty

            fee = order.price * order.qty * self.fee_rate
            if order.side == "buy":
                self.usdt_balance += order.qty
                self.available_usdt += order.qty
                self.twd_balance -= (order.price * order.qty + fee)
            else:
                self.usdt_balance -= order.qty
                self.available_usdt -= order.qty
                self.twd_balance += (order.price * order.qty - fee)

            self.trade_log.append(
                {
                    "index": bar_index,
                    "timestamp": self.current_time,
                    "side": order.side,
                    "price": float(order.price),
                    "quantity": float(order.qty),
                    "fee": float(fee),
                    "type": "grid_buy" if order.side == "buy" else "grid_sell",
                    "layer_idx": order.layer_idx,
                    "tag": order.tag,
                }
            )

            # 移除原訂單
            self.active_orders.pop(oid, None)

            # 標準網格：掛反向替補單（不在 TREND_FOLLOWING 模式下）
            if order.layer_idx is not None and self.strategy.strategy_state == "GRID":
                layer = self.strategy.grid_layers[order.layer_idx]
                gap = layer.gap_abs
                new_side = "sell" if order.side == "buy" else "buy"
                if new_side == "sell":
                    new_price = self.strategy.quantize_price(order.price + gap)
                else:
                    new_price = self.strategy.quantize_price(order.price - gap)

                total_eq = self._total_equity(self.current_price)
                new_qty = self.strategy.quantize_qty(
                    (layer.size_pct * total_eq) / self.current_price
                )
                if new_qty > 0:
                    self._place_order(
                        new_side,
                        new_price,
                        new_qty,
                        order.layer_idx,
                        tag="gr_repl",
                    )

    def _rebuild_grid(self, center_price: Decimal, full_rebuild: bool) -> None:
        """
        使用 GridStrategy 的 grid_layers 配置，重建靜態網格。
        僅依靠本地資金資訊與當前價格，不涉及 API / DB。
        """
        if full_rebuild:
            # 釋放掛單鎖定的資金
            for order in self.active_orders.values():
                if order.status != "open":
                    continue
                value = order.price * order.qty
                if order.side == "buy":
                    self.available_twd += value
                else:
                    self.available_usdt += order.qty
            self.active_orders.clear()

        total_eq = self._total_equity(center_price)

        for layer in self.strategy.grid_layers:
            qty_usdt = self.strategy.quantize_qty(
                (layer.size_pct * total_eq) / center_price
            )
            if qty_usdt <= 0:
                continue

            buy_levels = layer.levels_each_side
            sell_levels = layer.levels_each_side

            for i in range(1, buy_levels + 1):
                buy_price = self.strategy.quantize_price(
                    center_price - layer.gap_abs * i
                )
                if buy_price > 0:
                    self._place_order(
                        "buy", buy_price, qty_usdt, layer.idx, tag=f"gr{layer.idx}b"
                    )

            for i in range(1, sell_levels + 1):
                sell_price = self.strategy.quantize_price(
                    center_price + layer.gap_abs * i
                )
                if sell_price > 0:
                    self._place_order(
                        "sell", sell_price, qty_usdt, layer.idx, tag=f"gr{layer.idx}s"
                    )

    # ------------------------------------------------------------------ #
    # 主回測流程
    # ------------------------------------------------------------------ #
    def run(self, ohlc_df: pd.DataFrame) -> Dict[str, Any]:
        """
        執行回測。

        Args:
            ohlc_df: 必須包含 'high', 'low', 'close'，索引為 datetime。

        Returns:
            Dict: 包含 roi_pct, max_drawdown_pct, total_pnl, total_trades 等欄位。
        """
        if ohlc_df.empty:
            raise ValueError("ohlc_df is empty")

        price_series = ohlc_df["close"].astype(float).ffill()

        # 初始時間與價格
        first_ts = ohlc_df.index[0]
        self.current_time = (
            first_ts if isinstance(first_ts, datetime) else datetime.now(timezone.utc)
        )
        self.current_price = Decimal(str(price_series.iloc[0]))

        # 初始化網格
        self._rebuild_grid(self.current_price, full_rebuild=True)

        initial_equity = self._total_equity(self.current_price)

        for idx, (ts, row) in enumerate(ohlc_df.iterrows()):
            if self.is_halted:
                break

            self.current_time = ts if isinstance(ts, datetime) else datetime.now(
                timezone.utc
            )
            high = Decimal(str(row["high"]))
            low = Decimal(str(row["low"]))
            close = Decimal(str(row["close"]))
            self.current_price = close

            # 更新策略 price_history 以供 EMA / ATR 等計算使用
            ts_ms = int(self.current_time.timestamp() * 1000)
            self.strategy.price_history.append((ts_ms, close))

            # 檢查成交
            self._check_fills(high, low, idx)

            # 每根 K 線記錄一次權益
            self.equity_history.append(self._total_equity(close))

        # 統計
        final_equity = self._total_equity(self.current_price)
        total_pnl = final_equity - initial_equity
        roi_pct = (total_pnl / initial_equity * 100) if initial_equity > 0 else Decimal(
            "0"
        )

        # 最大回撤
        max_drawdown_pct = Decimal("0")
        if self.equity_history:
            peak = self.equity_history[0]
            max_dd = Decimal("0")
            for eq in self.equity_history:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else Decimal("0")
                if dd > max_dd:
                    max_dd = dd
            max_drawdown_pct = max_dd * 100

        result: Dict[str, Any] = {
            "total_pnl": float(total_pnl),
            "roi_pct": float(roi_pct),
            "max_drawdown_pct": float(max_drawdown_pct),
            "total_trades": len(self.trade_log),
            "final_equity": float(final_equity),
            "initial_equity": float(initial_equity),
            "trades_log": self.trade_log,
        }

        return result


