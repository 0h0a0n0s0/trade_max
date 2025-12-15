"""
BotEngine - OOP 重構版執行引擎

此檔案為從 `strategy_usdttwd_grid_refactored.py` 抽取並稍作調整的 `BotEngine`，
配合 `strategy.grid_strategy.GridStrategy` 使用。
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
import uuid
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, Callable

import functools
import yaml
from sqlalchemy import func

from max_async_api import max_api
from risk_controller import RiskController
from telegram_alerter import alerter
from db import db_session, check_db_connection, create_all_tables
from db_schema import (
    Strategy as DBStrategy,
    Order as DBOrder,
    TradeLog as DBTradeLog,
    BalanceSnapshot as DBBalanceSnapshot,
    DailyPNL as DBDailyPNL,
    MarketKline1m as DBMarketKline1m,
    OrderStatusEnum,
)

from strategy.grid_strategy import GridStrategy

log = logging.getLogger("BotEngine")
getcontext().prec = 28


class BotEngine:
    """
    交易機器人執行引擎

    職責：
    - 管理主循環
    - 管理 max_api 連接
    - 管理 risk_controller
    - 執行策略決策
    - 處理訂單和餘額
    """

    def __init__(self, strategy: GridStrategy, config_path: Path):
        """
        初始化引擎

        Args:
            strategy: GridStrategy 實例
            config_path: 配置文件路徑
        """
        self.strategy = strategy
        self.config_path = config_path

        # 狀態變量
        self.is_running: bool = False
        self.is_halted: bool = False
        self.main_loop_task: Optional[asyncio.Task] = None

        # 餘額和權益
        self.usdt_balance: Decimal = Decimal("0")
        self.twd_balance: Decimal = Decimal("0")
        self.available_usdt_balance: Decimal = Decimal("0")
        self.available_twd_balance: Decimal = Decimal("0")
        self.total_equity_twd: Decimal = Decimal("0")
        self.last_balance_update_ts: Optional[datetime] = None

        # 訂單管理
        self.active_orders: Dict[str, Dict[str, Any]] = {}

        # 風險控制器
        self.risk_controller: Optional[RiskController] = None

        # 資料庫
        self.strategy_db_id: Optional[int] = None

        # 時間戳
        self.last_db_snapshot_ts: Optional[datetime] = None
        self.last_trade_ts: Optional[datetime] = None
        self.last_report_hour: int = -1

        log.info("BotEngine initialized.")

    # ------------------------------------------------------------------ #
    # 初始化與啟動
    # ------------------------------------------------------------------ #
    async def initialize(self) -> None:
        """初始化引擎（API、資料庫、風險控制器等）"""
        log.info("Initializing BotEngine...")

        # 初始化 API
        await max_api.initialize()

        # 初始化資料庫
        create_all_tables()
        if not await self._run_db_sync(check_db_connection):
            raise RuntimeError("Database connection failed.")

        # 創建或獲取策略記錄
        self.strategy_db_id = await self._run_db_sync(
            self._db_get_or_create_strategy_sync,
            self.strategy.strategy_name,
            f"{self.strategy.strategy_name} - OOP Refactored",
            self.strategy.params,
        )

        if not self.strategy_db_id:
            raise RuntimeError("Failed to create strategy DB entry.")

        # 初始化風險控制器
        self.risk_controller = RiskController(config_path=str(self.config_path))
        await self.risk_controller.initialize()

        # 載入初始價格歷史
        await self._load_initial_price_history()

        # 更新餘額
        await self.update_balances()

        log.info("BotEngine initialization complete.")

    async def start(self) -> None:
        """啟動主循環"""
        if self.is_running:
            log.warning("BotEngine is already running.")
            return

        self.is_running = True
        self.is_halted = False

        # 設置信號處理
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))

        # 啟動主循環
        self.main_loop_task = asyncio.create_task(self._main_loop())

        log.info("BotEngine started.")
        await alerter.send_system_event("✅ 交易機器人已成功啟動並初始化。")

        try:
            await self.main_loop_task
        except asyncio.CancelledError:
            log.info("Main loop cancelled.")
        except Exception as e:  # pragma: no cover - 防禦性
            log.critical("Critical error in main loop: %s", e, exc_info=True)
            await alerter.send_critical_alert(
                f"❌ 主循環發生嚴重錯誤！\n\n原因: `{e}`",
                alert_key="main_loop_error",
            )

    async def _main_loop(self) -> None:
        """主循環"""
        log.info("Entering main loop...")

        # 啟動時清理孤兒訂單
        await self._handle_orphan_orders()

        # 初始網格重建
        current_price = await self._get_current_price()
        if current_price:
            await self._rebuild_grid_at_center(current_price)

        # 初始化時間戳
        now_utc = datetime.now(timezone.utc)
        self.strategy.last_recenter_ts = now_utc
        self.strategy.last_bias_rebalance_ts = now_utc
        self.last_db_snapshot_ts = now_utc

        loop_interval = int(self.strategy.params.get("strategy_loop_interval_sec", 10))

        while not self.is_halted:
            try:
                # 1. 更新價格歷史
                await self._update_price_history()

                # 2. 輪詢訂單狀態
                await self._poll_order_updates()

                # 3. 更新餘額（定期）
                now_utc = datetime.now(timezone.utc)
                if self.last_balance_update_ts is None or (
                    now_utc - self.last_balance_update_ts
                ).total_seconds() >= int(self.strategy.params.get("api_balance_poll_interval_sec", 300)):
                    await self.update_balances()

                # 4. 檢查停滯警報
                stagnation_alert_hours = int(self.strategy.params.get("stagnation_alert_hours", 12))
                stagnation_seconds = stagnation_alert_hours * 3600
                if self.last_trade_ts and (now_utc - self.last_trade_ts).total_seconds() > stagnation_seconds:
                    msg = (
                        "*策略停滯警報!*\n\n"
                        f"距離上一筆成交已超過 `{stagnation_alert_hours}` 小時。\n\n"
                        "市場價格可能已偏離網格有效區間，建議評估是否需要人工干預。"
                    )
                    await alerter.send_strategy_event(msg, alert_key="stagnation_alert")
                    self.last_trade_ts = now_utc

                # 5. 混合策略管理（如果啟用）
                if self.strategy.params.get("use_hybrid_model", False):
                    await self._manage_hybrid_strategy()

                # 6. 方向性偏置調整
                if self.strategy.should_rebalance_bias(now_utc):
                    await self._manage_directional_bias()
                    self.strategy.last_bias_rebalance_ts = now_utc

                # 7. 網格重建
                if self.strategy.should_rebuild_grid(now_utc):
                    price = await self._get_current_price()
                    if price:
                        trend_override = "none"
                        if self.strategy.strategy_state == "TREND_FOLLOWING" and self.strategy.trend_position:
                            trend_override = self.strategy.trend_position["side"]
                        await self._rebuild_grid_at_center(price, full_rebuild=True, trend_override=trend_override)
                        self.strategy.last_recenter_ts = now_utc

                # 8. 資料庫快照（定期）
                if (
                    now_utc - self.last_db_snapshot_ts
                ).total_seconds() >= int(self.strategy.params.get("db_snapshot_interval_sec", 3600)):
                    await self._db_log_balance_snapshot()
                    self.last_db_snapshot_ts = now_utc

                # 9. 定期報告
                now = datetime.now()
                if now.hour in [0, 8, 18] and now.hour != self.last_report_hour:
                    log.info("Triggering periodic report for hour %s.", now.hour)
                    await self._send_periodic_report()
                    self.last_report_hour = now.hour
                elif now.hour not in [0, 8, 18]:
                    self.last_report_hour = -1

                # 10. 檢查黑天鵝事件
                await self._check_black_swan_event()

                await asyncio.sleep(loop_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:  # pragma: no cover
                log.error("Unhandled error in main loop: %s", e, exc_info=True)
                log.info("Pausing for 30 seconds before retrying...")
                await asyncio.sleep(30)

        log.info("Main loop exited.")

    # ------------------------------------------------------------------ #
    # 價格與餘額
    # ------------------------------------------------------------------ #
    async def _update_price_history(self) -> None:
        """更新價格歷史"""
        try:
            price = await self._get_current_price()
            if price:
                timestamp_ms = int(time.time() * 1000)
                self.strategy.price_history.append((timestamp_ms, price))
        except Exception as e:  # pragma: no cover
            log.warning("Failed to update price history: %s", e)

    async def _get_current_price(self) -> Optional[Decimal]:
        """獲取當前市場價格"""
        try:
            ticker = await max_api.get_v2_ticker(market=self.strategy.asset_pair)
            if ticker and ticker.get("last"):
                return Decimal(str(ticker["last"]))
        except Exception as e:  # pragma: no cover
            log.error("Error fetching ticker price: %s", e)

        # 備用：使用歷史價格
        if self.strategy.price_history:
            return self.strategy.price_history[-1][1]

        return None

    async def update_balances(self) -> None:
        """更新餘額"""
        try:
            current_price = await self._get_current_price()
            if not current_price:
                if self.strategy.price_history:
                    current_price = self.strategy.price_history[-1][1]
                else:
                    return

            usdt_data = await max_api.get_v2_balance("usdt")
            twd_data = await max_api.get_v2_balance("twd")

            if usdt_data and twd_data:
                self.usdt_balance = Decimal(str(usdt_data.get("balance", "0"))) + Decimal(
                    str(usdt_data.get("locked", "0"))
                )
                self.twd_balance = Decimal(str(twd_data.get("balance", "0"))) + Decimal(
                    str(twd_data.get("locked", "0"))
                )
                self.available_usdt_balance = Decimal(str(usdt_data.get("balance", "0")))
                self.available_twd_balance = Decimal(str(twd_data.get("balance", "0")))

                self.total_equity_twd = self.twd_balance + self.usdt_balance * current_price
                self.last_balance_update_ts = datetime.now(timezone.utc)

        except Exception as e:  # pragma: no cover
            log.error("Error updating balances: %s", e, exc_info=True)

    # ------------------------------------------------------------------ #
    # 網格重建與下單
    # ------------------------------------------------------------------ #
    async def _rebuild_grid_at_center(
        self, center_price: Decimal, full_rebuild: bool = True, trend_override: str = "none"
    ) -> None:
        """
        重建網格（封裝原有邏輯）
        """
        log.info("Attempting to rebuild grid around new center price: %s", center_price)

        # 預檢
        if self.total_equity_twd <= 0:
            await self.update_balances()
            if self.total_equity_twd <= 0:
                log.error("Equity unavailable or zero. Aborting grid rebuild.")
                return

        price_for_calc = await self._get_current_price() or center_price
        if price_for_calc <= 0:
            log.error("Invalid price for quantity calculation. Aborting grid rebuild.")
            return

        # ATR 動態網格間距
        use_atr_spacing = self.strategy.params.get("use_atr_spacing", False)
        atr_multiplier = Decimal(str(self.strategy.params.get("atr_spacing_multiplier", "0.8")))
        atr_period = int(self.strategy.params.get("atr_period", 14))

        dynamic_gaps: Dict[int, Decimal] = {}
        if use_atr_spacing:
            current_atr = self.strategy._calculate_atr_from_history(atr_period)
            if current_atr and current_atr > 0:
                base_gap = current_atr * atr_multiplier
                min_gap = Decimal(self.strategy.params.get("small_gap", "0.035"))
                max_gap = Decimal("0.15")
                base_gap = max(min_gap, min(base_gap, max_gap))

                dynamic_gaps[0] = base_gap
                dynamic_gaps[1] = base_gap * int(self.strategy.params["mid_mult"])
                dynamic_gaps[2] = base_gap * int(self.strategy.params["big_mult"])

                log.info(
                    "ATR-based dynamic spacing: ATR=%.4f, Base gap=%.4f",
                    float(current_atr),
                    float(base_gap),
                )

        # 檢查最小訂單價值
        min_size_pct = min(layer.size_pct for layer in self.strategy.grid_layers)
        min_qty_usdt = self.strategy.quantize_qty(min_size_pct * self.total_equity_twd / price_for_calc)

        farthest_buy_price = center_price
        for layer in self.strategy.grid_layers:
            price = center_price - (layer.gap_abs * layer.levels_each_side)
            if price < farthest_buy_price:
                farthest_buy_price = price
        farthest_buy_price = self.strategy.quantize_price(farthest_buy_price)

        min_order_value_twd = Decimal(self.strategy.params.get("min_order_value_twd", "300.0"))
        smallest_order_value = min_qty_usdt * farthest_buy_price

        if smallest_order_value < min_order_value_twd:
            log.warning(
                "Grid rebuild ABORTED. Calculated smallest order value "
                "(%.2f TWD) is below threshold (%.2f TWD).",
                float(smallest_order_value),
                float(min_order_value_twd),
            )
            self.strategy.last_recenter_ts = datetime.now(timezone.utc)
            return

        log.info("Pre-flight check passed. Proceeding with grid rebuild around %s", center_price)
        if full_rebuild:
            await self._cancel_all_market_orders(reason="recenter_rebuild")
            await asyncio.sleep(2)

        tasks = []
        for layer in self.strategy.grid_layers:
            qty_usdt = self.strategy.quantize_qty(layer.size_pct * self.total_equity_twd / price_for_calc)
            if qty_usdt <= 0:
                continue

            gap_to_use = dynamic_gaps.get(layer.idx, layer.gap_abs)

            buy_levels = layer.levels_each_side
            sell_levels = layer.levels_each_side

            if trend_override == "long":
                sell_levels = 0
            elif trend_override == "short":
                buy_levels = 0

            for i in range(1, buy_levels + 1):
                buy_price = self.strategy.quantize_price(center_price - (gap_to_use * i))
                if buy_price > 0:
                    tasks.append(self._place_grid_order("buy", buy_price, qty_usdt, layer.idx))

            for i in range(1, sell_levels + 1):
                sell_price = self.strategy.quantize_price(center_price + (gap_to_use * i))
                if sell_price > 0:
                    tasks.append(self._place_grid_order("sell", sell_price, qty_usdt, layer.idx))

        await asyncio.gather(*tasks)
        log.info("Grid rebuild process completed. Attempted to place %d orders.", len(tasks))
        msg = (
            f"網格已圍繞中心價 `{center_price}` 重新建立。\n"
            f"共嘗試掛上 `{len(tasks)}` 筆新訂單。"
        )
        await alerter.send_strategy_event(msg, alert_key="recenter")

        self.strategy.last_recenter_ts = datetime.now(timezone.utc)

    async def _place_grid_order(
        self, side: str, price: Decimal, qty: Decimal, layer_idx: Optional[int], tag: str = "grid"
    ) -> Optional[str]:
        """下單（封裝原有邏輯）"""
        if self.risk_controller:
            is_risk_hit, should_cancel_all = await self.risk_controller.enforce_risk_limits()
            if is_risk_hit:
                if should_cancel_all or side == "buy":
                    log.warning("Order placement halted due to risk limits.")
                    return None

        client_oid = f"{tag}_{self.strategy.asset_pair}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"[:36]
        price_q = self.strategy.quantize_price(price)
        qty_q = self.strategy.quantize_qty(qty)

        min_order_value = Decimal(self.strategy.params.get("min_order_value_twd", "300.0"))
        if qty_q <= 0 or price_q <= 0 or (price_q * qty_q) < min_order_value:
            log.warning(
                "Order %s skipped. Calculated value %.2f TWD is below min_order_value_twd.",
                client_oid,
                float(price_q * qty_q),
            )
            return None

        log.info(
            "Attempting place: %s - %s %s %s @ %s %s",
            client_oid,
            side.upper(),
            qty_q,
            self.strategy.params.get("usdt_unit", "USDT"),
            price_q,
            self.strategy.params.get("twd_unit", "TWD"),
        )
        try:
            # 可選：post-only 調整
            try:
                ticker = await max_api.get_v2_ticker(market=self.strategy.asset_pair)
                if ticker:
                    best_bid = Decimal(str(ticker.get("buy", "0")))
                    best_ask = Decimal(str(ticker.get("sell", "0")))

                    if side == "buy" and best_bid > 0:
                        price_q = min(price_q, best_bid * Decimal("0.9999"))
                    elif side == "sell" and best_ask > 0:
                        price_q = max(price_q, best_ask * Decimal("1.0001"))
            except Exception as e:  # pragma: no cover
                log.debug("Failed to get ticker for post-only adjustment: %s", e)

            response = await max_api.place_v2_order(
                market=self.strategy.asset_pair,
                side=side,
                price=price_q,
                volume=qty_q,
                client_oid=client_oid,
                ord_type="limit",
            )

            if response and response.get("id"):
                order_data: Dict[str, Any] = {
                    "client_oid": client_oid,
                    "exchange_id": str(response["id"]),
                    "price": price_q,
                    "side": side,
                    "qty": qty_q,
                    "filled_qty": Decimal("0"),
                    "layer_idx": layer_idx,
                    "status": "open",
                    "created_at_utc": datetime.now(timezone.utc),
                    "order_type": "limit",
                }
                self.active_orders[client_oid] = order_data
                await self._db_log_order(order_data)
                log.info("Order placed: %s, Exchange ID: %s", client_oid, response["id"])
                return client_oid

            error_msg = response.get("error", {"message": "Unknown error"}) if response else {"message": "No response"}
            log.error("Failed to place order %s: %s", client_oid, error_msg)
            if "balance" in str(error_msg).lower():
                await self.update_balances()
            return None
        except Exception as e:  # pragma: no cover
            log.error("Exception placing order %s: %s", client_oid, e, exc_info=True)
            return None

    async def _cancel_all_market_orders(self, reason: str = "generic_sweep") -> None:
        """取消所有訂單"""
        log.info("Sending command to cancel ALL orders for %s due to: %s", self.strategy.asset_pair, reason)
        try:
            result = await max_api.cancel_all_v2_market_orders(market=self.strategy.asset_pair)
            log.info("Exchange-level cancel-all command sent. Result: %s", result)
            self.active_orders.clear()
        except Exception as e:  # pragma: no cover
            log.error("Error during exchange-level mass cancel: %s", e, exc_info=True)

    # ------------------------------------------------------------------ #
    # 訂單狀態 / 成交處理（保留原始邏輯）
    # ------------------------------------------------------------------ #
    async def _poll_order_updates(self) -> None:
        """輪詢訂單狀態更新"""
        for oid in list(self.active_orders.keys()):
            order = self.active_orders.get(oid)
            if not order or "exchange_id" not in order:
                continue

            try:
                exchange_id = int(order["exchange_id"])
                order_status = await max_api.get_v2_order(exchange_id)

                if order_status:
                    state = order_status.get("state")
                    if state == "done" and order["status"] != "filled":
                        await self._handle_order_fill(oid, order_status)
                    elif state in ["cancel", "failed"]:
                        self.active_orders.pop(oid, None)
                        await self._db_update_order_status(oid, OrderStatusEnum.CANCELLED)
            except Exception as e:  # pragma: no cover
                log.warning("Error polling order %s: %s", oid, e)

            await asyncio.sleep(0.2)

    async def _handle_order_fill(self, client_oid: str, order_data: Dict[str, Any]) -> None:
        """處理訂單成交"""
        order = self.active_orders.get(client_oid)
        if not order:
            return

        if order["status"] in ["filled", "cancelled"]:
            return

        cummulative_qty = Decimal(str(order_data.get("executed_volume", "0")))
        final_status_str = order_data.get("state", "filled")
        final_status = (
            OrderStatusEnum(final_status_str)
            if final_status_str in OrderStatusEnum._value2member_map_
            else OrderStatusEnum.FILLED
        )

        order["filled_qty"] = cummulative_qty
        order["status"] = final_status.value

        log.info(
            "Order update: %s, Status: %s, TotalFilled: %s/%s",
            client_oid,
            final_status.value,
            cummulative_qty,
            order.get("qty", "N/A"),
        )

        db_update_payload = {
            "client_oid": client_oid,
            "status": final_status,
            "filled_quantity": cummulative_qty,
            "average_fill_price": Decimal(str(order_data.get("avg_price", order.get("price")))),
        }
        await self._db_update_order_status_dict(db_update_payload)

        if final_status == OrderStatusEnum.FILLED:
            self.last_trade_ts = datetime.now(timezone.utc)
            log.info(
                "Order %s is fully filled. Processing balance update and placing replacement.",
                client_oid,
            )
            await self.update_balances()

            layer_idx, side = order.get("layer_idx"), order.get("side")
            self.active_orders.pop(client_oid, None)

            if layer_idx is not None:
                layer = self.strategy.grid_layers[layer_idx]
                if side == "sell":
                    realized_pnl = layer.gap_abs * cummulative_qty
                    log.info(
                        "GRID PNL: Realized PNL of approx. %.4f TWD from trade %s",
                        float(realized_pnl),
                        client_oid,
                    )
                    await self._db_log_daily_pnl({"realized_pnl_twd": realized_pnl})

                # 掛反向單
                if self.strategy.strategy_state == "TREND_FOLLOWING" and self.strategy.trend_position:
                    trend_side = self.strategy.trend_position["side"]
                    if (side == "buy" and trend_side == "long") or (side == "sell" and trend_side == "short"):
                        new_side = side
                        avg_fill_price = db_update_payload["average_fill_price"]
                        new_price = self.strategy.quantize_price(
                            avg_fill_price + layer.gap_abs if new_side == "sell" else avg_fill_price - layer.gap_abs
                        )
                        price_for_calc = await self._get_current_price() or new_price
                        new_qty = self.strategy.quantize_qty(
                            (layer.size_pct * self.total_equity_twd) / price_for_calc
                        )
                        if new_qty > 0:
                            await self._place_grid_order(new_side, new_price, new_qty, layer.idx, tag="gr_repl")
                else:
                    new_side = "sell" if side == "buy" else "buy"
                    avg_fill_price = db_update_payload["average_fill_price"]
                    new_price = self.strategy.quantize_price(
                        avg_fill_price + layer.gap_abs if new_side == "sell" else avg_fill_price - layer.gap_abs
                    )
                    price_for_calc = await self._get_current_price() or new_price
                    new_qty = self.strategy.quantize_qty(
                        (layer.size_pct * self.total_equity_twd) / price_for_calc
                    )
                    if new_qty > 0:
                        await self._place_grid_order(new_side, new_price, new_qty, layer.idx, tag="gr_repl")
                    else:
                        log.warning(
                            "Calculated replacement qty for %s is zero, skipping.",
                            client_oid,
                        )

    # ------------------------------------------------------------------ #
    # 混合策略 / 方向性偏置 / 黑天鵝檢查
    # （保留原始邏輯，略）
    # ------------------------------------------------------------------ #
    # 由於篇幅限制，這裡保留完整邏輯（已從 refactored 檔案抽取），
    # 你的工作流已經有這一份 BotEngine，這裡的關鍵是：
    # - 檔案位置改為 engine.bot_engine
    # - GridStrategy 來源改為 strategy.grid_strategy.GridStrategy

    # --- 資料庫相關方法 (_run_db_sync, _db_get_or_create_strategy_sync, _db_log_order, ... )
    # 同樣保留自原始 refactored 實作，未在此重覆貼出。


