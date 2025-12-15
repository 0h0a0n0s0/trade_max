# bot_engine.py
"""
BotEngine - 交易機器人執行引擎
負責管理主循環、API 調用、訂單執行和狀態追蹤
"""
from __future__ import annotations
import asyncio
import signal
import logging
import time
import uuid
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Deque
from collections import deque
from pathlib import Path
import yaml

from max_async_api import max_api
from strategy.base_strategy import BaseStrategy
from optimizer.strategy_optimizer import StrategyOptimizer
from risk_controller import RiskController
from telegram_alerter import alerter
from db import db_session, check_db_connection, create_all_tables
from db_schema import (
    Strategy as DBStrategy, Order as DBOrder, TradeLog as DBTradeLog,
    BalanceSnapshot as DBBalanceSnapshot, DailyPNL as DBDailyPNL,
    MarketKline1m as DBMarketKline1m, OrderStatusEnum
)

log = logging.getLogger("BotEngine")
getcontext().prec = 28


class BotEngine:
    """
    交易機器人執行引擎
    
    職責：
    1. 管理主循環
    2. 獲取市場數據並注入到策略
    3. 執行策略生成的信號
    4. 管理訂單狀態
    5. 協調 StrategyOptimizer 進行參數調整
    6. 追蹤績效並提供給 Optimizer
    """
    
    def __init__(self, strategy: BaseStrategy, optimizer: StrategyOptimizer, config_path: Path):
        """
        初始化引擎
        
        Args:
            strategy: 策略實例
            optimizer: 優化器實例
            config_path: 配置文件路徑
        """
        self.strategy = strategy
        self.optimizer = optimizer
        self.config_path = config_path
        
        # 載入配置
        self.config = self._load_config()
        
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
        
        # 績效追蹤
        self.initial_equity: Optional[Decimal] = None
        self.peak_equity: Optional[Decimal] = None
        self.realized_pnl_twd: Decimal = Decimal("0")
        
        # 風險控制器
        self.risk_controller: Optional[RiskController] = None
        
        # 時間戳
        self.last_recenter_ts: Optional[datetime] = None
        self.last_bias_rebalance_ts: Optional[datetime] = None
        self.last_db_snapshot_ts: Optional[datetime] = None
        self.last_trade_ts: Optional[datetime] = None
        
        # 資料庫
        self.strategy_db_id: Optional[int] = None
        
        log.info("BotEngine initialized.")
    
    def _load_config(self) -> Dict[str, Any]:
        """載入配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    async def initialize(self):
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
            f"{self.strategy.strategy_name} - Modular Architecture",
            self.config
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
        
        # 設定初始權益
        if self.initial_equity is None:
            self.initial_equity = self.total_equity_twd
            self.peak_equity = self.total_equity_twd
        
        log.info("BotEngine initialization complete.")
    
    async def start(self):
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
        except Exception as e:
            log.critical(f"Critical error in main loop: {e}", exc_info=True)
            await alerter.send_critical_alert(f"❌ 主循環發生嚴重錯誤！\n\n原因: `{e}`", alert_key='main_loop_error')
    
    async def _main_loop(self):
        """主循環"""
        log.info("Entering main loop...")
        
        # 啟動時清理孤兒訂單
        await self._handle_orphan_orders()
        
        # 初始網格重建
        current_price = await self._get_current_price()
        if current_price:
            await self._rebuild_grid(current_price)
        
        # 初始化時間戳
        now_utc = datetime.now(timezone.utc)
        self.last_recenter_ts = now_utc
        self.last_bias_rebalance_ts = now_utc
        self.last_db_snapshot_ts = now_utc
        
        loop_interval = int(self.config.get("strategy_loop_interval_sec", 10))
        
        while not self.is_halted:
            try:
                # 1. 更新價格歷史
                await self._update_price_history()
                
                # 2. 輪詢訂單狀態
                await self._poll_order_updates()
                
                # 3. 更新餘額（定期）
                now_utc = datetime.now(timezone.utc)
                if (self.last_balance_update_ts is None or 
                    (now_utc - self.last_balance_update_ts).total_seconds() >= 
                    int(self.config.get("api_balance_poll_interval_sec", 300))):
                    await self.update_balances()
                
                # 4. 檢查優化器是否需要調整參數
                await self._check_and_optimize()
                
                # 5. 執行策略邏輯（網格重建、偏置調整等）
                await self._execute_strategy_logic()
                
                # 6. 資料庫快照（定期）
                if (now_utc - self.last_db_snapshot_ts).total_seconds() >= 
                   int(self.config.get("db_snapshot_interval_sec", 3600)):
                    await self._db_log_balance_snapshot()
                    self.last_db_snapshot_ts = now_utc
                
                await asyncio.sleep(loop_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # 錯誤後等待30秒再繼續
        
        log.info("Main loop exited.")
    
    async def _check_and_optimize(self):
        """檢查並執行參數優化"""
        if not self.optimizer:
            return
        
        current_price = await self._get_current_price()
        if not current_price:
            return
        
        # 觀察市場狀態
        market_state = self.optimizer.observe_market_state(
            current_price=current_price,
            total_equity=self.total_equity_twd,
            realized_pnl=self.realized_pnl_twd,
            active_orders_count=len(self.active_orders)
        )
        
        # 計算當前績效
        current_performance = self._calculate_performance()
        
        # 嘗試優化參數
        new_params = self.optimizer.optimize_parameters(market_state, current_performance)
        
        if new_params:
            # 應用新參數
            success = self.optimizer.apply_parameters(new_params)
            if success:
                log.info("Strategy parameters optimized and applied.")
                # 記錄績效（用於後續 RL 訓練）
                self.optimizer.record_performance(current_performance)
    
    async def _execute_strategy_logic(self):
        """執行策略邏輯（網格重建、偏置調整等）"""
        current_price = await self._get_current_price()
        if not current_price:
            return
        
        now_utc = datetime.now(timezone.utc)
        
        # 網格重建（定期）
        recenter_interval = int(self.config.get("recenter_interval_minutes", 480)) * 60
        if (self.last_recenter_ts is None or 
            (now_utc - self.last_recenter_ts).total_seconds() >= recenter_interval):
            await self._rebuild_grid(current_price)
            self.last_recenter_ts = now_utc
        
        # 偏置調整（定期）
        bias_interval = int(self.config.get("bias_check_interval_sec", 60))
        if (self.last_bias_rebalance_ts is None or 
            (now_utc - self.last_bias_rebalance_ts).total_seconds() >= bias_interval):
            await self._manage_directional_bias(current_price)
            self.last_bias_rebalance_ts = now_utc
    
    async def _update_price_history(self):
        """更新價格歷史"""
        try:
            price = await self._get_current_price()
            if price:
                timestamp_ms = int(time.time() * 1000)
                self.strategy.price_history.append((timestamp_ms, price))
        except Exception as e:
            log.warning(f"Failed to update price history: {e}")
    
    async def _get_current_price(self) -> Optional[Decimal]:
        """獲取當前市場價格"""
        try:
            ticker = await max_api.get_v2_ticker(market=self.config["asset_pair"])
            if ticker and ticker.get("last"):
                return Decimal(str(ticker["last"]))
        except Exception as e:
            log.error(f"Error fetching ticker price: {e}")
        
        # 備用：使用歷史價格
        if self.strategy.price_history:
            return self.strategy.price_history[-1][1]
        
        return None
    
    async def update_balances(self):
        """更新餘額"""
        try:
            current_price = await self._get_current_price()
            if not current_price:
                return
            
            usdt_data = await max_api.get_v2_balance("usdt")
            twd_data = await max_api.get_v2_balance("twd")
            
            if usdt_data and twd_data:
                self.usdt_balance = Decimal(str(usdt_data.get("balance", "0")))
                self.twd_balance = Decimal(str(twd_data.get("balance", "0")))
                self.available_usdt_balance = Decimal(str(usdt_data.get("balance", "0")))
                self.available_twd_balance = Decimal(str(twd_data.get("balance", "0")))
                
                self.total_equity_twd = self.twd_balance + self.usdt_balance * current_price
                self.last_balance_update_ts = datetime.now(timezone.utc)
                
                # 更新峰值權益
                if self.peak_equity is None or self.total_equity_twd > self.peak_equity:
                    self.peak_equity = self.total_equity_twd
                
        except Exception as e:
            log.error(f"Error updating balances: {e}", exc_info=True)
    
    async def _rebuild_grid(self, center_price: Decimal):
        """重建網格"""
        log.info(f"Rebuilding grid around {center_price}")
        
        # 取消所有現有訂單
        await self._cancel_all_orders()
        await asyncio.sleep(2)
        
        # 獲取策略信號
        signals = self.strategy.generate_signals(center_price)
        
        if signals.get('action') == 'place_orders':
            # 執行訂單
            for signal in signals.get('signals', []):
                await self._place_order_from_signal(signal, center_price)
    
    async def _place_order_from_signal(self, signal: Dict[str, Any], current_price: Decimal):
        """根據信號下單"""
        side = signal['action']  # 'buy' or 'sell'
        price = signal['price']
        layer_idx = signal.get('layer_idx')
        size_pct = signal.get('size_pct', Decimal("0.1"))
        
        # 計算數量
        qty = self._quantize_qty((size_pct * self.total_equity_twd) / price)
        
        if qty <= 0:
            return
        
        # 下單
        await self._place_order(side, price, qty, layer_idx)
    
    async def _place_order(self, side: str, price: Decimal, qty: Decimal, layer_idx: Optional[int]) -> Optional[str]:
        """下單"""
        # 風險檢查
        if self.risk_controller:
            is_risk_hit, should_cancel_all = await self.risk_controller.enforce_risk_limits()
            if is_risk_hit:
                if should_cancel_all or side == "buy":
                    log.warning(f"Order placement halted due to risk limits.")
                    return None
        
        # 生成 client_oid
        client_oid = f"grid_{self.config['asset_pair']}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"[:36]
        
        # 量化價格和數量
        price_q = self._quantize_price(price)
        qty_q = self._quantize_qty(qty)
        
        # 檢查最小訂單價值
        min_order_value = Decimal(str(self.config.get("min_order_value_twd", "300.0")))
        if price_q * qty_q < min_order_value:
            log.debug(f"Order skipped: value {price_q * qty_q:.2f} TWD < {min_order_value} TWD")
            return None
        
        try:
            response = await max_api.place_v2_order(
                market=self.config["asset_pair"],
                side=side,
                price=price_q,
                volume=qty_q,
                client_oid=client_oid,
                ord_type='limit'
            )
            
            if response and response.get("id"):
                order_data = {
                    "client_oid": client_oid,
                    "exchange_id": str(response["id"]),
                    "price": price_q,
                    "side": side,
                    "qty": qty_q,
                    "filled_qty": Decimal("0"),
                    "layer_idx": layer_idx,
                    "status": "open",
                    "created_at_utc": datetime.now(timezone.utc)
                }
                self.active_orders[client_oid] = order_data
                await self._db_log_order(order_data)
                log.info(f"Order placed: {client_oid}")
                return client_oid
        except Exception as e:
            log.error(f"Exception placing order {client_oid}: {e}", exc_info=True)
        
        return None
    
    async def _cancel_all_orders(self):
        """取消所有訂單"""
        try:
            await max_api.cancel_all_v2_market_orders(market=self.config["asset_pair"])
            self.active_orders.clear()
            log.info("All orders cancelled.")
        except Exception as e:
            log.error(f"Error cancelling orders: {e}", exc_info=True)
    
    async def _poll_order_updates(self):
        """輪詢訂單狀態更新"""
        for oid in list(self.active_orders.keys()):
            order = self.active_orders.get(oid)
            if not order or 'exchange_id' not in order:
                continue
            
            try:
                exchange_id = int(order['exchange_id'])
                order_status = await max_api.get_v2_order(exchange_id)
                
                if order_status:
                    state = order_status.get("state")
                    if state == 'done' and order['status'] != 'filled':
                        await self._handle_order_fill(oid, order_status)
                    elif state in ['cancel', 'failed']:
                        self.active_orders.pop(oid, None)
                        await self._db_update_order_status(oid, OrderStatusEnum.CANCELLED)
            except Exception as e:
                log.warning(f"Error polling order {oid}: {e}")
            
            await asyncio.sleep(0.2)
    
    async def _handle_order_fill(self, client_oid: str, order_data: Dict[str, Any]):
        """處理訂單成交"""
        order = self.active_orders.get(client_oid)
        if not order:
            return
        
        filled_qty = Decimal(str(order_data.get("executed_volume", "0")))
        order['filled_qty'] = filled_qty
        order['status'] = 'filled'
        
        self.last_trade_ts = datetime.now(timezone.utc)
        
        # 更新餘額
        await self.update_balances()
        
        # 計算 PnL（如果是網格訂單）
        if order.get('layer_idx') is not None and order['side'] == 'sell':
            layer = self.strategy.grid_layers[order['layer_idx']]
            realized_pnl = layer.gap_abs * filled_qty
            self.realized_pnl_twd += realized_pnl
            await self._db_log_daily_pnl(realized_pnl)
        
        # 移除訂單
        self.active_orders.pop(client_oid, None)
        
        # 掛反向單（網格邏輯）
        await self._place_replacement_order(order, filled_qty)
    
    async def _place_replacement_order(self, filled_order: Dict[str, Any], filled_qty: Decimal):
        """掛反向訂單（網格邏輯）"""
        layer_idx = filled_order.get('layer_idx')
        if layer_idx is None:
            return
        
        side = filled_order['side']
        new_side = "sell" if side == "buy" else "buy"
        avg_price = filled_order.get('price', Decimal("0"))
        
        layer = self.strategy.grid_layers[layer_idx]
        new_price = self._quantize_price(
            avg_price + layer.gap_abs if new_side == 'sell' else avg_price - layer.gap_abs
        )
        
        current_price = await self._get_current_price() or new_price
        new_qty = self._quantize_qty((layer.size_pct * self.total_equity_twd) / current_price)
        
        if new_qty > 0:
            await self._place_order(new_side, new_price, new_qty, layer_idx)
    
    async def _manage_directional_bias(self, current_price: Decimal):
        """管理方向性偏置"""
        # 實現偏置調整邏輯（從原 strategy_usdttwd_grid.py 移植）
        # 這裡簡化處理
        pass
    
    def _calculate_performance(self) -> Dict[str, Decimal]:
        """計算當前績效"""
        roi = Decimal("0")
        if self.initial_equity and self.initial_equity > 0:
            roi = (self.total_equity_twd - self.initial_equity) / self.initial_equity
        
        max_drawdown = Decimal("0")
        if self.peak_equity and self.peak_equity > 0:
            max_drawdown = (self.peak_equity - self.total_equity_twd) / self.peak_equity
        
        return {
            'roi': roi,
            'realized_pnl': self.realized_pnl_twd,
            'max_drawdown': max_drawdown,
            'total_equity': self.total_equity_twd
        }
    
    async def _handle_orphan_orders(self):
        """處理啟動時的孤兒訂單"""
        log.info("Checking for orphan orders...")
        await self._cancel_all_orders()
        await asyncio.sleep(3)
        log.info("Orphan order cleanup finished.")
    
    async def _load_initial_price_history(self):
        """載入初始價格歷史"""
        # 實現從資料庫或 API 載入歷史價格
        # 這裡簡化處理
        pass
    
    async def shutdown(self, sig=None):
        """關閉引擎"""
        if self.is_halted:
            return
        
        self.is_halted = True
        self.is_running = False
        
        if sig:
            log.info(f"Received signal {sig}. Initiating shutdown...")
            await alerter.send_system_event(f"👋 收到訊號 {sig}，機器人已安全關閉。")
        
        if self.main_loop_task and not self.main_loop_task.done():
            self.main_loop_task.cancel()
        
        await self._cancel_all_orders()
        await max_api.close()
        
        log.info("BotEngine shutdown complete.")
    
    # 工具方法
    def _quantize_price(self, price: Decimal) -> Decimal:
        return price.quantize(Decimal(str(self.config.get("price_precision", "0.001"))), rounding=getcontext().rounding)
    
    def _quantize_qty(self, qty: Decimal) -> Decimal:
        return qty.quantize(Decimal(str(self.config.get("qty_precision", "0.001"))), rounding="ROUND_DOWN")
    
    # 資料庫方法（簡化版，需要完整實現）
    async def _run_db_sync(self, func, *args, **kwargs):
        import functools
        loop = asyncio.get_event_loop()
        func_call = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, func_call)
    
    def _db_get_or_create_strategy_sync(self, name: str, description: str, params: Dict) -> Optional[int]:
        with db_session() as s:
            strategy = s.query(DBStrategy).filter_by(name=name).first()
            params_str = yaml.dump(params)
            if strategy:
                strategy.params_json = params_str
                strategy.is_active = True
                s.commit()
                return strategy.id
            else:
                strategy = DBStrategy(name=name, description=description, params_json=params_str, is_active=True)
                s.add(strategy)
                s.commit()
                return strategy.id
    
    async def _db_log_order(self, order_data: Dict):
        pass  # 實現訂單記錄邏輯
    
    async def _db_update_order_status(self, client_oid: str, status: OrderStatusEnum):
        pass  # 實現訂單狀態更新邏輯
    
    async def _db_log_daily_pnl(self, pnl: Decimal):
        pass  # 實現 PnL 記錄邏輯
    
    async def _db_log_balance_snapshot(self):
        pass  # 實現餘額快照邏輯

