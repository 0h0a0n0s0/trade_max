# strategy_usdttwd_grid_refactored.py
"""
核心策略 (基於 backtester_grid.py): OOP 重構版
* 三層固定間隙網格 (small/mid/big gaps)。
* 每個網格層級的訂單數量基於當前總權益的百分比動態計算。
* EMA10h‑50h (分鐘線) 判斷趨勢，調整方向性倉位。
* 正確處理部分成交：只有當訂單完全成交後，才在另一側掛出新訂單。
* 啟動時清空舊掛單，避免孤兒訂單。
* 黑天鵝保護觸發後將永久停止，需要人工介入。
* 整合增強版風險控制器，監控TWD餘額。
* 【OOP重構】類式架構，支持動態參數調整（AI Agent熱更新）
"""
from __future__ import annotations
import asyncio
import os
import time
import signal
import logging
import uuid
import yaml
import functools
from collections import deque
from sqlalchemy import func
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta, date as DateObject
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Deque, Any, Callable
import traceback
import pandas as pd

# --- .env 檔案載入 (最優先) ---
from dotenv import load_dotenv
def find_and_load_dotenv():
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        env_path = current_dir / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"INFO: Successfully loaded .env file from: {env_path}")
            return
        current_dir = current_dir.parent
    print("CRITICAL: .env file not found. Please ensure it exists.")
find_and_load_dotenv()

from max_async_api import max_api
from risk_controller import RiskController
from telegram_alerter import alerter
from db import db_session, check_db_connection, create_all_tables
from db_schema import (
    Strategy as DBStrategy, Order as DBOrder, TradeLog as DBTradeLog,
    BalanceSnapshot as DBBalanceSnapshot, DailyPNL as DBDailyPNL,
    MarketKline1m as DBMarketKline1m, OrderStatusEnum
)
from indicators import ema, atr, adx, rsi, macd

# --- 設定 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger("GridStrategy")
getcontext().prec = 28


class GridLayer:
    """網格層級定義"""
    def __init__(self, idx: int, gap_abs: Decimal, size_pct: Decimal, levels_each_side: int):
        self.idx = idx
        self.gap_abs = gap_abs
        self.size_pct = size_pct
        self.levels_each_side = levels_each_side
    
    def __repr__(self):
        return f"GridLayer(idx={self.idx}, gap={self.gap_abs}, pct={self.size_pct*100:.2f}%)"


class GridStrategy:
    """
    網格交易策略類
    
    職責：
    - 持有策略參數（self.params）
    - 計算策略邏輯（should_rebuild, get_ema_target_bias等）
    - 管理內部狀態（grid_layers, strategy_state等）
    - 支持熱更新（update_config）
    - 提供市場觀察（get_market_observation）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略
        
        Args:
            config: 配置字典（從YAML載入）
        """
        self.params = config.copy()  # 保存完整配置
        self.asset_pair = config.get("asset_pair", "usdttwd")
        self.strategy_name = config.get("strategy_name", "Default_Grid_Strategy")
        
        # 精度設定
        self.price_precision = Decimal(str(config.get("price_precision", "0.001")))
        self.qty_precision = Decimal(str(config.get("qty_precision", "0.001")))
        
        # 價格歷史（由BotEngine注入）
        self.price_history: Deque[Tuple[int, Decimal]] = deque(maxlen=int(config.get("price_data_deque_size", 3100)))
        
        # 網格層級
        self.grid_layers: List[GridLayer] = []
        self._rebuild_grid_layers()
        
        # 策略狀態
        self.strategy_state: str = "GRID"  # "GRID" or "TREND_FOLLOWING"
        self.trend_position: Optional[Dict[str, Any]] = None
        self.cooldown_counter: int = 0
        
        # 時間戳（用於判斷是否需要重建網格）
        self.last_recenter_ts: Optional[datetime] = None
        self.last_bias_rebalance_ts: Optional[datetime] = None
        self.previous_ema_trend: Optional[str] = None
        
        log.info(f"GridStrategy '{self.strategy_name}' initialized.")
    
    def _rebuild_grid_layers(self):
        """根據當前參數重建網格層級"""
        self.grid_layers.clear()
        small_gap = Decimal(str(self.params["small_gap"]))
        levels_each = int(self.params["levels_each"])
        self.grid_layers.extend([
            GridLayer(0, small_gap, Decimal(str(self.params["size_pct_small"])), levels_each),
            GridLayer(1, small_gap * int(self.params["mid_mult"]), Decimal(str(self.params["size_pct_mid"])), levels_each),
            GridLayer(2, small_gap * int(self.params["big_mult"]), Decimal(str(self.params["size_pct_big"])), levels_each)
        ])
    
    def update_config(self, new_params: Dict[str, Any]) -> bool:
        """
        動態更新策略參數（熱更新，無需重啟）
        
        Args:
            new_params: 新的參數字典，例如：
            {
                'small_gap': '0.05',
                'ema_span_fast_bars': 120,
                'bias_high': '0.65',
                ...
            }
        
        Returns:
            bool: 是否成功更新
        """
        try:
            updated = False
            
            # 更新參數
            for key, value in new_params.items():
                if key in self.params:
                    old_value = self.params[key]
                    self.params[key] = value
                    updated = True
                    log.info(f"Parameter '{key}' updated: {old_value} -> {value}")
            
            # 如果網格參數改變，重建層級
            if updated and any(k in new_params for k in ['small_gap', 'mid_mult', 'big_mult', 'levels_each', 'size_pct_small', 'size_pct_mid', 'size_pct_big']):
                self._rebuild_grid_layers()
                log.info("Grid layers rebuilt due to parameter changes.")
            
            # 更新精度（如果改變）
            if 'price_precision' in new_params:
                self.price_precision = Decimal(str(new_params['price_precision']))
            if 'qty_precision' in new_params:
                self.qty_precision = Decimal(str(new_params['qty_precision']))
            
            return updated
            
        except Exception as e:
            log.error(f"Failed to update config: {e}", exc_info=True)
            return False
    
    def get_market_observation(self) -> Dict[str, Any]:
        """
        獲取當前市場觀察數據（供AI Agent使用）
        
        Returns:
            包含各種指標和狀態的字典
        """
        observation = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'price_history_length': len(self.price_history),
            'strategy_state': self.strategy_state,
            'grid_layers_count': len(self.grid_layers),
            'trend_position': self.trend_position.copy() if self.trend_position else None,
            'cooldown_counter': self.cooldown_counter,
            'indicators': {},
            'parameters': {}
        }
        
        # 計算技術指標
        if len(self.price_history) >= 10:
            prices = [float(p[1]) for p in self.price_history]
            series = pd.Series(prices)
            
            # EMA
            ema_fast_span = int(self.params.get("ema_span_fast_bars", 120))
            ema_slow_span = int(self.params.get("ema_span_slow_bars", 600))
            if len(series) >= ema_fast_span:
                ema_fast_series = ema(series, ema_fast_span)
                observation['indicators']['ema_fast'] = float(ema_fast_series.iloc[-1])
            if len(series) >= ema_slow_span:
                ema_slow_series = ema(series, ema_slow_span)
                observation['indicators']['ema_slow'] = float(ema_slow_series.iloc[-1])
            
            # ATR
            atr_period = int(self.params.get("atr_period", 14))
            if len(series) >= atr_period:
                atr_series = atr(series, series, series, atr_period)  # 簡化：使用收盤價作為高低價
                observation['indicators']['atr'] = float(atr_series.iloc[-1])
            
            # ADX
            dmi_period = int(self.params.get("dmi_period", 14))
            if len(series) >= dmi_period * 2:
                adx_series, plus_di, minus_di = adx(series, series, series, dmi_period)
                observation['indicators']['adx'] = float(adx_series.iloc[-1])
                observation['indicators']['plus_di'] = float(plus_di.iloc[-1])
                observation['indicators']['minus_di'] = float(minus_di.iloc[-1])
            
            # RSI
            rsi_period = int(self.params.get("rsi_period", 14))
            if len(series) >= rsi_period + 1:
                rsi_series = rsi(series, rsi_period)
                observation['indicators']['rsi'] = float(rsi_series.iloc[-1])
            
            # MACD
            macd_fast = int(self.params.get("macd_fast_period", 12))
            macd_slow = int(self.params.get("macd_slow_period", 26))
            if len(series) >= macd_slow:
                macd_line, signal_line, hist = macd(series, macd_fast, macd_slow)
                observation['indicators']['macd'] = float(macd_line.iloc[-1])
                observation['indicators']['macd_signal'] = float(signal_line.iloc[-1])
                observation['indicators']['macd_hist'] = float(hist.iloc[-1])
            
            # 波動率（簡化：使用價格標準差）
            if len(series) >= 20:
                observation['indicators']['volatility'] = float(series.tail(20).std())
        
        # 當前參數快照
        observation['parameters'] = {
            'small_gap': float(Decimal(str(self.params.get("small_gap", "0.035")))),
            'mid_mult': int(self.params.get("mid_mult", 3)),
            'big_mult': int(self.params.get("big_mult", 7)),
            'levels_each': int(self.params.get("levels_each", 6)),
            'ema_span_fast_bars': int(self.params.get("ema_span_fast_bars", 120)),
            'ema_span_slow_bars': int(self.params.get("ema_span_slow_bars", 600)),
            'bias_high': float(Decimal(str(self.params.get("bias_high", "0.60")))),
            'bias_low': float(Decimal(str(self.params.get("bias_low", "0.25")))),
            'use_atr_spacing': bool(self.params.get("use_atr_spacing", False)),
            'atr_spacing_multiplier': float(Decimal(str(self.params.get("atr_spacing_multiplier", "0.8")))),
            'use_hybrid_model': bool(self.params.get("use_hybrid_model", False)),
        }
        
        return observation
    
    def should_rebuild_grid(self, current_time: datetime) -> bool:
        """
        判斷是否應該重建網格
        
        Args:
            current_time: 當前時間
            
        Returns:
            bool: 是否應該重建
        """
        if self.last_recenter_ts is None:
            return True
        
        recenter_interval = int(self.params.get("recenter_interval_minutes", 480)) * 60
        elapsed = (current_time - self.last_recenter_ts).total_seconds()
        return elapsed >= recenter_interval
    
    def should_rebalance_bias(self, current_time: datetime) -> bool:
        """
        判斷是否應該調整方向性偏置
        
        Args:
            current_time: 當前時間
            
        Returns:
            bool: 是否應該調整
        """
        if self.last_bias_rebalance_ts is None:
            return True
        
        bias_interval = int(self.params.get("bias_check_interval_sec", 60))
        elapsed = (current_time - self.last_bias_rebalance_ts).total_seconds()
        return elapsed >= bias_interval
    
    def get_ema_target_bias(self, external_data: Optional[pd.Series] = None) -> Decimal:
        """
        根據EMA快慢線交叉，計算目標USDT曝險比例
        
        Args:
            external_data: 可選的外部數據（pandas Series），用於回測時的向量化計算
        
        Returns:
            Decimal: 目標USDT曝險比例
        """
        ema_fast = self._calculate_ema_from_history(
            int(self.params.get("ema_span_fast_bars", 120)),
            external_data=external_data
        )
        ema_slow = self._calculate_ema_from_history(
            int(self.params.get("ema_span_slow_bars", 600)),
            external_data=external_data
        )
        
        if ema_fast is None or ema_slow is None:
            return Decimal(str(self.params.get("bias_neutral_target", "0.40")))
        
        if ema_fast > ema_slow:
            return Decimal(str(self.params.get("bias_high", "0.60")))
        elif ema_fast < ema_slow:
            return Decimal(str(self.params.get("bias_low", "0.25")))
        else:
            return Decimal(str(self.params.get("bias_neutral_target", "0.40")))
    
    def _calculate_ema_from_history(self, span: int, external_data: Optional[pd.Series] = None) -> Optional[Decimal]:
        """
        計算EMA指標
        
        Args:
            span: EMA週期
            external_data: 可選的外部數據（pandas Series），用於回測時的向量化計算
        
        Returns:
            Optional[Decimal]: EMA值
        """
        if external_data is not None:
            # 使用外部數據（回測模式）
            try:
                if len(external_data) < span:
                    return None
                ema_val = external_data.ewm(span=span, adjust=False).mean().iloc[-1]
                return Decimal(str(ema_val))
            except Exception:
                return None
        else:
            # 使用內部歷史數據（實盤模式）
            if len(self.price_history) < span / 10 and len(self.price_history) < 10:
                return None
            prices = [p[1] for p in self.price_history]
            series = pd.Series(prices, dtype=float)
            try:
                ema_val = series.ewm(span=span, adjust=False).mean().iloc[-1]
                return Decimal(str(ema_val))
            except Exception:
                return None
    
    def _calculate_atr_from_history(self, period: int = 14, 
                                     external_high: Optional[pd.Series] = None,
                                     external_low: Optional[pd.Series] = None,
                                     external_close: Optional[pd.Series] = None) -> Optional[Decimal]:
        """
        計算ATR指標（簡化版）
        
        Args:
            period: ATR週期
            external_high: 可選的外部高價數據（回測模式）
            external_low: 可選的外部低價數據（回測模式）
            external_close: 可選的外部收盤價數據（回測模式）
        
        Returns:
            Optional[Decimal]: ATR值
        """
        if external_high is not None and external_low is not None and external_close is not None:
            # 使用外部數據（回測模式）
            try:
                if len(external_high) < period:
                    return None
                # 使用 indicators.py 中的 atr 函數
                from indicators import atr
                atr_series = atr(external_high, external_low, external_close, period)
                if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]):
                    return Decimal(str(atr_series.iloc[-1]))
                return None
            except Exception as e:
                log.warning(f"Failed to calculate ATR from external data: {e}")
                return None
        else:
            # 使用內部歷史數據（實盤模式，簡化版）
            if len(self.price_history) < period:
                return None
            try:
                prices = [p[1] for p in self.price_history]
                series = pd.Series(prices, dtype=float)
                high_low = series.rolling(window=period, min_periods=period).max() - \
                           series.rolling(window=period, min_periods=period).min()
                atr = high_low.rolling(window=period, min_periods=period).mean()
                if len(atr) > 0 and not pd.isna(atr.iloc[-1]):
                    return Decimal(str(atr.iloc[-1]))
                return None
            except Exception as e:
                log.warning(f"Failed to calculate ATR: {e}")
                return None
    
    def _calculate_adx_from_history(self, period: int = 14,
                                     external_high: Optional[pd.Series] = None,
                                     external_low: Optional[pd.Series] = None,
                                     external_close: Optional[pd.Series] = None) -> Optional[Decimal]:
        """
        計算ADX指標
        
        Args:
            period: ADX週期
            external_high: 可選的外部高價數據（回測模式）
            external_low: 可選的外部低價數據（回測模式）
            external_close: 可選的外部收盤價數據（回測模式）
        
        Returns:
            Optional[Decimal]: ADX值
        """
        if external_high is not None and external_low is not None and external_close is not None:
            # 使用外部數據（回測模式）
            try:
                if len(external_high) < period * 2:
                    return None
                # 使用 indicators.py 中的 adx 函數
                from indicators import adx
                adx_series, _, _ = adx(external_high, external_low, external_close, period)
                if len(adx_series) > 0 and not pd.isna(adx_series.iloc[-1]):
                    return Decimal(str(adx_series.iloc[-1]))
                return None
            except Exception as e:
                log.warning(f"Failed to calculate ADX from external data: {e}")
                return None
        else:
            # 使用內部歷史數據（實盤模式，簡化版）
            if len(self.price_history) < period * 2:
                return None
            try:
                prices = [p[1] for p in self.price_history]
                series = pd.Series(prices, dtype=float)
                price_changes = series.diff().abs()
                avg_change = price_changes.rolling(window=period, min_periods=period).mean()
                price_range = series.rolling(window=period, min_periods=period).max() - \
                             series.rolling(window=period, min_periods=period).min()
                if len(avg_change) > 0 and price_range.iloc[-1] > 0:
                    adx_approx = (avg_change.iloc[-1] / price_range.iloc[-1]) * 100
                    return Decimal(str(min(max(adx_approx, 0), 100)))
                return None
            except Exception as e:
                log.warning(f"Failed to calculate ADX: {e}")
                return None
    
    def quantize_price(self, price: Decimal) -> Decimal:
        """價格精度量化"""
        return price.quantize(self.price_precision, rounding=getcontext().rounding)
    
    def quantize_qty(self, qty: Decimal) -> Decimal:
        """數量精度量化"""
        return qty.quantize(self.qty_precision, rounding="ROUND_DOWN")


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
        self.is_running = False
        self.is_halted = False
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
            f"{self.strategy.strategy_name} - OOP Refactored",
            self.strategy.params
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
                if (self.last_balance_update_ts is None or 
                    (now_utc - self.last_balance_update_ts).total_seconds() >= 
                    int(self.strategy.params.get("api_balance_poll_interval_sec", 300))):
                    await self.update_balances()
                
                # 4. 檢查停滯警報
                stagnation_alert_hours = int(self.strategy.params.get("stagnation_alert_hours", 12))
                stagnation_seconds = stagnation_alert_hours * 3600
                if self.last_trade_ts and (now_utc - self.last_trade_ts).total_seconds() > stagnation_seconds:
                    msg = (f"*策略停滯警報!*\n\n"
                           f"距離上一筆成交已超過 `{stagnation_alert_hours}` 小時。\n\n"
                           f"市場價格可能已偏離網格有效區間，建議評估是否需要人工干預。")
                    await alerter.send_strategy_event(msg, alert_key='stagnation_alert')
                    self.last_trade_ts = now_utc
                
                # 5. 混合策略管理（如果啟用）
                if self.strategy.params.get('use_hybrid_model', False):
                    await self._manage_hybrid_strategy()
                
                # 6. 方向性偏置調整
                if self.strategy.should_rebalance_bias(now_utc):
                    await self._manage_directional_bias()
                    self.strategy.last_bias_rebalance_ts = now_utc
                
                # 7. 網格重建
                if self.strategy.should_rebuild_grid(now_utc):
                    price = await self._get_current_price()
                    if price:
                        trend_override = 'none'
                        if self.strategy.strategy_state == 'TREND_FOLLOWING' and self.strategy.trend_position:
                            trend_override = self.strategy.trend_position['side']
                        await self._rebuild_grid_at_center(price, full_rebuild=True, trend_override=trend_override)
                        self.strategy.last_recenter_ts = now_utc
                
                # 8. 資料庫快照（定期）
                if (now_utc - self.last_db_snapshot_ts).total_seconds() >= int(self.strategy.params.get("db_snapshot_interval_sec", 3600)):
                    await self._db_log_balance_snapshot()
                    self.last_db_snapshot_ts = now_utc
                
                # 9. 定期報告
                now = datetime.now()
                if now.hour in [0, 8, 18] and now.hour != self.last_report_hour:
                    log.info(f"Triggering periodic report for hour {now.hour}.")
                    await self._send_periodic_report()
                    self.last_report_hour = now.hour
                elif now.hour not in [0, 8, 18]:
                    self.last_report_hour = -1
                
                # 10. 檢查黑天鵝事件
                await self._check_black_swan_event()
                
                await asyncio.sleep(loop_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Unhandled error in main loop: {e}", exc_info=True)
                log.info("Pausing for 30 seconds before retrying...")
                await asyncio.sleep(30)
        
        log.info("Main loop exited.")
    
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
            ticker = await max_api.get_v2_ticker(market=self.strategy.asset_pair)
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
                if self.strategy.price_history:
                    current_price = self.strategy.price_history[-1][1]
                else:
                    return
            
            usdt_data = await max_api.get_v2_balance("usdt")
            twd_data = await max_api.get_v2_balance("twd")
            
            if usdt_data and twd_data:
                self.usdt_balance = Decimal(str(usdt_data.get("balance", "0"))) + Decimal(str(usdt_data.get("locked", "0")))
                self.twd_balance = Decimal(str(twd_data.get("balance", "0"))) + Decimal(str(twd_data.get("locked", "0")))
                self.available_usdt_balance = Decimal(str(usdt_data.get("balance", "0")))
                self.available_twd_balance = Decimal(str(twd_data.get("balance", "0")))
                
                self.total_equity_twd = self.twd_balance + self.usdt_balance * current_price
                self.last_balance_update_ts = datetime.now(timezone.utc)
                
        except Exception as e:
            log.error(f"Error updating balances: {e}", exc_info=True)
    
    async def _rebuild_grid_at_center(self, center_price: Decimal, full_rebuild: bool = True, trend_override: str = 'none'):
        """
        重建網格（封裝原有邏輯）
        
        注意：此方法需要訪問 BotEngine 的狀態（如 total_equity_twd, active_orders等）
        因此保留在 BotEngine 中，但使用 strategy 的參數和邏輯
        """
        log.info(f"Attempting to rebuild grid around new center price: {center_price}")
        
        # 預檢
        if self.total_equity_twd <= 0:
            if not await self.update_balances() or self.total_equity_twd <= 0:
                log.error("Equity unavailable or zero. Aborting grid rebuild.")
                return
        
        price_for_calc = await self._get_current_price() or center_price
        if price_for_calc <= 0:
            log.error("Invalid price for quantity calculation. Aborting grid rebuild.")
            return
        
        # ATR動態網格間距
        use_atr_spacing = self.strategy.params.get('use_atr_spacing', False)
        atr_multiplier = Decimal(str(self.strategy.params.get('atr_spacing_multiplier', '0.8')))
        atr_period = int(self.strategy.params.get('atr_period', 14))
        
        dynamic_gaps = {}
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
                
                log.info(f"ATR-based dynamic spacing: ATR={current_atr:.4f}, Base gap={base_gap:.4f}")
        
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
            log.warning("Grid rebuild ABORTED. Calculated smallest order value "
                       f"({smallest_order_value:.2f} TWD) is below threshold ({min_order_value_twd} TWD).")
            self.strategy.last_recenter_ts = datetime.now(timezone.utc)
            return
        
        log.info(f"Pre-flight check passed. Proceeding with grid rebuild around {center_price}")
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
            
            if trend_override == 'long':
                sell_levels = 0
            elif trend_override == 'short':
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
        log.info(f"Grid rebuild process completed. Attempted to place {len(tasks)} orders.")
        msg = (f"網格已圍繞中心價 `{center_price}` 重新建立。\n"
               f"共嘗試掛上 `{len(tasks)}` 筆新訂單。")
        await alerter.send_strategy_event(msg, alert_key='recenter')
        
        self.strategy.last_recenter_ts = datetime.now(timezone.utc)
    
    async def _place_grid_order(self, side: str, price: Decimal, qty: Decimal, layer_idx: Optional[int], tag: str = "grid") -> Optional[str]:
        """下單（封裝原有邏輯）"""
        if self.risk_controller:
            is_risk_hit, should_cancel_all = await self.risk_controller.enforce_risk_limits()
            if is_risk_hit:
                if should_cancel_all or side == "buy":
                    log.warning(f"Order placement halted due to risk limits.")
                    return None
        
        client_oid = f"{tag}_{self.strategy.asset_pair}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"[:36]
        price_q = self.strategy.quantize_price(price)
        qty_q = self.strategy.quantize_qty(qty)
        
        min_order_value = Decimal(self.strategy.params.get("min_order_value_twd", "300.0"))
        if qty_q <= 0 or price_q <= 0 or (price_q * qty_q) < min_order_value:
            log.warning(f"Order {client_oid} skipped. Calculated value {price_q * qty_q:.2f} TWD is below min_order_value_twd.")
            return None
        
        log.info(f"Attempting place: {client_oid} - {side.upper()} {qty_q} {self.strategy.params['usdt_unit']} @ {price_q} {self.strategy.params['twd_unit']}")
        try:
            use_post_only = self.strategy.params.get('use_post_only_orders', True)
            
            try:
                ticker = await max_api.get_v2_ticker(market=self.strategy.asset_pair)
                if ticker:
                    best_bid = Decimal(str(ticker.get("buy", "0")))
                    best_ask = Decimal(str(ticker.get("sell", "0")))
                    
                    if side == "buy" and best_bid > 0:
                        price_q = min(price_q, best_bid * Decimal("0.9999"))
                    elif side == "sell" and best_ask > 0:
                        price_q = max(price_q, best_ask * Decimal("1.0001"))
            except Exception as e:
                log.debug(f"Failed to get ticker for post-only adjustment: {e}")
            
            response = await max_api.place_v2_order(
                market=self.strategy.asset_pair,
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
                    "created_at_utc": datetime.now(timezone.utc),
                    "order_type": 'limit'
                }
                self.active_orders[client_oid] = order_data
                await self._db_log_order(order_data)
                log.info(f"Order placed: {client_oid}, Exchange ID: {response['id']}")
                return client_oid
            else:
                error_msg = response.get("error", {"message": "Unknown error"}) if response else {"message": "No response"}
                log.error(f"Failed to place order {client_oid}: {error_msg}")
                if "balance" in str(error_msg).lower():
                    await self.update_balances()
                return None
        except Exception as e:
            log.error(f"Exception placing order {client_oid}: {e}", exc_info=True)
            return None
    
    async def _cancel_all_market_orders(self, reason: str = "generic_sweep"):
        """取消所有訂單"""
        log.info(f"Sending command to cancel ALL orders for {self.strategy.asset_pair} due to: {reason}")
        try:
            result = await max_api.cancel_all_v2_market_orders(market=self.strategy.asset_pair)
            log.info(f"Exchange-level cancel-all command sent. Result: {result}")
            self.active_orders.clear()
        except Exception as e:
            log.error(f"Error during exchange-level mass cancel: {e}", exc_info=True)
    
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
        
        if order['status'] in ['filled', 'cancelled']:
            return
        
        cummulative_qty = Decimal(str(order_data.get("executed_volume", "0")))
        final_status_str = order_data.get("state", "filled")
        final_status = OrderStatusEnum(final_status_str) if final_status_str in OrderStatusEnum._value2member_map_ else OrderStatusEnum.FILLED
        
        order['filled_qty'] = cummulative_qty
        order['status'] = final_status.value
        
        log.info(f"Order update: {client_oid}, Status: {final_status.value}, TotalFilled: {cummulative_qty}/{order.get('qty', 'N/A')}")
        
        db_update_payload = {
            "client_oid": client_oid,
            "status": final_status,
            "filled_quantity": cummulative_qty,
            "average_fill_price": Decimal(str(order_data.get("avg_price", order.get('price'))))
        }
        await self._db_update_order_status_dict(db_update_payload)
        
        if final_status == OrderStatusEnum.FILLED:
            self.last_trade_ts = datetime.now(timezone.utc)
            log.info(f"Order {client_oid} is fully filled. Processing balance update and placing replacement.")
            await self.update_balances()
            
            layer_idx, side = order.get("layer_idx"), order.get("side")
            self.active_orders.pop(client_oid, None)
            
            if layer_idx is not None:
                layer = self.strategy.grid_layers[layer_idx]
                if side == "sell":
                    realized_pnl = layer.gap_abs * cummulative_qty
                    log.info(f"GRID PNL: Realized PNL of approx. {realized_pnl:.4f} TWD from trade {client_oid}")
                    await self._db_log_daily_pnl({"realized_pnl_twd": realized_pnl})
                
                # 掛反向單
                if self.strategy.strategy_state == 'TREND_FOLLOWING' and self.strategy.trend_position:
                    trend_side = self.strategy.trend_position['side']
                    if (side == 'buy' and trend_side == 'long') or (side == 'sell' and trend_side == 'short'):
                        new_side = side
                        avg_fill_price = db_update_payload['average_fill_price']
                        new_price = self.strategy.quantize_price(avg_fill_price + layer.gap_abs if new_side == 'sell' else avg_fill_price - layer.gap_abs)
                        price_for_calc = await self._get_current_price() or new_price
                        new_qty = self.strategy.quantize_qty((layer.size_pct * self.total_equity_twd) / price_for_calc)
                        if new_qty > 0:
                            await self._place_grid_order(new_side, new_price, new_qty, layer.idx, tag="gr_repl")
                else:
                    new_side = "sell" if side == "buy" else "buy"
                    avg_fill_price = db_update_payload['average_fill_price']
                    new_price = self.strategy.quantize_price(avg_fill_price + layer.gap_abs if new_side == 'sell' else avg_fill_price - layer.gap_abs)
                    price_for_calc = await self._get_current_price() or new_price
                    new_qty = self.strategy.quantize_qty((layer.size_pct * self.total_equity_twd) / price_for_calc)
                    if new_qty > 0:
                        await self._place_grid_order(new_side, new_price, new_qty, layer.idx, tag="gr_repl")
                    else:
                        log.warning(f"Calculated replacement qty for {client_oid} is zero, skipping.")
    
    async def _manage_hybrid_strategy(self):
        """管理混合策略模式（封裝原有邏輯）"""
        if not self.strategy.params.get('use_hybrid_model', False):
            return
        
        if self.strategy.cooldown_counter > 0:
            self.strategy.cooldown_counter -= 1
            return
        
        adx_period = int(self.strategy.params.get('dmi_period', 14))
        current_adx = self.strategy._calculate_adx_from_history(adx_period)
        if current_adx is None:
            return
        
        adx_threshold = int(self.strategy.params.get('adx_strength_threshold', 25))
        
        ema_fast = self.strategy._calculate_ema_from_history(int(self.strategy.params["ema_span_fast_bars"]))
        ema_slow = self.strategy._calculate_ema_from_history(int(self.strategy.params["ema_span_slow_bars"]))
        if ema_fast is None or ema_slow is None:
            return
        
        current_price = await self._get_current_price()
        if not current_price or current_price <= 0:
            return
        
        is_ema_bull = ema_fast > ema_slow
        is_ema_bear = ema_fast < ema_slow
        is_adx_strong = current_adx > adx_threshold
        
        if self.strategy.strategy_state == 'GRID':
            is_strong_uptrend = is_ema_bull and is_adx_strong
            is_strong_downtrend = is_ema_bear and is_adx_strong
            
            if is_strong_uptrend or is_strong_downtrend:
                self.strategy.strategy_state = 'TREND_FOLLOWING'
                trend_side = 'long' if is_strong_uptrend else 'short'
                
                await self._cancel_all_market_orders(reason="entering_trend_following")
                await asyncio.sleep(2)
                
                trend_equity_pct = Decimal(str(self.strategy.params.get('trend_trade_equity_pct', '0.4')))
                trade_value_twd = self.total_equity_twd * trend_equity_pct
                
                if trend_side == 'long':
                    qty_to_buy = self.strategy.quantize_qty(trade_value_twd / current_price)
                    if self.available_twd_balance >= trade_value_twd:
                        buy_price = current_price * Decimal("1.001")
                        client_oid = await self._place_grid_order("buy", buy_price, qty_to_buy, layer_idx=None, tag="trend_long")
                        if client_oid:
                            self.strategy.trend_position = {
                                'side': 'long',
                                'entry_price': current_price,
                                'qty': qty_to_buy,
                                'peak_price': current_price
                            }
                            log.info(f"Entered TREND_FOLLOWING mode (LONG): {qty_to_buy:.4f} USDT @ {current_price:.3f}")
                            msg = (f"📈 **進入趨勢跟隨模式（做多）**\n\n"
                                   f"ADX: `{current_adx:.2f}` (強趨勢)\n"
                                   f"EMA: 快線 > 慢線\n"
                                   f"建立多頭倉位: `{qty_to_buy:.4f} USDT` @ `{current_price:.3f}`")
                            await alerter.send_strategy_event(msg, alert_key='trend_entry')
                            await self._rebuild_grid_at_center(current_price, full_rebuild=False, trend_override='long')
                else:  # short
                    qty_to_sell = self.strategy.quantize_qty(trade_value_twd / current_price)
                    if self.available_usdt_balance >= qty_to_sell:
                        sell_price = current_price * Decimal("0.999")
                        client_oid = await self._place_grid_order("sell", sell_price, qty_to_sell, layer_idx=None, tag="trend_short")
                        if client_oid:
                            self.strategy.trend_position = {
                                'side': 'short',
                                'entry_price': current_price,
                                'qty': qty_to_sell,
                                'valley_price': current_price
                            }
                            log.info(f"Entered TREND_FOLLOWING mode (SHORT): {qty_to_sell:.4f} USDT @ {current_price:.3f}")
                            msg = (f"📉 **進入趨勢跟隨模式（做空）**\n\n"
                                   f"ADX: `{current_adx:.2f}` (強趨勢)\n"
                                   f"EMA: 快線 < 慢線\n"
                                   f"建立空頭倉位: `{qty_to_sell:.4f} USDT` @ `{current_price:.3f}`")
                            await alerter.send_strategy_event(msg, alert_key='trend_entry')
                            await self._rebuild_grid_at_center(current_price, full_rebuild=False, trend_override='short')
        
        elif self.strategy.strategy_state == 'TREND_FOLLOWING':
            if not self.strategy.trend_position:
                self.strategy.strategy_state = 'GRID'
                return
            
            trailing_stop_pct = Decimal(str(self.strategy.params.get('trend_trailing_stop_pct', '0.02')))
            side = self.strategy.trend_position['side']
            should_exit = False
            exit_reason = ""
            
            if side == 'long':
                peak_price = max(self.strategy.trend_position.get('peak_price', current_price), current_price)
                self.strategy.trend_position['peak_price'] = peak_price
                stop_loss_price = peak_price * (Decimal("1.0") - trailing_stop_pct)
                if current_price <= stop_loss_price:
                    should_exit = True
                    exit_reason = f"Trailing Stop Hit. Price ({current_price:.3f}) <= Stop ({stop_loss_price:.3f})"
            
            elif side == 'short':
                valley_price = min(self.strategy.trend_position.get('valley_price', current_price), current_price)
                self.strategy.trend_position['valley_price'] = valley_price
                stop_loss_price = valley_price * (Decimal("1.0") + trailing_stop_pct)
                if current_price >= stop_loss_price:
                    should_exit = True
                    exit_reason = f"Trailing Stop Hit. Price ({current_price:.3f}) >= Stop ({stop_loss_price:.3f})"
            
            if should_exit:
                qty = self.strategy.trend_position['qty']
                entry_price = self.strategy.trend_position['entry_price']
                
                if side == 'long':
                    sell_price = current_price * Decimal("0.999")
                    await self._place_grid_order("sell", sell_price, qty, layer_idx=None, tag="trend_exit")
                    pnl = (current_price - entry_price) * qty
                else:  # short
                    buy_price = current_price * Decimal("1.001")
                    await self._place_grid_order("buy", buy_price, qty, layer_idx=None, tag="trend_exit")
                    pnl = (entry_price - current_price) * qty
                
                log.info(f"Exited TREND_FOLLOWING mode. PNL: {pnl:.2f} TWD. Reason: {exit_reason}")
                msg = (f"🔄 **退出趨勢跟隨模式**\n\n"
                       f"原因: {exit_reason}\n"
                       f"已實現損益: `{pnl:+.2f} TWD`")
                await alerter.send_strategy_event(msg, alert_key='trend_exit')
                
                self.strategy.trend_position = None
                self.strategy.strategy_state = 'GRID'
                
                cooldown_bars = int(self.strategy.params.get('trend_cooldown_bars', 240))
                self.strategy.cooldown_counter = cooldown_bars
                
                await self._cancel_all_market_orders(reason="exiting_trend_following")
                await asyncio.sleep(2)
                await self._rebuild_grid_at_center(current_price, full_rebuild=False, trend_override='none')
    
    async def _manage_directional_bias(self):
        """管理方向性偏置（封裝原有邏輯）"""
        await self.update_balances()
        
        target_ratio = self.strategy.get_ema_target_bias()
        
        current_trend = "看漲" if target_ratio == Decimal(self.strategy.params["bias_high"]) else "看跌" if target_ratio == Decimal(self.strategy.params["bias_low"]) else "中性"
        if self.strategy.previous_ema_trend is None:
            self.strategy.previous_ema_trend = current_trend
        elif current_trend != self.strategy.previous_ema_trend:
            log.info(f"EMA trend has changed from '{self.strategy.previous_ema_trend}' to '{current_trend}'. Sending alert.")
            msg = (f"🧭 **趨勢變更: {current_trend}**\n\n"
                   f"EMA 指標已發生變化。\n"
                   f"目標 USDT 倉位比例已調整為: `{target_ratio:.0%}`")
            await alerter.send_strategy_event(msg, alert_key='trend_change')
            self.strategy.previous_ema_trend = current_trend
        
        price = await self._get_current_price()
        if not price or price <= 0 or self.total_equity_twd <= 0:
            return
        
        current_ratio = (self.usdt_balance * price) / self.total_equity_twd
        delta_value_target = (target_ratio - current_ratio) * self.total_equity_twd
        
        if abs(delta_value_target) > Decimal(self.strategy.params["bias_rebalance_threshold_twd"]):
            value_to_trade = delta_value_target * Decimal(self.strategy.params["bias_rebalance_fraction"])
            qty_to_trade = self.strategy.quantize_qty(value_to_trade / price)
            
            side = "buy" if qty_to_trade > 0 else "sell"
            qty_abs = abs(qty_to_trade)
            
            slip_price = price * (Decimal("1.001") if side == "buy" else Decimal("0.999"))
            order_value_twd = abs(qty_abs * slip_price)
            min_order_value = Decimal(self.strategy.params.get("min_order_value_twd", "300.0"))
            
            if order_value_twd < min_order_value:
                log.debug(f"Bias rebalance skipped. Calculated order value {order_value_twd:.2f} TWD is below threshold.")
                return
            
            SAFETY_MARGIN = Decimal("1.01")
            
            if side == 'buy' and self.available_twd_balance < (order_value_twd * SAFETY_MARGIN):
                log.debug(f"Bias rebalance BUY skipped. Insufficient available TWD with safety margin.")
                return
            if side == 'sell' and self.available_usdt_balance < (qty_abs * SAFETY_MARGIN):
                log.debug(f"Bias rebalance SELL skipped. Insufficient available USDT with safety margin.")
                return
            
            if qty_abs > 0:
                log.info(f"Bias rebalance: EMA trend suggests target {target_ratio:.0%}, trying to {side} {qty_abs} USDT.")
                await self._place_grid_order(side, slip_price, qty_abs, layer_idx=None, tag="bias_")
                self.strategy.last_bias_rebalance_ts = datetime.now(timezone.utc)
    
    async def _check_black_swan_event(self):
        """檢查黑天鵝事件"""
        if not self.strategy.params.get("use_black_swan_protection"):
            return
        
        check_minutes = int(self.strategy.params["black_swan_check_minutes"])
        threshold_pct = Decimal(self.strategy.params["black_swan_threshold_pct"])
        
        if len(self.strategy.price_history) < check_minutes * 5:
            return
        
        now_ts = time.time()
        past_ts = now_ts - (check_minutes * 60)
        
        relevant_prices = [p[1] for p in self.strategy.price_history if p[0]/1000 >= past_ts]
        if len(relevant_prices) < 2:
            return
        
        current_price = relevant_prices[-1]
        highest_price = max(relevant_prices)
        lowest_price = min(relevant_prices)
        
        if (highest_price - lowest_price) / lowest_price > threshold_pct:
            msg = (f"*USDTTWD 在 {check_minutes} 分鐘內波動超過 {threshold_pct:.1%}!*\n\n"
                   f"價格區間: `{lowest_price}` - `{highest_price}`\n\n"
                   f"策略已自動停止並撤銷所有訂單，請立即介入檢查！")
            await alerter.send_critical_alert(msg, alert_key='black_swan')
            
            log.critical("!!! BLACK SWAN EVENT DETECTED !!!")
            log.critical(f"Price fluctuated more than {threshold_pct:.2%} within {check_minutes} minutes.")
            log.critical("HALTING STRATEGY TO PREVENT FURTHER LOSSES. MANUAL INTERVENTION REQUIRED.")
            self.is_halted = True
            asyncio.create_task(self.shutdown(sig="BLACK_SWAN"))
    
    async def _handle_orphan_orders(self):
        """處理啟動時的孤兒訂單"""
        log.info("Checking for existing open orders (orphans) on startup...")
        try:
            await self._cancel_all_market_orders(reason="startup_cleanup")
            await asyncio.sleep(3)
            log.info("Orphan order cleanup finished.")
        except Exception as e:
            log.error(f"Critical error handling orphan orders on startup: {e}.", exc_info=True)
            raise SystemExit("Failed to handle orphan orders. Halting.")
    
    async def _load_initial_price_history(self):
        """載入初始價格歷史"""
        limit = self.strategy.price_history.maxlen or 3100
        
        with db_session() as s:
            kline_data_db = s.query(DBMarketKline1m.ts, DBMarketKline1m.close).filter(
                DBMarketKline1m.asset_pair == self.strategy.asset_pair
            ).order_by(DBMarketKline1m.ts.desc()).limit(limit).all()
        
        if not kline_data_db:
            log.info("DB has no K-line history, fetching from MAX API...")
            try:
                k_data_api = await max_api.get_v2_k_data(self.strategy.asset_pair, limit, 1)
                
                if k_data_api:
                    with db_session() as s:
                        for k in k_data_api:
                            try:
                                ts_dt = datetime.fromtimestamp(k[0], tz=timezone.utc)
                                open_p, high_p, low_p, close_p = (Decimal(str(p)) for p in k[1:5])
                                vol_asset = Decimal(str(k[5]))
                                vol_quote = vol_asset * close_p
                                self._db_save_kline_sync(ts_dt, open_p, high_p, low_p, close_p, vol_asset, vol_quote)
                            except Exception:
                                pass
                    
                    with db_session() as s:
                        kline_data_db = s.query(DBMarketKline1m.ts, DBMarketKline1m.close).filter(
                            DBMarketKline1m.asset_pair == self.strategy.asset_pair
                        ).order_by(DBMarketKline1m.ts.desc()).limit(limit).all()
            except Exception as e:
                log.error(f"Failed to fetch K-line data from API: {e}", exc_info=True)
                return
        
        history = [(int(row.ts.timestamp() * 1000), row.close) for row in reversed(kline_data_db or [])]
        self.strategy.price_history.extend(history)
        log.info(f"DB: Loaded {len(history)} K-line records for initial price history.")
    
    async def _send_periodic_report(self):
        """發送定期報告"""
        try:
            pnl_summary = await self._run_db_sync(self._db_get_pnl_summary_sync)
            
            current_price = await self._get_current_price() or (self.strategy.price_history[-1][1] if self.strategy.price_history else Decimal("30.0"))
            if self.total_equity_twd > 0:
                current_usdt_ratio = (self.usdt_balance * current_price) / self.total_equity_twd
            else:
                current_usdt_ratio = Decimal("0.0")
            
            target_usdt_ratio = self.strategy.get_ema_target_bias()
            current_trend = "看漲" if target_usdt_ratio == Decimal(self.strategy.params["bias_high"]) else "看跌" if target_usdt_ratio == Decimal(self.strategy.params["bias_low"]) else "中性"
            
            report_text = (
                f"📊 **USDTTWD 網格策略績效報告**\n"
                f"_(截至 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})_\n\n"
                f"--- *績效 (TWD)* ---\n"
                f"**今日已實現利潤:** `{pnl_summary['today']:+.2f}`\n"
                f"**今日總成交筆數:** `{pnl_summary['trades_today']}`\n"
                f"**近七日已實現利潤:** `{pnl_summary['last_7_days']:+.2f}`\n"
                f"**當月已實現利潤:** `{pnl_summary['this_month']:+.2f}`\n"
                f"**當前半年已實現利潤:** `{pnl_summary['this_half_year']:+.2f}`\n"
                f"**當年已實現利潤:** `{pnl_summary['this_year']:+.2f}`\n\n"
                f"--- *狀態* ---\n"
                f"**帳戶總權益:** `{self.total_equity_twd:,.2f} TWD`\n"
                f"**當前持倉:** `{self.usdt_balance:,.2f} USDT` / `{self.twd_balance:,.2f} TWD` ({current_usdt_ratio:.1%})\n"
                f"**在掛訂單數:** `{len(self.active_orders)}`\n"
                f"**當前趨勢判斷:** `{current_trend}` (目標 `{target_usdt_ratio:.0%} USDT`)"
            )
            
            await alerter.send_system_event(report_text)
            
        except Exception as e:
            log.error(f"Failed to send periodic report: {e}", exc_info=True)
            await alerter.send_critical_alert(f"❌ 產生績效報告時發生錯誤！\n\n原因: `{e}`", alert_key='report_fail')
    
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
        
        await self._cancel_all_market_orders()
        await max_api.close()
        
        log.info("BotEngine shutdown complete.")
    
    # --- 資料庫方法 ---
    async def _run_db_sync(self, func: Callable, *args, **kwargs):
        """執行同步資料庫函數"""
        loop = asyncio.get_event_loop()
        func_call = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, func_call)
    
    def _db_get_or_create_strategy_sync(self, name: str, description: str, params: Dict) -> Optional[int]:
        """獲取或創建策略記錄"""
        with db_session() as s:
            try:
                strategy = s.query(DBStrategy).filter_by(name=name).first()
                params_str = yaml.dump(params)
                
                if strategy:
                    strategy.description = description
                    strategy.params_json = params_str
                    strategy.is_active = True
                    log.info(f"DB: Updating existing strategy '{name}' (ID: {strategy.id}).")
                else:
                    strategy = DBStrategy(name=name, description=description, params_json=params_str, is_active=True)
                    s.add(strategy)
                    log.info(f"DB: Creating new strategy '{name}'.")
                
                s.commit()
                strategy_id = strategy.id
                log.info(f"DB: Strategy '{name}' successfully processed. ID: {strategy_id}")
                return strategy_id
                
            except Exception as e:
                log.error(f"DB error for strategy '{name}': {e}", exc_info=True)
                s.rollback()
                return None
    
    async def _db_log_order(self, order_data: Dict):
        """記錄訂單"""
        with db_session() as s:
            try:
                db_order = DBOrder(
                    strategy_id=self.strategy_db_id,
                    client_oid=order_data["client_oid"],
                    exchange_order_id=order_data.get("exchange_id"),
                    asset_pair=self.strategy.asset_pair,
                    side=order_data["side"],
                    order_type=order_data.get("order_type", "post_only"),
                    price=order_data["price"],
                    quantity=order_data["qty"],
                    status=OrderStatusEnum.NEW,
                    layer_idx=order_data.get("layer_idx")
                )
                s.add(db_order)
                s.commit()
                log.debug(f"DB: Logged new order: client_oid={db_order.client_oid}")
            except Exception as e:
                log.error(f"DB error logging order {order_data.get('client_oid')}: {e}", exc_info=True)
                s.rollback()
    
    async def _db_update_order_status_dict(self, update_data: dict):
        """更新訂單狀態"""
        await self._run_db_sync(self._db_update_order_status_sync, update_data)
    
    def _db_update_order_status_sync(self, update_data: dict) -> bool:
        """更新訂單狀態（同步版本）"""
        with db_session() as s:
            try:
                client_oid = update_data.get("client_oid")
                if not client_oid:
                    return False
                
                order = s.query(DBOrder).filter_by(client_oid=client_oid).first()
                if order:
                    for key, value in update_data.items():
                        if hasattr(order, key) and value is not None:
                            setattr(order, key, value)
                    s.commit()
                    log.debug(f"DB: Updated order {client_oid} with data: {update_data}")
                    return True
                log.warning(f"DB: Order {client_oid} not found for status update.")
                return False
            except Exception as e:
                log.error(f"DB error updating order {update_data.get('client_oid')} status: {e}", exc_info=True)
                s.rollback()
                return False
    
    async def _db_log_daily_pnl(self, pnl_data: dict):
        """記錄每日PNL"""
        await self._run_db_sync(self._db_log_daily_pnl_sync, pnl_data)
    
    def _db_log_daily_pnl_sync(self, pnl_data: dict) -> bool:
        """記錄每日PNL（同步版本）"""
        with db_session() as s:
            try:
                today = datetime.now(timezone.utc).date()
                realized_pnl_twd = pnl_data.get("realized_pnl_twd", Decimal("0.0"))
                
                pnl_entry = s.query(DBDailyPNL).filter_by(trade_date=today, strategy_id=self.strategy_db_id, asset_pair=self.strategy.asset_pair).first()
                if pnl_entry:
                    pnl_entry.realized_pnl += realized_pnl_twd
                    pnl_entry.net_pnl = pnl_entry.realized_pnl
                else:
                    pnl_entry = DBDailyPNL(
                        trade_date=today,
                        strategy_id=self.strategy_db_id,
                        asset_pair=self.strategy.asset_pair,
                        realized_pnl=realized_pnl_twd,
                        net_pnl=realized_pnl_twd,
                        pnl_currency=self.strategy.params['twd_unit'].upper(),
                    )
                    s.add(pnl_entry)
                
                s.commit()
                display_precision = self.strategy.params.get('price_precision', '0.001')
                log.info(f"DB: Logged/Updated daily PnL for {today}: {realized_pnl_twd:.{len(display_precision.split('.')[-1])}f} {self.strategy.params['twd_unit'].upper()}")
                return True
            except Exception as e:
                log.error(f"DB error logging daily PnL: {e}", exc_info=True)
                s.rollback()
                return False
    
    def _db_get_pnl_summary_sync(self) -> Dict[str, Decimal]:
        """獲取PNL匯總"""
        with db_session() as s:
            today = datetime.now(timezone.utc).date()
            
            seven_days_ago = today - timedelta(days=6)
            start_of_month = today.replace(day=1)
            start_of_half_year = today.replace(month=1, day=1) if today.month <= 6 else today.replace(month=7, day=1)
            start_of_year = today.replace(month=1, day=1)
            
            def query_pnl(start_date, end_date):
                result = s.query(func.sum(DBDailyPNL.realized_pnl)).filter(
                    DBDailyPNL.strategy_id == self.strategy_db_id,
                    DBDailyPNL.trade_date >= start_date,
                    DBDailyPNL.trade_date <= end_date
                ).scalar()
                return result or Decimal("0.0")
            
            pnl_today = query_pnl(today, today)
            pnl_7_days = query_pnl(seven_days_ago, today)
            pnl_month = query_pnl(start_of_month, today)
            pnl_half_year = query_pnl(start_of_half_year, today)
            pnl_year = query_pnl(start_of_year, today)
            
            pnl_entry_today = s.query(DBDailyPNL).filter_by(trade_date=today, strategy_id=self.strategy_db_id).first()
            trades_count_today = pnl_entry_today.trades_count if pnl_entry_today else 0
            
            return {
                "today": pnl_today,
                "trades_today": trades_count_today,
                "last_7_days": pnl_7_days,
                "this_month": pnl_month,
                "this_half_year": pnl_half_year,
                "this_year": pnl_year,
            }
    
    async def _db_log_balance_snapshot(self):
        """記錄餘額快照"""
        await self._run_db_sync(self._db_log_balance_snapshot_sync)
    
    def _db_log_balance_snapshot_sync(self) -> bool:
        """記錄餘額快照（同步版本）"""
        with db_session() as s:
            try:
                ts = datetime.now(timezone.utc)
                s.add(DBBalanceSnapshot(snapshot_ts=ts, currency=self.strategy.params["usdt_unit"].upper(), total_balance=self.usdt_balance, available_balance=self.usdt_balance))
                s.add(DBBalanceSnapshot(snapshot_ts=ts, currency=self.strategy.params["twd_unit"].upper(), total_balance=self.twd_balance, available_balance=self.twd_balance))
                s.commit()
                log.info("DB: Logged balance snapshot.")
                return True
            except Exception as e:
                log.error(f"DB error logging balance snapshot: {e}", exc_info=True)
                s.rollback()
                return False
    
    def _db_save_kline_sync(self, ts_dt: datetime, open_p: Decimal, high_p: Decimal, low_p: Decimal, close_p: Decimal, vol_asset: Decimal, vol_quote: Decimal):
        """保存K線數據（同步版本）"""
        with db_session() as s:
            try:
                kline = s.query(DBMarketKline1m).filter_by(ts=ts_dt, asset_pair=self.strategy.asset_pair).first()
                if kline:
                    kline.open = open_p
                    kline.high = high_p
                    kline.low = low_p
                    kline.close = close_p
                    kline.volume_asset = vol_asset
                    kline.volume_quote = vol_quote
                else:
                    kline = DBMarketKline1m(
                        ts=ts_dt, asset_pair=self.strategy.asset_pair,
                        open=open_p, high=high_p, low=low_p, close=close_p,
                        volume_asset=vol_asset, volume_quote=vol_quote
                    )
                    s.add(kline)
                s.commit()
            except Exception as e:
                log.error(f"DB error saving K-line for {ts_dt}: {e}", exc_info=True)
                s.rollback()


async def main():
    """主函數"""
    try:
        # 載入配置
        config_path = Path(os.getenv("STRATEGY_CFG", Path(__file__).resolve().parent / "config_usdttwd.yaml"))
        if not config_path.exists():
            raise SystemExit(f"CRITICAL: Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        log.info(f"Loading configuration from: {config_path}")
        
        # 創建策略實例
        strategy = GridStrategy(config)
        
        # 創建引擎實例
        engine = BotEngine(strategy, config_path)
        
        # 初始化並啟動
        await engine.initialize()
        await engine.start()
        
    except SystemExit as e:
        log.warning(f"SystemExit: {e}")
        await alerter.send_critical_alert(f"❌ 機器人啟動失敗！\n\n原因: `{e}`", alert_key='startup_fail')
    except Exception as e:
        log.critical(f"Critical error during startup: {e}", exc_info=True)
        await alerter.send_critical_alert(f"❌ 機器人啟動時發生嚴重錯誤！\n\n原因: `{e}`", alert_key='startup_fail')
    finally:
        if 'engine' in locals():
            await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

