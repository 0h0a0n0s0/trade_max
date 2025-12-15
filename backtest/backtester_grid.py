# backtester_grid.py (V9 - 優化版: 簡化進場 + 順勢網格)
import argparse
import logging
from decimal import Decimal, getcontext
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import yaml
import numpy as np

# --- [V7 新增] 引入繪圖函式庫 ---
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from indicators import ema, macd, adx, atr, rsi, bollinger, stochastic_oscillator

# --- 設定 ---
getcontext().prec = 28
LOG = logging.getLogger("GridBacktesterV9")
logging.basicConfig(format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s", level=logging.INFO)

# --- 全局狀態 ---
ACTIVE_ORDERS: Dict[str, Dict] = {}
USDT_BALANCE: Decimal = Decimal("0")
TWD_BALANCE: Decimal = Decimal("0")
TOTAL_EQUITY_TWD: Decimal = Decimal("0")

# --- 模擬類 ---
class GridLayer:
    def __init__(self, idx: int, gap_abs: Decimal, size_pct: Decimal, levels_each_side: int):
        self.idx, self.gap_abs, self.size_pct, self.levels_each_side = idx, gap_abs, size_pct, levels_each_side

def quantize(val: Decimal, precision: str) -> Decimal:
    return val.quantize(Decimal(precision))

# --- [V7.5 最終穩健版] 繪圖函數 (無變更) ---
def plot_backtest_results(price_df: pd.DataFrame, trade_log: List[Dict], output_filename: str = "backtest_results_v9.png"):
    """
    將回測結果視覺化（增強版：包含 EMA 與 ADX 指標）。
    """
    if len(price_df) == 0:
        LOG.warning("Price data is empty. Skipping plot generation.")
        return

    LOG.info(f"Generating enhanced backtest result plot, saving to {output_filename}...")
    
    trade_df = pd.DataFrame(trade_log)
    if not trade_df.empty:
        trade_df['price'] = pd.to_numeric(trade_df['price'])
    
    # 準備繪圖數據
    plot_indices = range(len(price_df))
    plot_prices = price_df['close'].astype(float)
    
    # 設定畫布：兩個子圖 (上圖價格 70%, 下圖 ADX 30%)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- 上圖：價格與 EMA ---
    ax1.plot(plot_indices, plot_prices, label='Close Price', color='skyblue', linewidth=1, alpha=0.6)
    
    # 繪製 EMA (如果存在)
    if 'ema_fast' in price_df.columns and 'ema_slow' in price_df.columns:
        ax1.plot(plot_indices, price_df['ema_fast'], label='EMA Fast', color='orange', linewidth=1.5, linestyle='--')
        ax1.plot(plot_indices, price_df['ema_slow'], label='EMA Slow', color='purple', linewidth=1.5, linestyle='--')

    # 標記交易點
    if not trade_df.empty:
        # 將 datetime 索引轉換為整數索引
        trade_df['plot_index'] = trade_df['index'].apply(lambda x: price_df.index.get_loc(x) if x in price_df.index else -1)
        valid_trades = trade_df[trade_df['plot_index'] != -1]

        # 網格交易
        grid_buys = valid_trades[valid_trades['type'] == 'grid_buy']
        grid_sells = valid_trades[valid_trades['type'] == 'grid_sell']
        ax1.scatter(grid_buys['plot_index'], grid_buys['price'], label='Grid Buy', marker='^', color='lime', s=30, zorder=5)
        ax1.scatter(grid_sells['plot_index'], grid_sells['price'], label='Grid Sell', marker='v', color='red', s=30, zorder=5)
        
        # 趨勢交易
        trend_entries = valid_trades[valid_trades['type'].str.contains('entry')]
        trend_exits = valid_trades[valid_trades['type'] == 'trend_exit']
        ax1.scatter(trend_entries['plot_index'], trend_entries['price'], label='Trend Entry', marker='o', color='blue', s=100, zorder=10, edgecolors='white', linewidth=2)
        ax1.scatter(trend_exits['plot_index'], trend_exits['price'], label='Trend Exit', marker='X', color='black', s=100, zorder=10, edgecolors='white', linewidth=2)

    ax1.set_title('Price Action with EMA Trend & Trades', fontsize=14)
    ax1.set_ylabel('Price (TWD)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 下圖：ADX 指標 ---
    if 'adx' in price_df.columns:
        ax2.plot(plot_indices, price_df['adx'], label='ADX Strength', color='magenta', linewidth=1.5)
        # 繪製閾值線 (假設預設 25，可從 config 讀取更好，這裡先寫死或用變數)
        ax2.axhline(y=25, color='gray', linestyle=':', label='Trend Threshold (25)')
        ax2.fill_between(plot_indices, price_df['adx'], 25, where=(price_df['adx'] > 25), color='magenta', alpha=0.1)
    
    ax2.set_title('ADX Trend Strength', fontsize=12)
    ax2.set_ylabel('ADX Value', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 處理 X 軸標籤
    num_ticks = 12
    tick_positions = [int(p) for p in np.linspace(0, len(price_df) - 1, num_ticks)]
    tick_labels = [price_df.index[pos].strftime('%Y-%m-%d %H:%M') for pos in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    LOG.info(f"Enhanced plot saved to {output_filename}")


# --- 核心回測邏輯 ---
class Backtester:
    def __init__(self, cfg: dict, init_usdt: Decimal, init_twd: Decimal, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.fee = Decimal(cfg['taker_fee'])
        self.min_order_value_twd = Decimal(cfg['min_order_value_twd'])
        self.grid_layers: List[GridLayer] = self._setup_grid_layers(cfg)
        self.use_hybrid = cfg.get('use_hybrid_model', False)
        self.trend_equity_pct = Decimal(str(cfg.get('trend_trade_equity_pct', '0.4')))
        self.trailing_stop_pct = Decimal(str(cfg.get('trend_trailing_stop_pct', '0.02')))
        self.cooldown_bars = int(cfg.get('trend_cooldown_bars', 240))
        self.adx_strength_threshold = int(cfg.get('adx_strength_threshold', 25))
        self.grid_aggression_threshold = int(cfg.get('grid_aggression_threshold', 20))
        self.grid_aggression_multiplier = Decimal(str(cfg.get('grid_aggression_multiplier', '1.0')))
        self.strategy_state = 'GRID'
        self.trend_position: Dict = {}
        self.cooldown_counter = 0
        self.ema_fast_span, self.ema_slow_span = int(cfg['ema_span_fast_bars']), int(cfg['ema_span_slow_bars'])
        self.macd_fast, self.macd_slow, self.macd_signal = int(cfg['macd_fast_period']), int(cfg['macd_slow_period']), int(cfg['macd_signal_period'])
        self.dmi_period = int(cfg['dmi_period'])
        # RSI參數（用於複合條件判斷）
        self.rsi_period = int(cfg.get('rsi_period', 14))
        self.rsi_bull_threshold = float(cfg.get('rsi_bull_threshold', 50.0))  # 多頭RSI門檻
        self.rsi_bear_threshold = float(cfg.get('rsi_bear_threshold', 50.0))  # 空頭RSI門檻
        # 複合條件參數（放寬進場條件）
        self.use_multi_indicator = cfg.get('use_multi_indicator', True)  # 是否使用多指標複合判斷
        self.adx_min_threshold = int(cfg.get('adx_min_threshold', 8))  # 最低ADX門檻（激進條件）
        # 布林帶參數
        self.bollinger_window = int(cfg.get('bollinger_window', 20))  # 布林帶週期
        self.bollinger_k = float(cfg.get('bollinger_k', 2.0))  # 布林帶標準差倍數
        self.bollinger_band_threshold = float(cfg.get('bollinger_band_threshold', 0.1))  # 接近布林帶邊界的閾值（0.1 = 10%）
        # 隨機震盪指標參數
        self.stochastic_k_period = int(cfg.get('stochastic_k_period', 14))  # 隨機指標K週期
        self.stochastic_d_period = int(cfg.get('stochastic_d_period', 3))  # 隨機指標D週期
        self.stochastic_oversold = float(cfg.get('stochastic_oversold', 30.0))  # 超賣門檻
        self.stochastic_overbought = float(cfg.get('stochastic_overbought', 70.0))  # 超買門檻
        # ATR動態網格參數
        self.use_atr_spacing = cfg.get('use_atr_spacing', False)
        self.atr_period = int(cfg.get('atr_period', 14))
        self.atr_spacing_multiplier = Decimal(str(cfg.get('atr_spacing_multiplier', '0.5')))
        # ADX過濾器參數（強趨勢時關閉網格）
        self.use_adx_filter = cfg.get('use_adx_filter', False)
        self.adx_filter_threshold = int(cfg.get('adx_threshold', 30))
        global USDT_BALANCE, TWD_BALANCE, TOTAL_EQUITY_TWD
        USDT_BALANCE = init_usdt; TWD_BALANCE = init_twd; TOTAL_EQUITY_TWD = TWD_BALANCE
        if self.verbose:
            LOG.info("Backtester V9 Initialized: Simplified Entry & Active Grid.")
            if self.use_hybrid: LOG.info(f"Hybrid mode ENABLED. Trend ADX Filter: >{self.adx_strength_threshold}, Aggressive Grid: <{self.grid_aggression_threshold}")

    def _setup_grid_layers(self, cfg: dict) -> List[GridLayer]:
        # (此函數無變更)
        small_gap = Decimal(cfg["small_gap"]); levels_each = int(cfg["levels_each"])
        return [ GridLayer(0, small_gap, Decimal(cfg["size_pct_small"]), levels_each), GridLayer(1, small_gap * int(cfg["mid_mult"]), Decimal(cfg["size_pct_mid"]), levels_each), GridLayer(2, small_gap * int(cfg["big_mult"]), Decimal(cfg["size_pct_big"]), levels_each) ]

    def _update_equity(self, price: Decimal):
        # (此函數無變更)
        global TOTAL_EQUITY_TWD; TOTAL_EQUITY_TWD = TWD_BALANCE + USDT_BALANCE * price

    def _place_grid_order(self, side: str, price: Decimal, qty: Decimal):
        # (此函數無變更)
        global ACTIVE_ORDERS
        if (price * qty) < self.min_order_value_twd: return
        ACTIVE_ORDERS[f"{side}_{price}"] = {"price": price, "qty": qty, "side": side}
        
    # --- V9 修改 ---
    # 新增 `trend_override` 參數以支援順勢網格
    # 新增 ATR 動態網格和 ADX 過濾器支持
    def _rebuild_grid(self, center_price: Decimal, trend: str, current_adx: Decimal, trend_override: str = 'none', current_atr: Decimal = None):
        global ACTIVE_ORDERS; ACTIVE_ORDERS.clear()
        
        # 判斷是否處於趨勢跟隨模式
        is_trend_following = trend_override in ['long', 'short']
        
        # 方向1優化：完全禁用ADX過濾器，讓策略主要依靠趨勢跟隨
        # 因為市場77.8%時間都是強趨勢，應該主要依靠趨勢跟隨而非網格
        grid_reduction_factor = Decimal('1.0')
        # 註：ADX過濾器已禁用（use_adx_filter: false），網格作為輔助不再被限制
        
        grid_mode, size_multiplier = 'NORMAL', Decimal('1.0')
        if not is_trend_following and current_adx < self.grid_aggression_threshold:
            grid_mode, size_multiplier = 'AGGRESSIVE', self.grid_aggression_multiplier
        
        # 只在verbose模式下輸出網格重建日誌
        if self.verbose:
            log_trend = trend.upper()
            if is_trend_following:
                log_trend = f"TREND FOLLOWING ({trend_override.upper()})"
            LOG.info(f"GRID MODE ({grid_mode}): Rebuilding grid for trend: '{log_trend}' @ {center_price:.3f}")

        for layer in self.grid_layers:
            # ATR動態網格間距
            if self.use_atr_spacing and current_atr is not None and current_atr > 0:
                # 使用ATR計算動態間距
                base_gap = current_atr * self.atr_spacing_multiplier
                # 確保間距不小於最小精度
                base_gap = max(base_gap, Decimal(self.cfg['price_precision']))
                # 根據層級應用倍數
                if layer.idx == 0:
                    effective_gap = base_gap
                elif layer.idx == 1:
                    effective_gap = base_gap * int(self.cfg.get('mid_mult', 2))
                else:
                    effective_gap = base_gap * int(self.cfg.get('big_mult', 5))
            else:
                # 使用固定間距
                effective_gap = layer.gap_abs
            
            # 應用網格縮減因子（ADX過濾器）
            effective_size_pct = layer.size_pct * size_multiplier * grid_reduction_factor
            qty = quantize(effective_size_pct * TOTAL_EQUITY_TWD / center_price, self.cfg['qty_precision'])
            if qty <= 0: continue
            
            # 應用網格縮減：減少層級數量
            base_levels = layer.levels_each_side
            buy_levels = max(1, int(base_levels * grid_reduction_factor))
            sell_levels = max(1, int(base_levels * grid_reduction_factor))

            # 根據總體趨勢調整掛單比例
            if not is_trend_following:
                if trend == 'up': sell_levels = buy_levels // 2
                elif trend == 'down': buy_levels = sell_levels // 2
            
            # 如果在趨勢跟隨中，只掛順勢單
            if trend_override == 'long':
                sell_levels = 0 # 不掛賣單
            elif trend_override == 'short':
                buy_levels = 0 # 不掛買單

            for i in range(1, buy_levels + 1): self._place_grid_order("buy", quantize(center_price - (effective_gap * i), self.cfg['price_precision']), qty)
            for i in range(1, sell_levels + 1): self._place_grid_order("sell", quantize(center_price + (effective_gap * i), self.cfg['price_precision']), qty)

    def _check_grid_fills(self, price: Decimal, bar_index: int, trade_log: list, diagnostic_stats: dict = None):
        # (此函數邏輯無變更，但會在每一次迴圈被呼叫)
        global ACTIVE_ORDERS, USDT_BALANCE, TWD_BALANCE
        filled_keys, new_orders = [], []
        for key, order in ACTIVE_ORDERS.items():
            order_price, order_qty = order['price'], order['qty']
            if order['side'] == 'buy' and price <= order_price:
                cost = order_price * order_qty
                if TWD_BALANCE >= cost:
                    fee_cost = cost * self.fee
                    TWD_BALANCE -= cost * (1 + self.fee); USDT_BALANCE += order_qty; filled_keys.append(key)
                    trade_log.append({'index': bar_index, 'price': order_price, 'type': 'grid_buy'}) # 記錄交易
                    # 診斷數據收集
                    if diagnostic_stats is not None:
                        diagnostic_stats['total_fee_cost'] += float(fee_cost)
                        diagnostic_stats['grid_fills'] += 1
                    # --- V9 修改 ---
                    # 如果在趨勢跟隨中，成交後不再掛反向單
                    if not self.trend_position:
                        new_orders.append(("sell", quantize(order_price + self.grid_layers[0].gap_abs, self.cfg['price_precision']), order_qty))
            elif order['side'] == 'sell' and price >= order_price:
                if USDT_BALANCE >= order_qty:
                    fee_cost = order_price * order_qty * self.fee
                    USDT_BALANCE -= order_qty; TWD_BALANCE += order_price * order_qty * (1 - self.fee); filled_keys.append(key)
                    trade_log.append({'index': bar_index, 'price': order_price, 'type': 'grid_sell'}) # 記錄交易
                    # 診斷數據收集
                    if diagnostic_stats is not None:
                        diagnostic_stats['total_fee_cost'] += float(fee_cost)
                        diagnostic_stats['grid_fills'] += 1
                    # --- V9 修改 ---
                    # 如果在趨勢跟隨中，成交後不再掛反向單
                    if not self.trend_position:
                        new_orders.append(("buy", quantize(order_price - self.grid_layers[0].gap_abs, self.cfg['price_precision']), order_qty))

        for key in filled_keys: ACTIVE_ORDERS.pop(key, None)
        for side, p, q in new_orders: self._place_grid_order(side, p, q)

    # --- V9 重大修改 ---
    # 重構 run 函數以整合順勢網格與簡化進場邏輯
    def run(self, ohlc_df: pd.DataFrame, collect_diagnostics: bool = False) -> Dict:
        global TWD_BALANCE, USDT_BALANCE, ACTIVE_ORDERS
        trade_log = []
        equity_history = []
        
        # 診斷數據收集初始化
        diagnostic_stats = {
            'total_fee_cost': 0.0,
            'grid_fills': 0,
            'grid_orders_placed': 0,
            'trend_entries': 0,
            'trend_exits': 0,
            'grid_rebuilds': 0,
            'price_min': float('inf'),
            'price_max': float('-inf'),
            'avg_gap_size': 0.0
        } if collect_diagnostics else None  # Track equity over time for drawdown calculation
        
        if self.verbose:
            if self.verbose:
                LOG.info("Calculating all required indicators for V9 Model...")
        price_series = ohlc_df['close'].ffill()
        
        # --- [修改開始] 將指標存入 ohlc_df 以便繪圖 ---
        ema_f = ema(price_series, span=self.ema_fast_span)
        ema_s = ema(price_series, span=self.ema_slow_span)
        adx_series, _, _ = adx(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], period=self.dmi_period)
        
        # 計算RSI和MACD（用於複合條件判斷）
        rsi_series = rsi(price_series, period=self.rsi_period)
        macd_line, macd_signal, macd_hist = macd(price_series, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        
        # 計算布林帶和隨機震盪指標（第六次優化新增）
        bollinger_upper, bollinger_middle, bollinger_lower = bollinger(price_series, window=self.bollinger_window, k=self.bollinger_k)
        stochastic_k, stochastic_d = stochastic_oscillator(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], 
                                                           k_period=self.stochastic_k_period, d_period=self.stochastic_d_period)
        
        # 計算ATR（如果啟用ATR動態網格）
        atr_series = None
        if self.use_atr_spacing:
            atr_series = atr(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], period=self.atr_period)
            ohlc_df['atr'] = atr_series
        
        # 存入 DataFrame
        ohlc_df['ema_fast'] = ema_f
        ohlc_df['ema_slow'] = ema_s
        ohlc_df['adx'] = adx_series
        ohlc_df['rsi'] = rsi_series
        ohlc_df['macd'] = macd_line
        ohlc_df['macd_signal'] = macd_signal
        ohlc_df['macd_hist'] = macd_hist
        ohlc_df['bollinger_upper'] = bollinger_upper
        ohlc_df['bollinger_middle'] = bollinger_middle
        ohlc_df['bollinger_lower'] = bollinger_lower
        ohlc_df['stochastic_k'] = stochastic_k
        ohlc_df['stochastic_d'] = stochastic_d
        # --- [修改結束] ---

        initial_price = Decimal(str(price_series.iloc[0]))
        initial_equity = TWD_BALANCE + USDT_BALANCE * initial_price
        
        self._update_equity(initial_price)
        initial_atr = Decimal(str(atr_series.iloc[0])) if atr_series is not None else None
        self._rebuild_grid(initial_price, trend='neutral', current_adx=adx_series.iloc[0], current_atr=initial_atr)
        if diagnostic_stats is not None:
            diagnostic_stats['grid_orders_placed'] += len(ACTIVE_ORDERS)
        recenter_interval = int(self.cfg['recenter_interval_minutes'])
        
        for i, price_val in enumerate(price_series):
            price = Decimal(str(price_val))
            self._update_equity(price)
            equity_history.append(float(TOTAL_EQUITY_TWD))  # Track equity for drawdown
            if self.cooldown_counter > 0: self.cooldown_counter -= 1
            
            current_adx = adx_series.iloc[i]
            current_atr = Decimal(str(atr_series.iloc[i])) if atr_series is not None else None
            
            # 診斷數據：價格範圍
            if diagnostic_stats is not None:
                price_float = float(price)
                diagnostic_stats['price_min'] = min(diagnostic_stats['price_min'], price_float)
                diagnostic_stats['price_max'] = max(diagnostic_stats['price_max'], price_float)
            
            # --- V9 修改: 網格檢查永遠執行 ---
            self._check_grid_fills(price, i, trade_log, diagnostic_stats)

            # 判斷網格是否需要重建
            if i > 0 and i % recenter_interval == 0:
                trend = 'neutral'
                if ema_f.iloc[i] > ema_s.iloc[i]: trend = 'up'
                elif ema_f.iloc[i] < ema_s.iloc[i]: trend = 'down'
                
                trend_override_state = 'none'
                if self.strategy_state == 'TREND_FOLLOWING' and self.trend_position:
                    trend_override_state = self.trend_position['side']

                self._rebuild_grid(price, trend, current_adx, trend_override=trend_override_state, current_atr=current_atr)
                if diagnostic_stats is not None:
                    diagnostic_stats['grid_rebuilds'] += 1
                    # 記錄重建時的掛單數
                    diagnostic_stats['grid_orders_placed'] += len(ACTIVE_ORDERS)

            # --- 狀態機邏輯 ---
            # 多指標複合判斷：使用OR邏輯放寬進場條件
            if self.strategy_state == 'GRID':
                if self.use_hybrid and self.cooldown_counter == 0:
                    is_ema_bull = ema_f.iloc[i] > ema_s.iloc[i]
                    is_ema_bear = ema_f.iloc[i] < ema_s.iloc[i]
                    current_rsi = rsi_series.iloc[i]
                    current_macd = macd_line.iloc[i]
                    current_macd_signal = macd_signal.iloc[i]
                    current_price = float(price)
                    
                    # 多指標複合判斷（OR邏輯，放寬條件）- 第六次優化：加入布林帶和隨機指標
                    if self.use_multi_indicator:
                        # 獲取當前布林帶和隨機指標值
                        current_bollinger_upper = bollinger_upper.iloc[i]
                        current_bollinger_middle = bollinger_middle.iloc[i]
                        current_bollinger_lower = bollinger_lower.iloc[i]
                        current_stochastic_k = stochastic_k.iloc[i]
                        current_stochastic_d = stochastic_d.iloc[i]
                        
                        # 計算布林帶位置（價格相對位置）
                        bollinger_range = current_bollinger_upper - current_bollinger_lower
                        price_position = (current_price - current_bollinger_lower) / bollinger_range if bollinger_range > 0 else 0.5
                        is_near_bollinger_lower = price_position < self.bollinger_band_threshold  # 接近下軌（超賣）
                        is_near_bollinger_upper = price_position > (1 - self.bollinger_band_threshold)  # 接近上軌（超買）
                        # 多頭進場條件（OR邏輯，第六次優化：擴展至4個條件）
                        # 條件A（主要）: EMA快線 > 慢線 AND ADX > 門檻 AND RSI > 50 AND MACD > 0
                        condition_a_bull = (is_ema_bull and 
                                           current_adx > self.adx_strength_threshold and 
                                           current_rsi > self.rsi_bull_threshold and
                                           current_macd > current_macd_signal)
                        # 條件B（輔助）: EMA快線 > 慢線 AND MACD > 0 AND RSI > 45 AND 價格接近布林帶下軌
                        condition_b_bull = (is_ema_bull and 
                                           current_macd > current_macd_signal and 
                                           current_rsi > (self.rsi_bull_threshold - 5) and
                                           is_near_bollinger_lower)
                        # 條件C（激進）: EMA快線 > 慢線 AND ADX > 8 AND 隨機指標%K < 30（超賣反彈）
                        condition_c_bull = (is_ema_bull and 
                                           current_adx > 8 and
                                           current_stochastic_k < self.stochastic_oversold)
                        # 條件D（極度放寬）: EMA快線 > 慢線 AND ADX > 最低門檻（降低ADX門檻至6）
                        condition_d_bull = (is_ema_bull and 
                                           current_adx > self.adx_min_threshold)
                        
                        # 空頭進場條件（OR邏輯，第六次優化：擴展至4個條件）
                        # 條件A（主要）: EMA快線 < 慢線 AND ADX > 門檻 AND RSI < 50 AND MACD < 0
                        condition_a_bear = (is_ema_bear and 
                                           current_adx > self.adx_strength_threshold and 
                                           current_rsi < self.rsi_bear_threshold and
                                           current_macd < current_macd_signal)
                        # 條件B（輔助）: EMA快線 < 慢線 AND MACD < 0 AND RSI < 55 AND 價格接近布林帶上軌
                        condition_b_bear = (is_ema_bear and 
                                           current_macd < current_macd_signal and 
                                           current_rsi < (self.rsi_bear_threshold + 5) and
                                           is_near_bollinger_upper)
                        # 條件C（激進）: EMA快線 < 慢線 AND ADX > 8 AND 隨機指標%K > 70（超買回調）
                        condition_c_bear = (is_ema_bear and 
                                           current_adx > 8 and
                                           current_stochastic_k > self.stochastic_overbought)
                        # 條件D（極度放寬）: EMA快線 < 慢線 AND ADX > 最低門檻（降低ADX門檻至6）
                        condition_d_bear = (is_ema_bear and 
                                           current_adx > self.adx_min_threshold)
                        
                        # OR邏輯：任一條件滿足即可進場（第六次優化：擴展至4個條件）
                        is_strong_uptrend = condition_a_bull or condition_b_bull or condition_c_bull or condition_d_bull
                        is_strong_downtrend = condition_a_bear or condition_b_bear or condition_c_bear or condition_d_bear
                    else:
                        # 單一指標判斷（原邏輯）
                        adjusted_adx_threshold = max(10, self.adx_strength_threshold - 3)
                        is_adx_strong = current_adx > adjusted_adx_threshold
                        is_strong_uptrend = is_ema_bull and is_adx_strong
                        is_strong_downtrend = is_ema_bear and is_adx_strong
                    
                    trend_side = None
                    if is_strong_uptrend: trend_side = 'buy'
                    elif is_strong_downtrend: trend_side = 'sell'

                    if trend_side:
                        self.strategy_state = 'TREND_FOLLOWING'
                        if diagnostic_stats is not None:
                            diagnostic_stats['trend_entries'] += 1
                        if self.verbose:
                            if self.verbose:
                                LOG.warning(f"--- Bar {i} | Price {price:.3f} | Trend Entry Signal ---")
                                LOG.warning(f"    - EMA={'BULL' if is_ema_bull else 'BEAR'}, ADX={current_adx:.2f} (> {self.adx_strength_threshold})")

                        # 清空網格，準備趨勢單
                        ACTIVE_ORDERS.clear()
                        trade_value_twd = TOTAL_EQUITY_TWD * self.trend_equity_pct
                        
                        if trend_side == 'buy':
                            qty_to_buy = quantize(trade_value_twd / price, self.cfg['qty_precision'])
                            if TWD_BALANCE >= trade_value_twd:
                                TWD_BALANCE -= qty_to_buy * price * (1 + self.fee); USDT_BALANCE += qty_to_buy
                                self.trend_position = {'side': 'long', 'entry_price': price, 'qty': qty_to_buy, 'peak_price': price}
                                trade_log.append({'index': i, 'price': price, 'type': 'trend_long_entry'})
                                if self.verbose:
                                    if self.verbose:
                                        LOG.info(f"    -> ACTION: Entered LONG position: {qty_to_buy:.4f} USDT @ {price:.3f}")
                                # 立即建立順勢網格
                                self._rebuild_grid(price, 'up', current_adx, trend_override='long', current_atr=current_atr)
                        else:
                            qty_to_sell = quantize(trade_value_twd / price, self.cfg['qty_precision'])
                            if USDT_BALANCE >= qty_to_sell:
                                USDT_BALANCE -= qty_to_sell; TWD_BALANCE += qty_to_sell * price * (1 - self.fee)
                                self.trend_position = {'side': 'short', 'entry_price': price, 'qty': qty_to_sell, 'valley_price': price}
                                trade_log.append({'index': i, 'price': price, 'type': 'trend_short_entry'})
                                if self.verbose:
                                    if self.verbose:
                                        LOG.info(f"    -> ACTION: Entered SHORT position: {qty_to_sell:.4f} USDT @ {price:.3f}")
                                # 立即建立順勢網格
                                self._rebuild_grid(price, 'down', current_adx, trend_override='short', current_atr=current_atr)

            elif self.strategy_state == 'TREND_FOLLOWING':
                if not self.trend_position:
                    self.strategy_state = 'GRID'
                    continue

                side, entry_price, qty = self.trend_position['side'], self.trend_position['entry_price'], self.trend_position['qty']
                
                should_exit = False
                exit_reason = ""
                
                # 多指標出場條件（OR邏輯）
                # 條件1：Trailing Stop（原有邏輯）
                if side == 'long':
                    peak_price = max(self.trend_position['peak_price'], price)
                    self.trend_position['peak_price'] = peak_price
                    stop_loss_price = peak_price * (1 - self.trailing_stop_pct)
                    if price <= stop_loss_price:
                        should_exit = True
                        exit_reason = f"Trailing Stop Hit. Price ({price:.3f}) <= Stop Price ({stop_loss_price:.3f}). Peak price was {peak_price:.3f}."
                
                elif side == 'short':
                    valley_price = min(self.trend_position['valley_price'], price)
                    self.trend_position['valley_price'] = valley_price
                    stop_loss_price = valley_price * (1 + self.trailing_stop_pct)
                    if price >= stop_loss_price:
                        should_exit = True
                        exit_reason = f"Trailing Stop Hit. Price ({price:.3f}) >= Stop Price ({stop_loss_price:.3f}). Valley price was {valley_price:.3f}."
                
                # 條件2：趨勢反轉（EMA交叉反向）
                if self.use_multi_indicator:
                    if side == 'long' and ema_f.iloc[i] < ema_s.iloc[i]:
                        should_exit = True
                        exit_reason = f"Trend Reversal: EMA Fast ({ema_f.iloc[i]:.3f}) < EMA Slow ({ema_s.iloc[i]:.3f})"
                    elif side == 'short' and ema_f.iloc[i] > ema_s.iloc[i]:
                        should_exit = True
                        exit_reason = f"Trend Reversal: EMA Fast ({ema_f.iloc[i]:.3f}) > EMA Slow ({ema_s.iloc[i]:.3f})"
                    
                    # 條件3：ADX弱化（趨勢結束）
                    if current_adx < self.adx_min_threshold:
                        should_exit = True
                        exit_reason = f"ADX Weakening: ADX ({current_adx:.2f}) < Min Threshold ({self.adx_min_threshold})"
                
                if should_exit:
                    if self.verbose:
                        if self.verbose:
                            LOG.warning(f"--- Bar {i} | Price {price:.3f} | Trend Exit Signal ---")
                            LOG.warning(f"    - REASON: {exit_reason}")
                    pnl = Decimal('0.0')
                    if side == 'long':
                        USDT_BALANCE -= qty; TWD_BALANCE += qty * price * (1 - self.fee)
                        pnl = (price - entry_price) * qty
                    else: # short
                        TWD_BALANCE -= qty * price * (1 + self.fee); USDT_BALANCE += qty
                        pnl = (entry_price - price) * qty
                    
                    if self.verbose:
                        if self.verbose:
                            LOG.info(f"    -> ACTION: Exited {side.upper()} position. PNL: {pnl:,.2f} TWD.")
                            LOG.info(f"    -> Switching to GRID mode (Cooldown: {self.cooldown_bars} bars).")
                    
                    trade_log.append({'index': i, 'price': price, 'type': 'trend_exit'})
                    if diagnostic_stats is not None:
                        diagnostic_stats['trend_exits'] += 1
                    self.trend_position = {}
                    self.strategy_state = 'GRID'
                    self.cooldown_counter = self.cooldown_bars
                    # 趨勢結束，重建一個中性的標準網格
                    self._rebuild_grid(price, 'neutral', current_adx, current_atr=current_atr)
                    if diagnostic_stats is not None:
                        diagnostic_stats['grid_orders_placed'] += len(ACTIVE_ORDERS)
                    
        # --- [V8 修改結束] 後續程式碼與 V8 相同 ---
        final_price = Decimal(str(price_series.iloc[-1]))
        if self.trend_position:
            if self.trend_position['side'] == 'long': USDT_BALANCE -= self.trend_position['qty']; TWD_BALANCE += self.trend_position['qty'] * final_price
            else: TWD_BALANCE -= self.trend_position['qty'] * final_price; USDT_BALANCE += self.trend_position['qty']
        
        final_equity = TWD_BALANCE + USDT_BALANCE * final_price
        pnl = final_equity - initial_equity
        roi_pct = (pnl / initial_equity) * 100 if initial_equity > 0 else 0
        
        # Calculate max drawdown
        if len(equity_history) > 0:
            equity_series = pd.Series(equity_history)
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown_pct = float(abs(drawdown.min())) * 100 if len(drawdown) > 0 else 0.0
        else:
            max_drawdown_pct = 0.0
        
        # Count total trades
        total_trades = len(trade_log)
        
        if self.verbose:
            # 只在verbose模式下輸出詳細結果
            if self.verbose:
                LOG.info("--- Backtest Finished ---")
                LOG.info(f"Initial Equity: {initial_equity:,.2f} TWD")
                LOG.info(f"Final Equity:   {final_equity:,.2f} TWD")
                LOG.info(f"Total PNL:      {pnl:,.2f} TWD")
                LOG.info(f"Total ROI:      {roi_pct:.2f}%")
                LOG.info(f"Max Drawdown:   {max_drawdown_pct:.2f}%")
                LOG.info(f"Total Trades:   {total_trades}")
                LOG.info(f"Final Balance:  {USDT_BALANCE:.2f} USDT, {TWD_BALANCE:,.2f} TWD")
        
        # 計算診斷指標
        if diagnostic_stats is not None:
            # 計算平均網格間距
            if len(self.grid_layers) > 0:
                avg_gap = float(sum([layer.gap_abs for layer in self.grid_layers]) / len(self.grid_layers))
                diagnostic_stats['avg_gap_size'] = avg_gap
            
            # 計算網格成交率
            if diagnostic_stats['grid_orders_placed'] > 0:
                diagnostic_stats['grid_fill_rate'] = diagnostic_stats['grid_fills'] / diagnostic_stats['grid_orders_placed']
            else:
                diagnostic_stats['grid_fill_rate'] = 0.0
            
            # 計算平均每筆交易利潤
            if total_trades > 0:
                diagnostic_stats['avg_profit_per_trade'] = float(pnl) / total_trades
            else:
                diagnostic_stats['avg_profit_per_trade'] = 0.0
            
            # 計算價格波動範圍
            diagnostic_stats['price_range_pct'] = ((diagnostic_stats['price_max'] - diagnostic_stats['price_min']) / diagnostic_stats['price_min'] * 100) if diagnostic_stats['price_min'] > 0 else 0.0
            
            # 計算扣除手續費後的淨利潤
            diagnostic_stats['net_profit_after_fee'] = float(pnl) - diagnostic_stats['total_fee_cost']
            diagnostic_stats['fee_to_profit_ratio'] = abs(diagnostic_stats['total_fee_cost'] / float(pnl)) if pnl != 0 else float('inf')
        
        # Return stats dictionary for optimization
        stats = {
            'total_pnl': float(pnl),
            'roi_pct': float(roi_pct),
            'max_drawdown_pct': max_drawdown_pct,
            'total_trades': total_trades,
            'final_equity': float(final_equity),
            # Keep original fields for backward compatibility
            'trade_log': trade_log,
            'initial_equity': float(initial_equity),
            'final_usdt_balance': float(USDT_BALANCE),
            'final_twd_balance': float(TWD_BALANCE),
            'final_price': float(final_price),
            'pnl': float(pnl)
        }
        
        # 合併診斷數據
        if diagnostic_stats is not None:
            stats.update(diagnostic_stats)
        
        return stats

# --- main 函數 (無變更) ---
def main():
    parser = argparse.ArgumentParser(description="V9 Hybrid Backtester: Simplified Entry & Active Grid")
    parser.add_argument("--csv", required=True, type=Path, help="Path to OHLC CSV file. Must contain 'ts', 'high', 'low', 'close'.")
    parser.add_argument("--config", default="config_usdttwd.yaml", type=Path, help="Path to the strategy config YAML file.")
    parser.add_argument("--init_usdt", default=10000.0, type=float, help="Initial USDT balance.")
    parser.add_argument("--init_twd", default=300000.0, type=float, help="Initial TWD balance.")
    args = parser.parse_args()

    if not args.csv.exists() or not args.config.exists():
        LOG.error(f"File not found.")
        return
        
    cfg = yaml.safe_load(args.config.read_text())
    
    try:
        temp_df = pd.read_csv(args.csv, usecols=['ts', 'high', 'low', 'close'])
        if pd.api.types.is_numeric_dtype(temp_df['ts']):
            try:
                tss = pd.to_datetime(temp_df['ts'], unit='ms')
                if tss.min().year < 2000:
                    raise ValueError("ts likely in seconds, not milliseconds.")
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                LOG.warning("Could not parse ts as milliseconds, trying seconds...")
                tss = pd.to_datetime(temp_df['ts'], unit='s')
            temp_df['ts'] = tss
        else:
            temp_df['ts'] = pd.to_datetime(temp_df['ts'])
        price_df = temp_df.set_index('ts')
        price_df['high'] = price_df['high'].astype(float)
        price_df['low'] = price_df['low'].astype(float)
        price_df['close'] = price_df['close'].astype(float)
        price_df.ffill(inplace=True)
    except Exception as e:
        LOG.error(f"A critical error occurred while reading or processing the CSV file: {e}", exc_info=True)
        return
    
    backtester = Backtester(cfg, Decimal(str(args.init_usdt)), Decimal(str(args.init_twd)), verbose=True)
    result = backtester.run(price_df)
    trade_log = result['trade_log']
    
    if trade_log:
        try:
            # 將索引從數字轉換回真實的時間戳
            processed_trade_log = []
            for trade in trade_log:
                if isinstance(trade['index'], int) and trade['index'] < len(price_df.index):
                     trade['index'] = price_df.index[trade['index']]
                     processed_trade_log.append(trade)
                # 可選擇性地處理索引超出範圍的情況，此處選擇忽略
            trade_log = processed_trade_log
        except IndexError:
             LOG.error("An index error occurred during trade log ts conversion. Skipping plot generation.")
             trade_log = []

    plot_backtest_results(price_df, trade_log)

if __name__ == "__main__":
    main()