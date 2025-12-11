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

from indicators import ema, macd, adx

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
    將回測結果視覺化（穩健版，手動處理X軸標籤以避免 'converter' 錯誤）。
    """
    if not trade_log and len(price_df) == 0:
        LOG.warning("Price data and trade log are empty. Skipping plot generation.")
        return

    LOG.info(f"Generating backtest result plot, saving to {output_filename}...")
    
    trade_df = pd.DataFrame(trade_log)
    if not trade_df.empty:
        trade_df['price'] = pd.to_numeric(trade_df['price'])
    
    # 建立一個從 0 開始的整數索引，用於繪圖
    plot_indices = range(len(price_df))
    plot_prices = price_df['close'].astype(float)

    fig, ax = plt.subplots(figsize=(20, 10))

    # 1. 繪製價格曲線 (使用整數索引作為 X 軸)
    ax.plot(plot_indices, plot_prices, label='Close Price', color='skyblue', linewidth=1)

    # 2. 標記交易點 (同樣使用整數索引作為 X 軸)
    if not trade_df.empty:
        marker_size_grid, marker_size_trend = 25, 80
        
        # 將 trade_log 中的 datetime 索引轉換為對應的整數位置索引
        trade_df['plot_index'] = trade_df['index'].apply(lambda x: price_df.index.get_loc(x) if x in price_df.index else -1)
        trade_df = trade_df[trade_df['plot_index'] != -1] # 移除無效的交易點

        grid_buys = trade_df[trade_df['type'] == 'grid_buy']
        grid_sells = trade_df[trade_df['type'] == 'grid_sell']
        ax.scatter(grid_buys['plot_index'], grid_buys['price'], label='Grid Buy', marker='^', color='lime', s=marker_size_grid, zorder=5)
        ax.scatter(grid_sells['plot_index'], grid_sells['price'], label='Grid Sell', marker='v', color='red', s=marker_size_grid, zorder=5)
        
        trend_long_entry = trade_df[trade_df['type'] == 'trend_long_entry']
        trend_short_entry = trade_df[trade_df['type'] == 'trend_short_entry']
        trend_exit = trade_df[trade_df['type'] == 'trend_exit']
        ax.scatter(trend_long_entry['plot_index'], trend_long_entry['price'], label='Trend Long Entry', marker='o', color='green', s=marker_size_trend, zorder=10, edgecolors='white')
        ax.scatter(trend_short_entry['plot_index'], trend_short_entry['price'], label='Trend Short Entry', marker='o', color='maroon', s=marker_size_trend, zorder=10, edgecolors='white')
        ax.scatter(trend_exit['plot_index'], trend_exit['price'], label='Trend Exit', marker='X', color='blue', s=marker_size_trend, zorder=10)

    # 3. 手動格式化 X 軸標籤
    # 選取大約 10 個點來顯示日期標籤
    num_ticks = 10
    tick_positions = [int(p) for p in np.linspace(0, len(price_df) - 1, num_ticks)]
    tick_labels = [price_df.index[pos].strftime('%Y-%m-%d') for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha='right')

    # 4. 格式化圖表
    ax.set_title('Backtest Trading Activity (V9)', fontsize=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (TWD)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    LOG.info("Plot generation complete.")


# --- 核心回測邏輯 ---
class Backtester:
    def __init__(self, cfg: dict, init_usdt: Decimal, init_twd: Decimal):
        # (此函數無變更)
        self.cfg = cfg
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
        global USDT_BALANCE, TWD_BALANCE, TOTAL_EQUITY_TWD
        USDT_BALANCE = init_usdt; TWD_BALANCE = init_twd; TOTAL_EQUITY_TWD = TWD_BALANCE
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
    def _rebuild_grid(self, center_price: Decimal, trend: str, current_adx: Decimal, trend_override: str = 'none'):
        global ACTIVE_ORDERS; ACTIVE_ORDERS.clear()
        
        # 判斷是否處於趨勢跟隨模式
        is_trend_following = trend_override in ['long', 'short']
        
        grid_mode, size_multiplier = 'NORMAL', Decimal('1.0')
        if not is_trend_following and current_adx < self.grid_aggression_threshold:
            grid_mode, size_multiplier = 'AGGRESSIVE', self.grid_aggression_multiplier
        
        log_trend = trend.upper()
        if is_trend_following:
            log_trend = f"TREND FOLLOWING ({trend_override.upper()})"
        
        LOG.info(f"GRID MODE ({grid_mode}): Rebuilding grid for trend: '{log_trend}' @ {center_price:.3f}")

        for layer in self.grid_layers:
            effective_size_pct = layer.size_pct * size_multiplier
            qty = quantize(effective_size_pct * TOTAL_EQUITY_TWD / center_price, self.cfg['qty_precision'])
            if qty <= 0: continue
            
            buy_levels, sell_levels = layer.levels_each_side, layer.levels_each_side

            # 根據總體趨勢調整掛單比例
            if not is_trend_following:
                if trend == 'up': sell_levels = buy_levels // 2
                elif trend == 'down': buy_levels = sell_levels // 2
            
            # 如果在趨勢跟隨中，只掛順勢單
            if trend_override == 'long':
                sell_levels = 0 # 不掛賣單
            elif trend_override == 'short':
                buy_levels = 0 # 不掛買單

            for i in range(1, buy_levels + 1): self._place_grid_order("buy", quantize(center_price - (layer.gap_abs * i), self.cfg['price_precision']), qty)
            for i in range(1, sell_levels + 1): self._place_grid_order("sell", quantize(center_price + (layer.gap_abs * i), self.cfg['price_precision']), qty)

    def _check_grid_fills(self, price: Decimal, bar_index: int, trade_log: list):
        # (此函數邏輯無變更，但會在每一次迴圈被呼叫)
        global ACTIVE_ORDERS, USDT_BALANCE, TWD_BALANCE
        filled_keys, new_orders = [], []
        for key, order in ACTIVE_ORDERS.items():
            order_price, order_qty = order['price'], order['qty']
            if order['side'] == 'buy' and price <= order_price:
                cost = order_price * order_qty
                if TWD_BALANCE >= cost:
                    TWD_BALANCE -= cost * (1 + self.fee); USDT_BALANCE += order_qty; filled_keys.append(key)
                    trade_log.append({'index': bar_index, 'price': order_price, 'type': 'grid_buy'}) # 記錄交易
                    # --- V9 修改 ---
                    # 如果在趨勢跟隨中，成交後不再掛反向單
                    if not self.trend_position:
                        new_orders.append(("sell", quantize(order_price + self.grid_layers[0].gap_abs, self.cfg['price_precision']), order_qty))
            elif order['side'] == 'sell' and price >= order_price:
                if USDT_BALANCE >= order_qty:
                    USDT_BALANCE -= order_qty; TWD_BALANCE += order_price * order_qty * (1 - self.fee); filled_keys.append(key)
                    trade_log.append({'index': bar_index, 'price': order_price, 'type': 'grid_sell'}) # 記錄交易
                    # --- V9 修改 ---
                    # 如果在趨勢跟隨中，成交後不再掛反向單
                    if not self.trend_position:
                        new_orders.append(("buy", quantize(order_price - self.grid_layers[0].gap_abs, self.cfg['price_precision']), order_qty))

        for key in filled_keys: ACTIVE_ORDERS.pop(key, None)
        for side, p, q in new_orders: self._place_grid_order(side, p, q)

    # --- V9 重大修改 ---
    # 重構 run 函數以整合順勢網格與簡化進場邏輯
    def run(self, ohlc_df: pd.DataFrame) -> List[Dict]:
        global TWD_BALANCE, USDT_BALANCE
        trade_log = [] 
        LOG.info("Calculating all required indicators for V9 Model...")
        price_series = ohlc_df['close'].ffill()
        
        ema_f, ema_s = ema(price_series, span=self.ema_fast_span), ema(price_series, span=self.ema_slow_span)
        adx_series, _, _ = adx(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], period=self.dmi_period)
        
        initial_price = Decimal(str(price_series.iloc[0]))
        initial_equity = TWD_BALANCE + USDT_BALANCE * initial_price
        
        self._update_equity(initial_price)
        self._rebuild_grid(initial_price, trend='neutral', current_adx=adx_series.iloc[0])
        recenter_interval = int(self.cfg['recenter_interval_minutes'])
        
        for i, price_val in enumerate(price_series):
            price = Decimal(str(price_val))
            self._update_equity(price)
            if self.cooldown_counter > 0: self.cooldown_counter -= 1
            
            current_adx = adx_series.iloc[i]
            
            # --- V9 修改: 網格檢查永遠執行 ---
            self._check_grid_fills(price, i, trade_log)

            # 判斷網格是否需要重建
            if i > 0 and i % recenter_interval == 0:
                trend = 'neutral'
                if ema_f.iloc[i] > ema_s.iloc[i]: trend = 'up'
                elif ema_f.iloc[i] < ema_s.iloc[i]: trend = 'down'
                
                trend_override_state = 'none'
                if self.strategy_state == 'TREND_FOLLOWING' and self.trend_position:
                    trend_override_state = self.trend_position['side']

                self._rebuild_grid(price, trend, current_adx, trend_override=trend_override_state)

            # --- 狀態機邏輯 ---
            if self.strategy_state == 'GRID':
                if self.use_hybrid and self.cooldown_counter == 0:
                    is_ema_bull = ema_f.iloc[i] > ema_s.iloc[i]
                    is_ema_bear = ema_f.iloc[i] < ema_s.iloc[i]
                    is_adx_strong = current_adx > self.adx_strength_threshold

                    # --- V9 修改: 簡化進場條件 ---
                    is_strong_uptrend = is_ema_bull and is_adx_strong
                    is_strong_downtrend = is_ema_bear and is_adx_strong
                    
                    trend_side = None
                    if is_strong_uptrend: trend_side = 'buy'
                    elif is_strong_downtrend: trend_side = 'sell'

                    if trend_side:
                        self.strategy_state = 'TREND_FOLLOWING'
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
                                LOG.info(f"    -> ACTION: Entered LONG position: {qty_to_buy:.4f} USDT @ {price:.3f}")
                                # 立即建立順勢網格
                                self._rebuild_grid(price, 'up', current_adx, trend_override='long')
                        else:
                            qty_to_sell = quantize(trade_value_twd / price, self.cfg['qty_precision'])
                            if USDT_BALANCE >= qty_to_sell:
                                USDT_BALANCE -= qty_to_sell; TWD_BALANCE += qty_to_sell * price * (1 - self.fee)
                                self.trend_position = {'side': 'short', 'entry_price': price, 'qty': qty_to_sell, 'valley_price': price}
                                trade_log.append({'index': i, 'price': price, 'type': 'trend_short_entry'})
                                LOG.info(f"    -> ACTION: Entered SHORT position: {qty_to_sell:.4f} USDT @ {price:.3f}")
                                # 立即建立順勢網格
                                self._rebuild_grid(price, 'down', current_adx, trend_override='short')

            elif self.strategy_state == 'TREND_FOLLOWING':
                if not self.trend_position:
                    self.strategy_state = 'GRID'
                    continue

                side, entry_price, qty = self.trend_position['side'], self.trend_position['entry_price'], self.trend_position['qty']
                
                should_exit = False
                exit_reason = ""
                
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
                
                if should_exit:
                    LOG.warning(f"--- Bar {i} | Price {price:.3f} | Trend Exit Signal ---")
                    LOG.warning(f"    - REASON: {exit_reason}")
                    pnl = Decimal('0.0')
                    if side == 'long':
                        USDT_BALANCE -= qty; TWD_BALANCE += qty * price * (1 - self.fee)
                        pnl = (price - entry_price) * qty
                    else: # short
                        TWD_BALANCE -= qty * price * (1 + self.fee); USDT_BALANCE += qty
                        pnl = (entry_price - price) * qty
                    
                    LOG.info(f"    -> ACTION: Exited {side.upper()} position. PNL: {pnl:,.2f} TWD.")
                    LOG.info(f"    -> Switching to GRID mode (Cooldown: {self.cooldown_bars} bars).")
                    
                    trade_log.append({'index': i, 'price': price, 'type': 'trend_exit'})
                    self.trend_position = {}
                    self.strategy_state = 'GRID'
                    self.cooldown_counter = self.cooldown_bars
                    # 趨勢結束，重建一個中性的標準網格
                    self._rebuild_grid(price, 'neutral', current_adx)
                    
        # --- [V8 修改結束] 後續程式碼與 V8 相同 ---
        final_price = Decimal(str(price_series.iloc[-1]))
        if self.trend_position:
            if self.trend_position['side'] == 'long': USDT_BALANCE -= self.trend_position['qty']; TWD_BALANCE += self.trend_position['qty'] * final_price
            else: TWD_BALANCE -= self.trend_position['qty'] * final_price; USDT_BALANCE += self.trend_position['qty']
        
        final_equity = TWD_BALANCE + USDT_BALANCE * final_price
        pnl = final_equity - initial_equity
        roi_pct = (pnl / initial_equity) * 100 if initial_equity > 0 else 0
        LOG.info("--- Backtest Finished ---"); LOG.info(f"Initial Equity: {initial_equity:,.2f} TWD"); LOG.info(f"Final Equity:   {final_equity:,.2f} TWD")
        LOG.info(f"Total PNL:      {pnl:,.2f} TWD"); LOG.info(f"Total ROI:      {roi_pct:.2f}%"); LOG.info(f"Final Balance:  {USDT_BALANCE:.2f} USDT, {TWD_BALANCE:,.2f} TWD")
        return trade_log

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
    
    backtester = Backtester(cfg, Decimal(str(args.init_usdt)), Decimal(str(args.init_twd)))
    trade_log = backtester.run(price_df)
    
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