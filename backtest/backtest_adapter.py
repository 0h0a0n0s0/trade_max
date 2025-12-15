# backtest_adapter.py
"""
Backtest Adapter: 使用與實盤完全相同的 GridStrategy 邏輯進行回測

這個適配器確保回測和實盤使用相同的策略邏輯，避免邏輯分歧導致的過擬合問題。
"""
import sys
from pathlib import Path
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import logging

# 添加父目錄到路徑以導入 GridStrategy
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_usdttwd_grid_refactored import GridStrategy
from indicators import ema, atr, adx

getcontext().prec = 28
log = logging.getLogger("BacktestAdapter")


class SimulatedOrder:
    """模擬訂單"""
    def __init__(self, order_id: str, side: str, price: Decimal, qty: Decimal, 
                 layer_idx: Optional[int] = None, tag: str = "grid"):
        self.order_id = order_id
        self.side = side  # "buy" or "sell"
        self.price = price
        self.qty = qty
        self.filled_qty = Decimal("0")
        self.layer_idx = layer_idx
        self.tag = tag
        self.status = "open"  # "open", "filled", "cancelled"
        self.created_at = None
    
    def __repr__(self):
        return f"SimulatedOrder({self.order_id}, {self.side}, {self.price}, {self.qty}, status={self.status})"


class BacktestAdapter:
    """
    回測適配器：模擬 BotEngine，使用相同的 GridStrategy 邏輯
    
    這個類確保回測和實盤使用完全相同的策略邏輯，避免邏輯分歧。
    """
    
    def __init__(self, strategy: GridStrategy, init_usdt: Decimal, init_twd: Decimal, 
                 fee_rate: Decimal = Decimal("0.0004"), verbose: bool = False):
        """
        初始化回測適配器
        
        Args:
            strategy: GridStrategy 實例（與實盤使用相同的類）
            init_usdt: 初始 USDT 餘額
            init_twd: 初始 TWD 餘額
            fee_rate: 交易手續費率
            verbose: 是否輸出詳細日誌
        """
        self.strategy = strategy
        self.verbose = verbose
        
        # 餘額
        self.usdt_balance = init_usdt
        self.twd_balance = init_twd
        self.available_usdt_balance = init_usdt
        self.available_twd_balance = init_twd
        
        # 訂單管理
        self.active_orders: Dict[str, SimulatedOrder] = {}
        self.order_counter = 0
        
        # 交易記錄
        self.trade_log: List[Dict[str, Any]] = []
        
        # 權益歷史（用於計算回撤）
        self.equity_history: List[Tuple[int, Decimal]] = []  # (index, equity)
        
        # 時間戳（模擬）
        self.current_time: Optional[datetime] = None
        self.current_index: int = 0
        
        # 費用
        self.fee_rate = fee_rate
        self.min_order_value_twd = Decimal(str(strategy.params.get("min_order_value_twd", "300.0")))
        
        # 黑天鵝保護
        self.is_halted = False
        self.halt_reason = None
        
        log.info(f"BacktestAdapter initialized: USDT={init_usdt}, TWD={init_twd}")
    
    def _gen_order_id(self, tag: str = "grid") -> str:
        """生成訂單ID"""
        self.order_counter += 1
        return f"{tag}_{self.order_counter:06d}"
    
    def _calculate_total_equity(self, current_price: Decimal) -> Decimal:
        """計算總權益"""
        return self.twd_balance + self.usdt_balance * current_price
    
    def _place_simulated_order(self, side: str, price: Decimal, qty: Decimal, 
                               layer_idx: Optional[int] = None, tag: str = "grid") -> Optional[str]:
        """
        模擬下單
        
        Returns:
            Optional[str]: 訂單ID，如果失敗則返回 None
        """
        # 檢查最小訂單價值
        order_value = price * qty
        if order_value < self.min_order_value_twd:
            if self.verbose:
                log.debug(f"Order skipped: value {order_value:.2f} < min {self.min_order_value_twd}")
            return None
        
        # 檢查餘額
        if side == "buy":
            required_twd = order_value
            if self.available_twd_balance < required_twd:
                if self.verbose:
                    log.debug(f"Order skipped: insufficient TWD. Need {required_twd:.2f}, have {self.available_twd_balance:.2f}")
                return None
            self.available_twd_balance -= required_twd
        else:  # sell
            if self.available_usdt_balance < qty:
                if self.verbose:
                    log.debug(f"Order skipped: insufficient USDT. Need {qty:.4f}, have {self.available_usdt_balance:.4f}")
                return None
            self.available_usdt_balance -= qty
        
        # 創建訂單
        order_id = self._gen_order_id(tag)
        order = SimulatedOrder(order_id, side, price, qty, layer_idx, tag)
        order.created_at = self.current_time
        self.active_orders[order_id] = order
        
        if self.verbose:
            log.debug(f"Placed order: {order}")
        
        return order_id
    
    def _check_order_fills(self, high: Decimal, low: Decimal, close: Decimal, 
                           current_index: int) -> List[Dict[str, Any]]:
        """
        檢查訂單是否成交
        
        Returns:
            List[Dict]: 成交記錄列表
        """
        fills = []
        
        for order_id, order in list(self.active_orders.items()):
            if order.status != "open":
                continue
            
            # 檢查是否觸及訂單價格
            if order.side == "buy" and low <= order.price <= high:
                # 買單成交
                fill_price = order.price
                fill_qty = order.qty
                
                # 計算費用
                fee = fill_qty * fill_price * self.fee_rate
                
                # 更新餘額
                self.usdt_balance += fill_qty
                self.available_usdt_balance += fill_qty
                self.twd_balance -= (fill_qty * fill_price + fee)
                self.available_twd_balance -= (fill_qty * fill_price + fee)
                
                # 更新訂單
                order.filled_qty = fill_qty
                order.status = "filled"
                
                # 記錄交易
                trade_record = {
                    'index': current_index,
                    'timestamp': self.current_time,
                    'order_id': order_id,
                    'type': f'grid_buy' if order.layer_idx is not None else 'trend_buy',
                    'side': 'buy',
                    'price': float(fill_price),
                    'quantity': float(fill_qty),
                    'fee': float(fee),
                    'layer_idx': order.layer_idx,
                    'tag': order.tag
                }
                fills.append(trade_record)
                self.trade_log.append(trade_record)
                
                # 移除訂單
                self.active_orders.pop(order_id)
                
                if self.verbose:
                    log.debug(f"Order filled: {order_id} - BUY {fill_qty} @ {fill_price}")
                
                # 掛反向單（網格模式）
                if order.layer_idx is not None and self.strategy.strategy_state == 'GRID':
                    self._place_replacement_order(order, close)
            
            elif order.side == "sell" and low <= order.price <= high:
                # 賣單成交
                fill_price = order.price
                fill_qty = order.qty
                
                # 計算費用
                fee = fill_qty * fill_price * self.fee_rate
                
                # 更新餘額
                self.twd_balance += (fill_qty * fill_price - fee)
                self.available_twd_balance += (fill_qty * fill_price - fee)
                self.usdt_balance -= fill_qty
                self.available_usdt_balance -= fill_qty
                
                # 更新訂單
                order.filled_qty = fill_qty
                order.status = "filled"
                
                # 記錄交易
                trade_record = {
                    'index': current_index,
                    'timestamp': self.current_time,
                    'order_id': order_id,
                    'type': f'grid_sell' if order.layer_idx is not None else 'trend_sell',
                    'side': 'sell',
                    'price': float(fill_price),
                    'quantity': float(fill_qty),
                    'fee': float(fee),
                    'layer_idx': order.layer_idx,
                    'tag': order.tag
                }
                fills.append(trade_record)
                self.trade_log.append(trade_record)
                
                # 移除訂單
                self.active_orders.pop(order_id)
                
                if self.verbose:
                    log.debug(f"Order filled: {order_id} - SELL {fill_qty} @ {fill_price}")
                
                # 掛反向單（網格模式）
                if order.layer_idx is not None and self.strategy.strategy_state == 'GRID':
                    self._place_replacement_order(order, close)
        
        return fills
    
    def _place_replacement_order(self, filled_order: SimulatedOrder, current_price: Decimal):
        """掛反向替換訂單"""
        if filled_order.layer_idx is None:
            return
        
        layer = self.strategy.grid_layers[filled_order.layer_idx]
        
        # 計算新訂單價格和數量
        new_side = "sell" if filled_order.side == "buy" else "buy"
        
        # 使用動態間距（如果啟用ATR）
        gap_to_use = layer.gap_abs
        if self.strategy.params.get('use_atr_spacing', False):
            # 這裡簡化處理，實際應該使用當前時間點的 ATR
            # 為了簡化，我們使用固定間距
            pass
        
        if new_side == "sell":
            new_price = self.strategy.quantize_price(filled_order.price + gap_to_use)
        else:
            new_price = self.strategy.quantize_price(filled_order.price - gap_to_use)
        
        if new_price <= 0:
            return
        
        # 計算數量
        total_equity = self._calculate_total_equity(current_price)
        new_qty = self.strategy.quantize_qty((layer.size_pct * total_equity) / current_price)
        
        if new_qty > 0:
            self._place_simulated_order(new_side, new_price, new_qty, layer.idx, tag="gr_repl")
    
    def _rebuild_grid_simulated(self, center_price: Decimal, full_rebuild: bool = True, 
                                trend_override: str = 'none'):
        """模擬重建網格"""
        if full_rebuild:
            # 取消所有訂單
            for order in list(self.active_orders.values()):
                if order.status == "open":
                    # 釋放鎖定的餘額
                    if order.side == "buy":
                        self.available_twd_balance += order.price * order.qty
                    else:
                        self.available_usdt_balance += order.qty
            self.active_orders.clear()
        
        # 計算動態間距（如果啟用ATR）
        dynamic_gaps = {}
        if self.strategy.params.get('use_atr_spacing', False):
            # 簡化處理：使用固定間距
            pass
        
        total_equity = self._calculate_total_equity(center_price)
        
        for layer in self.strategy.grid_layers:
            qty_usdt = self.strategy.quantize_qty(layer.size_pct * total_equity / center_price)
            if qty_usdt <= 0:
                continue
            
            gap_to_use = dynamic_gaps.get(layer.idx, layer.gap_abs)
            
            buy_levels = layer.levels_each_side
            sell_levels = layer.levels_each_side
            
            if trend_override == 'long':
                sell_levels = 0
            elif trend_override == 'short':
                buy_levels = 0
            
            # 掛買單
            for i in range(1, buy_levels + 1):
                buy_price = self.strategy.quantize_price(center_price - (gap_to_use * i))
                if buy_price > 0:
                    self._place_simulated_order("buy", buy_price, qty_usdt, layer.idx, tag=f"gr{layer.idx}b")
            
            # 掛賣單
            for i in range(1, sell_levels + 1):
                sell_price = self.strategy.quantize_price(center_price + (gap_to_use * i))
                if sell_price > 0:
                    self._place_simulated_order("sell", sell_price, qty_usdt, layer.idx, tag=f"gr{layer.idx}s")
    
    def _check_black_swan(self, price_history_slice: pd.Series) -> bool:
        """檢查黑天鵝事件"""
        if not self.strategy.params.get("use_black_swan_protection", False):
            return False
        
        check_minutes = int(self.strategy.params.get("black_swan_check_minutes", 5))
        threshold_pct = Decimal(str(self.strategy.params.get("black_swan_threshold_pct", "0.03")))
        
        if len(price_history_slice) < check_minutes:
            return False
        
        # 檢查最近 N 分鐘的價格波動
        recent_prices = price_history_slice.tail(check_minutes)
        if len(recent_prices) < 2:
            return False
        
        highest = Decimal(str(recent_prices.max()))
        lowest = Decimal(str(recent_prices.min()))
        
        if lowest > 0:
            volatility_pct = (highest - lowest) / lowest
            if volatility_pct > threshold_pct:
                self.is_halted = True
                self.halt_reason = f"Black swan: {volatility_pct:.2%} volatility in {check_minutes} minutes"
                if self.verbose:
                    log.warning(f"BLACK SWAN EVENT: {self.halt_reason}")
                return True
        
        return False
    
    def _simulate_hybrid_strategy(self, current_price: Decimal, ema_fast: Optional[Decimal], 
                                  ema_slow: Optional[Decimal], adx_val: Optional[Decimal]):
        """模擬混合策略邏輯"""
        if not self.strategy.params.get('use_hybrid_model', False):
            return
        
        if self.strategy.cooldown_counter > 0:
            self.strategy.cooldown_counter -= 1
            return
        
        if ema_fast is None or ema_slow is None or adx_val is None:
            return
        
        adx_threshold = int(self.strategy.params.get('adx_strength_threshold', 25))
        is_ema_bull = ema_fast > ema_slow
        is_ema_bear = ema_fast < ema_slow
        is_adx_strong = adx_val > adx_threshold
        
        if self.strategy.strategy_state == 'GRID':
            is_strong_uptrend = is_ema_bull and is_adx_strong
            is_strong_downtrend = is_ema_bear and is_adx_strong
            
            if is_strong_uptrend or is_strong_downtrend:
                self.strategy.strategy_state = 'TREND_FOLLOWING'
                trend_side = 'long' if is_strong_uptrend else 'short'
                
                # 取消所有網格訂單
                for order in list(self.active_orders.values()):
                    if order.status == "open":
                        if order.side == "buy":
                            self.available_twd_balance += order.price * order.qty
                        else:
                            self.available_usdt_balance += order.qty
                self.active_orders.clear()
                
                # 建立趨勢倉位
                trend_equity_pct = Decimal(str(self.strategy.params.get('trend_trade_equity_pct', '0.4')))
                total_equity = self._calculate_total_equity(current_price)
                trade_value_twd = total_equity * trend_equity_pct
                
                if trend_side == 'long':
                    qty_to_buy = self.strategy.quantize_qty(trade_value_twd / current_price)
                    if self.available_twd_balance >= trade_value_twd:
                        buy_price = current_price * Decimal("1.001")
                        order_id = self._place_simulated_order("buy", buy_price, qty_to_buy, layer_idx=None, tag="trend_long")
                        if order_id:
                            # 假設立即成交（簡化）
                            order = self.active_orders[order_id]
                            order.status = "filled"
                            order.filled_qty = order.qty
                            self.usdt_balance += order.qty
                            self.available_usdt_balance += order.qty
                            self.twd_balance -= (order.qty * buy_price)
                            self.available_twd_balance -= (order.qty * buy_price)
                            self.active_orders.pop(order_id)
                            
                            self.strategy.trend_position = {
                                'side': 'long',
                                'entry_price': current_price,
                                'qty': qty_to_buy,
                                'peak_price': current_price
                            }
                            # 建立順勢網格
                            self._rebuild_grid_simulated(current_price, full_rebuild=False, trend_override='long')
                else:  # short
                    qty_to_sell = self.strategy.quantize_qty(trade_value_twd / current_price)
                    if self.available_usdt_balance >= qty_to_sell:
                        sell_price = current_price * Decimal("0.999")
                        order_id = self._place_simulated_order("sell", sell_price, qty_to_sell, layer_idx=None, tag="trend_short")
                        if order_id:
                            # 假設立即成交（簡化）
                            order = self.active_orders[order_id]
                            order.status = "filled"
                            order.filled_qty = order.qty
                            self.twd_balance += (order.qty * sell_price)
                            self.available_twd_balance += (order.qty * sell_price)
                            self.usdt_balance -= order.qty
                            self.available_usdt_balance -= order.qty
                            self.active_orders.pop(order_id)
                            
                            self.strategy.trend_position = {
                                'side': 'short',
                                'entry_price': current_price,
                                'qty': qty_to_sell,
                                'valley_price': current_price
                            }
                            # 建立順勢網格
                            self._rebuild_grid_simulated(current_price, full_rebuild=False, trend_override='short')
        
        elif self.strategy.strategy_state == 'TREND_FOLLOWING':
            if not self.strategy.trend_position:
                self.strategy.strategy_state = 'GRID'
                return
            
            # 檢查止損
            trailing_stop_pct = Decimal(str(self.strategy.params.get('trend_trailing_stop_pct', '0.02')))
            side = self.strategy.trend_position['side']
            should_exit = False
            
            if side == 'long':
                peak_price = max(self.strategy.trend_position.get('peak_price', current_price), current_price)
                self.strategy.trend_position['peak_price'] = peak_price
                stop_loss_price = peak_price * (Decimal("1.0") - trailing_stop_pct)
                if current_price <= stop_loss_price:
                    should_exit = True
            elif side == 'short':
                valley_price = min(self.strategy.trend_position.get('valley_price', current_price), current_price)
                self.strategy.trend_position['valley_price'] = valley_price
                stop_loss_price = valley_price * (Decimal("1.0") + trailing_stop_pct)
                if current_price >= stop_loss_price:
                    should_exit = True
            
            if should_exit:
                # 平倉
                qty = self.strategy.trend_position['qty']
                if side == 'long':
                    sell_price = current_price * Decimal("0.999")
                    # 簡化：直接更新餘額
                    self.twd_balance += (qty * sell_price)
                    self.available_twd_balance += (qty * sell_price)
                    self.usdt_balance -= qty
                    self.available_usdt_balance -= qty
                else:
                    buy_price = current_price * Decimal("1.001")
                    self.usdt_balance += qty
                    self.available_usdt_balance += qty
                    self.twd_balance -= (qty * buy_price)
                    self.available_twd_balance -= (qty * buy_price)
                
                self.strategy.trend_position = None
                self.strategy.strategy_state = 'GRID'
                
                cooldown_bars = int(self.strategy.params.get('trend_cooldown_bars', 240))
                self.strategy.cooldown_counter = cooldown_bars
                
                # 取消所有訂單並重建網格
                for order in list(self.active_orders.values()):
                    if order.status == "open":
                        if order.side == "buy":
                            self.available_twd_balance += order.price * order.qty
                        else:
                            self.available_usdt_balance += order.qty
                self.active_orders.clear()
                self._rebuild_grid_simulated(current_price, full_rebuild=False, trend_override='none')
    
    def _simulate_directional_bias(self, current_price: Decimal, ema_fast: Optional[Decimal], 
                                    ema_slow: Optional[Decimal]):
        """模擬方向性偏置調整"""
        target_ratio = self.strategy.get_ema_target_bias(
            external_data=pd.Series([float(p[1]) for p in self.strategy.price_history])
        )
        
        current_ratio = (self.usdt_balance * current_price) / self._calculate_total_equity(current_price) if self._calculate_total_equity(current_price) > 0 else Decimal("0")
        delta_value_target = (target_ratio - current_ratio) * self._calculate_total_equity(current_price)
        
        threshold = Decimal(str(self.strategy.params.get("bias_rebalance_threshold_twd", "1660.0")))
        if abs(delta_value_target) > threshold:
            value_to_trade = delta_value_target * Decimal(str(self.strategy.params.get("bias_rebalance_fraction", "0.25")))
            qty_to_trade = self.strategy.quantize_qty(value_to_trade / current_price)
            
            side = "buy" if qty_to_trade > 0 else "sell"
            qty_abs = abs(qty_to_trade)
            
            slip_price = current_price * (Decimal("1.001") if side == "buy" else Decimal("0.999"))
            order_value_twd = abs(qty_abs * slip_price)
            
            if order_value_twd < self.min_order_value_twd:
                return
            
            SAFETY_MARGIN = Decimal("1.01")
            
            if side == 'buy' and self.available_twd_balance < (order_value_twd * SAFETY_MARGIN):
                return
            if side == 'sell' and self.available_usdt_balance < (qty_abs * SAFETY_MARGIN):
                return
            
            if qty_abs > 0:
                # 簡化：直接執行交易（不掛單）
                if side == "buy":
                    self.usdt_balance += qty_abs
                    self.available_usdt_balance += qty_abs
                    fee = qty_abs * slip_price * self.fee_rate
                    self.twd_balance -= (qty_abs * slip_price + fee)
                    self.available_twd_balance -= (qty_abs * slip_price + fee)
                else:
                    self.twd_balance += (qty_abs * slip_price - qty_abs * slip_price * self.fee_rate)
                    self.available_twd_balance += (qty_abs * slip_price - qty_abs * slip_price * self.fee_rate)
                    self.usdt_balance -= qty_abs
                    self.available_usdt_balance -= qty_abs
                
                self.strategy.last_bias_rebalance_ts = self.current_time
    
    def run(self, ohlc_df: pd.DataFrame) -> Dict[str, Any]:
        """
        執行回測
        
        Args:
            ohlc_df: OHLC DataFrame，必須包含 'high', 'low', 'close' 列，索引為時間戳
        
        Returns:
            Dict: 回測結果，包含 total_pnl, roi_pct, max_drawdown_pct, trades_log 等
        """
        log.info("Starting backtest with BacktestAdapter...")
        
        # 重置狀態
        self.trade_log.clear()
        self.equity_history.clear()
        self.active_orders.clear()
        self.is_halted = False
        self.halt_reason = None
        
        # 初始化策略時間戳
        initial_time = ohlc_df.index[0] if isinstance(ohlc_df.index[0], datetime) else datetime.now(timezone.utc)
        self.strategy.last_recenter_ts = initial_time
        self.strategy.last_bias_rebalance_ts = initial_time
        
        # 計算指標（向量化，一次性計算）
        price_series = ohlc_df['close'].ffill()
        high_series = ohlc_df['high'].ffill()
        low_series = ohlc_df['low'].ffill()
        
        ema_fast_span = int(self.strategy.params.get("ema_span_fast_bars", 120))
        ema_slow_span = int(self.strategy.params.get("ema_span_slow_bars", 600))
        dmi_period = int(self.strategy.params.get("dmi_period", 14))
        atr_period = int(self.strategy.params.get("atr_period", 14))
        
        # 預計算指標
        ema_fast_series = ema(price_series, ema_fast_span)
        ema_slow_series = ema(price_series, ema_slow_span)
        adx_series, _, _ = adx(high_series, low_series, price_series, dmi_period)
        atr_series = atr(high_series, low_series, price_series, atr_period)
        
        # 初始網格重建
        initial_price = Decimal(str(price_series.iloc[0]))
        self._rebuild_grid_simulated(initial_price, full_rebuild=True)
        
        # 主循環
        for idx, (timestamp, row) in enumerate(ohlc_df.iterrows()):
            if self.is_halted:
                if self.verbose:
                    log.warning(f"Backtest halted at index {idx}: {self.halt_reason}")
                break
            
            self.current_index = idx
            self.current_time = timestamp if isinstance(timestamp, datetime) else datetime.now(timezone.utc)
            
            high = Decimal(str(row['high']))
            low = Decimal(str(row['low']))
            close = Decimal(str(row['close']))
            
            # 更新策略價格歷史（用於實盤模式的指標計算）
            timestamp_ms = int(timestamp.timestamp() * 1000) if isinstance(timestamp, datetime) else int(self.current_time.timestamp() * 1000)
            self.strategy.price_history.append((timestamp_ms, close))
            
            # 檢查黑天鵝事件
            if idx >= 5:  # 至少需要一些歷史數據
                price_slice = price_series.iloc[max(0, idx-60):idx+1]  # 最近60個數據點
                if self._check_black_swan(price_slice):
                    break
            
            # 獲取當前指標值（使用預計算的指標）
            ema_fast_val = Decimal(str(ema_fast_series.iloc[idx])) if idx < len(ema_fast_series) and not pd.isna(ema_fast_series.iloc[idx]) else None
            ema_slow_val = Decimal(str(ema_slow_series.iloc[idx])) if idx < len(ema_slow_series) and not pd.isna(ema_slow_series.iloc[idx]) else None
            adx_val = Decimal(str(adx_series.iloc[idx])) if idx < len(adx_series) and not pd.isna(adx_series.iloc[idx]) else None
            
            # 檢查訂單成交
            self._check_order_fills(high, low, close, idx)
            
            # 混合策略管理
            if self.strategy.params.get('use_hybrid_model', False):
                self._simulate_hybrid_strategy(close, ema_fast_val, ema_slow_val, adx_val)
            
            # 方向性偏置調整
            if self.strategy.should_rebalance_bias(self.current_time):
                self._simulate_directional_bias(close, ema_fast_val, ema_slow_val)
                self.strategy.last_bias_rebalance_ts = self.current_time
            
            # 網格重建
            if self.strategy.should_rebuild_grid(self.current_time):
                trend_override = 'none'
                if self.strategy.strategy_state == 'TREND_FOLLOWING' and self.strategy.trend_position:
                    trend_override = self.strategy.trend_position['side']
                self._rebuild_grid_simulated(close, full_rebuild=True, trend_override=trend_override)
                self.strategy.last_recenter_ts = self.current_time
            
            # 記錄權益
            total_equity = self._calculate_total_equity(close)
            self.equity_history.append((idx, total_equity))
        
        # 計算統計結果
        initial_equity = self._calculate_total_equity(Decimal(str(price_series.iloc[0])))
        final_equity = self._calculate_total_equity(Decimal(str(price_series.iloc[-1])))
        total_pnl = final_equity - initial_equity
        roi_pct = (total_pnl / initial_equity) * 100 if initial_equity > 0 else Decimal("0")
        
        # 計算最大回撤
        max_drawdown_pct = Decimal("0")
        if len(self.equity_history) > 0:
            equity_values = [e[1] for e in self.equity_history]
            peak = equity_values[0]
            max_dd = Decimal("0")
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else Decimal("0")
                if dd > max_dd:
                    max_dd = dd
            max_drawdown_pct = max_dd * 100
        
        # 計算夏普比率（簡化版）
        sharpe_ratio = Decimal("0")
        if len(self.equity_history) > 1:
            returns = []
            for i in range(1, len(self.equity_history)):
                prev_equity = self.equity_history[i-1][1]
                curr_equity = self.equity_history[i][1]
                if prev_equity > 0:
                    ret = (curr_equity - prev_equity) / prev_equity
                    returns.append(float(ret))
            
            if len(returns) > 0:
                import numpy as np
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    sharpe_ratio = Decimal(str(mean_return / std_return * np.sqrt(252)))  # 年化
        
        result = {
            'total_pnl': float(total_pnl),
            'roi_pct': float(roi_pct),
            'max_drawdown_pct': float(max_drawdown_pct),
            'sharpe_ratio': float(sharpe_ratio),
            'trades_log': self.trade_log,
            'final_equity': float(final_equity),
            'initial_equity': float(initial_equity),
            'total_trades': len(self.trade_log),
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason
        }
        
        log.info(f"Backtest completed: ROI={roi_pct:.2f}%, MaxDD={max_drawdown_pct:.2f}%, Trades={len(self.trade_log)}")
        
        return result

