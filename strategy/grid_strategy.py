# grid_strategy.py
"""
GridStrategy - 網格交易策略實現
繼承 BaseStrategy，實現三層固定間隙網格邏輯
"""
from __future__ import annotations
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Deque, Tuple
from datetime import datetime, timezone
from collections import deque
import logging
import pandas as pd

from strategy.base_strategy import BaseStrategy
from indicators import ema, atr, adx

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


class GridStrategy(BaseStrategy):
    """
    網格交易策略
    
    核心特性：
    - 三層固定間隙網格（small/mid/big gaps）
    - EMA 趨勢判斷
    - 動態參數調整（通過 update_config）
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # ========== 可調整參數（類屬性，可由 StrategyOptimizer 修改）==========
        # 網格參數
        self.small_gap: Decimal = Decimal(str(config.get("small_gap", "0.035")))
        self.mid_multiplier: int = int(config.get("mid_mult", 3))
        self.big_multiplier: int = int(config.get("big_mult", 5))
        self.levels_each: int = int(config.get("levels_each", 5))
        
        # 網格層級大小百分比
        self.size_pct_small: Decimal = Decimal(str(config.get("size_pct_small", "0.15")))
        self.size_pct_mid: Decimal = Decimal(str(config.get("size_pct_mid", "0.20")))
        self.size_pct_big: Decimal = Decimal(str(config.get("size_pct_big", "0.25")))
        
        # EMA 參數
        self.ema_span_fast_bars: int = int(config.get("ema_span_fast_bars", 10))
        self.ema_span_slow_bars: int = int(config.get("ema_span_slow_bars", 50))
        
        # 方向性偏置參數
        self.bias_high: Decimal = Decimal(str(config.get("bias_high", "0.6")))
        self.bias_low: Decimal = Decimal(str(config.get("bias_low", "0.4")))
        self.bias_neutral_target: Decimal = Decimal(str(config.get("bias_neutral_target", "0.5")))
        self.bias_rebalance_threshold_twd: Decimal = Decimal(str(config.get("bias_rebalance_threshold_twd", "500.0")))
        self.bias_rebalance_fraction: Decimal = Decimal(str(config.get("bias_rebalance_fraction", "0.5")))
        
        # ATR 動態間距參數
        self.use_atr_spacing: bool = config.get("use_atr_spacing", False)
        self.atr_spacing_multiplier: Decimal = Decimal(str(config.get("atr_spacing_multiplier", "0.5")))
        self.atr_period: int = int(config.get("atr_period", 14))
        
        # 混合策略參數
        self.use_hybrid_model: bool = config.get("use_hybrid_model", False)
        self.dmi_period: int = int(config.get("dmi_period", 14))
        self.adx_strength_threshold: int = int(config.get("adx_strength_threshold", 25))
        self.trend_trade_equity_pct: Decimal = Decimal(str(config.get("trend_trade_equity_pct", "0.4")))
        self.trend_trailing_stop_pct: Decimal = Decimal(str(config.get("trend_trailing_stop_pct", "0.02")))
        self.trend_cooldown_bars: int = int(config.get("trend_cooldown_bars", 240))
        
        # 其他配置
        self.price_precision: Decimal = Decimal(str(config.get("price_precision", "0.001")))
        self.qty_precision: Decimal = Decimal(str(config.get("qty_precision", "0.001")))
        self.min_order_value_twd: Decimal = Decimal(str(config.get("min_order_value_twd", "300.0")))
        
        # 內部狀態
        self.grid_layers: List[GridLayer] = []
        self._rebuild_grid_layers()
        
        # 策略狀態
        self.strategy_state: str = "GRID"  # "GRID" or "TREND_FOLLOWING"
        self.trend_position: Optional[Dict[str, Any]] = None
        self.cooldown_counter: int = 0
        
        log.info(f"GridStrategy initialized with {len(self.grid_layers)} layers.")
    
    def _rebuild_grid_layers(self):
        """根據當前參數重建網格層級"""
        self.grid_layers.clear()
        self.grid_layers.extend([
            GridLayer(0, self.small_gap, self.size_pct_small, self.levels_each),
            GridLayer(1, self.small_gap * self.mid_multiplier, self.size_pct_mid, self.levels_each),
            GridLayer(2, self.small_gap * self.big_multiplier, self.size_pct_big, self.levels_each)
        ])
    
    def calculate_indicators(self) -> Dict[str, Optional[Decimal]]:
        """計算技術指標"""
        if not self.validate_price_history(min_length=max(self.ema_span_slow_bars, self.atr_period)):
            return {
                'ema_fast': None,
                'ema_slow': None,
                'atr': None,
                'adx': None,
                'plus_di': None,
                'minus_di': None
            }
        
        prices = [float(p[1]) for p in self.price_history]
        series = pd.Series(prices)
        
        # 計算 EMA
        ema_fast_val = None
        ema_slow_val = None
        if len(series) >= self.ema_span_fast_bars:
            ema_fast_series = ema(series, self.ema_span_fast_bars)
            ema_fast_val = Decimal(str(float(ema_fast_series.iloc[-1])))
        if len(series) >= self.ema_span_slow_bars:
            ema_slow_series = ema(series, self.ema_span_slow_bars)
            ema_slow_val = Decimal(str(float(ema_slow_series.iloc[-1])))
        
        # 計算 ATR（簡化版，使用價格變化）
        atr_val = None
        if self.use_atr_spacing and len(series) >= self.atr_period:
            high_series = series  # 簡化：使用收盤價作為高低價
            low_series = series
            atr_series = atr(high_series, low_series, series, self.atr_period)
            atr_val = Decimal(str(float(atr_series.iloc[-1])))
        
        # 計算 ADX（簡化版）
        adx_val = None
        plus_di_val = None
        minus_di_val = None
        if self.use_hybrid_model and len(series) >= self.dmi_period * 2:
            high_series = series
            low_series = series
            adx_series, plus_di_series, minus_di_series = adx(high_series, low_series, series, self.dmi_period)
            adx_val = Decimal(str(float(adx_series.iloc[-1])))
            plus_di_val = Decimal(str(float(plus_di_series.iloc[-1])))
            minus_di_val = Decimal(str(float(minus_di_series.iloc[-1])))
        
        return {
            'ema_fast': ema_fast_val,
            'ema_slow': ema_slow_val,
            'atr': atr_val,
            'adx': adx_val,
            'plus_di': plus_di_val,
            'minus_di': minus_di_val
        }
    
    def get_ema_target_bias(self) -> Decimal:
        """根據 EMA 快慢線交叉，計算目標 USDT 曝險比例"""
        indicators = self.calculate_indicators()
        ema_fast = indicators.get('ema_fast')
        ema_slow = indicators.get('ema_slow')
        
        if ema_fast is None or ema_slow is None:
            return self.bias_neutral_target
        
        if ema_fast > ema_slow:
            return self.bias_high
        elif ema_fast < ema_slow:
            return self.bias_low
        else:
            return self.bias_neutral_target
    
    def generate_signals(self, current_price: Decimal) -> Dict[str, Any]:
        """
        生成交易信號
        
        注意：此方法只返回信號，不執行實際下單。
        實際下單由 BotEngine 負責。
        """
        if not self.validate_price_history():
            return {'action': 'hold', 'reason': 'insufficient_price_history'}
        
        signals = []
        indicators = self.calculate_indicators()
        
        # 根據策略狀態生成不同信號
        if self.strategy_state == "GRID":
            # 標準網格模式：生成網格訂單信號
            for layer in self.grid_layers:
                # 計算動態間距（如果啟用 ATR）
                gap_to_use = layer.gap_abs
                if self.use_atr_spacing and indicators.get('atr'):
                    base_gap = indicators['atr'] * self.atr_spacing_multiplier
                    min_gap = self.small_gap
                    max_gap = Decimal("0.15")
                    base_gap = max(min_gap, min(base_gap, max_gap))
                    gap_to_use = base_gap * (self.mid_multiplier if layer.idx == 1 else self.big_multiplier if layer.idx == 2 else 1)
                
                # 生成買單信號
                for i in range(1, layer.levels_each_side + 1):
                    buy_price = self._quantize_price(current_price - (gap_to_use * i))
                    if buy_price > 0:
                        signals.append({
                            'action': 'buy',
                            'price': buy_price,
                            'layer_idx': layer.idx,
                            'size_pct': layer.size_pct,
                            'reason': f'grid_layer_{layer.idx}_buy_level_{i}'
                        })
                
                # 生成賣單信號
                for i in range(1, layer.levels_each_side + 1):
                    sell_price = self._quantize_price(current_price + (gap_to_use * i))
                    if sell_price > 0:
                        signals.append({
                            'action': 'sell',
                            'price': sell_price,
                            'layer_idx': layer.idx,
                            'size_pct': layer.size_pct,
                            'reason': f'grid_layer_{layer.idx}_sell_level_{i}'
                        })
        
        elif self.strategy_state == "TREND_FOLLOWING":
            # 趨勢跟隨模式：只生成順勢單
            if self.trend_position:
                trend_side = self.trend_position['side']
                # 只生成與趨勢方向一致的訂單
                # （具體邏輯可根據需求擴展）
                pass
        
        return {
            'action': 'place_orders' if signals else 'hold',
            'signals': signals,
            'strategy_state': self.strategy_state,
            'indicators': indicators
        }
    
    def update_config(self, new_params: Dict[str, Any]) -> bool:
        """
        動態更新策略參數
        
        這是 AI Agent 調整參數的關鍵入口點
        """
        try:
            updated = False
            
            # 更新網格參數
            if 'small_gap' in new_params:
                self.small_gap = Decimal(str(new_params['small_gap']))
                updated = True
            if 'mid_multiplier' in new_params:
                self.mid_multiplier = int(new_params['mid_multiplier'])
                updated = True
            if 'big_multiplier' in new_params:
                self.big_multiplier = int(new_params['big_multiplier'])
                updated = True
            if 'levels_each' in new_params:
                self.levels_each = int(new_params['levels_each'])
                updated = True
            
            # 更新大小百分比
            if 'size_pct_small' in new_params:
                self.size_pct_small = Decimal(str(new_params['size_pct_small']))
                updated = True
            if 'size_pct_mid' in new_params:
                self.size_pct_mid = Decimal(str(new_params['size_pct_mid']))
                updated = True
            if 'size_pct_big' in new_params:
                self.size_pct_big = Decimal(str(new_params['size_pct_big']))
                updated = True
            
            # 更新 EMA 參數
            if 'ema_span_fast_bars' in new_params:
                self.ema_span_fast_bars = int(new_params['ema_span_fast_bars'])
                updated = True
            if 'ema_span_slow_bars' in new_params:
                self.ema_span_slow_bars = int(new_params['ema_span_slow_bars'])
                updated = True
            
            # 更新偏置參數
            if 'bias_high' in new_params:
                self.bias_high = Decimal(str(new_params['bias_high']))
                updated = True
            if 'bias_low' in new_params:
                self.bias_low = Decimal(str(new_params['bias_low']))
                updated = True
            
            # 如果網格參數改變，重建層級
            if updated and any(k in new_params for k in ['small_gap', 'mid_multiplier', 'big_multiplier', 'levels_each']):
                self._rebuild_grid_layers()
            
            self.last_update_ts = datetime.now(timezone.utc)
            
            if updated:
                log.info(f"Strategy parameters updated: {list(new_params.keys())}")
            
            return updated
            
        except Exception as e:
            log.error(f"Failed to update strategy config: {e}", exc_info=True)
            return False
    
    def _get_config_snapshot(self) -> Dict[str, Any]:
        """獲取當前可調整參數的快照"""
        return {
            'small_gap': float(self.small_gap),
            'mid_multiplier': self.mid_multiplier,
            'big_multiplier': self.big_multiplier,
            'levels_each': self.levels_each,
            'size_pct_small': float(self.size_pct_small),
            'size_pct_mid': float(self.size_pct_mid),
            'size_pct_big': float(self.size_pct_big),
            'ema_span_fast_bars': self.ema_span_fast_bars,
            'ema_span_slow_bars': self.ema_span_slow_bars,
            'bias_high': float(self.bias_high),
            'bias_low': float(self.bias_low),
            'use_atr_spacing': self.use_atr_spacing,
            'atr_spacing_multiplier': float(self.atr_spacing_multiplier),
            'atr_period': self.atr_period
        }
    
    def _quantize_price(self, price: Decimal) -> Decimal:
        """價格精度量化"""
        return price.quantize(self.price_precision, rounding=getcontext().rounding)
    
    def _quantize_qty(self, qty: Decimal) -> Decimal:
        """數量精度量化"""
        return qty.quantize(self.qty_precision, rounding="ROUND_DOWN")

