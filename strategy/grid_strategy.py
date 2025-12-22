"""
GridStrategy - OOP 重構版網格策略

此檔案為從 `strategy_usdttwd_grid_refactored.py` 抽取的 `GridStrategy` 實作，
用於模組化架構（`main_modular.py`、`engine.bot_engine` 等）。
"""
from __future__ import annotations

from collections import deque
from decimal import Decimal, getcontext
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Deque, Tuple

import logging
import pandas as pd

from core.indicators import ema, atr, adx, rsi, macd

log = logging.getLogger("GridStrategy")
getcontext().prec = 28


class GridLayer:
    """網格層級定義"""

    def __init__(self, idx: int, gap_abs: Decimal, size_pct: Decimal, levels_each_side: int):
        self.idx = idx
        self.gap_abs = gap_abs
        self.size_pct = size_pct
        self.levels_each_side = levels_each_side

    def __repr__(self) -> str:
        return f"GridLayer(idx={self.idx}, gap={self.gap_abs}, pct={self.size_pct * 100:.2f}%)"


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

        # 價格歷史（由 BotEngine 注入）
        self.price_history: Deque[Tuple[int, Decimal]] = deque(
            maxlen=int(config.get("price_data_deque_size", 3100))
        )

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

        log.info("GridStrategy '%s' initialized.", self.strategy_name)

    # ------------------------------------------------------------------ #
    # 網格結構 & 參數更新
    # ------------------------------------------------------------------ #
    def _rebuild_grid_layers(self) -> None:
        """根據當前參數重建網格層級"""
        self.grid_layers.clear()
        small_gap = Decimal(str(self.params["small_gap"]))
        levels_each = int(self.params["levels_each"])
        self.grid_layers.extend(
            [
                GridLayer(
                    0,
                    small_gap,
                    Decimal(str(self.params["size_pct_small"])),
                    levels_each,
                ),
                GridLayer(
                    1,
                    small_gap * int(self.params["mid_mult"]),
                    Decimal(str(self.params["size_pct_mid"])),
                    levels_each,
                ),
                GridLayer(
                    2,
                    small_gap * int(self.params["big_mult"]),
                    Decimal(str(self.params["size_pct_big"])),
                    levels_each,
                ),
            ]
        )

    def update_config(self, new_params: Dict[str, Any]) -> bool:
        """
        動態更新策略參數（熱更新，無需重啟）

        Args:
            new_params: 新的參數字典

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
                    log.info("Parameter '%s' updated: %s -> %s", key, old_value, value)

            # 如果網格參數改變，重建層級
            if updated and any(
                k
                in new_params
                for k in [
                    "small_gap",
                    "mid_mult",
                    "big_mult",
                    "levels_each",
                    "size_pct_small",
                    "size_pct_mid",
                    "size_pct_big",
                ]
            ):
                self._rebuild_grid_layers()
                log.info("Grid layers rebuilt due to parameter changes.")

            # 更新精度（如果改變）
            if "price_precision" in new_params:
                self.price_precision = Decimal(str(new_params["price_precision"]))
            if "qty_precision" in new_params:
                self.qty_precision = Decimal(str(new_params["qty_precision"]))

            return updated

        except Exception as e:  # pragma: no cover - 防禦性
            log.error("Failed to update config: %s", e, exc_info=True)
            return False

    # ------------------------------------------------------------------ #
    # 觀察 / 指標
    # ------------------------------------------------------------------ #
    def get_market_observation(self) -> Dict[str, Any]:
        """
        獲取當前市場觀察數據（供 AI Agent 使用）
        """
        observation: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price_history_length": len(self.price_history),
            "strategy_state": self.strategy_state,
            "grid_layers_count": len(self.grid_layers),
            "trend_position": self.trend_position.copy()
            if self.trend_position
            else None,
            "cooldown_counter": self.cooldown_counter,
            "indicators": {},
            "parameters": {},
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
                observation["indicators"]["ema_fast"] = float(ema_fast_series.iloc[-1])
            if len(series) >= ema_slow_span:
                ema_slow_series = ema(series, ema_slow_span)
                observation["indicators"]["ema_slow"] = float(ema_slow_series.iloc[-1])

            # ATR
            atr_period = int(self.params.get("atr_period", 14))
            if len(series) >= atr_period:
                atr_series = atr(
                    series, series, series, atr_period
                )  # 簡化：使用收盤價作為高低價
                observation["indicators"]["atr"] = float(atr_series.iloc[-1])

            # ADX
            dmi_period = int(self.params.get("dmi_period", 14))
            if len(series) >= dmi_period * 2:
                adx_series, plus_di, minus_di = adx(series, series, series, dmi_period)
                observation["indicators"]["adx"] = float(adx_series.iloc[-1])
                observation["indicators"]["plus_di"] = float(plus_di.iloc[-1])
                observation["indicators"]["minus_di"] = float(minus_di.iloc[-1])

            # RSI
            rsi_period = int(self.params.get("rsi_period", 14))
            if len(series) >= rsi_period + 1:
                rsi_series = rsi(series, rsi_period)
                observation["indicators"]["rsi"] = float(rsi_series.iloc[-1])

            # MACD
            macd_fast = int(self.params.get("macd_fast_period", 12))
            macd_slow = int(self.params.get("macd_slow_period", 26))
            if len(series) >= macd_slow:
                macd_line, signal_line, hist = macd(series, macd_fast, macd_slow)
                observation["indicators"]["macd"] = float(macd_line.iloc[-1])
                observation["indicators"]["macd_signal"] = float(signal_line.iloc[-1])
                observation["indicators"]["macd_hist"] = float(hist.iloc[-1])

            # 波動率（簡化：使用價格標準差）
            if len(series) >= 20:
                observation["indicators"]["volatility"] = float(series.tail(20).std())

        # 當前參數快照
        observation["parameters"] = {
            "small_gap": float(Decimal(str(self.params.get("small_gap", "0.035")))),
            "mid_mult": int(self.params.get("mid_mult", 3)),
            "big_mult": int(self.params.get("big_mult", 7)),
            "levels_each": int(self.params.get("levels_each", 6)),
            "ema_span_fast_bars": int(self.params.get("ema_span_fast_bars", 120)),
            "ema_span_slow_bars": int(self.params.get("ema_span_slow_bars", 600)),
            "bias_high": float(Decimal(str(self.params.get("bias_high", "0.60")))),
            "bias_low": float(Decimal(str(self.params.get("bias_low", "0.25")))),
            "use_atr_spacing": bool(self.params.get("use_atr_spacing", False)),
            "atr_spacing_multiplier": float(
                Decimal(str(self.params.get("atr_spacing_multiplier", "0.8")))
            ),
            "use_hybrid_model": bool(self.params.get("use_hybrid_model", False)),
        }

        return observation

    # ------------------------------------------------------------------ #
    # 時間驅動條件
    # ------------------------------------------------------------------ #
    def should_rebuild_grid(self, current_time: datetime) -> bool:
        """判斷是否應該重建網格"""
        if self.last_recenter_ts is None:
            return True

        recenter_interval = int(self.params.get("recenter_interval_minutes", 480)) * 60
        elapsed = (current_time - self.last_recenter_ts).total_seconds()
        return elapsed >= recenter_interval

    def should_rebalance_bias(self, current_time: datetime) -> bool:
        """判斷是否應該調整方向性偏置"""
        if self.last_bias_rebalance_ts is None:
            return True

        bias_interval = int(self.params.get("bias_check_interval_sec", 60))
        elapsed = (current_time - self.last_bias_rebalance_ts).total_seconds()
        return elapsed >= bias_interval

    # ------------------------------------------------------------------ #
    # EMA / ATR / ADX 計算
    # ------------------------------------------------------------------ #
    def get_ema_target_bias(self, external_data: Optional[pd.Series] = None) -> Decimal:
        """
        根據 EMA 快慢線交叉，計算目標 USDT 曝險比例
        """
        ema_fast = self._calculate_ema_from_history(
            int(self.params.get("ema_span_fast_bars", 120)), external_data=external_data
        )
        ema_slow = self._calculate_ema_from_history(
            int(self.params.get("ema_span_slow_bars", 600)), external_data=external_data
        )

        if ema_fast is None or ema_slow is None:
            return Decimal(str(self.params.get("bias_neutral_target", "0.40")))

        if ema_fast > ema_slow:
            return Decimal(str(self.params.get("bias_high", "0.60")))
        if ema_fast < ema_slow:
            return Decimal(str(self.params.get("bias_low", "0.25")))
        return Decimal(str(self.params.get("bias_neutral_target", "0.40")))

    def _calculate_ema_from_history(
        self, span: int, external_data: Optional[pd.Series] = None
    ) -> Optional[Decimal]:
        """
        計算 EMA 指標
        """
        if external_data is not None:
            # 使用外部數據（回測模式）
            try:
                if len(external_data) < span:
                    return None
                ema_val = external_data.ewm(span=span, adjust=False).mean().iloc[-1]
                return Decimal(str(ema_val))
            except Exception:  # pragma: no cover - 防禦性
                return None

        # 使用內部歷史數據（實盤模式）
        if len(self.price_history) < span / 10 and len(self.price_history) < 10:
            return None
        prices = [p[1] for p in self.price_history]
        series = pd.Series(prices, dtype=float)
        try:
            ema_val = series.ewm(span=span, adjust=False).mean().iloc[-1]
            return Decimal(str(ema_val))
        except Exception:  # pragma: no cover
            return None

    def _calculate_atr_from_history(
        self,
        period: int = 14,
        external_high: Optional[pd.Series] = None,
        external_low: Optional[pd.Series] = None,
        external_close: Optional[pd.Series] = None,
    ) -> Optional[Decimal]:
        """
        計算 ATR 指標（簡化版）
        """
        if (
            external_high is not None
            and external_low is not None
            and external_close is not None
        ):
            # 使用外部數據（回測模式）
            try:
                if len(external_high) < period:
                    return None
                atr_series = atr(external_high, external_low, external_close, period)
                if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]):
                    return Decimal(str(atr_series.iloc[-1]))
                return None
            except Exception as e:  # pragma: no cover
                log.warning("Failed to calculate ATR from external data: %s", e)
                return None

        # 使用內部歷史數據（實盤模式，簡化版）
        if len(self.price_history) < period:
            return None
        try:
            prices = [p[1] for p in self.price_history]
            series = pd.Series(prices, dtype=float)
            high_low = series.rolling(window=period, min_periods=period).max() - series.rolling(
                window=period, min_periods=period
            ).min()
            atr_series = high_low.rolling(window=period, min_periods=period).mean()
            if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]):
                return Decimal(str(atr_series.iloc[-1]))
            return None
        except Exception as e:  # pragma: no cover
            log.warning("Failed to calculate ATR: %s", e)
            return None

    def _calculate_adx_from_history(
        self,
        period: int = 14,
        external_high: Optional[pd.Series] = None,
        external_low: Optional[pd.Series] = None,
        external_close: Optional[pd.Series] = None,
    ) -> Optional[Decimal]:
        """
        計算 ADX 指標
        """
        if (
            external_high is not None
            and external_low is not None
            and external_close is not None
        ):
            try:
                if len(external_high) < period * 2:
                    return None
                adx_series, _, _ = adx(external_high, external_low, external_close, period)
                if len(adx_series) > 0 and not pd.isna(adx_series.iloc[-1]):
                    return Decimal(str(adx_series.iloc[-1]))
                return None
            except Exception as e:  # pragma: no cover
                log.warning("Failed to calculate ADX from external data: %s", e)
                return None

        if len(self.price_history) < period * 2:
            return None
        try:
            prices = [p[1] for p in self.price_history]
            series = pd.Series(prices, dtype=float)
            price_changes = series.diff().abs()
            avg_change = price_changes.rolling(window=period, min_periods=period).mean()
            price_range = series.rolling(window=period, min_periods=period).max() - series.rolling(
                window=period, min_periods=period
            ).min()
            if len(avg_change) > 0 and price_range.iloc[-1] > 0:
                adx_approx = (avg_change.iloc[-1] / price_range.iloc[-1]) * 100
                return Decimal(str(min(max(adx_approx, 0), 100)))
            return None
        except Exception as e:  # pragma: no cover
            log.warning("Failed to calculate ADX: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # 量化工具
    # ------------------------------------------------------------------ #
    def quantize_price(self, price: Decimal) -> Decimal:
        """價格精度量化"""
        return price.quantize(self.price_precision, rounding=getcontext().rounding)

    def quantize_qty(self, qty: Decimal) -> Decimal:
        """數量精度量化"""
        return qty.quantize(self.qty_precision, rounding="ROUND_DOWN")


