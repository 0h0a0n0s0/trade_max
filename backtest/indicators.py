# indicators.py (V7.0 最終版)
import pandas as pd
import numpy as np
from typing import Tuple

# ---------- 基礎 ----------
def ema(series: pd.Series, span: int, adjust: bool = False) -> pd.Series:
    """計算指數移動平均線 - V7.0: 返回完整序列"""
    if series.empty or len(series) < 1:
        return pd.Series([0.0] * len(series), index=series.index)
    return series.ewm(span=span, adjust=adjust, min_periods=1).mean().ffill().fillna(0.0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """計算MACD指標 - V7.0: 返回三個完整的序列"""
    if series.empty or len(series) < slow:
        default_series = pd.Series([0.0] * len(series), index=series.index)
        return default_series, default_series, default_series
    ema_f = series.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_s = series.ewm(span=slow, adjust=False, min_periods=1).mean()
    diff = ema_f - ema_s
    dea = diff.ewm(span=signal, adjust=False, min_periods=1).mean()
    hist = diff - dea
    return diff.ffill().fillna(0.0), dea.ffill().fillna(0.0), hist.ffill().fillna(0.0)

def bollinger(series: pd.Series, window: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """計算布林帶指標 - V7.0: 返回完整序列"""
    if series.empty or len(series) < 2:
        return series, series, series
    ma = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std(ddof=0)
    upper_band = ma + k * std
    lower_band = ma - k * std
    return upper_band.ffill().fillna(series), ma.ffill().fillna(series), lower_band.ffill().fillna(series)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """計算相對強弱指數 (RSI) - V7.0: 返回完整序列"""
    if series.empty or len(series) < period + 1:
        return pd.Series([50.0] * len(series), index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    safe_avg_loss = avg_loss.replace(0, 1e-9)
    rs = avg_gain / safe_avg_loss
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    rsi_series.loc[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    rsi_series.loc[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return rsi_series.ffill().fillna(50.0)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """計算平均真實波幅 (ATR) - V7.0: 返回完整序列"""
    if high.empty or len(high) < period:
        return pd.Series([0.0] * len(high), index=high.index)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr_series = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    tr_series.iloc[0] = high.iloc[0] - low.iloc[0]
    atr_series = tr_series.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    return atr_series.ffill().fillna(0.0)

# ---------- 進階 ----------
def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """計算平均方向指數 (ADX), +DI, -DI - V7.0: 返回完整序列"""
    if high.empty or len(high) < period * 2:
        default_series = pd.Series([20.0] * len(high), index=high.index)
        return default_series, default_series, default_series
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    atr_values = tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    move_up = high.diff(1)
    move_down = -low.diff(1)
    plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0), index=high.index)
    plus_dm_smooth = plus_dm.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    safe_atr = atr_values.replace(0, 1e-9)
    plus_di = 100 * (plus_dm_smooth / safe_atr)
    minus_di = 100 * (minus_dm_smooth / safe_atr)
    di_sum = (plus_di + minus_di).replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / di_sum)
    adx_series = dx.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    return adx_series.ffill().fillna(20.0), plus_di.ffill().fillna(20.0), minus_di.ffill().fillna(20.0)

def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """計算隨機震盪指標 (%K, %D) - V7.0: 返回完整序列"""
    if high.empty or len(high) < k_period:
        default_series = pd.Series([50.0] * len(high), index=high.index)
        return default_series, default_series
    lowest_low_k = low.rolling(window=k_period, min_periods=1).min()
    highest_high_k = high.rolling(window=k_period, min_periods=1).max()
    denominator_k = (highest_high_k - lowest_low_k).replace(0, 1e-9)
    percent_k = 100 * ((close - lowest_low_k) / denominator_k)
    percent_k.replace([np.inf, -np.inf], np.nan, inplace=True)
    percent_k.fillna(50.0, inplace=True)
    percent_d = percent_k.rolling(window=d_period, min_periods=1).mean()
    percent_d.fillna(50.0, inplace=True)
    return percent_k, percent_d

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """計算能量潮指標 (On-Balance Volume, OBV) - (此函式本來就返回序列，無需修改)"""
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()