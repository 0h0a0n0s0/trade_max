#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市場環境分析工具
分析K線數據特徵，判斷市場是否適合網格交易策略
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from indicators import ema, adx, atr, rsi

LOG = logging.getLogger("MarketAnalyzer")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    level=logging.INFO
)


class MarketAnalyzer:
    """市場環境分析器"""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.df = None
        self.analysis = {}
        
        self._load_data()
    
    def _load_data(self):
        """載入OHLC數據"""
        LOG.info(f"載入數據: {self.csv_path.name}...")
        try:
            temp_df = pd.read_csv(self.csv_path, usecols=['ts', 'open', 'high', 'low', 'close'])
            
            # 處理時間戳
            if pd.api.types.is_numeric_dtype(temp_df['ts']):
                try:
                    tss = pd.to_datetime(temp_df['ts'], unit='ms')
                    if tss.min().year < 2000:
                        raise ValueError("ts likely in seconds, not milliseconds.")
                except (ValueError, pd.errors.OutOfBoundsDatetime):
                    LOG.warning("無法解析毫秒時間戳，嘗試秒...")
                    tss = pd.to_datetime(temp_df['ts'], unit='s')
                temp_df['ts'] = tss
            else:
                temp_df['ts'] = pd.to_datetime(temp_df['ts'])
            
            self.df = temp_df.set_index('ts').sort_index()
            self.df['high'] = self.df['high'].astype(float)
            self.df['low'] = self.df['low'].astype(float)
            self.df['close'] = self.df['close'].astype(float)
            if 'open' in self.df.columns:
                self.df['open'] = self.df['open'].astype(float)
            else:
                self.df['open'] = self.df['close']  # 如果沒有open，使用close
            
            self.df.ffill(inplace=True)
            
            LOG.info(f"✓ 數據載入完成: {len(self.df):,} 根K線")
            LOG.info(f"  時間範圍: {self.df.index[0]} 至 {self.df.index[-1]}")
        except Exception as e:
            LOG.error(f"載入數據失敗: {e}", exc_info=True)
            raise
    
    def analyze_price_trend(self) -> Dict:
        """分析價格趨勢"""
        LOG.info("分析價格趨勢...")
        
        close = self.df['close']
        initial_price = close.iloc[0]
        final_price = close.iloc[-1]
        price_change_pct = ((final_price - initial_price) / initial_price) * 100
        
        # 計算EMA判斷趨勢
        ema_fast = ema(close, span=60)
        ema_slow = ema(close, span=300)
        
        # 統計EMA交叉次數
        ema_crosses = 0
        prev_fast_above = None
        for i in range(1, len(close)):
            fast_above = ema_fast.iloc[i] > ema_slow.iloc[i]
            if prev_fast_above is not None and fast_above != prev_fast_above:
                ema_crosses += 1
            prev_fast_above = fast_above
        
        # 計算趨勢持續時間
        trend_duration = []
        current_trend = None
        trend_start = 0
        for i in range(len(close)):
            is_uptrend = ema_fast.iloc[i] > ema_slow.iloc[i]
            if current_trend != is_uptrend:
                if current_trend is not None:
                    trend_duration.append(i - trend_start)
                current_trend = is_uptrend
                trend_start = i
        if current_trend is not None:
            trend_duration.append(len(close) - trend_start)
        
        avg_trend_duration = np.mean(trend_duration) if trend_duration else 0
        
        return {
            'initial_price': float(initial_price),
            'final_price': float(final_price),
            'price_change_pct': float(price_change_pct),
            'is_uptrend': price_change_pct > 0,
            'ema_crosses': ema_crosses,
            'avg_trend_duration': float(avg_trend_duration),
            'trend_type': '上漲' if price_change_pct > 5 else ('下跌' if price_change_pct < -5 else '震盪')
        }
    
    def analyze_volatility(self) -> Dict:
        """分析波動率"""
        LOG.info("分析波動率...")
        
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        
        # 計算ATR
        atr_series = atr(high, low, close, period=14)
        avg_atr = atr_series.mean()
        atr_pct = (avg_atr / close.mean()) * 100
        
        # 計算日內波動
        daily_range = (high - low) / close
        avg_daily_range = daily_range.mean() * 100
        
        # 計算價格波動範圍
        price_range = (close.max() - close.min()) / close.min() * 100
        
        # 計算連續上漲/下跌天數
        returns = close.pct_change()
        consecutive_up = 0
        consecutive_down = 0
        max_consecutive_up = 0
        max_consecutive_down = 0
        
        for ret in returns:
            if pd.isna(ret):
                continue
            if ret > 0:
                consecutive_up += 1
                consecutive_down = 0
                max_consecutive_up = max(max_consecutive_up, consecutive_up)
            elif ret < 0:
                consecutive_down += 1
                consecutive_up = 0
                max_consecutive_down = max(max_consecutive_down, consecutive_down)
            else:
                consecutive_up = 0
                consecutive_down = 0
        
        return {
            'avg_atr': float(avg_atr),
            'atr_pct': float(atr_pct),
            'avg_daily_range_pct': float(avg_daily_range),
            'price_range_pct': float(price_range),
            'max_consecutive_up': int(max_consecutive_up),
            'max_consecutive_down': int(max_consecutive_down),
            'volatility_level': '高' if atr_pct > 1.0 else ('中' if atr_pct > 0.5 else '低')
        }
    
    def analyze_trend_strength(self) -> Dict:
        """分析趨勢強度"""
        LOG.info("分析趨勢強度...")
        
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        # 計算ADX
        adx_series, plus_di, minus_di = adx(high, low, close, period=14)
        avg_adx = adx_series.mean()
        
        # 統計強趨勢時段比例
        strong_trend_pct = (adx_series > 25).sum() / len(adx_series) * 100
        weak_trend_pct = (adx_series < 20).sum() / len(adx_series) * 100
        
        # 計算RSI
        rsi_series = rsi(close, period=14)
        avg_rsi = rsi_series.mean()
        overbought_pct = (rsi_series > 70).sum() / len(rsi_series) * 100
        oversold_pct = (rsi_series < 30).sum() / len(rsi_series) * 100
        
        return {
            'avg_adx': float(avg_adx),
            'strong_trend_pct': float(strong_trend_pct),
            'weak_trend_pct': float(weak_trend_pct),
            'avg_rsi': float(avg_rsi),
            'overbought_pct': float(overbought_pct),
            'oversold_pct': float(oversold_pct),
            'market_type': '強趨勢' if avg_adx > 25 else ('弱趨勢' if avg_adx < 20 else '震盪')
        }
    
    def analyze_grid_suitability(self) -> Dict:
        """分析網格交易適合度"""
        LOG.info("分析網格交易適合度...")
        
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        
        # 計算適合網格交易的條件
        atr_series = atr(high, low, close, period=14)
        adx_series, _, _ = adx(high, low, close, period=14)
        
        # 理想網格條件：
        # 1. 波動率適中（ATR在0.5%-2%之間）
        atr_pct = (atr_series / close) * 100
        suitable_volatility = ((atr_pct > 0.5) & (atr_pct < 2.0)).sum() / len(atr_series) * 100
        
        # 2. 弱趨勢或震盪市場（ADX < 25）
        suitable_trend = (adx_series < 25).sum() / len(adx_series) * 100
        
        # 3. 價格在區間內震盪（計算價格在布林帶內的時間）
        from indicators import bollinger
        upper, middle, lower = bollinger(close, window=20, k=2.0)
        in_band = ((close >= lower) & (close <= upper)).sum() / len(close) * 100
        
        # 綜合評分
        suitability_score = (suitable_volatility * 0.4 + suitable_trend * 0.4 + in_band * 0.2)
        
        return {
            'suitable_volatility_pct': float(suitable_volatility),
            'suitable_trend_pct': float(suitable_trend),
            'price_in_band_pct': float(in_band),
            'suitability_score': float(suitability_score),
            'suitability_level': '適合' if suitability_score > 60 else ('一般' if suitability_score > 40 else '不適合')
        }
    
    def run_full_analysis(self) -> Dict:
        """運行完整分析"""
        LOG.info("=" * 80)
        LOG.info("開始市場環境分析")
        LOG.info("=" * 80)
        
        self.analysis = {
            'data_info': {
                'total_bars': len(self.df),
                'start_date': str(self.df.index[0]),
                'end_date': str(self.df.index[-1]),
                'timeframe': '1分鐘' if (self.df.index[1] - self.df.index[0]).total_seconds() == 60 else '未知'
            },
            'price_trend': self.analyze_price_trend(),
            'volatility': self.analyze_volatility(),
            'trend_strength': self.analyze_trend_strength(),
            'grid_suitability': self.analyze_grid_suitability()
        }
        
        return self.analysis
    
    def generate_report(self, output_path: Path):
        """生成分析報告"""
        LOG.info("\n" + "=" * 80)
        LOG.info("市場環境分析報告")
        LOG.info("=" * 80)
        
        # 數據信息
        data_info = self.analysis['data_info']
        LOG.info(f"\n📊 數據信息:")
        LOG.info(f"  總K線數: {data_info['total_bars']:,}")
        LOG.info(f"  時間範圍: {data_info['start_date']} 至 {data_info['end_date']}")
        LOG.info(f"  K線週期: {data_info['timeframe']}")
        
        # 價格趨勢
        trend = self.analysis['price_trend']
        LOG.info(f"\n📈 價格趨勢:")
        LOG.info(f"  初始價格: {trend['initial_price']:.3f} TWD")
        LOG.info(f"  最終價格: {trend['final_price']:.3f} TWD")
        LOG.info(f"  價格變化: {trend['price_change_pct']:.2f}%")
        LOG.info(f"  趨勢類型: {trend['trend_type']}")
        LOG.info(f"  EMA交叉次數: {trend['ema_crosses']}")
        LOG.info(f"  平均趨勢持續時間: {trend['avg_trend_duration']:.0f} 根K線")
        
        # 波動率
        vol = self.analysis['volatility']
        LOG.info(f"\n📊 波動率分析:")
        LOG.info(f"  平均ATR: {vol['avg_atr']:.3f} TWD ({vol['atr_pct']:.2f}%)")
        LOG.info(f"  平均日內波動: {vol['avg_daily_range_pct']:.2f}%")
        LOG.info(f"  價格波動範圍: {vol['price_range_pct']:.2f}%")
        LOG.info(f"  波動率水平: {vol['volatility_level']}")
        LOG.info(f"  最大連續上漲: {vol['max_consecutive_up']} 根K線")
        LOG.info(f"  最大連續下跌: {vol['max_consecutive_down']} 根K線")
        
        # 趨勢強度
        strength = self.analysis['trend_strength']
        LOG.info(f"\n💪 趨勢強度:")
        LOG.info(f"  平均ADX: {strength['avg_adx']:.2f}")
        LOG.info(f"  強趨勢時段: {strength['strong_trend_pct']:.1f}%")
        LOG.info(f"  弱趨勢時段: {strength['weak_trend_pct']:.1f}%")
        LOG.info(f"  市場類型: {strength['market_type']}")
        LOG.info(f"  平均RSI: {strength['avg_rsi']:.1f}")
        LOG.info(f"  超買時段: {strength['overbought_pct']:.1f}%")
        LOG.info(f"  超賣時段: {strength['oversold_pct']:.1f}%")
        
        # 網格適合度
        suitability = self.analysis['grid_suitability']
        LOG.info(f"\n🎯 網格交易適合度:")
        LOG.info(f"  適合波動率時段: {suitability['suitable_volatility_pct']:.1f}%")
        LOG.info(f"  適合趨勢時段: {suitability['suitable_trend_pct']:.1f}%")
        LOG.info(f"  價格在區間內: {suitability['price_in_band_pct']:.1f}%")
        LOG.info(f"  綜合適合度評分: {suitability['suitability_score']:.1f}/100")
        LOG.info(f"  適合度等級: {suitability['suitability_level']}")
        
        # 建議
        LOG.info(f"\n💡 策略建議:")
        suggestions = []
        
        if suitability['suitability_score'] < 40:
            suggestions.append("⚠️  市場環境不太適合網格交易，建議：")
            suggestions.append("   - 考慮使用趨勢跟隨策略")
            suggestions.append("   - 或等待更適合的市場環境")
        elif suitability['suitability_score'] < 60:
            suggestions.append("⚠️  市場環境一般適合網格交易，建議：")
            suggestions.append("   - 使用較大的網格間距")
            suggestions.append("   - 啟用混合模式（趨勢+網格）")
        else:
            suggestions.append("✓ 市場環境適合網格交易")
        
        if vol['atr_pct'] < 0.3:
            suggestions.append("⚠️  波動率過低，網格間距應設置較小")
        elif vol['atr_pct'] > 2.0:
            suggestions.append("⚠️  波動率過高，網格間距應設置較大，並加強風險控制")
        
        if strength['avg_adx'] > 30:
            suggestions.append("⚠️  強趨勢市場，建議啟用混合模式，在強趨勢時使用趨勢跟隨")
        
        if not suggestions:
            suggestions.append("✓ 未發現明顯問題")
        
        for suggestion in suggestions:
            LOG.info(f"  {suggestion}")
        
        # 保存報告
        import json
        report_path = output_path / 'market_analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis, f, indent=2, ensure_ascii=False, default=str)
        LOG.info(f"\n💾 詳細報告已保存至: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="市場環境分析工具")
    parser.add_argument("--csv", required=True, type=Path, help="OHLC CSV文件路徑")
    parser.add_argument("--output", default=".", type=Path, help="輸出目錄")
    
    args = parser.parse_args()
    
    if not args.csv.exists():
        LOG.error(f"CSV文件不存在: {args.csv}")
        return
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 運行分析
    analyzer = MarketAnalyzer(args.csv)
    analyzer.run_full_analysis()
    analyzer.generate_report(args.output)
    
    LOG.info("\n" + "=" * 80)
    LOG.info("分析完成")
    LOG.info("=" * 80)


if __name__ == "__main__":
    main()

