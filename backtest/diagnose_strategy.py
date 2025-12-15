#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略診斷分析工具
運行樣本回測，收集診斷數據，識別策略問題
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal
from typing import Dict, List
import yaml
import random
from datetime import datetime

import sys
from pathlib import Path

# 添加當前目錄到路徑以便導入
sys.path.insert(0, str(Path(__file__).parent))

from backtester_grid import Backtester

LOG = logging.getLogger("StrategyDiagnostic")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    level=logging.INFO
)


class StrategyDiagnostic:
    """策略診斷分析器"""
    
    def __init__(self, csv_path: Path, base_config_path: Path, init_usdt: float, init_twd: float):
        self.csv_path = csv_path
        self.base_config_path = base_config_path
        self.init_usdt = Decimal(str(init_usdt))
        self.init_twd = Decimal(str(init_twd))
        self.price_df = None
        self.base_config = None
        self.results = []
        
        self._load_data()
        self._load_base_config()
    
    def _load_data(self):
        """載入OHLC數據"""
        LOG.info(f"載入數據: {self.csv_path.name}...")
        try:
            temp_df = pd.read_csv(self.csv_path, usecols=['ts', 'high', 'low', 'close'])
            
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
            
            self.price_df = temp_df.set_index('ts')
            self.price_df['high'] = self.price_df['high'].astype(float)
            self.price_df['low'] = self.price_df['low'].astype(float)
            self.price_df['close'] = self.price_df['close'].astype(float)
            self.price_df.ffill(inplace=True)
            
            LOG.info(f"✓ 數據載入完成: {len(self.price_df):,} 根K線")
            LOG.info(f"  時間範圍: {self.price_df.index[0]} 至 {self.price_df.index[-1]}")
        except Exception as e:
            LOG.error(f"載入數據失敗: {e}", exc_info=True)
            raise
    
    def _load_base_config(self):
        """載入基礎配置"""
        try:
            with open(self.base_config_path, 'r', encoding='utf-8') as f:
                self.base_config = yaml.safe_load(f) or {}
        except Exception as e:
            LOG.error(f"載入配置失敗: {e}", exc_info=True)
            raise
    
    def _generate_sample_params(self, n: int = 20) -> List[Dict]:
        """生成樣本參數組合"""
        samples = []
        for i in range(n):
            params = self.base_config.copy()
            
            # 隨機生成參數（使用與優化器相同的範圍）
            params['small_gap'] = str(round(random.uniform(0.01, 0.10), 4))
            params['mid_mult'] = random.randint(2, 6)
            params['big_mult'] = random.randint(5, 15)
            params['size_pct_small'] = str(round(random.uniform(0.01, 0.05), 4))
            params['size_pct_mid'] = str(round(random.uniform(0.015, 0.06), 4))
            params['size_pct_big'] = str(round(random.uniform(0.02, 0.08), 4))
            params['ema_span_fast_bars'] = random.randint(30, 1200)
            params['ema_span_slow_bars'] = random.randint(600, 8000)
            params['bias_high'] = str(round(random.uniform(0.50, 0.90), 3))
            params['bias_low'] = str(round(random.uniform(0.05, 0.50), 3))
            params['bias_neutral_target'] = str(round(random.uniform(0.30, 0.60), 3))
            
            # 確保必要參數存在
            if 'macd_fast_period' not in params:
                params['macd_fast_period'] = 12
            if 'macd_slow_period' not in params:
                params['macd_slow_period'] = 26
            if 'macd_signal_period' not in params:
                params['macd_signal_period'] = 9
            if 'dmi_period' not in params:
                params['dmi_period'] = 14
            if 'grid_aggression_threshold' not in params:
                params['grid_aggression_threshold'] = 20
            if 'grid_aggression_multiplier' not in params:
                params['grid_aggression_multiplier'] = '1.0'
            if 'use_hybrid_model' not in params:
                params['use_hybrid_model'] = True
            
            samples.append(params)
        
        return samples
    
    def run_diagnosis(self, n_samples: int = 20) -> pd.DataFrame:
        """運行診斷分析"""
        LOG.info("=" * 80)
        LOG.info("開始策略診斷分析")
        LOG.info(f"樣本數量: {n_samples}")
        LOG.info("=" * 80)
        
        sample_params = self._generate_sample_params(n_samples)
        
        for i, params in enumerate(sample_params, 1):
            LOG.info(f"\n[{i}/{n_samples}] 運行樣本回測...")
            try:
                backtester = Backtester(params, self.init_usdt, self.init_twd, verbose=False)
                stats = backtester.run(self.price_df, collect_diagnostics=True)
                
                # 添加參數信息
                result = {
                    'sample_id': i,
                    'small_gap': float(params['small_gap']),
                    'mid_mult': params['mid_mult'],
                    'big_mult': params['big_mult'],
                    'size_pct_small': float(params['size_pct_small']),
                    'ema_span_fast': params['ema_span_fast_bars'],
                    'ema_span_slow': params['ema_span_slow_bars'],
                    **stats  # 包含所有回測統計和診斷數據
                }
                
                self.results.append(result)
                
                LOG.info(f"  ROI: {stats['roi_pct']:.2f}% | "
                        f"Max DD: {stats['max_drawdown_pct']:.2f}% | "
                        f"交易次數: {stats['total_trades']} | "
                        f"手續費: {stats.get('total_fee_cost', 0):.2f} TWD")
                
            except Exception as e:
                LOG.error(f"  樣本 {i} 回測失敗: {e}")
                continue
        
        # 轉換為DataFrame
        if not self.results:
            LOG.warning("沒有成功完成的回測樣本")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        return df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """分析診斷結果"""
        if results_df.empty:
            return {}
        
        analysis = {}
        
        # 基本統計
        analysis['avg_roi'] = results_df['roi_pct'].mean()
        analysis['median_roi'] = results_df['roi_pct'].median()
        analysis['min_roi'] = results_df['roi_pct'].min()
        analysis['max_roi'] = results_df['roi_pct'].max()
        analysis['positive_roi_count'] = (results_df['roi_pct'] > 0).sum()
        analysis['positive_roi_pct'] = (results_df['roi_pct'] > 0).sum() / len(results_df) * 100
        
        analysis['avg_max_dd'] = results_df['max_drawdown_pct'].mean()
        analysis['max_dd_max'] = results_df['max_drawdown_pct'].max()
        
        # 手續費分析
        if 'total_fee_cost' in results_df.columns:
            analysis['avg_fee_cost'] = results_df['total_fee_cost'].mean()
            analysis['total_fee_cost_max'] = results_df['total_fee_cost'].max()
            analysis['avg_fee_to_profit_ratio'] = results_df['fee_to_profit_ratio'].replace([np.inf, -np.inf], np.nan).mean()
        
        # 交易頻率分析
        if 'total_trades' in results_df.columns:
            analysis['avg_trades'] = results_df['total_trades'].mean()
            analysis['avg_profit_per_trade'] = results_df.get('avg_profit_per_trade', pd.Series([0])).mean()
        
        # 網格成交率分析
        if 'grid_fill_rate' in results_df.columns:
            analysis['avg_grid_fill_rate'] = results_df['grid_fill_rate'].mean()
            analysis['avg_grid_fills'] = results_df['grid_fills'].mean()
        
        # 趨勢模式分析
        if 'trend_entries' in results_df.columns:
            analysis['avg_trend_entries'] = results_df['trend_entries'].mean()
            analysis['avg_trend_exits'] = results_df['trend_exits'].mean()
        
        # 價格波動分析
        if 'price_range_pct' in results_df.columns:
            analysis['avg_price_range_pct'] = results_df['price_range_pct'].mean()
        
        return analysis
    
    def generate_report(self, results_df: pd.DataFrame, analysis: Dict, output_path: Path):
        """生成診斷報告"""
        LOG.info("\n" + "=" * 80)
        LOG.info("診斷報告")
        LOG.info("=" * 80)
        
        # 基本統計
        LOG.info(f"\n📊 基本統計:")
        LOG.info(f"  平均 ROI: {analysis.get('avg_roi', 0):.2f}%")
        LOG.info(f"  中位數 ROI: {analysis.get('median_roi', 0):.2f}%")
        LOG.info(f"  ROI 範圍: {analysis.get('min_roi', 0):.2f}% ~ {analysis.get('max_roi', 0):.2f}%")
        LOG.info(f"  盈利樣本: {analysis.get('positive_roi_count', 0)}/{len(results_df)} ({analysis.get('positive_roi_pct', 0):.1f}%)")
        
        LOG.info(f"\n📉 風險指標:")
        LOG.info(f"  平均最大回撤: {analysis.get('avg_max_dd', 0):.2f}%")
        LOG.info(f"  最大回撤: {analysis.get('max_dd_max', 0):.2f}%")
        
        # 手續費分析
        if 'avg_fee_cost' in analysis:
            LOG.info(f"\n💰 手續費分析:")
            LOG.info(f"  平均手續費成本: {analysis['avg_fee_cost']:,.2f} TWD")
            LOG.info(f"  最大手續費成本: {analysis.get('total_fee_cost_max', 0):,.2f} TWD")
            if 'avg_fee_to_profit_ratio' in analysis and not pd.isna(analysis['avg_fee_to_profit_ratio']):
                LOG.info(f"  手續費/利潤比: {analysis['avg_fee_to_profit_ratio']:.2f}")
                if analysis['avg_fee_to_profit_ratio'] > 1.0:
                    LOG.warning("  ⚠️  手續費超過利潤！這是主要問題！")
        
        # 交易分析
        if 'avg_trades' in analysis:
            LOG.info(f"\n📈 交易分析:")
            LOG.info(f"  平均交易次數: {analysis['avg_trades']:.0f}")
            LOG.info(f"  平均每筆利潤: {analysis.get('avg_profit_per_trade', 0):.2f} TWD")
        
        # 網格分析
        if 'avg_grid_fill_rate' in analysis:
            LOG.info(f"\n🔲 網格分析:")
            LOG.info(f"  平均網格成交率: {analysis['avg_grid_fill_rate']:.2%}")
            LOG.info(f"  平均網格成交次數: {analysis.get('avg_grid_fills', 0):.0f}")
        
        # 趨勢模式分析
        if 'avg_trend_entries' in analysis:
            LOG.info(f"\n📊 趨勢模式分析:")
            LOG.info(f"  平均趨勢進場次數: {analysis['avg_trend_entries']:.1f}")
            LOG.info(f"  平均趨勢出場次數: {analysis.get('avg_trend_exits', 0):.1f}")
        
        # 問題診斷
        LOG.info(f"\n🔍 問題診斷:")
        issues = []
        
        if analysis.get('positive_roi_pct', 0) < 10:
            issues.append("❌ 盈利樣本比例極低 (<10%)，策略可能不適合當前市場環境")
        
        if analysis.get('avg_fee_to_profit_ratio', 0) > 1.0:
            issues.append("❌ 手續費成本超過利潤，需要降低交易頻率或使用maker訂單")
        
        if analysis.get('avg_grid_fill_rate', 0) < 0.1:
            issues.append("⚠️  網格成交率過低 (<10%)，網格間距可能太大")
        
        if analysis.get('avg_grid_fill_rate', 0) > 0.9:
            issues.append("⚠️  網格成交率過高 (>90%)，網格間距可能太小，導致頻繁交易")
        
        if analysis.get('avg_max_dd', 0) > 20:
            issues.append("⚠️  平均最大回撤過高 (>20%)，風險控制需要加強")
        
        if not issues:
            issues.append("✓ 未發現明顯問題，建議進一步擴大樣本數量或調整參數範圍")
        
        for issue in issues:
            LOG.info(f"  {issue}")
        
        # 保存CSV報告
        results_df.to_csv(output_path / 'diagnosis_results.csv', index=False, encoding='utf-8-sig')
        LOG.info(f"\n💾 詳細結果已保存至: {output_path / 'diagnosis_results.csv'}")
        
        return issues


def main():
    parser = argparse.ArgumentParser(description="策略診斷分析工具")
    parser.add_argument("--csv", required=True, type=Path, help="OHLC CSV文件路徑")
    parser.add_argument("--config", default="config_usdttwd.yaml", type=Path, help="配置檔案路徑")
    parser.add_argument("--init_usdt", default=10000.0, type=float, help="初始USDT餘額")
    parser.add_argument("--init_twd", default=300000.0, type=float, help="初始TWD餘額")
    parser.add_argument("--samples", default=20, type=int, help="樣本數量")
    parser.add_argument("--output", default=".", type=Path, help="輸出目錄")
    
    args = parser.parse_args()
    
    if not args.csv.exists():
        LOG.error(f"CSV文件不存在: {args.csv}")
        return
    
    if not args.config.exists():
        LOG.error(f"配置檔案不存在: {args.config}")
        return
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 運行診斷
    diagnostic = StrategyDiagnostic(
        csv_path=args.csv,
        base_config_path=args.config,
        init_usdt=args.init_usdt,
        init_twd=args.init_twd
    )
    
    results_df = diagnostic.run_diagnosis(n_samples=args.samples)
    
    if results_df.empty:
        LOG.error("診斷失敗：沒有成功完成的回測")
        return
    
    # 分析結果
    analysis = diagnostic.analyze_results(results_df)
    
    # 生成報告
    issues = diagnostic.generate_report(results_df, analysis, args.output)
    
    LOG.info("\n" + "=" * 80)
    LOG.info("診斷完成")
    LOG.info("=" * 80)


if __name__ == "__main__":
    main()

