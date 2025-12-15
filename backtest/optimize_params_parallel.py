# optimize_params_parallel.py
"""
並行版本的參數優化腳本（針對 Mac M1 優化）
使用 multiprocessing 實現真正的並行執行
"""
import argparse
import csv
import logging
import random
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional
from multiprocessing import Pool, Manager
import pandas as pd
import yaml
import time

import sys
from pathlib import Path

# 添加父目錄到路徑以導入 GridStrategy
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 嘗試兩種導入方式（支持從根目錄或 backtest 目錄運行）
try:
    from backtest.backtest_adapter import BacktestAdapter
except ImportError:
    from backtest_adapter import BacktestAdapter

from strategy_usdttwd_grid_refactored import GridStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger("ParamOptimizerParallel")


def run_single_backtest(args_tuple):
    """單個回測任務（用於多進程）- 使用 BacktestAdapter 確保邏輯一致性"""
    params, init_usdt, init_twd, csv_path = args_tuple
    
    try:
        # 在每個進程中重新載入數據（避免序列化問題）
        temp_df = pd.read_csv(csv_path, usecols=['ts', 'high', 'low', 'close'])
        
        # Handle timestamp
        if pd.api.types.is_numeric_dtype(temp_df['ts']):
            try:
                tss = pd.to_datetime(temp_df['ts'], unit='ms')
                if tss.min().year < 2000:
                    raise ValueError("ts likely in seconds, not milliseconds.")
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                tss = pd.to_datetime(temp_df['ts'], unit='s')
            temp_df['ts'] = tss
        else:
            temp_df['ts'] = pd.to_datetime(temp_df['ts'])
        
        price_df = temp_df.set_index('ts')
        price_df['high'] = price_df['high'].astype(float)
        price_df['low'] = price_df['low'].astype(float)
        price_df['close'] = price_df['close'].astype(float)
        price_df.ffill(inplace=True)
        
        # 使用 BacktestAdapter 和 GridStrategy（與實盤相同的邏輯）
        strategy = GridStrategy(params)
        adapter = BacktestAdapter(
            strategy=strategy,
            init_usdt=Decimal(str(init_usdt)),
            init_twd=Decimal(str(init_twd)),
            fee_rate=Decimal(str(params.get('taker_fee', '0.0004'))),
            verbose=False
        )
        
        stats = adapter.run(price_df)
        
        # 計算 Robustness Score（穩健性分數）
        # Formula: score = roi_pct * 0.4 + (100 / (max_drawdown_pct + 1)) * 0.6
        roi_pct = stats['roi_pct']
        max_dd_pct = stats['max_drawdown_pct']
        robustness_score = roi_pct * 0.4 + (100 / (max_dd_pct + 1)) * 0.6
        
        stats['robustness_score'] = robustness_score
        
        # 使用 Robustness Score 和基本閾值進行篩選
        # 要求：ROI > 0.5%, MaxDD < 15%, Robustness Score > 10
        if stats['roi_pct'] > 0.5 and stats['max_drawdown_pct'] < 15.0 and robustness_score > 10.0:
            return {
                'params': params,
                'stats': stats,
                'success': True
            }
    except Exception as e:
        LOG.warning(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
    
    return {'success': False}


class ParameterOptimizerParallel:
    """並行版本的參數優化器"""
    
    def __init__(self, csv_path: Path, base_config_path: Path, 
                 init_usdt: float = 10000.0, init_twd: float = 300000.0,
                 num_workers: int = 4):
        self.csv_path = csv_path
        self.base_config_path = base_config_path
        self.init_usdt = Decimal(str(init_usdt))
        self.init_twd = Decimal(str(init_twd))
        self.num_workers = num_workers
        self.base_config = {}
        self.price_df = None
        self.valid_results = []
        self.iteration_count = 0
        self.max_iterations = 2000
        self.target_valid_sets = 100
        
        self._load_data()
        self._load_base_config()
    
    def _load_data(self):
        """Load OHLC data from CSV"""
        print(f"📂 載入數據: {self.csv_path.name}...")
        try:
            temp_df = pd.read_csv(self.csv_path, usecols=['ts', 'high', 'low', 'close'])
            
            # Handle timestamp
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
            
            self.price_df = temp_df.set_index('ts')
            self.price_df['high'] = self.price_df['high'].astype(float)
            self.price_df['low'] = self.price_df['low'].astype(float)
            self.price_df['close'] = self.price_df['close'].astype(float)
            self.price_df.ffill(inplace=True)
            
            print(f"   ✓ 數據載入完成: {len(self.price_df):,} 根K線")
        except Exception as e:
            LOG.error(f"Failed to load data: {e}", exc_info=True)
            raise
    
    def _load_base_config(self):
        """Load base configuration"""
        try:
            with open(self.base_config_path, 'r', encoding='utf-8') as f:
                self.base_config = yaml.safe_load(f) or {}
            # Config loaded silently
        except Exception as e:
            LOG.error(f"Failed to load base config: {e}", exc_info=True)
            raise
    
    def _generate_random_params(self) -> Dict:
        """Generate a random set of parameters within defined ranges"""
        params = self.base_config.copy()
        
        # 方向1優化：重點優化趨勢跟隨參數，網格作為輔助
        # 策略調整：完全轉向趨勢跟隨，網格作為輔助
        
        # ATR動態網格乘數（網格作為輔助，間距可以稍大）
        if params.get('use_atr_spacing', False):
            params['atr_spacing_multiplier'] = str(round(random.uniform(0.5, 1.5), 3))  # 0.5-1.5（網格作為輔助）
        
        # 網格倍數（影響網格層級間距）
        params['mid_mult'] = random.randint(2, 5)  # 擴大：2-5
        params['big_mult'] = random.randint(5, 12)  # 擴大：5-12
        
        # 訂單大小（影響資金利用率和單筆利潤）
        # 找到的有效參數在0.029-0.068，需要提高以增加收益
        params['size_pct_small'] = str(round(random.uniform(0.03, 0.08), 4))  # 提高：0.03-0.08
        params['size_pct_mid'] = str(round(random.uniform(0.04, 0.10), 4))  # 提高：0.04-0.10
        params['size_pct_big'] = str(round(random.uniform(0.05, 0.12), 4))  # 提高：0.05-0.12
        
        # EMA參數（第六次優化：調整至業界標準）
        # 業界標準：快線12-50，慢線26-200
        params['ema_span_fast_bars'] = random.randint(12, 50)  # 業界標準：12-50
        params['ema_span_slow_bars'] = random.randint(26, 200)  # 業界標準：26-200
        
        # 趨勢偏好（影響倉位分配）
        params['bias_high'] = str(round(random.uniform(0.50, 0.80), 3))  # 擴大：0.50-0.80
        params['bias_low'] = str(round(random.uniform(0.10, 0.45), 3))  # 擴大：0.10-0.45
        params['bias_neutral_target'] = str(round(random.uniform(0.35, 0.60), 3))  # 擴大：0.35-0.60
        
        # small_gap保留作為備選（如果禁用ATR動態網格）
        params['small_gap'] = str(round(random.uniform(0.001, 0.01), 4))  # 0.001-0.01
        
        # Ensure required parameters
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
            params['use_hybrid_model'] = True  # 啟用混合模式
        if 'use_atr_spacing' not in params:
            params['use_atr_spacing'] = True  # 啟用ATR動態網格
        if 'use_adx_filter' not in params:
            params['use_adx_filter'] = False  # 方向1優化：禁用ADX過濾器，主要依靠趨勢跟隨
        if 'atr_spacing_multiplier' not in params:
            params['atr_spacing_multiplier'] = str(round(random.uniform(0.3, 1.5), 3))
        
        return params
    
    def _mutate_params(self, base_params: Dict, mutation_rate: float = 0.1) -> Dict:
        """Create a mutated variant of parameters"""
        params = base_params.copy()
        
        if 'small_gap' in params:
            val = float(params['small_gap'])
            params['small_gap'] = str(round(max(0.01, min(0.10, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 4))
        
        if 'size_pct_small' in params:
            val = float(params['size_pct_small'])
            params['size_pct_small'] = str(round(max(0.03, min(0.08, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 4))
        
        if 'size_pct_mid' in params:
            val = float(params['size_pct_mid'])
            params['size_pct_mid'] = str(round(max(0.04, min(0.10, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 4))
        
        if 'size_pct_big' in params:
            val = float(params['size_pct_big'])
            params['size_pct_big'] = str(round(max(0.05, min(0.12, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 4))
        
        if 'mid_mult' in params:
            params['mid_mult'] = max(2, min(5, params['mid_mult'] + random.randint(-1, 1)))
        
        if 'big_mult' in params:
            params['big_mult'] = max(5, min(12, params['big_mult'] + random.randint(-1, 1)))
        
        if 'ema_span_fast_bars' in params:
            change = int(params['ema_span_fast_bars'] * mutation_rate)
            params['ema_span_fast_bars'] = max(100, min(600, params['ema_span_fast_bars'] + random.randint(-change, change)))
        
        if 'ema_span_slow_bars' in params:
            change = int(params['ema_span_slow_bars'] * mutation_rate)
            params['ema_span_slow_bars'] = max(300, min(2000, params['ema_span_slow_bars'] + random.randint(-change, change)))
        
        # 趨勢跟隨倉位比例變異
        if 'trend_trade_equity_pct' in params:
            val = float(params['trend_trade_equity_pct'])
            params['trend_trade_equity_pct'] = str(round(max(0.6, min(0.85, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 3))
        
        # ADX趨勢進場門檻變異（第六次優化：進一步降低）
        if 'adx_strength_threshold' in params:
            params['adx_strength_threshold'] = max(6, min(12, params['adx_strength_threshold'] + random.randint(-1, 1)))
        
        # 多指標複合判斷參數變異（第六次優化：調整至業界標準）
        if 'rsi_period' in params:
            params['rsi_period'] = max(14, min(21, params['rsi_period'] + random.randint(-2, 2)))  # 業界標準：14-21
        if 'rsi_bull_threshold' in params:
            val = float(params['rsi_bull_threshold'])
            params['rsi_bull_threshold'] = round(max(50.0, min(60.0, val + random.uniform(-2.0, 2.0))), 1)  # 業界標準：50-60
        if 'rsi_bear_threshold' in params:
            val = float(params['rsi_bear_threshold'])
            params['rsi_bear_threshold'] = round(max(40.0, min(50.0, val + random.uniform(-2.0, 2.0))), 1)  # 業界標準：40-50
        if 'adx_min_threshold' in params:
            params['adx_min_threshold'] = max(5, min(10, params['adx_min_threshold'] + random.randint(-1, 1)))  # 極度放寬：5-10
        # 布林帶參數變異（第六次優化新增）
        if 'bollinger_window' in params:
            params['bollinger_window'] = max(18, min(22, params['bollinger_window'] + random.randint(-1, 1)))
        if 'bollinger_k' in params:
            val = float(params['bollinger_k'])
            params['bollinger_k'] = round(max(1.8, min(2.2, val + random.uniform(-0.1, 0.1))), 1)
        if 'bollinger_band_threshold' in params:
            val = float(params['bollinger_band_threshold'])
            params['bollinger_band_threshold'] = round(max(0.05, min(0.15, val + random.uniform(-0.02, 0.02))), 2)
        # 隨機震盪指標參數變異（第六次優化新增）
        if 'stochastic_k_period' in params:
            params['stochastic_k_period'] = max(12, min(16, params['stochastic_k_period'] + random.randint(-1, 1)))
        if 'stochastic_d_period' in params:
            params['stochastic_d_period'] = max(2, min(4, params['stochastic_d_period'] + random.randint(-1, 1)))
        if 'stochastic_oversold' in params:
            val = float(params['stochastic_oversold'])
            params['stochastic_oversold'] = round(max(25.0, min(35.0, val + random.uniform(-2.0, 2.0))), 1)
        if 'stochastic_overbought' in params:
            val = float(params['stochastic_overbought'])
            params['stochastic_overbought'] = round(max(65.0, min(75.0, val + random.uniform(-2.0, 2.0))), 1)
        
        # ATR動態網格乘數變異
        if 'atr_spacing_multiplier' in params:
            val = float(params['atr_spacing_multiplier'])
            params['atr_spacing_multiplier'] = str(round(max(0.3, min(1.5, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 3))
        
        # 網格層級數量變異
        if 'levels_each' in params:
            params['levels_each'] = max(16, min(24, params['levels_each'] + random.randint(-2, 2)))
        
        if 'bias_high' in params:
            val = float(params['bias_high'])
            params['bias_high'] = str(round(max(0.50, min(0.80, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 3))
        
        if 'bias_low' in params:
            val = float(params['bias_low'])
            params['bias_low'] = str(round(max(0.10, min(0.45, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 3))
        
        if 'bias_neutral_target' in params:
            val = float(params['bias_neutral_target'])
            params['bias_neutral_target'] = str(round(max(0.35, min(0.60, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 3))
        
        return params
    
    def optimize(self):
        """Run parallel optimization"""
        print("=" * 80)
        print("🚀 參數優化開始")
        print(f"   工作進程數: {self.num_workers}")
        print(f"   目標有效參數組數: {self.target_valid_sets}")
        print(f"   最大迭代次數: {self.max_iterations}")
        print(f"   篩選條件: ROI > 0.5% AND Max Drawdown < 15% AND Robustness Score > 10")
        print("=" * 80)
        
        start_time = time.time()
        batch_size = self.num_workers * 2  # Process in batches
        
        with Pool(processes=self.num_workers) as pool:
            while len(self.valid_results) < self.target_valid_sets and self.iteration_count < self.max_iterations:
                # Generate batch of parameters
                batch_params = []
                
                # Generate random parameters
                for _ in range(batch_size):
                    if self.iteration_count >= self.max_iterations:
                        break
                    params = self._generate_random_params()
                    batch_params.append((
                        params,
                        float(self.init_usdt),
                        float(self.init_twd),
                        str(self.csv_path)  # Pass CSV path instead of DataFrame
                    ))
                    self.iteration_count += 1
                
                # Run batch in parallel
                results = pool.map(run_single_backtest, batch_params)
                
                # Process results
                for result in results:
                    if result and result.get('success'):
                        self.valid_results.append(result)
                        stats = result['stats']
                        robustness = stats.get('robustness_score', 0)
                        print(f"✅ 找到有效參數 [{len(self.valid_results)}/{self.target_valid_sets}] | "
                              f"ROI: {stats['roi_pct']:.2f}% | Max DD: {stats['max_drawdown_pct']:.2f}% | "
                              f"Robustness: {robustness:.2f}")
                        
                        # Generate mutations for successful params
                        if len(self.valid_results) < self.target_valid_sets:
                            mutation_params = []
                            for i in range(5):
                                if self.iteration_count >= self.max_iterations:
                                    break
                                mutated = self._mutate_params(result['params'], mutation_rate=0.1)
                                mutation_params.append((
                                    mutated,
                                    float(self.init_usdt),
                                    float(self.init_twd),
                                    str(self.csv_path)
                                ))
                                self.iteration_count += 1
                            
                            # Run mutations in parallel
                            mut_results = pool.map(run_single_backtest, mutation_params)
                            for mut_result in mut_results:
                                if mut_result and mut_result.get('success'):
                                    self.valid_results.append(mut_result)
                                    mut_stats = mut_result['stats']
                                    mut_robustness = mut_stats.get('robustness_score', 0)
                                    print(f"   └─ 變異成功 | ROI: {mut_stats['roi_pct']:.2f}% | Max DD: {mut_stats['max_drawdown_pct']:.2f}% | Robustness: {mut_robustness:.2f}")
                
                # Progress update with progress bar
                elapsed = time.time() - start_time
                if self.iteration_count % batch_size == 0:
                    rate = self.iteration_count / elapsed if elapsed > 0 else 0
                    remaining = (self.max_iterations - self.iteration_count) / rate if rate > 0 else 0
                    progress_pct = (self.iteration_count / self.max_iterations) * 100
                    bar_length = 50
                    filled = int(bar_length * self.iteration_count / self.max_iterations)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r進度: [{bar}] {progress_pct:.1f}% | "
                          f"迭代: {self.iteration_count}/{self.max_iterations} | "
                          f"有效: {len(self.valid_results)} | "
                          f"速度: {rate:.1f} iter/s | "
                          f"剩餘: {remaining/60:.1f} min", end='', flush=True)
        
        print()  # New line after progress bar
        total_time = time.time() - start_time
        print("=" * 80)
        print("✅ 優化完成")
        print(f"   總迭代次數: {self.iteration_count}")
        print(f"   找到有效參數組數: {len(self.valid_results)}")
        print(f"   總耗時: {total_time/60:.1f} 分鐘 ({total_time/3600:.2f} 小時)")
        print(f"   平均每次迭代: {total_time/self.iteration_count:.2f} 秒")
        print("=" * 80)
    
    def save_results(self, output_path: Path):
        """Save all valid results to CSV"""
        if not self.valid_results:
            print("\n⚠️  未找到有效參數，無法保存結果")
            print("   建議：")
            print("   1. 放寬篩選條件（降低ROI要求或提高MaxDD容忍度）")
            print("   2. 擴大參數搜索範圍")
            print("   3. 檢查數據質量")
            return
        
        # 使用 Robustness Score 排序（優先考慮穩健性）
        sorted_results = sorted(
            self.valid_results, 
            key=lambda x: x['stats'].get('robustness_score', 0), 
            reverse=True
        )
        
        csv_data = []
        for result in sorted_results:
            params = result['params']
            stats = result['stats']
            
            row = {
                'robustness_score': stats.get('robustness_score', 0),
                'roi_pct': stats['roi_pct'],
                'max_drawdown_pct': stats['max_drawdown_pct'],
                'sharpe_ratio': stats.get('sharpe_ratio', 0),
                'total_pnl': stats['total_pnl'],
                'total_trades': stats['total_trades'],
                'final_equity': stats['final_equity'],
                'small_gap': params.get('small_gap', ''),
                'mid_mult': params.get('mid_mult', ''),
                'big_mult': params.get('big_mult', ''),
                'size_pct_small': params.get('size_pct_small', ''),
                'size_pct_mid': params.get('size_pct_mid', ''),
                'size_pct_big': params.get('size_pct_big', ''),
                'ema_span_fast_bars': params.get('ema_span_fast_bars', ''),
                'ema_span_slow_bars': params.get('ema_span_slow_bars', ''),
                'bias_high': params.get('bias_high', ''),
                'bias_low': params.get('bias_low', ''),
                'bias_neutral_target': params.get('bias_neutral_target', ''),
                'atr_spacing_multiplier': params.get('atr_spacing_multiplier', ''),
                'trend_trade_equity_pct': params.get('trend_trade_equity_pct', ''),
                'adx_strength_threshold': params.get('adx_strength_threshold', ''),
                'rsi_period': params.get('rsi_period', ''),
                'rsi_bull_threshold': params.get('rsi_bull_threshold', ''),
                'rsi_bear_threshold': params.get('rsi_bear_threshold', ''),
                'adx_min_threshold': params.get('adx_min_threshold', ''),
                'bollinger_window': params.get('bollinger_window', ''),
                'bollinger_k': params.get('bollinger_k', ''),
                'bollinger_band_threshold': params.get('bollinger_band_threshold', ''),
                'stochastic_k_period': params.get('stochastic_k_period', ''),
                'stochastic_d_period': params.get('stochastic_d_period', ''),
                'stochastic_oversold': params.get('stochastic_oversold', ''),
                'stochastic_overbought': params.get('stochastic_overbought', ''),
            }
            csv_data.append(row)
        
        # 處理輸出路徑：如果是目錄，在目錄中創建文件
        if output_path.is_dir():
            output_file = output_path / 'optimization_results.csv'
        else:
            output_file = output_path
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        print(f"\n📊 結果已保存至: {output_file}")
        print(f"\n🏆 Top 5 參數組合（按 Robustness Score 排序）:")
        for i, result in enumerate(sorted_results[:5], 1):
            stats = result['stats']
            robustness = stats.get('robustness_score', 0)
            print(f"   {i}. Robustness: {robustness:>6.2f} | ROI: {stats['roi_pct']:>6.2f}% | Max DD: {stats['max_drawdown_pct']:>5.2f}% | "
                  f"Sharpe: {stats.get('sharpe_ratio', 0):>5.2f} | 交易次數: {stats['total_trades']:>4} | 總損益: {stats['total_pnl']:>10.2f} TWD")


def main():
    parser = argparse.ArgumentParser(description="Parallel Parameter Optimization")
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "usdttwd_1m_25y7m.csv",
                        help="Path to OHLC CSV file")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config_usdttwd.yaml",
                        help="Path to base config YAML file")
    parser.add_argument("--init-usdt", type=float, default=10000.0, help="Initial USDT balance")
    parser.add_argument("--init-twd", type=float, default=300000.0, help="Initial TWD balance")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "optimization_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--target", type=int, default=100, help="Target number of valid parameter sets")
    parser.add_argument("--max-iter", type=int, default=2000, help="Maximum iterations")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (recommended: 4-6 for M1)")
    
    args = parser.parse_args()
    
    if not args.csv.exists():
        LOG.error(f"CSV file not found: {args.csv}")
        return
    
    if not args.config.exists():
        LOG.error(f"Config file not found: {args.config}")
        return
    
    optimizer = ParameterOptimizerParallel(
        csv_path=args.csv,
        base_config_path=args.config,
        init_usdt=args.init_usdt,
        init_twd=args.init_twd,
        num_workers=args.workers
    )
    
    optimizer.target_valid_sets = args.target
    optimizer.max_iterations = args.max_iter
    
    optimizer.optimize()
    optimizer.save_results(args.output)


if __name__ == "__main__":
    main()

