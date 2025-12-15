# optimize_params.py
"""
Random Search + Local Mutation Parameter Optimization System
Finds optimal trading strategy parameters using random search with local mutation
"""
import argparse
import csv
import logging
import random
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yaml

from backtester_grid import Backtester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger("ParamOptimizer")


class ParameterOptimizer:
    """Random Search + Local Mutation optimizer"""
    
    def __init__(self, csv_path: Path, base_config_path: Path, 
                 init_usdt: float = 10000.0, init_twd: float = 300000.0):
        self.csv_path = csv_path
        self.base_config_path = base_config_path
        self.init_usdt = Decimal(str(init_usdt))
        self.init_twd = Decimal(str(init_twd))
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
        LOG.info(f"Loading data from {self.csv_path}...")
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
            
            LOG.info(f"Data loaded: {len(self.price_df)} rows")
        except Exception as e:
            LOG.error(f"Failed to load data: {e}", exc_info=True)
            raise
    
    def _load_base_config(self):
        """Load base configuration"""
        try:
            with open(self.base_config_path, 'r', encoding='utf-8') as f:
                self.base_config = yaml.safe_load(f) or {}
            LOG.info(f"Base config loaded from {self.base_config_path}")
        except Exception as e:
            LOG.error(f"Failed to load base config: {e}", exc_info=True)
            raise
    
    def _generate_random_params(self) -> Dict:
        """Generate a random set of parameters within defined ranges"""
        params = self.base_config.copy()
        
        # 調整後的參數範圍（擴大搜索空間）
        params['small_gap'] = str(round(random.uniform(0.01, 0.10), 4))  # 擴大：0.01-0.10
        params['mid_mult'] = random.randint(2, 6)  # 擴大：2-6
        params['big_mult'] = random.randint(5, 15)  # 擴大：5-15
        params['size_pct_small'] = str(round(random.uniform(0.01, 0.05), 4))  # 擴大：0.01-0.05
        params['size_pct_mid'] = str(round(random.uniform(0.015, 0.06), 4))  # 擴大：0.015-0.06
        params['size_pct_big'] = str(round(random.uniform(0.02, 0.08), 4))  # 擴大：0.02-0.08
        params['ema_span_fast_bars'] = random.randint(30, 1200)  # 擴大：30-1200
        params['ema_span_slow_bars'] = random.randint(600, 8000)  # 擴大：600-8000
        params['bias_high'] = str(round(random.uniform(0.50, 0.90), 3))  # 擴大：0.50-0.90
        params['bias_low'] = str(round(random.uniform(0.05, 0.50), 3))  # 擴大：0.05-0.50
        params['bias_neutral_target'] = str(round(random.uniform(0.30, 0.60), 3))  # 擴大：0.30-0.60
        
        # Ensure required parameters exist
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
            params['use_hybrid_model'] = False
        
        return params
    
    def _mutate_params(self, base_params: Dict, mutation_rate: float = 0.1) -> Dict:
        """Create a mutated variant of parameters (±10% perturbation)"""
        params = base_params.copy()
        
        # Mutate float parameters
        if 'small_gap' in params:
            val = float(params['small_gap'])
            params['small_gap'] = str(round(max(0.005, min(0.05, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 4))
        
        if 'size_pct_small' in params:
            val = float(params['size_pct_small'])
            params['size_pct_small'] = str(round(max(0.005, min(0.02, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 4))
        
        if 'size_pct_mid' in params:
            val = float(params['size_pct_mid'])
            params['size_pct_mid'] = str(round(max(0.01, min(0.04, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 4))
        
        if 'size_pct_big' in params:
            val = float(params['size_pct_big'])
            params['size_pct_big'] = str(round(max(0.02, min(0.06, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 4))
        
        # Mutate int parameters
        if 'mid_mult' in params:
            params['mid_mult'] = max(2, min(5, params['mid_mult'] + random.randint(-1, 1)))
        
        if 'big_mult' in params:
            params['big_mult'] = max(6, min(12, params['big_mult'] + random.randint(-1, 1)))
        
        if 'ema_span_fast_bars' in params:
            change = int(params['ema_span_fast_bars'] * mutation_rate)
            params['ema_span_fast_bars'] = max(60, min(800, params['ema_span_fast_bars'] + random.randint(-change, change)))
        
        if 'ema_span_slow_bars' in params:
            change = int(params['ema_span_slow_bars'] * mutation_rate)
            params['ema_span_slow_bars'] = max(1500, min(6000, params['ema_span_slow_bars'] + random.randint(-change, change)))
        
        # Mutate bias parameters
        if 'bias_high' in params:
            val = float(params['bias_high'])
            params['bias_high'] = str(round(max(0.55, min(0.85, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 3))
        
        if 'bias_low' in params:
            val = float(params['bias_low'])
            params['bias_low'] = str(round(max(0.10, min(0.40, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 3))
        
        if 'bias_neutral_target' in params:
            val = float(params['bias_neutral_target'])
            params['bias_neutral_target'] = str(round(max(0.35, min(0.55, val * (1 + random.uniform(-mutation_rate, mutation_rate)))), 3))
        
        return params
    
    def _run_backtest(self, params: Dict) -> Optional[Dict]:
        """Run a single backtest and return stats if valid"""
        try:
            backtester = Backtester(params, self.init_usdt, self.init_twd, verbose=False)
            stats = backtester.run(self.price_df)
            
            # 調整後的篩選條件（放寬以找到更多候選參數）
            if stats['roi_pct'] > 5.0 and stats['max_drawdown_pct'] < 15.0:
                result = {
                    'params': params,
                    'stats': stats
                }
                return result
        except Exception as e:
            LOG.debug(f"Backtest failed: {e}")
            return None
        
        return None
    
    def optimize(self, use_parallel: bool = False, max_workers: int = 4):
        """Run optimization with random search + local mutation"""
        LOG.info("=" * 80)
        LOG.info("Starting Random Search + Local Mutation Optimization")
        LOG.info(f"Target: {self.target_valid_sets} valid parameter sets")
        LOG.info(f"Max iterations: {self.max_iterations}")
        LOG.info("=" * 80)
        
        while len(self.valid_results) < self.target_valid_sets and self.iteration_count < self.max_iterations:
            # Generate random parameters
            params = self._generate_random_params()
            self.iteration_count += 1
            
            # Run backtest
            result = self._run_backtest(params)
            
            if result:
                self.valid_results.append(result)
                LOG.info(f"Found {len(self.valid_results)}/{self.target_valid_sets} valid candidates (Iteration {self.iteration_count})")
                LOG.info(f"  ROI: {result['stats']['roi_pct']:.2f}%, Max DD: {result['stats']['max_drawdown_pct']:.2f}%")
                
                # Generate 5 mutations and test them
                for i in range(5):
                    mutated_params = self._mutate_params(params, mutation_rate=0.1)
                    self.iteration_count += 1
                    
                    mut_result = self._run_backtest(mutated_params)
                    
                    if mut_result:
                        self.valid_results.append(mut_result)
                        LOG.info(f"  Mutation {i+1}: ROI {mut_result['stats']['roi_pct']:.2f}%, Max DD {mut_result['stats']['max_drawdown_pct']:.2f}%")
                
                if len(self.valid_results) >= self.target_valid_sets:
                    break
            
            # Progress update every 100 iterations
            if self.iteration_count % 100 == 0:
                LOG.info(f"Progress: {self.iteration_count}/{self.max_iterations} iterations, {len(self.valid_results)} valid sets found")
        
        LOG.info("=" * 80)
        LOG.info(f"Optimization completed!")
        LOG.info(f"Total iterations: {self.iteration_count}")
        LOG.info(f"Valid parameter sets found: {len(self.valid_results)}")
        LOG.info("=" * 80)
    
    def save_results(self, output_path: Path):
        """Save all valid results to CSV"""
        if not self.valid_results:
            LOG.warning("No valid results to save")
            return
        
        # Sort by ROI descending
        sorted_results = sorted(self.valid_results, key=lambda x: x['stats']['roi_pct'], reverse=True)
        
        # Prepare CSV data
        csv_data = []
        for result in sorted_results:
            params = result['params']
            stats = result['stats']
            
            row = {
                'roi_pct': stats['roi_pct'],
                'max_drawdown_pct': stats['max_drawdown_pct'],
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
            }
            csv_data.append(row)
        
        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        LOG.info(f"Results saved to {output_path}")
        LOG.info(f"Top 5 results by ROI:")
        for i, result in enumerate(sorted_results[:5], 1):
            stats = result['stats']
            LOG.info(f"  {i}. ROI: {stats['roi_pct']:.2f}%, Max DD: {stats['max_drawdown_pct']:.2f}%, Trades: {stats['total_trades']}")


def main():
    parser = argparse.ArgumentParser(description="Random Search + Local Mutation Parameter Optimization")
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "usdttwd_1m_25y7m.csv",
                        help="Path to OHLC CSV file")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config_usdttwd.yaml",
                        help="Path to base config YAML file")
    parser.add_argument("--init-usdt", type=float, default=10000.0, help="Initial USDT balance")
    parser.add_argument("--init-twd", type=float, default=300000.0, help="Initial TWD balance")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "optimization_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--target", type=int, default=100, help="Target number of valid parameter sets")
    parser.add_argument("--max-iter", type=int, default=5000, help="Maximum iterations (default: 5000 for more results)")
    
    args = parser.parse_args()
    
    if not args.csv.exists():
        LOG.error(f"CSV file not found: {args.csv}")
        return
    
    if not args.config.exists():
        LOG.error(f"Config file not found: {args.config}")
        return
    
    optimizer = ParameterOptimizer(
        csv_path=args.csv,
        base_config_path=args.config,
        init_usdt=args.init_usdt,
        init_twd=args.init_twd
    )
    
    optimizer.target_valid_sets = args.target
    optimizer.max_iterations = args.max_iter
    
    optimizer.optimize(use_parallel=False)
    optimizer.save_results(args.output)


if __name__ == "__main__":
    main()

