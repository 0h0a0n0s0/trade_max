# optimize_grid.py
"""
Intelligent Grid Parameter Optimization using Optuna
Searches only within mathematically profitable parameter ranges
Saves ALL trial results to CSV for analysis

Recommended execution command (4-core optimization):
    python backtest/optimize_grid.py \
        --csv backtest/usdttwd_1m_2025.csv \
        --config backtest/config_usdttwd.yaml \
        --strategy-mode pure_grid \
        --n-jobs 4 \
        --n-trials 100 \
        --output-yaml backtest/best_params.yaml \
        --output-csv backtest/optimization_results.csv
"""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any
import yaml
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

def objective(
    trial: optuna.Trial,
    csv_path: Path,
    base_config_path: Path,
    init_usdt: float,
    init_twd: float,
    strategy_mode: str = 'hybrid',
    min_gap: float = 0.05,
    max_gap: float = 0.3,
    min_bias_rebalance: float = 100.0,
    max_bias_rebalance: float = 5000.0,
) -> float:
    """
    Objective function for Optuna optimization.
    Returns ROI percentage (to be maximized).
    """
    # Define search space (CRITICAL: Aggressive profitability optimization)
    # Force pure_grid mode for high-frequency trading
    if strategy_mode == 'pure_grid' or True:  # Always use aggressive grid mode
        # Unlock Aggressive Spacing: Allow tighter gaps to harvest micro-volatility
        # Use dynamic gap range based on initial_price to support any asset (USDT, BTC, ETH, DOGE, ...)
        small_gap = trial.suggest_float("small_gap", min_gap, max_gap)
        mid_mult = trial.suggest_int("mid_mult", 4, 8)  # Expanded range
        big_mult = trial.suggest_int("big_mult", 5, 12)
        size_pct_small = trial.suggest_float("size_pct_small", 0.01, 0.06)
        # Unlock Aggressive Gridding: Allow aggressive grid rebuilding
        grid_aggression_multiplier = trial.suggest_float("grid_aggression_multiplier", 1.0, 3.0)
        # Unlock Position Bias: Allow bot to hold very little USDT if trend is down
        bias_neutral_target = trial.suggest_float("bias_neutral_target", 0.05, 0.6)  # CRITICAL: Unlock short-bias
        # Dynamic rebalance threshold based on asset price
        bias_rebalance_threshold_twd = trial.suggest_float(
            "bias_rebalance_threshold_twd", min_bias_rebalance, max_bias_rebalance
        )
        # Fix trend params to defaults (not used in pure_grid, save compute)
        ema_span_fast = 200  # Default
        ema_span_slow = 1000  # Default
        trend_trade_equity_pct = 0.4  # Default
    elif strategy_mode == 'pure_trend':
        # Search trend params normally
        ema_span_fast = trial.suggest_int("ema_span_fast_bars", 60, 300)
        ema_span_slow = trial.suggest_int("ema_span_slow_bars", 400, 3000)
        trend_trade_equity_pct = trial.suggest_float("trend_trade_equity_pct", 0.3, 0.5)
        # Fix grid params to defaults (not used in pure_trend)
        small_gap = 0.1  # Default
        mid_mult = 3  # Default
        big_mult = 7  # Default
        size_pct_small = 0.05  # Default
        grid_aggression_multiplier = 1.0  # Default
        bias_neutral_target = 0.4  # Default
        bias_rebalance_threshold_twd = (min_bias_rebalance + max_bias_rebalance) / 2.0
    else:  # hybrid mode
        # Refined search space based on latest data (sweet spot), but still dynamic by price
        small_gap = trial.suggest_float("small_gap", min_gap, max_gap)
        mid_mult = trial.suggest_int("mid_mult", 4, 8)  # Expanded range
        big_mult = trial.suggest_int("big_mult", 6, 12)
        size_pct_small = trial.suggest_float("size_pct_small", 0.01, 0.05)
        # Re-enable EMA params for hybrid mode
        ema_span_fast = trial.suggest_int("ema_span_fast_bars", 60, 300)
        ema_span_slow = trial.suggest_int("ema_span_slow_bars", 400, 3000)
        trend_trade_equity_pct = trial.suggest_float("trend_trade_equity_pct", 0.3, 0.5)  # Conservative range
        grid_aggression_multiplier = 1.0  # Default for hybrid
        bias_neutral_target = 0.4  # Default for hybrid
        bias_rebalance_threshold_twd = (min_bias_rebalance + max_bias_rebalance) / 2.0
    
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with trial suggestions
    config['small_gap'] = str(small_gap)
    config['mid_mult'] = mid_mult
    config['big_mult'] = big_mult
    config['size_pct_small'] = str(size_pct_small)
    config['ema_span_fast_bars'] = ema_span_fast
    config['ema_span_slow_bars'] = ema_span_slow
    config['trend_trade_equity_pct'] = str(trend_trade_equity_pct)
    # Dynamic / aggressive parameters
    config['grid_aggression_multiplier'] = str(grid_aggression_multiplier)
    config['bias_neutral_target'] = str(bias_neutral_target)
    config['bias_rebalance_threshold_twd'] = str(bias_rebalance_threshold_twd)
    
    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.yaml', 
        delete=False,
        prefix=f'temp_config_{trial.number}_'
    )
    temp_config_path = Path(temp_config.name)
    
    try:
        # Write config to temp file
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Run backtester
        cmd = [
            'python',
            'backtest/backtester_grid.py',
            '--csv', str(csv_path),
            '--config', str(temp_config_path),
            '--init_usdt', str(init_usdt),
            '--init_twd', str(init_twd),
            '--strategy-mode', strategy_mode
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent  # Run from project root
        )
        
        # Parse output for JSON result
        stdout_lines = result.stdout.split('\n')
        stderr_lines = result.stderr.split('\n') if result.stderr else []
        
        # Search for __BACKTEST_RESULT__: pattern
        result_json = None
        for line in stdout_lines + stderr_lines:
            if '__BACKTEST_RESULT__:' in line:
                match = re.search(r'__BACKTEST_RESULT__:(.+)', line)
                if match:
                    try:
                        result_json = json.loads(match.group(1))
                        break
                    except json.JSONDecodeError:
                        continue
        
        if result_json is None:
            # No result found, return penalty
            print(f"Trial {trial.number}: No result found in output")
            return -100.0
        
        status = result_json.get('status', 'error')
        roi_pct = result_json.get('roi_pct', 0.0)
        
        # Extract custom metrics from JSON output
        alpha_pct = result_json.get('alpha_pct', 0.0)
        bh_roi_pct = result_json.get('bh_roi_pct', 0.0)
        trades = result_json.get('trades', 0)
        total_pnl = result_json.get('total_pnl', 0.0)
        
        # Set user attributes so they appear in trials_dataframe
        trial.set_user_attr("alpha_pct", alpha_pct)
        trial.set_user_attr("bh_roi_pct", bh_roi_pct)
        trial.set_user_attr("trades", trades)
        trial.set_user_attr("total_pnl", total_pnl)
        
        # Pruning: If invalid params or error, return penalty immediately
        if status in ['invalid_params', 'error']:
            error_msg = result_json.get('error', 'Unknown error')
            print(f"Trial {trial.number}: {status} - {error_msg}")
            return -100.0
        
        # CRITICAL: Pruning for low-frequency strategies
        # We need HIGH frequency (>= 500 trades) to generate enough profit to overcome the trend
        if trades < 500:
            print(f"Trial {trial.number}: Low frequency (trades={trades} < 500). Pruning.")
            return -100.0
        
        # Report intermediate value for pruning
        trial.report(roi_pct, step=trial.number)
        
        # Check if should prune
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        print(f"Trial {trial.number}: ROI = {roi_pct:.2f}% | Alpha = {alpha_pct:.2f}% | Trades = {trades}")
        return roi_pct
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number}: Exception - {e}")
        return -100.0
    finally:
        # Clean up temp config file
        if temp_config_path.exists():
            temp_config_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Grid Parameter Optimization using Optuna"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to OHLC CSV file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("backtest/config_usdttwd.yaml"),
        help="Path to base config YAML file"
    )
    parser.add_argument(
        "--init-usdt",
        type=float,
        default=10000.0,
        help="Initial USDT balance"
    )
    parser.add_argument(
        "--init-twd",
        type=float,
        default=300000.0,
        help="Initial TWD balance"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=Path("backtest/best_params.yaml"),
        help="Output path for best parameters YAML"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("backtest/optimization_results.csv"),
        help="Output path for all trial results CSV"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="grid_optimization",
        help="Optuna study name"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for optimization (default: 1)"
    )
    parser.add_argument(
        "--strategy-mode",
        choices=['hybrid', 'pure_grid', 'pure_trend'],
        default='pure_grid',  # Focus on grid optimization based on isolation test results
        help="Strategy execution mode: 'pure_grid' (default, grid only), 'hybrid', 'pure_trend' (trend only)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return

    # ------------------------------------------------------------------
    # Read initial price from CSV to build dynamic, asset-aware ranges
    # ------------------------------------------------------------------
    try:
        # Read only the first row for efficiency
        temp_df = pd.read_csv(args.csv, usecols=['close'], nrows=1)
        initial_price = float(temp_df['close'].iloc[0])
    except Exception as e:
        print(f"⚠️  Warning: Failed to read initial price from {args.csv}: {e}")
        print("   Falling back to initial_price = 1.0 (dynamic ranges may be suboptimal).")
        initial_price = 1.0

    # Dynamic ranges based on asset price
    # small_gap: 0.1% ~ 2.0% of price
    min_gap = initial_price * 0.001
    max_gap = initial_price * 0.02

    # bias_rebalance_threshold_twd: 0.5x ~ 5x price
    min_bias_rebalance = initial_price * 0.5
    max_bias_rebalance = initial_price * 5.0
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize ROI
        study_name=args.study_name,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    print("=" * 80)
    print("🚀 Starting Grid Parameter Optimization")
    print(f"   CSV: {args.csv}")
    print(f"   Base Config: {args.config}")
    print(f"   Strategy Mode: {args.strategy_mode}")
    print(f"   Trials: {args.n_trials}")
    print(f"   Parallel Jobs: {args.n_jobs}")
    print(f"   Detected initial_price: {initial_price:.6f}")
    print(f"   Dynamic small_gap range: {min_gap:.6f} to {max_gap:.6f}")
    print(f"   Dynamic bias_rebalance_threshold_twd range: {min_bias_rebalance:.2f} to {max_bias_rebalance:.2f}")
    print(f"   Search Space:")
    if args.strategy_mode == 'pure_grid':
        print(f"     - small_gap: {min_gap:.6f} to {max_gap:.6f} (0.1% - 2.0% of price)")
        print(f"     - mid_mult: 4 to 8")
        print(f"     - big_mult: 5 to 12")
        print(f"     - size_pct_small: 0.01 to 0.06")
        print(f"     - grid_aggression_multiplier: 1.0 to 3.0 (aggressive gridding)")
        print(f"     - bias_neutral_target: 0.05 to 0.6 (unlock short-bias)")
        print(f"     - bias_rebalance_threshold_twd: {min_bias_rebalance:.2f} to {max_bias_rebalance:.2f}")
        print(f"     - (Trend params fixed to defaults, saving compute)")
        print(f"     - MIN TRADES: 500 (high-frequency requirement)")
    elif args.strategy_mode == 'pure_trend':
        print(f"     - ema_span_fast: 60 to 300")
        print(f"     - ema_span_slow: 400 to 3000")
        print(f"     - trend_trade_equity_pct: 0.3 to 0.5")
        print(f"     - (Grid params fixed to defaults)")
    else:  # hybrid
        print(f"     - small_gap: {min_gap:.6f} to {max_gap:.6f} (0.1% - 2.0% of price)")
        print(f"     - mid_mult: 4 to 8")
        print(f"     - big_mult: 6 to 12")
        print(f"     - size_pct_small: 0.01 to 0.05")
        print(f"     - ema_span_fast: 60 to 300")
        print(f"     - ema_span_slow: 400 to 3000")
        print(f"     - trend_trade_equity_pct: 0.3 to 0.5")
    print("=" * 80)
    
    # Optimize
    study.optimize(
        lambda trial: objective(
            trial,
            args.csv,
            args.config,
            args.init_usdt,
            args.init_twd,
            args.strategy_mode,
            min_gap,
            max_gap,
            min_bias_rebalance,
            max_bias_rebalance,
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("✅ Optimization Complete")
    print(f"   Best Trial: {study.best_trial.number}")
    print(f"   Best ROI: {study.best_value:.2f}%")
    
    # Print custom metrics from best trial
    best_alpha = study.best_trial.user_attrs.get('alpha_pct', 'N/A')
    best_bh_roi = study.best_trial.user_attrs.get('bh_roi_pct', 'N/A')
    best_trades = study.best_trial.user_attrs.get('trades', 'N/A')
    print(f"   Best Alpha: {best_alpha}%")
    print(f"   Best Buy & Hold ROI: {best_bh_roi}%")
    print(f"   Best Trades: {best_trades}")
    
    print("\n   Best Parameters:")
    for key, value in study.best_params.items():
        print(f"     {key}: {value}")
    print("=" * 80)
    
    # 1. Save Best Parameters to YAML
    best_config = yaml.safe_load(args.config.read_text())
    best_config.update(study.best_params)
    
    # Convert numeric values to strings where needed (for YAML compatibility)
    if 'small_gap' in best_config:
        best_config['small_gap'] = str(best_config['small_gap'])
    if 'size_pct_small' in best_config:
        best_config['size_pct_small'] = str(best_config['size_pct_small'])
    
    yaml_output_path = args.output_yaml
    if yaml_output_path.is_dir():
        yaml_output_path = yaml_output_path / 'best_params.yaml'
    
    with open(yaml_output_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    
    # 2. Save ALL Trial Results to CSV
    try:
        # Get trials dataframe from Optuna
        df = study.trials_dataframe()
        
        # Clean up column names (remove "params_" and "user_attrs_" prefixes)
        df.columns = [
            col.replace('params_', '').replace('user_attrs_', '') 
            if col.startswith('params_') or col.startswith('user_attrs_') 
            else col 
            for col in df.columns
        ]
        
        # Ensure output directory exists
        csv_output_path = args.output_csv
        if csv_output_path.is_dir():
            csv_output_path = csv_output_path / 'optimization_results.csv'
        else:
            # Create parent directory if it doesn't exist
            csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(csv_output_path, index=False, encoding='utf-8')
        
        print(f"\n📊 Optimization complete. Results saved to:")
        print(f"   CSV: {csv_output_path}")
        print(f"   YAML: {yaml_output_path}")
        print(f"\n   Total trials: {len(df)}")
        print(f"   Successful trials: {len(df[df['state'] == 'COMPLETE'])}")
        print(f"   Pruned trials: {len(df[df['state'] == 'PRUNED'])}")
        print(f"   Failed trials: {len(df[df['state'] == 'FAIL'])}")
        
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to save CSV results: {e}")
        print(f"   Best parameters still saved to: {yaml_output_path}")


if __name__ == "__main__":
    main()

