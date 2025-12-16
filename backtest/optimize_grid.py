# optimize_grid.py
"""
Intelligent Grid Parameter Optimization using Optuna
Searches only within mathematically profitable parameter ranges
Saves ALL trial results to CSV for analysis
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

def objective(trial: optuna.Trial, csv_path: Path, base_config_path: Path, 
              init_usdt: float, init_twd: float) -> float:
    """
    Objective function for Optuna optimization.
    Returns ROI percentage (to be maximized).
    """
    # Define search space (CRITICAL: Only profitable ranges)
    small_gap = trial.suggest_float("small_gap", 0.1, 1.5)  # TWD, strictly >= 0.1
    mid_mult = trial.suggest_int("mid_mult", 2, 5)
    big_mult = trial.suggest_int("big_mult", 6, 12)
    size_pct_small = trial.suggest_float("size_pct_small", 0.01, 0.05)
    ema_span_fast = trial.suggest_int("ema_span_fast_bars", 60, 300)
    ema_span_slow = trial.suggest_int("ema_span_slow_bars", 400, 3000)
    
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
            '--init_twd', str(init_twd)
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
        
        # Pruning: If invalid params or error, return penalty immediately
        if status in ['invalid_params', 'error']:
            error_msg = result_json.get('error', 'Unknown error')
            print(f"Trial {trial.number}: {status} - {error_msg}")
            return -100.0
        
        # Report intermediate value for pruning
        trial.report(roi_pct, step=trial.number)
        
        # Check if should prune
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        print(f"Trial {trial.number}: ROI = {roi_pct:.2f}%")
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
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return
    
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
    print(f"   Trials: {args.n_trials}")
    print(f"   Search Space:")
    print(f"     - small_gap: 0.1 to 1.5 TWD (profitable range)")
    print(f"     - mid_mult: 2 to 5")
    print(f"     - big_mult: 6 to 12")
    print(f"     - size_pct_small: 0.01 to 0.05")
    print(f"     - ema_span_fast: 60 to 300")
    print(f"     - ema_span_slow: 400 to 3000")
    print("=" * 80)
    
    # Optimize
    study.optimize(
        lambda trial: objective(
            trial, args.csv, args.config, args.init_usdt, args.init_twd
        ),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("✅ Optimization Complete")
    print(f"   Best Trial: {study.best_trial.number}")
    print(f"   Best ROI: {study.best_value:.2f}%")
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
        
        # Clean up column names (remove "params_" prefix)
        df.columns = [col.replace('params_', '') if col.startswith('params_') else col 
                     for col in df.columns]
        
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

