# optimize_params.py
"""
Optuna 參數優化腳本
自動搜尋最佳策略參數組合
"""
import optuna
import yaml
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import sys
import os

# 添加 backtest 目錄到路徑
sys.path.insert(0, str(Path(__file__).parent / "backtest"))
from backtester_grid import Backtester
from indicators import adx, ema

getcontext().prec = 28
LOG = logging.getLogger("OptunaOptimizer")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    level=logging.INFO
)


class OptunaOptimizer:
    def __init__(self, csv_path: Path, base_config_path: Path, train_ratio: float = 0.7):
        self.csv_path = csv_path
        self.base_config_path = base_config_path
        self.train_ratio = train_ratio
        self.price_df = None
        self.train_df = None
        self.test_df = None
        self.base_config = {}
        self._load_data()
        self._load_base_config()
    
    def _load_data(self):
        """載入並分割數據"""
        LOG.info(f"Loading data from {self.csv_path}...")
        try:
            temp_df = pd.read_csv(self.csv_path, usecols=['ts', 'high', 'low', 'close'])
            
            # 處理時間戳
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
            
            # 分割訓練集和測試集
            split_idx = int(len(self.price_df) * self.train_ratio)
            self.train_df = self.price_df.iloc[:split_idx].copy()
            self.test_df = self.price_df.iloc[split_idx:].copy()
            
            LOG.info(f"Data loaded: Total={len(self.price_df)}, Train={len(self.train_df)}, Test={len(self.test_df)}")
        except Exception as e:
            LOG.error(f"Failed to load data: {e}", exc_info=True)
            raise
    
    def _load_base_config(self):
        """載入基礎配置"""
        try:
            with open(self.base_config_path, 'r', encoding='utf-8') as f:
                self.base_config = yaml.safe_load(f) or {}
            LOG.info(f"Base config loaded from {self.base_config_path}")
        except Exception as e:
            LOG.error(f"Failed to load base config: {e}", exc_info=True)
            raise
    
    def _create_config(self, trial) -> Dict:
        """根據trial建議創建配置"""
        cfg = self.base_config.copy()
        
        # 網格參數
        cfg['small_gap'] = str(trial.suggest_float('small_gap', 0.03, 0.08, step=0.005))
        cfg['mid_mult'] = trial.suggest_int('mid_mult', 2, 5)
        cfg['big_mult'] = trial.suggest_int('big_mult', 5, 10)
        cfg['levels_each'] = trial.suggest_int('levels_each', 8, 16)
        
        # 訂單大小
        cfg['size_pct_small'] = str(trial.suggest_float('size_pct_small', 0.01, 0.05, step=0.005))
        cfg['size_pct_mid'] = str(trial.suggest_float('size_pct_mid', 0.02, 0.06, step=0.005))
        cfg['size_pct_big'] = str(trial.suggest_float('size_pct_big', 0.03, 0.08, step=0.005))
        
        # EMA參數
        cfg['ema_span_fast_bars'] = trial.suggest_int('ema_span_fast_bars', 10, 100, step=10)
        cfg['ema_span_slow_bars'] = trial.suggest_int('ema_span_slow_bars', 100, 500, step=50)
        
        # 重建間隔
        cfg['recenter_interval_minutes'] = trial.suggest_int('recenter_interval_minutes', 240, 960, step=60)
        
        # 混合模式參數（如果啟用）
        if cfg.get('use_hybrid_model', False):
            cfg['adx_strength_threshold'] = trial.suggest_int('adx_strength_threshold', 25, 40)
            cfg['trend_trade_equity_pct'] = str(trial.suggest_float('trend_trade_equity_pct', 0.3, 0.5, step=0.05))
            cfg['trend_trailing_stop_pct'] = str(trial.suggest_float('trend_trailing_stop_pct', 0.01, 0.03, step=0.005))
        
        # 確保必要的技術指標參數存在（Backtester需要）
        if 'macd_fast_period' not in cfg:
            cfg['macd_fast_period'] = 12
        if 'macd_slow_period' not in cfg:
            cfg['macd_slow_period'] = 26
        if 'macd_signal_period' not in cfg:
            cfg['macd_signal_period'] = 9
        if 'dmi_period' not in cfg:
            cfg['dmi_period'] = 14
        if 'grid_aggression_threshold' not in cfg:
            cfg['grid_aggression_threshold'] = 20
        if 'grid_aggression_multiplier' not in cfg:
            cfg['grid_aggression_multiplier'] = '1.0'
        if 'use_hybrid_model' not in cfg:
            cfg['use_hybrid_model'] = False
        
        return cfg
    
    def _calculate_metrics(self, initial_equity: Decimal, final_equity: Decimal, 
                          trade_log: List[Dict], price_df: pd.DataFrame) -> Dict:
        """計算績效指標"""
        # ROI
        roi = float((final_equity - initial_equity) / initial_equity) if initial_equity > 0 else 0.0
        
        # 計算每日收益序列（用於夏普比率）
        if trade_log and len(trade_log) > 0:
            trade_df = pd.DataFrame(trade_log)
            if 'index' in trade_df.columns and len(trade_df) > 1:
                # 簡化計算：使用交易次數和平均收益
                num_trades = len(trade_df)
                avg_trade_pnl = float((final_equity - initial_equity) / num_trades) if num_trades > 0 else 0.0
                
                # 估算夏普比率（簡化版）
                if num_trades > 1 and avg_trade_pnl > 0:
                    # 假設收益標準差為平均收益的30%
                    std_dev = abs(avg_trade_pnl * 0.3)
                    sharpe_ratio = (avg_trade_pnl / std_dev) if std_dev > 0 else 0.0
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # 最大回撤（簡化計算）
        # 使用價格波動作為代理
        if len(price_df) > 0:
            price_returns = price_df['close'].pct_change().dropna()
            if len(price_returns) > 0:
                cumulative = (1 + price_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = float(abs(drawdown.min())) if len(drawdown) > 0 else 0.0
            else:
                max_drawdown = 0.0
        else:
            max_drawdown = 0.0
        
        return {
            'roi': roi,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trade_log) if trade_log else 0
        }
    
    def objective(self, trial) -> float:
        """Optuna目標函數"""
        try:
            # 創建配置
            cfg = self._create_config(trial)
            
            # 初始化回測器
            init_usdt = Decimal(str(self.base_config.get('init_usdt', 10000.0)))
            init_twd = Decimal(str(self.base_config.get('init_twd', 300000.0)))
            
            backtester = Backtester(cfg, init_usdt, init_twd)
            
            # 在訓練集上執行回測
            trade_log = backtester.run(self.train_df)
            
            # 計算最終權益
            from backtester_grid import USDT_BALANCE, TWD_BALANCE, TOTAL_EQUITY_TWD
            final_price = Decimal(str(self.train_df['close'].iloc[-1]))
            final_equity = TWD_BALANCE + USDT_BALANCE * final_price
            initial_equity = init_twd + init_usdt * Decimal(str(self.train_df['close'].iloc[0]))
            
            # 計算指標
            metrics = self._calculate_metrics(initial_equity, final_equity, trade_log, self.train_df)
            
            # 綜合評分：ROI * 0.6 + Sharpe * 0.3 - MaxDrawdown * 0.1
            score = (
                metrics['roi'] * 0.6 +
                min(metrics['sharpe_ratio'] / 2.0, 1.0) * 0.3 -  # 正規化夏普比率
                metrics['max_drawdown'] * 0.1
            )
            
            # 記錄中間值
            trial.set_user_attr('roi', metrics['roi'])
            trial.set_user_attr('sharpe_ratio', metrics['sharpe_ratio'])
            trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
            trial.set_user_attr('num_trades', metrics['num_trades'])
            
            LOG.info(f"Trial {trial.number}: ROI={metrics['roi']:.4f}, Sharpe={metrics['sharpe_ratio']:.2f}, Score={score:.4f}")
            
            return score
            
        except Exception as e:
            LOG.error(f"Trial {trial.number} failed: {e}", exc_info=True)
            return -999.0  # 返回極低分數
    
    def optimize(self, n_trials: int = 100, study_name: str = "grid_strategy_optimization"):
        """執行優化"""
        # 創建或載入study
        storage_url = f"sqlite:///{Path(__file__).parent / 'optuna_study.db'}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True
        )
        
        LOG.info(f"Starting optimization with {n_trials} trials...")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # 輸出最佳結果
        LOG.info("=" * 80)
        LOG.info("Optimization completed!")
        LOG.info(f"Best trial: {study.best_trial.number}")
        LOG.info(f"Best score: {study.best_value:.4f}")
        LOG.info(f"Best ROI: {study.best_trial.user_attrs.get('roi', 0):.4f}")
        LOG.info(f"Best Sharpe: {study.best_trial.user_attrs.get('sharpe_ratio', 0):.2f}")
        LOG.info(f"Best Max Drawdown: {study.best_trial.user_attrs.get('max_drawdown', 0):.4f}")
        LOG.info("Best parameters:")
        for key, value in study.best_params.items():
            LOG.info(f"  {key}: {value}")
        
        # 在測試集上驗證
        LOG.info("=" * 80)
        LOG.info("Validating on test set...")
        best_cfg = self._create_config_from_params(study.best_params)
        validation_result = self.validate_on_test_set(best_cfg)
        LOG.info(f"Test set ROI: {validation_result['roi']:.4f}")
        LOG.info(f"Test set Sharpe: {validation_result['sharpe_ratio']:.2f}")
        LOG.info(f"Test set Max Drawdown: {validation_result['max_drawdown']:.4f}")
        
        return study, validation_result
    
    def _create_config_from_params(self, params: Dict) -> Dict:
        """從參數字典創建配置（用於驗證）"""
        cfg = self.base_config.copy()
        for key, value in params.items():
            if key in ['small_gap', 'size_pct_small', 'size_pct_mid', 'size_pct_big', 
                      'trend_trade_equity_pct', 'trend_trailing_stop_pct']:
                cfg[key] = str(value)
            else:
                cfg[key] = value
        
        # 確保必要的技術指標參數存在
        if 'macd_fast_period' not in cfg:
            cfg['macd_fast_period'] = 12
        if 'macd_slow_period' not in cfg:
            cfg['macd_slow_period'] = 26
        if 'macd_signal_period' not in cfg:
            cfg['macd_signal_period'] = 9
        if 'dmi_period' not in cfg:
            cfg['dmi_period'] = 14
        if 'grid_aggression_threshold' not in cfg:
            cfg['grid_aggression_threshold'] = 20
        if 'grid_aggression_multiplier' not in cfg:
            cfg['grid_aggression_multiplier'] = '1.0'
        if 'use_hybrid_model' not in cfg:
            cfg['use_hybrid_model'] = False
        
        return cfg
    
    def validate_on_test_set(self, cfg: Dict) -> Dict:
        """在測試集上驗證配置"""
        init_usdt = Decimal(str(self.base_config.get('init_usdt', 10000.0)))
        init_twd = Decimal(str(self.base_config.get('init_twd', 300000.0)))
        
        backtester = Backtester(cfg, init_usdt, init_twd)
        trade_log = backtester.run(self.test_df)
        
        from backtester_grid import USDT_BALANCE, TWD_BALANCE
        final_price = Decimal(str(self.test_df['close'].iloc[-1]))
        final_equity = TWD_BALANCE + USDT_BALANCE * final_price
        initial_equity = init_twd + init_usdt * Decimal(str(self.test_df['close'].iloc[0]))
        
        metrics = self._calculate_metrics(initial_equity, final_equity, trade_log, self.test_df)
        return metrics
    
    def save_best_config(self, study, output_path: Path, min_test_roi: float = 0.15):
        """保存最佳配置到檔案"""
        validation_result = self.validate_on_test_set(
            self._create_config_from_params(study.best_params)
        )
        
        if validation_result['roi'] >= min_test_roi:
            best_cfg = self._create_config_from_params(study.best_params)
            
            # 保存到YAML檔案
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(best_cfg, f, default_flow_style=False, allow_unicode=True)
            
            LOG.info(f"Best config saved to {output_path} (Test ROI: {validation_result['roi']:.4f})")
            return True
        else:
            LOG.warning(f"Test ROI ({validation_result['roi']:.4f}) below threshold ({min_test_roi}). Config not saved.")
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optuna參數優化")
    parser.add_argument("--csv", required=True, type=Path, help="CSV數據檔案路徑")
    parser.add_argument("--config", default="config_usdttwd.yaml", type=Path, help="基礎配置檔案")
    parser.add_argument("--trials", type=int, default=100, help="優化試驗次數")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="訓練集比例")
    parser.add_argument("--output", type=Path, help="輸出配置檔案路徑（可選）")
    parser.add_argument("--min-test-roi", type=float, default=0.15, help="最小測試集ROI閾值")
    
    args = parser.parse_args()
    
    optimizer = OptunaOptimizer(args.csv, args.config, args.train_ratio)
    study, validation_result = optimizer.optimize(n_trials=args.trials)
    
    if args.output:
        optimizer.save_best_config(study, args.output, args.min_test_roi)


if __name__ == "__main__":
    main()

