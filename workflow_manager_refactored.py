# workflow_manager_refactored.py
"""
自動化工作流程管理器（OOP 重構版）
支持熱更新策略參數，無需重啟進程
"""
import asyncio
import schedule
import logging
import yaml
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict
import sys

from optimize_params import OptunaOptimizer
from telegram_alerter import alerter

LOG = logging.getLogger("WorkflowManager")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    level=logging.INFO
)


class WorkflowManager:
    def __init__(self, bot_engine=None, config_path: Path = None, csv_path: Path = None):
        """
        初始化工作流程管理器
        
        Args:
            bot_engine: BotEngine 實例（可選，如果提供則支持熱更新）
            config_path: 配置檔案路徑
            csv_path: CSV數據檔案路徑
        """
        self.bot_engine = bot_engine  # BotEngine 實例（用於熱更新）
        self.config_path = config_path or Path(__file__).parent / "config_usdttwd.yaml"
        self.csv_path = csv_path or Path(__file__).parent / "backtest" / "usdttwd_1m_25y7m.csv"
        self.backup_dir = Path(__file__).parent / "config_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # 載入配置
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}
    
    def setup_schedule(self):
        """設置定時任務"""
        # 每週日凌晨2點執行參數優化
        schedule.every().sunday.at("02:00").do(self._run_weekly_optimization)
        
        # 每日凌晨3點執行回測驗證
        schedule.every().day.at("03:00").do(self._run_daily_validation)
        
        LOG.info("Workflow schedule initialized:")
        LOG.info("  - Weekly optimization: Sunday 02:00")
        LOG.info("  - Daily validation: 03:00")
        if self.bot_engine:
            LOG.info("  - Hot update enabled: Parameters can be updated without restart")
    
    async def _run_weekly_optimization(self):
        """每週參數優化"""
        LOG.info("=" * 80)
        LOG.info("Starting weekly parameter optimization...")
        
        try:
            # 創建優化器
            optimizer = OptunaOptimizer(
                self.csv_path,
                self.config_path,
                train_ratio=0.7
            )
            
            # 執行優化
            study, validation_result = optimizer.optimize(n_trials=100)
            
            # 檢查測試集ROI
            min_test_roi = float(self.config.get('auto_optimize_min_test_roi', 0.15))
            
            if validation_result['roi'] >= min_test_roi:
                # 備份當前配置
                backup_path = self.backup_dir / f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                import shutil
                shutil.copy(self.config_path, backup_path)
                LOG.info(f"Current config backed up to {backup_path}")
                
                # 提取最佳參數
                best_params = self._extract_params_from_study(study)
                
                # 如果提供了 bot_engine，使用熱更新
                if self.bot_engine:
                    LOG.info("Using hot update: Updating strategy parameters without restart...")
                    success = self.bot_engine.strategy.update_config(best_params)
                    
                    if success:
                        # 保存最佳配置到文件（用於記錄）
                        output_path = self.config_path.parent / "config_optimized.yaml"
                        optimizer.save_best_config(study, output_path, min_test_roi)
                        
                        msg = (f"✅ **參數優化完成（熱更新）！**\n\n"
                               f"測試集ROI: `{validation_result['roi']:.2%}`\n"
                               f"測試集夏普比率: `{validation_result['sharpe_ratio']:.2f}`\n"
                               f"測試集最大回撤: `{validation_result['max_drawdown']:.2%}`\n\n"
                               f"參數已熱更新，無需重啟進程。\n"
                               f"新配置已保存至: `config_optimized.yaml`")
                        await alerter.send_strategy_event(msg, alert_key='optimization_complete')
                    else:
                        msg = (f"⚠️ **參數優化完成，但熱更新失敗**\n\n"
                               f"測試集ROI: `{validation_result['roi']:.2%}`\n"
                               f"請手動檢查並重啟進程以應用新參數。")
                        await alerter.send_strategy_event(msg, alert_key='optimization_failed')
                else:
                    # 沒有 bot_engine，使用傳統方式（更新配置文件）
                    output_path = self.config_path.parent / "config_optimized.yaml"
                    if optimizer.save_best_config(study, output_path, min_test_roi):
                        msg = (f"✅ **參數優化完成！**\n\n"
                               f"測試集ROI: `{validation_result['roi']:.2%}`\n"
                               f"測試集夏普比率: `{validation_result['sharpe_ratio']:.2f}`\n"
                               f"測試集最大回撤: `{validation_result['max_drawdown']:.2%}`\n\n"
                               f"新配置已保存至: `config_optimized.yaml`\n"
                               f"請手動檢查並應用新配置。")
                        await alerter.send_strategy_event(msg, alert_key='optimization_complete')
            else:
                msg = (f"⚠️ **參數優化完成，但測試集ROI未達閾值**\n\n"
                       f"測試集ROI: `{validation_result['roi']:.2%}` (閾值: {min_test_roi:.2%})\n"
                       f"保持當前配置。")
                await alerter.send_strategy_event(msg, alert_key='optimization_failed')
            
            LOG.info("Weekly optimization completed.")
            
        except Exception as e:
            LOG.error(f"Weekly optimization failed: {e}", exc_info=True)
            await alerter.send_critical_alert(
                f"❌ **參數優化失敗！**\n\n原因: `{e}`",
                alert_key='optimization_error'
            )
    
    def _extract_params_from_study(self, study) -> Dict[str, any]:
        """
        從 Optuna study 中提取最佳參數
        
        Args:
            study: Optuna study 對象
            
        Returns:
            參數字典（格式與 config_usdttwd.yaml 兼容）
        """
        best_params = study.best_params
        
        # 轉換為配置格式
        config_params = {}
        
        # 網格參數
        if 'small_gap' in best_params:
            config_params['small_gap'] = str(best_params['small_gap'])
        if 'mid_mult' in best_params:
            config_params['mid_mult'] = int(best_params['mid_mult'])
        if 'big_mult' in best_params:
            config_params['big_mult'] = int(best_params['big_mult'])
        if 'levels_each' in best_params:
            config_params['levels_each'] = int(best_params['levels_each'])
        
        # 訂單大小
        if 'size_pct_small' in best_params:
            config_params['size_pct_small'] = str(best_params['size_pct_small'])
        if 'size_pct_mid' in best_params:
            config_params['size_pct_mid'] = str(best_params['size_pct_mid'])
        if 'size_pct_big' in best_params:
            config_params['size_pct_big'] = str(best_params['size_pct_big'])
        
        # EMA參數
        if 'ema_span_fast_bars' in best_params:
            config_params['ema_span_fast_bars'] = int(best_params['ema_span_fast_bars'])
        if 'ema_span_slow_bars' in best_params:
            config_params['ema_span_slow_bars'] = int(best_params['ema_span_slow_bars'])
        
        # 偏置參數
        if 'bias_high' in best_params:
            config_params['bias_high'] = str(best_params['bias_high'])
        if 'bias_low' in best_params:
            config_params['bias_low'] = str(best_params['bias_low'])
        
        # ATR參數
        if 'atr_spacing_multiplier' in best_params:
            config_params['atr_spacing_multiplier'] = str(best_params['atr_spacing_multiplier'])
        
        return config_params
    
    async def _run_daily_validation(self):
        """每日回測驗證"""
        LOG.info("=" * 80)
        LOG.info("Starting daily validation...")
        
        try:
            # 使用當前配置執行回測
            from backtest.backtester_grid import Backtester
            import pandas as pd
            
            # 載入數據
            temp_df = pd.read_csv(self.csv_path, usecols=['ts', 'high', 'low', 'close'])
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
            
            # 使用最近30%的數據作為驗證集
            split_idx = int(len(price_df) * 0.7)
            validation_df = price_df.iloc[split_idx:].copy()
            
            # 執行回測
            init_usdt = Decimal(str(self.config.get('init_usdt', 10000.0)))
            init_twd = Decimal(str(self.config.get('init_twd', 300000.0)))
            
            backtester = Backtester(self.config, init_usdt, init_twd)
            result = backtester.run(validation_df)
            trade_log = result['trade_log']
            final_equity = result['final_equity']
            initial_equity = result['initial_equity']
            
            validation_roi = float((final_equity - initial_equity) / initial_equity) if initial_equity > 0 else 0.0
            
            LOG.info(f"Daily validation completed. ROI: {validation_roi:.4f}")
            
            # 如果ROI異常低，發送警報
            if validation_roi < -0.05:  # 虧損超過5%
                msg = (f"⚠️ **回測驗證異常！**\n\n"
                       f"驗證集ROI: `{validation_roi:.2%}`\n"
                       f"當前配置可能不適合當前市場環境，建議重新優化參數。")
                await alerter.send_strategy_event(msg, alert_key='validation_alert')
            
        except Exception as e:
            LOG.error(f"Daily validation failed: {e}", exc_info=True)
            await alerter.send_critical_alert(
                f"❌ **每日驗證失敗！**\n\n原因: `{e}`",
                alert_key='validation_error'
            )
    
    async def run(self):
        """運行工作流程管理器"""
        self.setup_schedule()
        LOG.info("Workflow manager started. Waiting for scheduled tasks...")
        
        while True:
            schedule.run_pending()
            await asyncio.sleep(60)  # 每分鐘檢查一次


async def main():
    """主函數（示例：如何整合 BotEngine）"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自動化工作流程管理器（OOP重構版）")
    parser.add_argument("--config", type=Path, help="配置檔案路徑")
    parser.add_argument("--csv", type=Path, help="CSV數據檔案路徑")
    parser.add_argument("--run-optimization", action="store_true", help="立即執行一次優化")
    parser.add_argument("--run-validation", action="store_true", help="立即執行一次驗證")
    parser.add_argument("--with-bot", action="store_true", help="整合 BotEngine（支持熱更新）")
    
    args = parser.parse_args()
    
    bot_engine = None
    if args.with_bot:
        # 如果提供 --with-bot，需要先啟動 BotEngine
        # 這裡只是示例，實際使用時需要通過消息隊列或其他方式共享 bot_engine 實例
        LOG.warning("--with-bot flag requires BotEngine instance. "
                   "In production, use message queue or shared memory to access bot_engine.")
        # bot_engine = get_bot_engine_from_shared_memory()  # 示例
    
    manager = WorkflowManager(bot_engine, args.config, args.csv)
    
    if args.run_optimization:
        await manager._run_weekly_optimization()
    elif args.run_validation:
        await manager._run_daily_validation()
    else:
        await manager.run()


if __name__ == "__main__":
    asyncio.run(main())

