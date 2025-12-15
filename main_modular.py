# main_modular.py
"""
模組化架構主入口文件
展示如何使用新的模組化架構運行交易機器人
"""
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

from strategy.grid_strategy import GridStrategy
from optimizer.strategy_optimizer import StrategyOptimizer
from engine.bot_engine import BotEngine
import yaml

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
)
log = logging.getLogger("Main")


async def main():
    """主函數"""
    try:
        # 1. 載入配置
        config_path = Path(__file__).parent / "config_usdttwd.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        log.info(f"Configuration loaded from: {config_path}")
        
        # 2. 創建策略實例
        strategy = GridStrategy(config)
        log.info("Strategy instance created.")
        
        # 3. 創建優化器實例
        optimizer_config = config.get("optimizer", {})
        optimizer = StrategyOptimizer(strategy, optimizer_config)
        log.info("Optimizer instance created.")
        
        # 4. 創建引擎實例
        engine = BotEngine(strategy, optimizer, config_path)
        log.info("Engine instance created.")
        
        # 5. 初始化引擎
        await engine.initialize()
        
        # 6. 啟動引擎
        await engine.start()
        
    except KeyboardInterrupt:
        log.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        log.critical(f"Critical error: {e}", exc_info=True)
    finally:
        log.info("Application exited.")


if __name__ == "__main__":
    asyncio.run(main())

