#!/usr/bin/env python3
"""
TradeMax - Rank 77 Strategy 實盤交易主程式

使用模組化架構啟動交易機器人
"""
import asyncio
import logging
import yaml
from pathlib import Path
import sys

from strategy.grid_strategy import GridStrategy
from engine.bot_engine import BotEngine

# 設定日誌
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger("Main")


async def main():
    """主函數：啟動交易機器人"""
    # 1. 載入配置
    config_path = Path("configs/config_rank77.yaml")
    
    if not config_path.exists():
        log.error(f"配置文件不存在: {config_path}")
        log.error("請確認配置文件路徑正確")
        sys.exit(1)
    
    log.info(f"載入配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 創建策略實例
    log.info("初始化 GridStrategy...")
    strategy = GridStrategy(config)
    log.info(f"策略名稱: {strategy.strategy_name}")
    log.info(f"交易對: {strategy.asset_pair}")
    log.info(f"基礎貨幣: {config.get('usdt_unit', 'USDT')}")
    log.info(f"報價貨幣: {config.get('twd_unit', 'TWD')}")
    
    # 3. 創建執行引擎
    log.info("初始化 BotEngine...")
    engine = BotEngine(strategy, config_path)
    
    # 4. 初始化引擎（API、資料庫等）
    log.info("開始初始化引擎...")
    try:
        await engine.initialize()
        log.info("✅ 引擎初始化完成")
    except Exception as e:
        log.critical(f"❌ 引擎初始化失敗: {e}", exc_info=True)
        sys.exit(1)
    
    # 5. 啟動交易機器人
    log.info("🚀 啟動交易機器人...")
    try:
        await engine.start()
    except KeyboardInterrupt:
        log.info("收到中斷信號，正在關閉...")
    except Exception as e:
        log.critical(f"❌ 交易機器人運行錯誤: {e}", exc_info=True)
    finally:
        log.info("交易機器人已停止")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程式已中斷")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        sys.exit(1)

