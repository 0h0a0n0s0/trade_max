# hot_update_example.py
"""
展示如何使用 OOP 重構版本的熱更新功能

此範例展示：
1. 如何創建策略和引擎
2. 如何在運行時動態更新參數
3. 如何獲取市場觀察數據
"""
import asyncio
import yaml
from pathlib import Path
from decimal import Decimal
import sys

# 添加父目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_usdttwd_grid_refactored import GridStrategy, BotEngine


async def example_basic_usage():
    """基本使用範例"""
    print("=" * 80)
    print("範例 1: 基本使用")
    print("=" * 80)
    
    # 1. 載入配置
    config_path = Path(__file__).parent.parent / "config_usdttwd.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 創建策略和引擎
    strategy = GridStrategy(config)
    engine = BotEngine(strategy, config_path)
    
    print(f"策略名稱: {strategy.strategy_name}")
    print(f"網格層數: {len(strategy.grid_layers)}")
    print(f"當前參數 small_gap: {strategy.params['small_gap']}")
    
    # 注意：實際運行需要初始化 API 和資料庫
    # await engine.initialize()
    # await engine.start()


async def example_hot_update():
    """熱更新參數範例"""
    print("\n" + "=" * 80)
    print("範例 2: 熱更新參數")
    print("=" * 80)
    
    # 1. 創建策略
    config_path = Path(__file__).parent.parent / "config_usdttwd.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    strategy = GridStrategy(config)
    
    print(f"更新前 small_gap: {strategy.params['small_gap']}")
    print(f"更新前 grid_layers[0].gap_abs: {strategy.grid_layers[0].gap_abs}")
    
    # 2. 熱更新參數
    new_params = {
        'small_gap': '0.08',  # 從 0.06 增加到 0.08
        'ema_span_fast_bars': 150,  # 從 120 增加到 150
        'bias_high': '0.65'  # 從 0.60 增加到 0.65
    }
    
    success = strategy.update_config(new_params)
    
    if success:
        print(f"\n✅ 參數更新成功！")
        print(f"更新後 small_gap: {strategy.params['small_gap']}")
        print(f"更新後 grid_layers[0].gap_abs: {strategy.grid_layers[0].gap_abs}")
        print(f"更新後 ema_span_fast_bars: {strategy.params['ema_span_fast_bars']}")
        print(f"更新後 bias_high: {strategy.params['bias_high']}")
    else:
        print("\n❌ 參數更新失敗")


async def example_market_observation():
    """獲取市場觀察數據範例"""
    print("\n" + "=" * 80)
    print("範例 3: 獲取市場觀察數據")
    print("=" * 80)
    
    # 1. 創建策略並模擬價格歷史
    config_path = Path(__file__).parent.parent / "config_usdttwd.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    strategy = GridStrategy(config)
    
    # 2. 模擬價格歷史（實際運行時由 BotEngine 注入）
    import time
    base_price = Decimal("30.0")
    for i in range(100):
        timestamp_ms = int(time.time() * 1000) + i * 60000  # 每分鐘一個價格點
        price = base_price + Decimal(str(i * 0.01))  # 模擬價格上升
        strategy.price_history.append((timestamp_ms, price))
    
    # 3. 獲取市場觀察數據
    observation = strategy.get_market_observation()
    
    print("市場觀察數據:")
    print(f"  時間戳: {observation['timestamp']}")
    print(f"  價格歷史長度: {observation['price_history_length']}")
    print(f"  策略狀態: {observation['strategy_state']}")
    print(f"  網格層數: {observation['grid_layers_count']}")
    
    print("\n技術指標:")
    for key, value in observation['indicators'].items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
    
    print("\n當前參數:")
    for key, value in observation['parameters'].items():
        print(f"  {key}: {value}")


async def example_workflow_integration():
    """workflow_manager 整合範例"""
    print("\n" + "=" * 80)
    print("範例 4: workflow_manager 整合（熱更新）")
    print("=" * 80)
    
    # 模擬 workflow_manager 的優化流程
    print("1. 執行參數優化...")
    # best_params = optimizer.optimize(...)
    
    # 模擬優化結果
    best_params = {
        'small_gap': '0.07',
        'ema_span_fast_bars': 130,
        'bias_high': '0.62'
    }
    
    print(f"2. 優化完成，最佳參數: {best_params}")
    
    # 假設 bot_engine 是全局可訪問的 BotEngine 實例
    # 在實際應用中，可以通過消息隊列、共享內存等方式訪問
    print("\n3. 熱更新策略參數（無需重啟）...")
    print("   # 在 workflow_manager.py 中：")
    print("   # success = bot_engine.strategy.update_config(best_params)")
    print("   # if success:")
    print("   #     log.info('Parameters updated successfully!')")
    print("   #     # 可選：立即觸發網格重建")
    print("   #     current_price = await bot_engine._get_current_price()")
    print("   #     await bot_engine._rebuild_grid_at_center(current_price)")


async def main():
    """運行所有範例"""
    await example_basic_usage()
    await example_hot_update()
    await example_market_observation()
    await example_workflow_integration()
    
    print("\n" + "=" * 80)
    print("所有範例運行完成！")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

