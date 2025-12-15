# OOP 重構：支援 AI 動態參數優化

**日期**: 2025-01-23  
**版本**: 2.0  
**狀態**: ✅ 已完成

## 📋 背景

### 問題

原始的 `strategy_usdttwd_grid.py` 使用全域變數（`CFG`, `GRID_LAYERS`, `ACTIVE_ORDERS`）來管理策略狀態和參數。這導致：

1. **無法動態調整參數**：AI Agent 無法在不重啟進程的情況下更新策略參數
2. **狀態管理混亂**：全域變數散佈在各處，難以追蹤
3. **無法進行單元測試**：邏輯與執行耦合，難以測試
4. **無法並行運行多個策略**：全域狀態導致無法實例化多個策略

### 目標

將腳本式架構重構為類式架構，實現：

1. ✅ **熱更新參數**：AI Agent 可以動態調整參數，無需重啟
2. ✅ **狀態封裝**：策略狀態封裝在類中，易於管理
3. ✅ **可測試性**：邏輯與執行分離，易於單元測試
4. ✅ **可擴展性**：支持多策略實例並行運行

## 🏗️ 架構設計

### 類別結構

```
GridStrategy (策略類)
├── 職責：策略邏輯和參數管理
├── 狀態：grid_layers, strategy_state, trend_position
├── 方法：
│   ├── update_config() - 熱更新參數
│   ├── get_market_observation() - 提供市場觀察數據
│   ├── should_rebuild_grid() - 判斷是否需要重建網格
│   ├── get_ema_target_bias() - 計算EMA目標偏置
│   └── _rebuild_grid_layers() - 重建網格層級
└── 無 API 調用（純邏輯）

BotEngine (執行引擎)
├── 職責：主循環、API調用、訂單執行
├── 狀態：active_orders, balances, risk_controller
├── 方法：
│   ├── initialize() - 初始化
│   ├── start() - 啟動主循環
│   ├── _main_loop() - 主循環邏輯
│   ├── _rebuild_grid_at_center() - 重建網格（使用策略邏輯）
│   ├── _place_grid_order() - 下單
│   ├── _manage_hybrid_strategy() - 管理混合策略
│   └── _manage_directional_bias() - 管理方向性偏置
└── 持有 GridStrategy 實例
```

### 關鍵特性

#### 1. 參數熱更新

```python
# AI Agent 可以動態調整參數
new_params = {
    'small_gap': '0.05',
    'ema_span_fast_bars': 120,
    'bias_high': '0.65'
}

# 無需重啟，立即生效
success = strategy.update_config(new_params)
if success:
    # 參數已更新，grid_layers 已重建
    # 下次重建網格時會使用新參數
    pass
```

#### 2. 市場觀察

```python
# AI Agent 可以獲取當前市場狀態
observation = strategy.get_market_observation()

# 返回：
# {
#     'timestamp': '2025-01-23T10:00:00',
#     'indicators': {
#         'ema_fast': 30.5,
#         'ema_slow': 30.2,
#         'atr': 0.05,
#         'adx': 25.3,
#         'rsi': 55.2,
#         'macd': 0.02,
#         'volatility': 0.001
#     },
#     'parameters': {
#         'small_gap': 0.035,
#         'ema_span_fast_bars': 120,
#         ...
#     },
#     'strategy_state': 'GRID',
#     ...
# }
```

#### 3. 配置注入

```python
# 不再使用 load_cfg()，而是直接傳遞配置字典
with open('config_usdttwd.yaml', 'r') as f:
    config = yaml.safe_load(f)

strategy = GridStrategy(config)
engine = BotEngine(strategy, config_path)
```

## 🔄 遷移指南

### 從舊架構遷移

#### 舊代碼（腳本式）

```python
# 舊代碼
load_cfg()  # 載入配置到全域變數 CFG
# ... 使用 CFG, GRID_LAYERS, ACTIVE_ORDERS 等全域變數
```

#### 新代碼（類式）

```python
# 新代碼
with open('config_usdttwd.yaml', 'r') as f:
    config = yaml.safe_load(f)

strategy = GridStrategy(config)
engine = BotEngine(strategy, Path('config_usdttwd.yaml'))
await engine.initialize()
await engine.start()
```

### workflow_manager.py 整合

#### 舊方式（需要重啟）

```python
# 舊方式：需要重啟進程
# 1. 優化參數
best_params = optimizer.optimize(...)

# 2. 更新配置文件
with open('config_usdttwd.yaml', 'w') as f:
    yaml.dump(best_params, f)

# 3. 重啟進程（需要外部腳本）
# subprocess.call(['pkill', '-f', 'strategy_usdttwd_grid.py'])
# subprocess.call(['python', 'strategy_usdttwd_grid.py'])
```

#### 新方式（熱更新）

```python
# 新方式：直接更新參數，無需重啟
# 假設 bot 是 BotEngine 實例（可以通過全局變數或消息隊列訪問）

# 1. 優化參數
best_params = optimizer.optimize(...)

# 2. 直接更新策略參數
success = bot.strategy.update_config(best_params)

if success:
    log.info("Strategy parameters updated successfully!")
    # 參數已更新，下次重建網格時會使用新參數
    # 或者可以立即觸發重建：
    # current_price = await bot._get_current_price()
    # await bot._rebuild_grid_at_center(current_price)
```

## 📝 使用範例

### 基本使用

```python
import asyncio
import yaml
from pathlib import Path
from strategy_usdttwd_grid_refactored import GridStrategy, BotEngine

async def main():
    # 1. 載入配置
    config_path = Path("config_usdttwd.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. 創建策略和引擎
    strategy = GridStrategy(config)
    engine = BotEngine(strategy, config_path)
    
    # 3. 初始化並啟動
    await engine.initialize()
    await engine.start()

if __name__ == "__main__":
    asyncio.run(main())
```

### AI Agent 熱更新參數

```python
# 在另一個進程或線程中（例如 workflow_manager.py）

# 獲取 bot 實例（可以通過消息隊列、共享內存等方式）
# 這裡假設有一個全局的 bot 實例

async def update_strategy_parameters(new_params: dict):
    """AI Agent 更新策略參數"""
    # 驗證參數
    validated_params = validate_parameters(new_params)
    
    # 熱更新
    success = bot.strategy.update_config(validated_params)
    
    if success:
        log.info(f"Parameters updated: {list(validated_params.keys())}")
        
        # 可選：立即觸發網格重建
        current_price = await bot._get_current_price()
        if current_price:
            await bot._rebuild_grid_at_center(current_price)
    else:
        log.error("Failed to update parameters")
```

### 獲取市場觀察數據

```python
# AI Agent 獲取市場狀態
observation = bot.strategy.get_market_observation()

# 使用觀察數據進行決策
if observation['indicators']['adx'] > 25:
    # 強趨勢市場，可能需要調整參數
    new_params = {
        'use_hybrid_model': True,
        'trend_trade_equity_pct': '0.5'
    }
    bot.strategy.update_config(new_params)
```

## 🔍 關鍵改進

### 1. 參數管理

**舊方式**：
```python
# 全域變數
CFG = {}
load_cfg()  # 載入到 CFG
# 無法動態更新
```

**新方式**：
```python
# 類屬性
strategy.params = {...}
strategy.update_config(new_params)  # 可以動態更新
```

### 2. 狀態管理

**舊方式**：
```python
# 全域變數散佈
GRID_LAYERS = []
ACTIVE_ORDERS = {}
STRATEGY_STATE = "GRID"
```

**新方式**：
```python
# 封裝在類中
strategy.grid_layers
strategy.strategy_state
engine.active_orders
```

### 3. 邏輯分離

**舊方式**：
```python
# 邏輯和執行混在一起
async def rebuild_grid_at_center(...):
    # 直接使用全域變數 CFG, GRID_LAYERS
    # 直接調用 API
    pass
```

**新方式**：
```python
# 策略類：純邏輯
class GridStrategy:
    def should_rebuild_grid(self, current_time):
        # 只判斷邏輯，不執行
        pass

# 引擎類：執行
class BotEngine:
    async def _rebuild_grid_at_center(self, ...):
        # 使用 strategy 的邏輯
        # 執行 API 調用
        pass
```

## ✅ 保留的功能

所有原有功能都已保留：

- ✅ `rebuild_grid_at_center` 邏輯（封裝在 `BotEngine._rebuild_grid_at_center`）
- ✅ `manage_hybrid_strategy` 邏輯（封裝在 `BotEngine._manage_hybrid_strategy`）
- ✅ `max_async_api` 集成（保留在 `BotEngine` 中）
- ✅ `db.py` 和 `db_schema.py` 集成（保留在 `BotEngine` 中）
- ✅ 所有風險控制和黑天鵝保護邏輯

## 📊 對比表

| 特性 | 舊架構（腳本式） | 新架構（類式） |
|------|----------------|--------------|
| 參數更新 | 需要重啟進程 | ✅ 熱更新（`update_config()`） |
| 狀態管理 | 全域變數 | ✅ 類屬性 |
| 市場觀察 | 無統一介面 | ✅ `get_market_observation()` |
| 可測試性 | 困難 | ✅ 易於單元測試 |
| 多實例 | 不支持 | ✅ 支持多策略實例 |
| 配置載入 | `load_cfg()` | ✅ 構造函數注入 |

## 🚀 下一步

### 1. 整合到 workflow_manager.py

修改 `workflow_manager.py` 以支持熱更新：

```python
# workflow_manager.py 修改示例
class WorkflowManager:
    def __init__(self, bot_engine: BotEngine):
        self.bot_engine = bot_engine  # 持有 BotEngine 實例
    
    async def _run_weekly_optimization(self):
        # ... 優化邏輯 ...
        
        if validation_result['roi'] >= min_test_roi:
            # 熱更新參數，無需重啟
            new_params = extract_params_from_study(study)
            success = self.bot_engine.strategy.update_config(new_params)
            
            if success:
                msg = f"✅ 參數已熱更新！無需重啟。"
                await alerter.send_strategy_event(msg)
```

### 2. 添加參數驗證

在 `GridStrategy.update_config()` 中添加參數驗證：

```python
def update_config(self, new_params: Dict[str, Any]) -> bool:
    # 驗證參數範圍
    if 'small_gap' in new_params:
        gap = Decimal(str(new_params['small_gap']))
        if gap < Decimal("0.01") or gap > Decimal("0.10"):
            log.error(f"Invalid small_gap: {gap}")
            return False
    # ...
```

### 3. 添加參數變更通知

當參數更新時，發送 Telegram 通知：

```python
def update_config(self, new_params: Dict[str, Any]) -> bool:
    # ... 更新邏輯 ...
    
    if updated:
        # 發送通知
        msg = f"策略參數已更新：{list(new_params.keys())}"
        asyncio.create_task(alerter.send_strategy_event(msg))
    
    return updated
```

## 📚 相關文檔

- [`architecture_design.md`](../architecture_design.md) - 模組化架構設計
- [`quick_start.md`](../quick_start.md) - 快速開始指南

## ⚠️ 注意事項

1. **向後兼容**：原始 `strategy_usdttwd_grid.py` 保留，新版本為 `strategy_usdttwd_grid_refactored.py`
2. **測試**：建議先在測試環境中驗證新架構
3. **遷移**：可以逐步遷移，先運行新版本並行測試

---

**最後更新**: 2025-01-23  
**狀態**: ✅ 完成

