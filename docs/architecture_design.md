# Trade_Max 模組化架構設計文檔

## 📋 概述

本文檔描述 Trade_Max 專案從線性腳本重構為模組化、AI-ready 架構的設計方案。該架構參考 Freqtrade 的設計理念，但保留原有的 Grid + Trend 交易邏輯。

## 🏗️ 架構原則

### 1. 三層分離架構

```
┌─────────────────────────────────────┐
│   Strategy Layer (策略層)           │
│   - 純邏輯計算                      │
│   - 技術指標計算                    │
│   - 信號生成                        │
│   - NO API 調用                    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Execution Layer (執行層)           │
│   - BotEngine                       │
│   - 主循環管理                      │
│   - API 調用                        │
│   - 訂單執行                        │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Optimization Layer (優化層)        │
│   - StrategyOptimizer               │
│   - 市場狀態觀察                    │
│   - 參數動態調整                    │
│   - 績效追蹤                        │
└─────────────────────────────────────┘
```

### 2. 可注入參數設計

所有策略參數必須是**類屬性**，可以通過 `update_config()` 方法動態修改：

```python
class GridStrategy(BaseStrategy):
    # 可調整參數（類屬性）
    small_gap: Decimal = Decimal("0.035")
    ema_span_fast_bars: int = 10
    bias_high: Decimal = Decimal("0.6")
    
    def update_config(self, new_params: Dict[str, Any]) -> bool:
        # AI Agent 通過此方法調整參數
        if 'small_gap' in new_params:
            self.small_gap = Decimal(str(new_params['small_gap']))
        # ...
```

## 📂 文件結構

```
NoAI/
├── strategy/                    # 策略層
│   ├── __init__.py
│   ├── base_strategy.py         # 抽象基類
│   └── grid_strategy.py         # 網格策略實現
│
├── optimizer/                   # 優化層
│   ├── __init__.py
│   └── strategy_optimizer.py   # AI Agent
│
├── engine/                      # 執行層
│   ├── __init__.py
│   └── bot_engine.py           # 執行引擎
│
├── main_modular.py             # 模組化架構入口
├── strategy_usdttwd_grid.py    # 原始腳本（保留）
│
└── docs/
    └── architecture_design.md  # 本文檔
```

## 🔧 核心類別設計

### 1. BaseStrategy（抽象基類）

**職責：**
- 定義策略介面
- 提供狀態向量導出（供 AI Agent 觀察）
- 管理價格歷史

**關鍵方法：**
- `calculate_indicators()` - 計算技術指標
- `generate_signals()` - 生成交易信號
- `update_config()` - 動態更新參數
- `get_state_vector()` - 導出狀態向量

### 2. GridStrategy（網格策略）

**職責：**
- 實現三層網格邏輯
- EMA 趨勢判斷
- 混合策略模式（網格 + 趨勢跟隨）

**可調整參數：**
- `small_gap` - 小網格間距
- `mid_multiplier` - 中網格倍數
- `big_multiplier` - 大網格倍數
- `ema_span_fast_bars` - EMA 快線週期
- `ema_span_slow_bars` - EMA 慢線週期
- `bias_high` - 看漲偏置比例
- `bias_low` - 看跌偏置比例
- `use_atr_spacing` - 是否使用 ATR 動態間距
- `atr_spacing_multiplier` - ATR 間距倍數

### 3. StrategyOptimizer（AI Agent）

**職責：**
- 觀察市場狀態和策略表現
- 決定是否需要調整參數
- 計算新的參數值
- 追蹤參數調整的結果（用於 RL 訓練）

**工作流程：**
```
1. observe_market_state() - 觀察市場
   ↓
2. optimize_parameters() - 計算新參數
   ↓
3. apply_parameters() - 應用參數
   ↓
4. record_performance() - 記錄績效（用於 RL）
```

**當前實現：**
- 基於規則的優化（簡化版）
- 未來可替換為：
  - 強化學習模型
  - 遺傳算法
  - 貝葉斯優化

### 4. BotEngine（執行引擎）

**職責：**
- 管理主循環
- 獲取市場數據並注入到策略
- 執行策略生成的信號
- 管理訂單狀態
- 協調 StrategyOptimizer 進行參數調整
- 追蹤績效並提供給 Optimizer

**主循環流程：**
```
1. 更新價格歷史
   ↓
2. 輪詢訂單狀態
   ↓
3. 更新餘額
   ↓
4. 檢查優化器（是否需要調整參數）
   ↓
5. 執行策略邏輯（網格重建、偏置調整）
   ↓
6. 資料庫快照
```

## 🔄 參數調整流程

### 安全調整機制

1. **參數驗證**
   - 檢查參數是否在允許範圍內
   - 防止無效值導致策略崩潰

2. **漸進式調整**
   - 不一次性大幅調整
   - 記錄調整歷史

3. **績效追蹤**
   - 記錄調整前的績效
   - 記錄調整後的績效
   - 計算獎勵（用於 RL）

### 範例：動態調整網格間距

```python
# 1. Optimizer 觀察到高波動性
market_state = optimizer.observe_market_state(...)
# market_state['volatility'] = 0.025  # 2.5% 波動

# 2. Optimizer 計算新參數
new_params = optimizer.optimize_parameters(market_state, performance)
# new_params = {'small_gap': Decimal('0.042')}  # 從 0.035 增加到 0.042

# 3. Optimizer 應用參數
optimizer.apply_parameters(new_params)
# → 調用 strategy.update_config(new_params)

# 4. Strategy 更新參數並重建網格層級
strategy.update_config(new_params)
# → self.small_gap = Decimal('0.042')
# → self._rebuild_grid_layers()

# 5. BotEngine 在下一個循環中重建網格
# → 使用新的間距參數
```

## 📊 狀態觀察與獎勵反饋

### 狀態向量（State Vector）

策略導出的狀態向量包含：

```python
{
    'strategy_name': str,
    'is_active': bool,
    'last_update_ts': str,
    'indicators': {
        'ema_fast': float,
        'ema_slow': float,
        'atr': float,
        'adx': float,
        ...
    },
    'config_snapshot': {
        'small_gap': float,
        'ema_span_fast_bars': int,
        'bias_high': float,
        ...
    }
}
```

### 績效指標（Performance Metrics）

```python
{
    'roi': Decimal,              # 總收益率
    'realized_pnl': Decimal,     # 已實現損益
    'max_drawdown': Decimal,     # 最大回撤
    'total_equity': Decimal,     # 總權益
    'sharpe_ratio': float,      # 夏普比率
    ...
}
```

### 獎勵計算（未來 RL 實現）

```python
# 簡化版獎勵函數
def calculate_reward(performance_before, performance_after, params):
    roi_change = performance_after['roi'] - performance_before['roi']
    drawdown_change = performance_before['max_drawdown'] - performance_after['max_drawdown']
    
    reward = roi_change * 10 + drawdown_change * 5
    return reward
```

## 🚀 使用方式

### 基本使用

```python
from strategy.grid_strategy import GridStrategy
from optimizer.strategy_optimizer import StrategyOptimizer
from engine.bot_engine import BotEngine
import yaml

# 1. 載入配置
with open('config_usdttwd.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. 創建實例
strategy = GridStrategy(config)
optimizer = StrategyOptimizer(strategy, config.get('optimizer', {}))
engine = BotEngine(strategy, optimizer, Path('config_usdttwd.yaml'))

# 3. 啟動
await engine.initialize()
await engine.start()
```

### 手動調整參數（測試用）

```python
# 直接調整策略參數
new_params = {
    'small_gap': Decimal('0.05'),
    'ema_span_fast_bars': 12
}
strategy.update_config(new_params)
```

## 🔐 安全機制

### 1. 參數範圍限制

所有可調整參數都有範圍限制：

```python
param_bounds = {
    'small_gap': {'min': 0.01, 'max': 0.10},
    'ema_span_fast_bars': {'min': 5, 'max': 20},
    ...
}
```

### 2. 錯誤處理

- 策略層：參數驗證、指標計算錯誤處理
- 執行層：API 超時重試、訂單失敗處理
- 優化層：參數調整失敗回滾

### 3. 安全模式

當發生嚴重錯誤時，BotEngine 可以進入「安全模式」：
- 停止下單
- 保持監控
- 等待人工介入

## 🔮 未來擴展

### 1. 強化學習整合

```python
class RLStrategyOptimizer(StrategyOptimizer):
    def __init__(self, strategy, config):
        super().__init__(strategy, config)
        self.rl_model = load_rl_model()  # 載入訓練好的 RL 模型
    
    def optimize_parameters(self, market_state, performance):
        # 使用 RL 模型選擇動作（參數調整）
        action = self.rl_model.predict(market_state)
        return self._action_to_params(action)
```

### 2. 多策略支援

```python
# BotEngine 可以管理多個策略
strategies = [
    GridStrategy(config),
    TrendFollowingStrategy(config),
    ArbitrageStrategy(config)
]

# Optimizer 可以為每個策略獨立優化
optimizers = [StrategyOptimizer(s, config) for s in strategies]
```

### 3. 分散式優化

```python
# 多個 Optimizer 實例協同工作
# 使用共享的參數歷史和績效數據
```

## 📝 遷移指南

### 從舊架構遷移

1. **保留現有功能**
   - `max_async_api.py` - 保留
   - `db.py` - 保留
   - `telegram_alerter.py` - 保留

2. **逐步遷移**
   - 先運行 `main_modular.py` 測試
   - 確認功能正常後，逐步遷移配置
   - 最後替換 `strategy_usdttwd_grid.py`

3. **配置調整**
   - 在 `config_usdttwd.yaml` 中添加 `optimizer` 區塊：

```yaml
optimizer:
  optimization_enabled: true
  optimization_interval_sec: 3600
  min_performance_change_threshold: 0.05
  param_bounds:
    small_gap:
      min: 0.01
      max: 0.10
    ema_span_fast_bars:
      min: 5
      max: 20
    # ...
```

## ✅ 檢查清單

- [x] BaseStrategy 抽象基類
- [x] GridStrategy 實現
- [x] StrategyOptimizer AI Agent
- [x] BotEngine 執行引擎
- [x] 參數動態調整機制
- [x] 狀態觀察與獎勵反饋
- [x] 架構設計文檔
- [ ] 完整測試（待實現）
- [ ] RL 模型整合（未來）

## 📚 參考資料

- Freqtrade Strategy Interface: `freqtrade-develop/freqtrade/strategy/interface.py`
- 原始策略實現: `strategy_usdttwd_grid.py`
- 專案規則: `.cursor/rules/trade-rules.mdc`

