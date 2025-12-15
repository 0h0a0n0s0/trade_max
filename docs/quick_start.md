# 模組化架構快速開始指南

## 🎯 概述

本文檔說明如何使用新的模組化架構運行 Trade_Max 交易機器人。

## 📋 前置需求

1. Python 3.8+
2. 已安裝所有依賴（`requirements.txt`）
3. 配置好 `.env` 文件（API 金鑰等）
4. 配置好 `config_usdttwd.yaml`

## 🚀 快速開始

### 1. 基本運行

```bash
python main_modular.py
```

### 2. 架構組件說明

#### Strategy Layer（策略層）

```python
from strategy.grid_strategy import GridStrategy
import yaml

# 載入配置
with open('config_usdttwd.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 創建策略實例
strategy = GridStrategy(config)

# 策略參數是可調整的類屬性
print(f"當前網格間距: {strategy.small_gap}")
print(f"EMA 快線週期: {strategy.ema_span_fast_bars}")
```

#### Optimizer Layer（優化層）

```python
from optimizer.strategy_optimizer import StrategyOptimizer

# 創建優化器（需要策略實例）
optimizer_config = config.get('optimizer', {})
optimizer = StrategyOptimizer(strategy, optimizer_config)

# 優化器會自動：
# 1. 觀察市場狀態
# 2. 計算新參數
# 3. 應用參數到策略
```

#### Engine Layer（執行層）

```python
from engine.bot_engine import BotEngine
from pathlib import Path

# 創建引擎
engine = BotEngine(strategy, optimizer, Path('config_usdttwd.yaml'))

# 初始化和啟動
await engine.initialize()
await engine.start()
```

## ⚙️ 配置優化器

在 `config_usdttwd.yaml` 中添加以下配置：

```yaml
optimizer:
  # 是否啟用自動優化
  optimization_enabled: true
  
  # 優化檢查間隔（秒）
  optimization_interval_sec: 3600  # 每小時檢查一次
  
  # 最小績效變化閾值（低於此值不調整）
  min_performance_change_threshold: 0.05  # 5%
  
  # 參數調整範圍限制
  param_bounds:
    small_gap:
      min: 0.01
      max: 0.10
    ema_span_fast_bars:
      min: 5
      max: 20
    ema_span_slow_bars:
      min: 30
      max: 100
    bias_high:
      min: 0.5
      max: 0.8
    bias_low:
      min: 0.2
      max: 0.5
```

## 🔧 手動調整參數（測試用）

```python
# 在運行時動態調整參數
new_params = {
    'small_gap': Decimal('0.05'),
    'ema_span_fast_bars': 12,
    'bias_high': Decimal('0.65')
}

# 應用參數
success = strategy.update_config(new_params)
if success:
    print("參數已成功更新")
```

## 📊 監控優化器狀態

```python
# 獲取優化報告
report = optimizer.get_optimization_report()
print(f"優化次數: {report['total_optimizations']}")
print(f"最近績效: {report['recent_performance']}")
```

## 🔍 狀態觀察

```python
# 獲取策略狀態向量
state_vector = strategy.get_state_vector()
print(f"當前指標: {state_vector['indicators']}")
print(f"當前參數: {state_vector['config_snapshot']}")

# 獲取市場狀態
market_state = optimizer.observe_market_state(
    current_price=Decimal('30.5'),
    total_equity=Decimal('1000000'),
    realized_pnl=Decimal('5000'),
    active_orders_count=10
)
print(f"市場波動性: {market_state['volatility']}")
print(f"趨勢強度: {market_state['trend_strength']}")
```

## 🛡️ 安全機制

### 參數驗證

所有參數調整都會經過驗證：

```python
# 如果參數超出範圍，會被自動限制
new_params = {'small_gap': Decimal('0.20')}  # 超出 max: 0.10
optimizer.apply_parameters(new_params)
# → small_gap 會被限制為 0.10
```

### 錯誤處理

- API 超時：自動重試
- 參數調整失敗：記錄錯誤，不影響運行
- 嚴重錯誤：進入安全模式（停止下單）

## 📈 績效追蹤

```python
# 計算當前績效
performance = engine._calculate_performance()
print(f"ROI: {performance['roi']:.2%}")
print(f"已實現損益: {performance['realized_pnl']:.2f} TWD")
print(f"最大回撤: {performance['max_drawdown']:.2%}")

# 記錄績效（用於 RL 訓練）
optimizer.record_performance(performance)
```

## 🔄 與舊架構的差異

### 舊架構（strategy_usdttwd_grid.py）

- 線性腳本
- 參數硬編碼在配置文件中
- 無法動態調整

### 新架構（模組化）

- 三層分離
- 參數作為類屬性，可動態調整
- AI Agent 自動優化

## 🐛 故障排除

### 問題：策略參數無法更新

**解決方案：**
1. 檢查參數名稱是否正確
2. 檢查參數範圍是否在 `param_bounds` 內
3. 查看日誌中的錯誤訊息

### 問題：優化器不工作

**解決方案：**
1. 檢查 `optimization_enabled` 是否為 `true`
2. 檢查 `optimization_interval_sec` 是否足夠長
3. 查看優化器日誌

### 問題：訂單無法下單

**解決方案：**
1. 檢查 API 連接
2. 檢查餘額是否足夠
3. 檢查風險控制器設置

## 📚 更多資訊

- 詳細架構設計：`docs/architecture_design.md`
- 原始策略實現：`strategy_usdttwd_grid.py`
- Freqtrade 參考：`freqtrade-develop/freqtrade/strategy/interface.py`

## ✅ 檢查清單

運行前確認：

- [ ] `.env` 文件已配置
- [ ] `config_usdttwd.yaml` 已配置
- [ ] 資料庫連接正常
- [ ] API 金鑰有效
- [ ] 優化器配置已添加（可選）

## 🎓 下一步

1. **測試基本功能**
   - 運行 `main_modular.py`
   - 確認策略正常運行

2. **測試參數調整**
   - 手動調整參數
   - 觀察策略行為變化

3. **啟用自動優化**
   - 配置優化器
   - 觀察自動調整效果

4. **整合 RL 模型**（未來）
   - 訓練 RL 模型
   - 替換規則式優化

