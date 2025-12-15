# Backtest Adapter 實現：確保回測與實盤邏輯一致性

**日期**: 2025-01-23  
**版本**: 3.0  
**狀態**: ✅ 已完成

## 📋 背景

### 問題：邏輯分歧（Logic Divergence）

原始的優化腳本使用獨立的 `Backtester` 類進行回測，該類實現了簡化版的策略邏輯。這導致：

1. **過擬合風險**：回測結果無法轉移到實盤
2. **邏輯不一致**：回測和實盤使用不同的代碼路徑
3. **維護困難**：需要同時維護兩套邏輯

### 目標

創建一個 **Backtest Adapter**，使用與實盤完全相同的 `GridStrategy` 邏輯，確保：

- ✅ 回測和實盤使用相同的策略決策邏輯
- ✅ 避免邏輯分歧導致的過擬合
- ✅ 提高回測結果的可信度

## 🏗️ 架構設計

### 核心概念

```
實盤模式：
BotEngine → GridStrategy → 決策邏輯

回測模式：
BacktestAdapter → GridStrategy → 決策邏輯（相同！）
```

### 實現步驟

#### 1. 修改 GridStrategy 支持數據注入

**修改的方法**：
- `_calculate_ema_from_history()` - 支持 `external_data` 參數
- `_calculate_atr_from_history()` - 支持 `external_high/low/close` 參數
- `_calculate_adx_from_history()` - 支持 `external_high/low/close` 參數
- `get_ema_target_bias()` - 支持 `external_data` 參數

**邏輯**：
- 如果提供外部數據（回測模式），使用向量化計算（快速）
- 如果沒有外部數據（實盤模式），使用 `self.price_history`（原有邏輯）

#### 2. 創建 BacktestAdapter 類

**職責**：
- 模擬 `BotEngine` 的執行環境
- 使用相同的 `GridStrategy` 實例
- 模擬訂單匹配、PnL 計算等

**關鍵方法**：
- `run(ohlc_df)` - 執行回測主循環
- `_check_order_fills()` - 檢查訂單成交
- `_rebuild_grid_simulated()` - 模擬網格重建
- `_simulate_hybrid_strategy()` - 模擬混合策略
- `_simulate_directional_bias()` - 模擬方向性偏置
- `_check_black_swan()` - 檢查黑天鵝事件

#### 3. 更新 optimize_params_parallel.py

**改進**：
- 使用 `BacktestAdapter` 替代 `Backtester`
- 添加 **Robustness Score** 計算
- 使用 Robustness Score 進行排序和篩選

**Robustness Score 公式**：
```
score = roi_pct * 0.4 + (100 / (max_drawdown_pct + 1)) * 0.6
```

這個公式平衡了收益和風險，優先考慮穩健性。

## 📝 實現細節

### 1. GridStrategy 數據注入

```python
def _calculate_ema_from_history(self, span: int, external_data: Optional[pd.Series] = None) -> Optional[Decimal]:
    """
    計算EMA指標
    
    Args:
        span: EMA週期
        external_data: 可選的外部數據（pandas Series），用於回測時的向量化計算
    
    Returns:
        Optional[Decimal]: EMA值
    """
    if external_data is not None:
        # 使用外部數據（回測模式）
        ema_val = external_data.ewm(span=span, adjust=False).mean().iloc[-1]
        return Decimal(str(ema_val))
    else:
        # 使用內部歷史數據（實盤模式）
        prices = [p[1] for p in self.price_history]
        series = pd.Series(prices, dtype=float)
        ema_val = series.ewm(span=span, adjust=False).mean().iloc[-1]
        return Decimal(str(ema_val))
```

### 2. BacktestAdapter 主循環

```python
def run(self, ohlc_df: pd.DataFrame) -> Dict[str, Any]:
    """執行回測"""
    # 預計算指標（向量化，一次性計算）
    ema_fast_series = ema(price_series, ema_fast_span)
    ema_slow_series = ema(price_series, ema_slow_span)
    adx_series, _, _ = adx(high_series, low_series, price_series, dmi_period)
    
    # 主循環
    for idx, (timestamp, row) in enumerate(ohlc_df.iterrows()):
        # 1. 檢查訂單成交
        self._check_order_fills(high, low, close, idx)
        
        # 2. 混合策略管理（使用相同的 GridStrategy 邏輯）
        if self.strategy.params.get('use_hybrid_model', False):
            self._simulate_hybrid_strategy(close, ema_fast_val, ema_slow_val, adx_val)
        
        # 3. 方向性偏置調整（使用相同的 GridStrategy 邏輯）
        if self.strategy.should_rebalance_bias(self.current_time):
            self._simulate_directional_bias(close, ema_fast_val, ema_slow_val)
        
        # 4. 網格重建（使用相同的 GridStrategy 邏輯）
        if self.strategy.should_rebuild_grid(self.current_time):
            self._rebuild_grid_simulated(close, full_rebuild=True)
```

### 3. Robustness Score 計算

```python
# 在 run_single_backtest 中
roi_pct = stats['roi_pct']
max_dd_pct = stats['max_drawdown_pct']
robustness_score = roi_pct * 0.4 + (100 / (max_dd_pct + 1)) * 0.6

stats['robustness_score'] = robustness_score

# 篩選條件
if stats['roi_pct'] > 0.5 and stats['max_drawdown_pct'] < 15.0 and robustness_score > 10.0:
    return {'params': params, 'stats': stats, 'success': True}
```

## ✅ 完成的工作

### 1. 修改 GridStrategy

- ✅ `_calculate_ema_from_history()` - 支持外部數據注入
- ✅ `_calculate_atr_from_history()` - 支持外部數據注入
- ✅ `_calculate_adx_from_history()` - 支持外部數據注入
- ✅ `get_ema_target_bias()` - 支持外部數據注入

### 2. 創建 BacktestAdapter

- ✅ `BacktestAdapter` 類實現
- ✅ 模擬訂單匹配邏輯
- ✅ 模擬混合策略邏輯
- ✅ 模擬方向性偏置邏輯
- ✅ 黑天鵝事件檢查
- ✅ PnL 和回撤計算

### 3. 更新 optimize_params_parallel.py

- ✅ 使用 `BacktestAdapter` 替代 `Backtester`
- ✅ 添加 Robustness Score 計算
- ✅ 使用 Robustness Score 排序
- ✅ 更新輸出格式

## 📊 對比

### 舊架構（邏輯分歧）

```
回測：Backtester (獨立邏輯) → 簡化版策略
實盤：BotEngine → GridStrategy → 完整策略

問題：兩套邏輯，容易過擬合
```

### 新架構（邏輯一致）

```
回測：BacktestAdapter → GridStrategy → 完整策略
實盤：BotEngine → GridStrategy → 完整策略

優勢：同一套邏輯，避免過擬合
```

## 🚀 使用方式

### 基本回測

```python
from backtest_adapter import BacktestAdapter
from strategy_usdttwd_grid_refactored import GridStrategy
import yaml

# 載入配置
with open('config_usdttwd.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 創建策略（與實盤相同）
strategy = GridStrategy(config)

# 創建適配器
adapter = BacktestAdapter(
    strategy=strategy,
    init_usdt=Decimal("10000"),
    init_twd=Decimal("300000"),
    fee_rate=Decimal("0.0004")
)

# 執行回測
result = adapter.run(ohlc_df)

print(f"ROI: {result['roi_pct']:.2f}%")
print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
print(f"Robustness Score: {result.get('robustness_score', 0):.2f}")
```

### 參數優化

```bash
python backtest/optimize_params_parallel.py \
    --csv backtest/usdttwd_1m_25y7m.csv \
    --config backtest/config_usdttwd.yaml \
    --target 100 \
    --max-iter 2000 \
    --workers 4
```

## 📈 Robustness Score 說明

### 公式

```
score = roi_pct * 0.4 + (100 / (max_drawdown_pct + 1)) * 0.6
```

### 解釋

- **ROI 權重 (40%)**：考慮收益
- **回撤權重 (60%)**：優先考慮風險控制
- **分母 +1**：避免除零錯誤，並平滑小回撤的影響

### 示例

| ROI | Max DD | Robustness Score |
|-----|--------|------------------|
| 5%  | 10%    | 5 * 0.4 + (100/11) * 0.6 = 7.45 |
| 10% | 5%     | 10 * 0.4 + (100/6) * 0.6 = 14.00 |
| 3%  | 2%     | 3 * 0.4 + (100/3) * 0.6 = 20.20 |

**結論**：低回撤的策略會獲得更高的 Robustness Score，即使 ROI 較低。

## ⚠️ 注意事項

1. **性能**：BacktestAdapter 使用向量化指標計算，比實盤模式更快
2. **簡化**：某些實盤細節（如訂單部分成交）在回測中簡化處理
3. **一致性**：策略決策邏輯完全一致，但執行細節可能略有不同

## 📚 相關文檔

- [`20250123_OOP_Refactor_for_AI.md`](20250123_OOP_Refactor_for_AI.md) - OOP 重構文檔
- [`backtest/README.md`](../backtest/README.md) - 回測目錄說明

---

**最後更新**: 2025-01-23  
**狀態**: ✅ 完成

