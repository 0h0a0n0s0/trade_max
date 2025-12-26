# 回測結果不一致問題修復指南

## 問題描述

使用優化結果 rank_176 的參數進行回測時，無法重現優化時的高 ROI 成績。

實際回測結果：
- 2024: 50.36%
- 2025: 2.49%

## 根本原因

**手續費率不一致**：
- 優化時使用的基礎配置：`backtest/config_usdttwd.yaml` (taker_fee: 0.0004, 0.04%)
- 實戰配置：`configs/config_rank77.yaml` (taker_fee: 0.0002, 0.02%)

優化結果是基於 0.04% 手續費找到的最優參數，使用 0.02% 手續費回測會導致結果偏差。

## 解決方案

### 方案1：驗證優化結果（用於確認問題）

將 `configs/config_rank77.yaml` 的 `taker_fee` 臨時改為 `0.0004`，重新回測：

```bash
# 回測 2024 數據
python core/backtester.py \
    --csv data/btctwd_1m_2024.csv \
    --config configs/config_rank77.yaml \
    --strategy-mode pure_grid \
    --init_usdt 10000.0 \
    --init_twd 300000.0

# 回測 2025 數據
python core/backtester.py \
    --csv data/btctwd_1m_2025.csv \
    --config configs/config_rank77.yaml \
    --strategy-mode pure_grid \
    --init_usdt 10000.0 \
    --init_twd 300000.0
```

如果使用 0.04% 手續費能重現高 ROI，則確認問題是手續費率不一致。

### 方案2：使用正確手續費重新優化（用於實戰）

由於實戰時手續費是 0.02%，應該使用正確的手續費率重新優化：

```bash
# 首先確保 config_rank77.yaml 的 taker_fee 為 0.0002
# 然後使用實戰配置作為基礎（taker_fee: 0.0002）
python core/optimizer.py \
    --csv data/btctwd_1m_2024.csv \
    --config configs/config_rank77.yaml \
    --strategy-mode pure_grid \
    --n-trials 200 \
    --n-jobs 4 \
    --output-yaml configs/best_params_fee_002.yaml \
    --output-csv backtest/optimization_results_fee_002.csv
```

這樣找到的參數才是適用於 0.02% 手續費的最優參數。

## 重要提醒

1. **策略模式一致性**：
   - 優化時使用的 `--strategy-mode` 必須與回測時一致
   - 優化器默認：`pure_grid`
   - 回測器默認：`hybrid`
   - **必須手動指定相同的模式**

2. **初始餘額一致性**：
   - 確保優化和回測使用相同的 `--init_usdt` 和 `--init_twd`
   - 默認值都是：`--init_usdt 10000.0 --init_twd 300000.0`

3. **基礎配置一致性**：
   - 優化時使用的 `--config` 基礎配置中的所有非優化參數
   - 應該與回測時使用的配置完全一致（特別是手續費率）

## 手續費率影響

0.04% vs 0.02% 的手續費差異會影響：
- 網格間距的有效性（較高手續費需要較大間距才有利潤）
- 交易頻率（較高手續費會減少交易次數）
- 最終 ROI（較高手續費會降低淨收益）

因此，基於 0.04% 手續費優化的參數，可能不適用於 0.02% 手續費的環境。

## 驗證步驟

1. **使用 0.04% 手續費驗證**（確認問題根源）：
   ```bash
   # 將 config_rank77.yaml 的 taker_fee 改為 0.0004
   # 然後使用 rank_176 參數回測
   python core/backtester.py --csv data/btctwd_1m_2024.csv --config configs/config_rank77.yaml --strategy-mode pure_grid
   ```

2. **如果驗證成功**（能重現高 ROI），說明問題確是手續費率不一致

3. **使用 0.02% 手續費重新優化**（用於實戰）：
   ```bash
   # 將 config_rank77.yaml 的 taker_fee 改回 0.0002
   # 然後重新執行優化
   python core/optimizer.py --csv data/btctwd_1m_2024.csv --config configs/config_rank77.yaml --strategy-mode pure_grid
   ```

