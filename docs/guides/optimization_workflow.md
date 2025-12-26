# 參數優化工作流程指南

## 完整優化流程

### 步驟 1：準備配置

確保 `configs/config_rank77.yaml` 的手續費率設置正確：
- 實戰手續費：`taker_fee: 0.0002` (0.02%)
- 其他參數已設置為當前最佳值

### 步驟 2：執行優化

使用正確的手續費率進行參數優化：

```bash
# 方式1：使用優化腳本（推薦）
bash scripts/optimize_with_fee_002.sh

# 方式2：直接執行命令
python core/optimizer.py \
    --csv data/btctwd_1m_2024.csv \
    --config configs/config_rank77.yaml \
    --strategy-mode pure_grid \
    --n-trials 200 \
    --n-jobs 4 \
    --output-yaml configs/best_params_fee_002.yaml \
    --output-csv backtest/optimization_results_fee_002.csv
```

**參數說明：**
- `--csv`: 訓練數據（使用 2024 年數據）
- `--config`: 基礎配置文件（必須包含正確的手續費率）
- `--strategy-mode`: 策略模式（必須與回測時一致）
- `--n-trials`: 優化試驗次數（建議 200+）
- `--n-jobs`: 並行任務數（建議 4，根據 CPU 核心數調整）
- `--output-yaml`: 最佳參數輸出路徑
- `--output-csv`: 所有試驗結果 CSV 路徑

### 步驟 3：驗證新參數

使用新找到的最佳參數回測 2024 和 2025 數據：

```bash
# 使用驗證腳本（推薦）
bash scripts/validate_new_params.sh

# 或手動驗證
python core/backtester.py \
    --csv data/btctwd_1m_2024.csv \
    --config configs/best_params_fee_002.yaml \
    --strategy-mode pure_grid

python core/backtester.py \
    --csv data/btctwd_1m_2025.csv \
    --config configs/best_params_fee_002.yaml \
    --strategy-mode pure_grid
```

### 步驟 4：應用新參數

如果驗證結果滿意，將最佳參數合併到實戰配置：

```bash
# 使用 Python 腳本合併（需要創建）
python scripts/merge_best_params.py \
    configs/best_params_fee_002.yaml \
    configs/config_rank77.yaml
```

或手動將最佳參數複製到 `config_rank77.yaml`，但保留以下設置：
- `asset_pair: btctwd`
- `usdt_unit: BTC`
- `twd_unit: TWD`
- `taker_fee: 0.0002`

## 重要注意事項

### 1. 手續費率一致性

**關鍵原則**：優化時使用的手續費率必須與實戰時一致！

- ❌ **錯誤**：使用 0.04% 手續費優化，但實戰使用 0.02%
- ✅ **正確**：優化和實戰都使用 0.02% 手續費

### 2. 策略模式一致性

**關鍵原則**：優化時使用的策略模式必須與回測時一致！

- 優化器默認：`pure_grid`
- 回測器默認：`hybrid`
- **必須手動指定相同的模式**

### 3. 初始餘額一致性

確保優化和回測使用相同的初始餘額：
- `--init_usdt 10000.0`
- `--init_twd 300000.0`

### 4. 基礎配置一致性

優化時使用的基礎配置中的所有非優化參數，應該與實戰配置完全一致。

## 常見問題

### Q: 為什麼優化結果和回測結果不一致？

**A:** 最常見的原因是：
1. 手續費率不一致（0.04% vs 0.02%）
2. 策略模式不一致（pure_grid vs hybrid）
3. 初始餘額不一致
4. 基礎配置中的其他參數不一致

### Q: 優化需要多長時間？

**A:** 取決於：
- 試驗次數（`--n-trials`）
- 並行任務數（`--n-jobs`）
- 數據大小
- CPU 性能

例如：200 trials，4 並行任務，可能需要 2-4 小時。

### Q: 如何選擇試驗次數？

**A:** 
- 初步探索：50-100 trials
- 深度優化：200-500 trials
- 生產環境：500+ trials

### Q: 優化結果不理想怎麼辦？

**A:**
1. 檢查基礎配置是否正確
2. 增加試驗次數
3. 調整搜索空間範圍
4. 檢查數據質量
5. 考慮使用不同的策略模式

## 最佳實踐

1. **記錄每次優化**：
   - 保存優化結果 CSV
   - 記錄使用的基礎配置
   - 記錄優化參數（trials, jobs, mode等）

2. **驗證優化結果**：
   - 在訓練數據上驗證
   - 在測試數據上驗證（2025）
   - 檢查是否過擬合

3. **逐步優化**：
   - 先進行小規模試驗（50-100 trials）
   - 如果結果有希望，再進行大規模優化（200+ trials）

4. **版本控制**：
   - 為每次優化創建版本標籤
   - 保存最佳參數的備份

