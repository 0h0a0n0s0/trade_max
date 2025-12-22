# 回測功能使用指南

## 📁 數據文件存放位置

**所有 K線數據文件應放在 `data/` 目錄**

### 建議的文件命名格式

- `btctwd_1m_2024.csv` - 2024年1分鐘K線
- `btctwd_1m_2025.csv` - 2025年1分鐘K線
- `btctwd_5m_2024.csv` - 2024年5分鐘K線
- `btctwd_1h_2024.csv` - 2024年1小時K線

### CSV 格式要求

必須包含以下欄位（按順序）：

```csv
ts,open,high,low,close,volume
1735660800,3135780.1,3135780.1,3133460.4,3133460.4,0.0023
1735660860,3133228.5,3134988.8,3132758.6,3134988.8,0.0035
```

**欄位說明：**
- `ts`: Unix 時間戳（秒）
- `open`: 開盤價
- `high`: 最高價
- `low`: 最低價
- `close`: 收盤價
- `volume`: 成交量（可選）

## 🚀 回測執行指令

### 1. 基本回測（使用 BTC/TWD 配置）

```bash
python core/backtester.py \
    --csv data/btctwd_1m_2024.csv \
    --config configs/config_rank77.yaml \
    --init_usdt 10000.0 \
    --init_twd 300000.0
```

### 2. 純網格模式回測

```bash
python core/backtester.py \
    --csv data/btctwd_1m_2024.csv \
    --config configs/config_rank77.yaml \
    --strategy-mode pure_grid \
    --init_usdt 10000.0 \
    --init_twd 300000.0
```

### 3. 純趨勢模式回測

```bash
python core/backtester.py \
    --csv data/btctwd_1m_2024.csv \
    --config configs/config_rank77.yaml \
    --strategy-mode pure_trend \
    --init_usdt 10000.0 \
    --init_twd 300000.0
```

### 4. 混合模式回測（默認）

```bash
python core/backtester.py \
    --csv data/btctwd_1m_2024.csv \
    --config configs/config_rank77.yaml \
    --strategy-mode hybrid \
    --init_usdt 10000.0 \
    --init_twd 300000.0
```

## 📊 參數說明

| 參數 | 說明 | 必填 | 默認值 |
|------|------|------|--------|
| `--csv` | CSV 文件路徑 | ✅ 是 | - |
| `--config` | 配置文件路徑 | ❌ 否 | `backtest/config_usdttwd.yaml` |
| `--init_usdt` | 初始 USDT 餘額 | ❌ 否 | `10000.0` |
| `--init_twd` | 初始 TWD 餘額 | ❌ 否 | `300000.0` |
| `--strategy-mode` | 策略模式 | ❌ 否 | `hybrid` |

**策略模式選項：**
- `hybrid`: 混合模式（網格 + 趨勢）
- `pure_grid`: 純網格模式
- `pure_trend`: 純趨勢模式

## 🔧 參數優化執行指令

### 基本優化

```bash
python core/optimizer.py \
    --csv data/btctwd_1m_2024.csv \
    --config configs/config_rank77.yaml \
    --n-trials 100
```

### 並行優化（推薦，速度快）

```bash
python core/optimizer.py \
    --csv data/btctwd_1m_2024.csv \
    --config configs/config_rank77.yaml \
    --n-trials 100 \
    --n-jobs 4 \
    --output-yaml configs/best_params.yaml \
    --output-csv backtest/optimization_results.csv
```

### 優化器參數說明

| 參數 | 說明 | 必填 | 默認值 |
|------|------|------|--------|
| `--csv` | CSV 文件路徑 | ✅ 是 | - |
| `--config` | 基礎配置文件 | ❌ 否 | `backtest/config_usdttwd.yaml` |
| `--n-trials` | 優化試驗次數 | ❌ 否 | `100` |
| `--n-jobs` | 並行工作數 | ❌ 否 | `1` |
| `--output-yaml` | 最佳參數輸出路徑 | ❌ 否 | `backtest/best_params.yaml` |
| `--output-csv` | 所有結果 CSV 路徑 | ❌ 否 | `backtest/optimization_results.csv` |
| `--strategy-mode` | 策略模式 | ❌ 否 | `pure_grid` |

## 📝 輸出結果

### 回測結果

回測結束後會輸出 JSON 格式的結果：

```json
__BACKTEST_RESULT__:{
  "status": "success",
  "roi_pct": 15.65,
  "total_pnl": 46800.0,
  "trades": 1234,
  "bh_roi_pct": -4.73,
  "alpha_pct": 20.38
}
```

### 優化結果

優化完成後會生成：
- **最佳參數 YAML**: `configs/best_params.yaml`（或指定路徑）
- **所有試驗結果 CSV**: `backtest/optimization_results.csv`（或指定路徑）

## 💡 使用建議

1. **數據準備**：確保 CSV 文件格式正確，時間戳按升序排列
2. **初始餘額**：根據實際交易資金設置 `--init_usdt` 和 `--init_twd`
3. **策略模式**：根據需求選擇合適的模式
   - `pure_grid`: 專注網格交易性能
   - `pure_trend`: 專注趨勢跟隨性能
   - `hybrid`: 平衡兩種策略
4. **優化參數**：使用 `--n-jobs 4` 可以大幅加速優化過程（使用 4 個 CPU 核心）

