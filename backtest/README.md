# Backtest 目錄說明

## 📁 檔案說明

### 核心檔案

| 檔案 | 用途 | 狀態 |
|------|------|------|
| `backtester_grid.py` | 回測系統主程式 | ✅ 使用中 |
| `optimize_params_parallel.py` | 並行參數優化（推薦） | ✅ 使用中 |
| `optimize_params.py` | Random Search優化（舊版） | ⚠️ 已棄用，使用 `optimize_params_parallel.py` |
| `analyze_market.py` | 市場環境分析工具 | ✅ 使用中 |
| `diagnose_strategy.py` | 策略診斷工具 | ✅ 使用中 |
| `config_usdttwd.yaml` | 回測配置檔案 | ✅ 使用中 |

### 數據檔案

| 檔案 | 說明 |
|------|------|
| `usdttwd_1m_2025.csv` | 2025年1分鐘K線數據 |
| `usdttwd_1m_25y7m.csv` | 2025年7月1分鐘K線數據 |
| `usdttwd_5m_25.csv` | 2025年5分鐘K線數據 |

### 結果檔案

| 檔案 | 說明 |
|------|------|
| `optimization_results.csv` | 參數優化結果 |
| `diagnosis_results.csv` | 策略診斷結果 |
| `market_analysis_report.json` | 市場分析報告 |

## 🚀 使用方式

### 執行回測

```bash
python backtester_grid.py \
    --csv usdttwd_1m_2025.csv \
    --config config_usdttwd.yaml \
    --init_usdt 10000.0 \
    --init_twd 300000.0
```

### 參數優化（推薦：並行版本）

```bash
python optimize_params_parallel.py \
    --csv usdttwd_1m_2025.csv \
    --config config_usdttwd.yaml \
    --target 100 \
    --max-iter 20000 \
    --workers 4
```

### 市場分析

```bash
python analyze_market.py \
    --csv usdttwd_1m_2025.csv \
    --output market_analysis_report.json
```

### 策略診斷

```bash
python diagnose_strategy.py \
    --csv usdttwd_1m_2025.csv \
    --config config_usdttwd.yaml \
    --samples 10
```

## 📝 注意事項

1. **指標計算**：使用根目錄的 `indicators.py`（已統一）
2. **參數優化**：優先使用 `optimize_params_parallel.py`（速度快）
3. **配置檔案**：`config_usdttwd.yaml` 與根目錄的配置檔案相同

## 🔗 相關文檔

- [`../docs/iteration_history/`](../docs/iteration_history/) - 優化歷程記錄
- [`../README.md`](../README.md) - 專案總覽

---

**最後更新**：2025-01-23

