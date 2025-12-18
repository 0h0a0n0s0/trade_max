# 專案運行指南

> **最後更新**: 2025-01-23

## 快速開始

### 1. 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt

# 建立 .env 檔案
cat > .env << EOF
MAX_API_KEY="your_api_key"
MAX_API_SECRET="your_api_secret"
TG_TOKEN="your_telegram_token"
TG_CHAT_ID="your_chat_id"
DB_URI="sqlite+aiosqlite:///trading.db"
EOF
```

### 2. 初始化資料庫

```bash
python db.py
```

### 3. 運行實盤策略

#### 方式A：模組化架構（推薦，新架構）

```bash
python main_modular.py
```

**特點**：
- 三層分離架構（Strategy / Execution / Optimization）
- AI Agent 動態參數調整
- 狀態觀察與獎勵反饋

#### 方式B：原始架構（穩定，已驗證）

```bash
python strategy_usdttwd_grid.py
```

**特點**：
- 線性腳本，易於理解
- 經過實盤驗證
- 功能完整

### 4. 執行回測

```bash
cd backtest
python backtester_grid.py \
    --csv usdttwd_1m_25y7m.csv \
    --config config_usdttwd.yaml \
    --init_usdt 10000.0 \
    --init_twd 300000.0
```

### 5. 參數優化

#### 並行優化（推薦，速度快）

```bash
cd backtest
python optimize_params_parallel.py \
    --csv usdttwd_1m_2025.csv \
    --config config_usdttwd.yaml \
    --target 100 \
    --max-iter 20000 \
    --workers 4
```

#### Optuna優化（用於workflow）

```bash
python optimize_params.py \
    --csv backtest/usdttwd_1m_2025.csv \
    --config config_usdttwd.yaml \
    --trials 100
```

## 檔案說明

| 檔案 | 用途 | 運行方式 |
|------|------|----------|
| `main_modular.py` | 模組化架構入口（推薦） | `python main_modular.py` |
| `strategy_usdttwd_grid.py` | 原始實盤交易主程式 | `python strategy_usdttwd_grid.py` |
| `backtest/backtester_grid.py` | 回測系統 | `python backtester_grid.py --csv <檔案> --config <配置>` |
| `backtest/optimize_params_parallel.py` | 並行參數優化（推薦） | `python optimize_params_parallel.py --csv <檔案> --target 100` |
| `backtest/optimize_params.py` | Random Search優化 | `python optimize_params.py --csv <檔案>` |
| `optimize_params.py` | Optuna優化（用於workflow） | `python optimize_params.py --csv <檔案> --trials 100` |
| `workflow_manager.py` | 自動化工作流程管理器 | `python workflow_manager.py` |
| `risk_controller.py` | 風險控制（自動調用） | 無需單獨運行 |
| `db.py` | 資料庫初始化 | `python db.py` |
| `check_db_pnl.py` | 查詢PNL | `python check_db_pnl.py` |
| `test_telegram.py` | 測試通知 | `python test_telegram.py` |

## 配置檔案

主要配置在 `config_usdttwd.yaml`，包含：

- **網格參數**：間距、層數、訂單大小
- **EMA參數**：快慢線週期
- **風險控制參數**：USDT限制、TWD門檻
- **黑天鵝保護設定**：波動閾值、檢查間隔
- **優化器配置**：`optimizer` 區塊（模組化架構）

## 停止策略

按 `Ctrl+C` 或發送 SIGTERM 信號，程式會：

1. 撤銷所有掛單
2. 保存當前狀態
3. 安全退出

## 模組化架構使用

詳見 [`docs/quick_start.md`](docs/quick_start.md)

**核心概念**：
- **Strategy Layer**：純邏輯計算，無API調用
- **Execution Layer**：管理主循環、API調用、訂單執行
- **Optimization Layer**：AI Agent，動態調整參數

## 自動化工作流程

使用 `workflow_manager.py` 實現：

- 每週日凌晨2點執行參數優化
- 每日凌晨3點執行回測驗證
- 自動備份配置
- 自動應用新參數（如果通過驗證）

```bash
python workflow_manager.py
```

## 故障排除

### 問題：策略無法啟動

**解決方案**：
1. 檢查 `.env` 文件是否正確配置
2. 檢查資料庫連接是否正常
3. 檢查 API 金鑰是否有效
4. 查看日誌錯誤訊息

### 問題：訂單無法下單

**解決方案**：
1. 檢查 API 連接
2. 檢查餘額是否足夠
3. 檢查風險控制器設置
4. 查看日誌錯誤訊息

### 問題：參數優化無結果

**解決方案**：
1. 檢查 CSV 數據文件是否存在
2. 檢查配置檔案是否正確
3. 調整篩選條件（降低ROI閾值）
4. 查看優化歷程文檔了解歷史優化結果

## 相關文檔

- [`README.md`](README.md) - 專案總覽
- [`docs/architecture_design.md`](docs/architecture_design.md) - 架構設計
- [`docs/quick_start.md`](docs/quick_start.md) - 快速開始
- [`策略說明.md`](策略說明.md) - 策略運作原理
- [`專案架構與優化分析.md`](專案架構與優化分析.md) - 架構分析

---

**最後更新**：2025-01-23
