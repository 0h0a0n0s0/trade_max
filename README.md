# Trade_Max - USDTTWD 網格交易系統

> **版本**: 2.0（模組化架構）  
> **最後更新**: 2025-01-23

## 📋 專案簡介

自動化網格交易系統，使用三層固定間隙網格策略 + EMA趨勢判斷，適用於 USDTTWD 交易對。

**核心特性**：
- ✅ 模組化架構（Strategy / Execution / Optimization 三層分離）
- ✅ AI Agent 動態參數調整
- ✅ 混合模式（網格 + 趨勢跟隨）
- ✅ ATR動態網格間距
- ✅ 自動化參數優化流程

## 🚀 快速開始

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

#### 方式A：模組化架構（推薦）

```bash
python main_modular.py
```

#### 方式B：原始架構（保留）

```bash
python strategy_usdttwd_grid.py
```

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

```bash
# 使用並行版本（推薦，速度快）
cd backtest
python optimize_params_parallel.py \
    --csv usdttwd_1m_2025.csv \
    --config config_usdttwd.yaml \
    --target 100 \
    --max-iter 20000 \
    --workers 4
```

## 📁 專案結構

```
NoAI/
├── strategy/                    # 策略層（模組化架構）
│   ├── base_strategy.py         # 抽象基類
│   └── grid_strategy.py         # 網格策略實現
├── optimizer/                   # 優化層（AI Agent）
│   └── strategy_optimizer.py   # 動態參數優化器
├── engine/                      # 執行層
│   └── bot_engine.py            # 執行引擎
├── strategy_usdttwd_grid.py    # 原始實盤交易策略（保留）
├── main_modular.py             # 模組化架構入口（推薦）
├── backtest/                   # 回測相關
│   ├── backtester_grid.py      # 回測系統
│   ├── optimize_params_parallel.py  # 並行參數優化（推薦）
│   └── optimize_params.py      # Optuna參數優化
├── risk_controller.py          # 風險控制模組
├── max_async_api.py            # MAX交易所API封裝
├── db.py                       # 資料庫連線管理
├── db_schema.py                # 資料庫模型定義
├── indicators.py               # 技術指標計算（統一）
├── telegram_alerter.py         # Telegram通知系統
├── workflow_manager.py         # 自動化工作流程管理器
├── optimize_params.py          # Optuna優化器（用於workflow）
├── config_usdttwd.yaml         # 策略配置檔案
└── docs/                       # 文檔目錄
    ├── architecture_design.md  # 架構設計文檔
    ├── quick_start.md          # 快速開始指南
    ├── 架構重構總結.md         # 架構重構總結
    └── iteration_history/      # 優化歷程記錄
        ├── 20251213_優化歷程完整記錄.md
        └── 20251213_第六次優化方案.md
```

## 📊 策略說明

### 核心策略

1. **三層固定間隙網格**
   - 小網格：`small_gap` TWD
   - 中網格：`small_gap * mid_mult` TWD
   - 大網格：`small_gap * big_mult` TWD
   - 每層上下各 `levels_each` 個掛單

2. **EMA趨勢判斷**
   - 快線：`ema_span_fast_bars` 分鐘
   - 慢線：`ema_span_slow_bars` 分鐘
   - 根據趨勢調整USDT/TWD持倉比例

3. **混合模式（可選）**
   - 強趨勢市場自動進入趨勢跟隨模式
   - 使用ADX指標判斷市場狀態
   - 多指標複合判斷（EMA + ADX + RSI + MACD）

### 風險控制

- USDT淨持倉限制
- TWD餘額最低門檻
- 黑天鵝事件保護（價格劇烈波動時自動停止）

## 🔧 配置說明

主要配置在 `config_usdttwd.yaml`：

- **網格參數**：`small_gap`, `mid_mult`, `big_mult`, `levels_each`
- **訂單大小**：`size_pct_small`, `size_pct_mid`, `size_pct_big`
- **EMA參數**：`ema_span_fast_bars`, `ema_span_slow_bars`
- **趨勢偏好**：`bias_high`, `bias_low`, `bias_neutral_target`
- **優化器配置**：`optimizer` 區塊（模組化架構）

詳細說明請參考配置檔案中的註釋。

## 📈 參數優化

### 使用並行優化（推薦）

```bash
cd backtest
python optimize_params_parallel.py \
    --csv usdttwd_1m_2025.csv \
    --target 100 \
    --max-iter 20000 \
    --workers 4
```

**預估時間（Mac M1 16GB）：**
- 2000次迭代：約 30-45分鐘
- 5000次迭代：約 1-1.5小時
- 20000次迭代：約 4-5小時

### 優化歷程

詳見 `docs/iteration_history/20251213_優化歷程完整記錄.md`

**關鍵發現**：
- 2025年市場環境：77.8%時間為強趨勢市場
- 網格交易在強趨勢市場表現不佳
- 最佳ROI約0.4%（ROI > 0%條件下）
- 需要更多依賴趨勢跟隨策略

## 📝 常用指令

```bash
# 啟動實盤策略（模組化架構）
python main_modular.py

# 啟動實盤策略（原始架構）
python strategy_usdttwd_grid.py

# 執行回測
cd backtest && python backtester_grid.py --csv usdttwd_1m_2025.csv --config config_usdttwd.yaml

# 參數優化
cd backtest && python optimize_params_parallel.py --csv usdttwd_1m_2025.csv --target 100 --max-iter 20000

# 檢查資料庫PNL
python check_db_pnl.py

# 測試Telegram通知
python test_telegram.py

# 自動化工作流程管理器
python workflow_manager.py
```

## ⚠️ 注意事項

1. **首次使用**：建議先小資金測試
2. **參數優化**：使用最近6-12個月的1分鐘K線數據
3. **風險控制**：確保有足夠的TWD餘額應對單邊行情
4. **監控**：定期檢查策略運行狀態和PNL
5. **模組化架構**：新架構仍在測試階段，建議先使用原始架構

## 🔗 相關文檔

- [`docs/architecture_design.md`](docs/architecture_design.md) - 模組化架構詳細設計
- [`docs/quick_start.md`](docs/quick_start.md) - 模組化架構快速開始指南
- [`docs/架構重構總結.md`](docs/架構重構總結.md) - 架構重構完整總結
- [`專案架構與優化分析.md`](專案架構與優化分析.md) - 專案架構與優化分析（已更新）
- [`策略說明.md`](策略說明.md) - 詳細的策略運作原理
- [`運行指南.md`](運行指南.md) - 運行說明

## 📚 優化歷程

所有優化歷程記錄在 `docs/iteration_history/` 目錄：

- `20251213_優化歷程完整記錄.md` - 完整優化歷程（6次優化）
- `20251213_第六次優化方案.md` - 第六次優化方案

---

**最後更新**：2025-01-23  
**維護者**：Trade_Max Team
