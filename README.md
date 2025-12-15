# USDTTWD 網格交易系統

## 📋 專案簡介

自動化網格交易系統，使用三層固定間隙網格策略 + EMA趨勢判斷，適用於 USDTTWD 交易對。

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
    --csv usdttwd_1m_6m.csv \
    --config config_usdttwd.yaml \
    --target 100 \
    --max-iter 5000 \
    --workers 4
```

## 📁 專案結構

```
NoAI/
├── strategy_usdttwd_grid.py    # 實盤交易策略主程式
├── backtest/
│   ├── backtester_grid.py      # 回測系統
│   ├── optimize_params_parallel.py  # 並行參數優化（推薦）
│   └── optimize_params.py      # Optuna參數優化
├── risk_controller.py          # 風險控制模組
├── max_async_api.py            # MAX交易所API封裝
├── db.py                       # 資料庫連線管理
├── db_schema.py                # 資料庫模型定義
├── indicators.py               # 技術指標計算
├── telegram_alerter.py         # Telegram通知系統
├── workflow_manager.py         # 自動化工作流程管理器
├── config_usdttwd.yaml         # 策略配置檔案
└── trading.db                  # SQLite資料庫
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

詳細說明請參考配置檔案中的註釋。

## 📈 參數優化

### 使用並行優化（推薦）

```bash
cd backtest
python optimize_params_parallel.py \
    --csv usdttwd_1m_6m.csv \
    --target 100 \
    --max-iter 5000 \
    --workers 4
```

**預估時間（Mac M1 16GB）：**
- 2000次迭代：約 30-45分鐘
- 5000次迭代：約 1-1.5小時
- 10000次迭代：約 2-2.5小時

### 優化參數範圍（已調整）

- `small_gap`: 0.01 - 0.10 TWD
- `size_pct_*`: 0.01 - 0.08
- `ema_span_fast_bars`: 30 - 1200 分鐘
- `ema_span_slow_bars`: 600 - 8000 分鐘
- `bias_high`: 0.50 - 0.90
- `bias_low`: 0.05 - 0.50

### 篩選條件

- ROI > 5% AND Max Drawdown < 15%
- 結果保存至 `optimization_results.csv`

## 📝 常用指令

```bash
# 啟動實盤策略
python strategy_usdttwd_grid.py

# 執行回測
cd backtest && python backtester_grid.py --csv usdttwd_1m_6m.csv --config config_usdttwd.yaml

# 參數優化
cd backtest && python optimize_params_parallel.py --csv usdttwd_1m_6m.csv --target 100 --max-iter 5000

# 檢查資料庫PNL
python check_db_pnl.py

# 測試Telegram通知
python test_telegram.py
```

## ⚠️ 注意事項

1. **首次使用**：建議先小資金測試
2. **參數優化**：使用最近6-12個月的1分鐘K線數據
3. **風險控制**：確保有足夠的TWD餘額應對單邊行情
4. **監控**：定期檢查策略運行狀態和PNL

## 🔗 相關文檔

- `策略說明.md` - 詳細的策略運作原理
- `專案架構與優化分析.md` - 完整的架構分析（部分內容已實施）

---

**最後更新**：2025-01-XX

