# USDTTWD 網格交易專案 - 架構與優化分析

> **最後更新**: 2025-01-23  
> **狀態**: 已重構為模組化架構，部分優化建議已實施

## 📋 目錄
1. [專案架構總覽](#專案架構總覽)
2. [檔案運行方式](#檔案運行方式)
3. [模組化架構（新）](#模組化架構新)
4. [策略優化建議](#策略優化建議)
5. [參數優化系統](#參數優化系統)
6. [整體流程優化](#整體流程優化)

---

## 專案架構總覽

### 核心模組

```
NoAI/
├── strategy/                    # 策略層（模組化架構）
│   ├── base_strategy.py         # 抽象基類
│   └── grid_strategy.py         # 網格策略實現
├── optimizer/                   # 優化層（AI Agent）
│   └── strategy_optimizer.py    # 動態參數優化器
├── engine/                      # 執行層
│   └── bot_engine.py            # 執行引擎
├── strategy_usdttwd_grid.py    # 原始實盤交易策略（保留）
├── main_modular.py             # 模組化架構入口（新）
├── backtest/
│   ├── backtester_grid.py      # 回測系統
│   ├── optimize_params_parallel.py  # 並行參數優化（推薦）
│   └── optimize_params.py      # Optuna參數優化
├── risk_controller.py          # 風險控制模組
├── max_async_api.py            # MAX交易所API封裝
├── db.py                       # 資料庫連線管理
├── db_schema.py                # 資料庫模型定義
├── indicators.py               # 技術指標計算（統一）
├── telegram_alerter.py        # Telegram通知系統
├── workflow_manager.py         # 自動化工作流程管理器
├── optimize_params.py          # Optuna優化器（用於workflow）
├── config_usdttwd.yaml         # 策略配置檔案
└── docs/                       # 文檔目錄
    ├── architecture_design.md  # 架構設計文檔
    ├── quick_start.md          # 快速開始指南
    └── iteration_history/      # 優化歷程記錄
```

### 策略核心邏輯

**三層固定間隙網格策略 + EMA趨勢偏好**

1. **網格層級**：
   - 小網格 (Small Grid): `small_gap = 0.035 TWD`
   - 中網格 (Mid Grid): `small_gap * 3 = 0.105 TWD`
   - 大網格 (Big Grid): `small_gap * 7 = 0.245 TWD`
   - 每層上下各 6 個掛單

2. **EMA趨勢判斷**：
   - 快線：600分鐘 (10小時)
   - 慢線：3000分鐘 (50小時)
   - 看漲：目標USDT曝險 60%
   - 看跌：目標USDT曝險 25%

3. **風險控制**：
   - USDT淨持倉限制
   - TWD餘額最低門檻
   - 黑天鵝事件保護

---

## 檔案運行方式

### 1. 實盤交易策略

#### 方式A：原始架構（保留）

```bash
python strategy_usdttwd_grid.py
```

#### 方式B：模組化架構（推薦）

```bash
python main_modular.py
```

**運行流程**：
1. 載入配置檔案 `config_usdttwd.yaml`
2. 初始化資料庫連線
3. 初始化MAX API客戶端
4. 創建策略、優化器、引擎實例
5. 清理舊掛單（避免孤兒訂單）
6. 建立初始網格
7. 進入主迴圈：
   - 輪詢訂單狀態
   - 更新餘額
   - 檢查優化器（動態調整參數）
   - 執行策略邏輯（網格重建、偏置調整）
   - 檢查黑天鵝事件

### 2. 回測系統

```bash
cd backtest
python backtester_grid.py \
    --csv usdttwd_1m_25y7m.csv \
    --config config_usdttwd.yaml \
    --init_usdt 10000.0 \
    --init_twd 300000.0
```

### 3. 參數優化

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

### 4. 其他工具

```bash
# 檢查資料庫PNL
python check_db_pnl.py

# 測試Telegram通知
python test_telegram.py

# 自動化工作流程管理器
python workflow_manager.py
```

---

## 模組化架構（新）

### 架構設計

參考 `docs/architecture_design.md` 了解詳細設計。

**三層分離**：
1. **Strategy Layer** - 純邏輯計算，無API調用
2. **Execution Layer** - 管理主循環、API調用、訂單執行
3. **Optimization Layer** - AI Agent，動態調整參數

### 核心特性

- ✅ **可注入參數**：所有策略參數作為類屬性，可動態調整
- ✅ **狀態觀察**：策略導出狀態向量供AI Agent觀察
- ✅ **獎勵反饋**：追蹤參數調整前後的績效
- ✅ **安全機制**：參數範圍驗證、錯誤處理

### 使用方式

詳見 `docs/quick_start.md`

---

## 策略優化建議

> **狀態**：✅ 已實施混合模式、ATR動態網格、資本利用率優化、Maker訂單偏好

### 1. 網格間距優化 ⭐⭐⭐⭐⭐ ✅ 已實施

**實施內容**：
- ✅ ATR動態網格間距
- ✅ 波動率分層調整

**配置**：
```yaml
use_atr_spacing: true
atr_period: 14
atr_spacing_multiplier: "0.8"
```

### 2. 資金利用率優化 ⭐⭐⭐⭐ ✅ 已實施

**實施內容**：
- ✅ 增加訂單大小百分比
- ✅ 增加網格層級

### 3. 實現混合模式（趨勢跟隨） ⭐⭐⭐⭐⭐ ✅ 已實施

**實施內容**：
- ✅ ADX指標判斷市場狀態
- ✅ 強趨勢市場自動進入趨勢跟隨模式
- ✅ 多指標複合判斷（EMA + ADX + RSI + MACD）

**配置**：
```yaml
use_hybrid_model: true
adx_strength_threshold: 25
trend_trade_equity_pct: "0.4"
```

### 4. 手續費優化 ⭐⭐⭐⭐ ✅ 已實施

**實施內容**：
- ✅ Post-only訂單偏好
- ✅ 動態調整掛單價格確保maker訂單

---

## 參數優化系統

### 當前狀況

✅ **已實施兩套優化系統**：

1. **Random Search + Local Mutation**（`backtest/optimize_params_parallel.py`）
   - 並行執行，速度快
   - 適合大規模參數搜尋
   - 當前使用的主要優化工具

2. **Optuna優化**（`optimize_params.py`）
   - 使用TPE採樣器
   - 適合精細優化
   - 整合到 `workflow_manager.py` 自動化流程

### 優化歷程

詳見 `docs/iteration_history/20251213_優化歷程完整記錄.md`

**關鍵發現**：
- 2025年市場環境：77.8%時間為強趨勢市場
- 網格交易在強趨勢市場表現不佳
- 最佳ROI約0.4%（ROI > 0%條件下）
- 需要更多依賴趨勢跟隨策略

---

## 整體流程優化

### 當前流程

✅ **已實施自動化工作流程管理器**（`workflow_manager.py`）

**功能**：
- 每週日凌晨2點執行參數優化
- 每日凌晨3點執行回測驗證
- 自動備份配置
- 自動應用新參數（如果通過驗證）

**流程**：
```
1. 定期參數優化（每週）
   ├── 執行Optuna優化
   ├── 在訓練集上找到最佳參數
   └── 在測試集上驗證

2. 自動驗證與應用
   ├── 如果測試集ROI > 閾值（如15%）
   │   ├── 自動更新config_usdttwd.yaml
   │   ├── 備份舊配置
   │   └── 發送Telegram通知
   └── 如果測試集ROI < 閾值
       └── 記錄結果，不更新配置

3. 實盤監控
   ├── 每日計算實際ROI
   ├── 與回測ROI對比
   └── 如果偏差過大，觸發重新優化
```

---

## 總結與優先級

### 已完成項目 ✅

1. ✅ 模組化架構重構
2. ✅ Optuna參數優化系統
3. ✅ 自動化工作流程管理器
4. ✅ 混合模式（趨勢跟隨）
5. ✅ ATR動態網格間距
6. ✅ 資金利用率優化
7. ✅ 手續費優化（maker訂單）

### 未來改進方向

1. **強化學習整合** ⭐⭐⭐⭐⭐
   - 將規則式優化器替換為RL模型
   - 使用歷史績效數據訓練

2. **多策略支援** ⭐⭐⭐⭐
   - 支援多個策略實例
   - 獨立優化每個策略

3. **分散式優化** ⭐⭐⭐
   - 多個優化器實例協同工作

### 預期效果

實施所有優化後：
- **年化ROI目標**：15-25%（保守估計）
- **風險控制**：最大回撤 < 10%
- **自動化程度**：90%以上流程自動化

---

## 📚 相關文檔

- [`docs/architecture_design.md`](docs/architecture_design.md) - 詳細架構設計
- [`docs/quick_start.md`](docs/quick_start.md) - 快速開始指南
- [`docs/iteration_history/`](docs/iteration_history/) - 優化歷程記錄
- [`策略說明.md`](策略說明.md) - 策略運作原理
- [`運行指南.md`](運行指南.md) - 運行說明

---

**最後更新**：2025-01-23  
**版本**：2.0（模組化架構）
