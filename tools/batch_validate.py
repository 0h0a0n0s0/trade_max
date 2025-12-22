import pandas as pd
import yaml
import os
import subprocess
import sys
import time

# ================= 設定區 =================
RESULTS_CSV = 'backtest/optimization_results.csv'  # 您的 2024 訓練結果
BASE_CONFIG = 'backtest/config_usdttwd.yaml'                    # 您的基礎設定檔
TEST_CSV_2025 = 'data/btctwd_1m_2025.csv'                   # 要驗證的 2025 數據
OUTPUT_DIR = 'backtest/candidates'                              # 存放候選參數的資料夾
TOP_N = 200                                                     # 要取前幾名來驗證
# ============================================

def main():
    # 0. 檢查檔案是否存在
    if not os.path.exists(RESULTS_CSV):
        print(f"❌ 找不到訓練結果 CSV: {RESULTS_CSV}")
        print("   請確認您是否有加上 --output-csv 參數執行訓練")
        return
    
    if not os.path.exists(TEST_CSV_2025):
        print(f"❌ 找不到測試數據 CSV: {TEST_CSV_2025}")
        return

    # 1. 讀取並篩選前 N 名
    try:
        df = pd.read_csv(RESULTS_CSV)
        # 排除失敗的訓練
        df = df[df['value'] > -90]
        # 依照 ROI (value) 排序
        top_df = df.sort_values(by='value', ascending=False).head(TOP_N)
    except Exception as e:
        print(f"❌ 讀取 CSV 失敗: {e}")
        return
    
    print(f"✅ 載入 {len(df)} 筆訓練資料，選取 Top {len(top_df)} 進行 2025 壓力測試...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 讀取基礎設定檔
    try:
        with open(BASE_CONFIG, 'r') as f:
            base_config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 讀取 Config 失敗: {e}")
        return

    # 2. 迴圈測試
    for index, row in top_df.iterrows():
        rank = list(top_df.index).index(index) + 1
        roi_2024 = row['value']
        
        # 建立 Config
        new_config = base_config.copy()
        
        # 覆蓋參數
        ignore_cols = ['number', 'value', 'alpha_pct', 'bh_roi_pct', 'total_pnl', 'trades', 'state', 'duration', 'datetime_start', 'datetime_complete']
        for col in df.columns:
            if col not in ignore_cols and col in row:
                val = row[col]
                # 轉換 numpy 類型為 python 原生類型 (避免 yaml 報錯)
                if hasattr(val, 'item'): 
                    val = val.item()
                new_config[col] = val

        # 存檔
        candidate_config_file = f"{OUTPUT_DIR}/rank_{rank}_roi_{roi_2024:.2f}.yaml"
        with open(candidate_config_file, 'w') as f:
            yaml.dump(new_config, f)

        log_file = f"{candidate_config_file}.log"
        print(f"\n🚀 [Rank {rank}/{len(top_df)}] 測試中... (2024 ROI: {roi_2024:.2f}%)")
        print(f"   Config: {candidate_config_file}")

        # 3. 執行回測 (即時輸出)
        cmd = [
            sys.executable, "core/backtester.py",
            "--csv", TEST_CSV_2025,
            "--config", candidate_config_file,
            "--init_usdt", "0.2",   # 改成底線 _
            "--init_twd", "300000"    # 改成底線 _
        ]
        
        try:
            # 開啟檔案準備寫入 Log
            with open(log_file, "w", encoding='utf-8') as f_log:
                # 啟動子進程，合併 stderr 到 stdout
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, # 關鍵修正：捕捉錯誤輸出
                    text=True, 
                    bufsize=1, # 行緩衝，即時輸出
                    encoding='utf-8',
                    errors='replace'
                )

                # 即時讀取輸出
                captured_lines = []
                for line in process.stdout:
                    # 1. 寫入 Log 檔
                    f_log.write(line)
                    # 2. 存入記憶體以供簡單分析
                    captured_lines.append(line)
                    # 3. (選用) 印在螢幕上，如果您不想看太多字可以註解掉下面這行
                    # print(f"   | {line.strip()}") 

                process.wait() # 等待程式結束
                
                # 簡單分析結果
                full_log = "".join(captured_lines)
                is_stopped = "HARD STOP" in full_log or "STOPPED" in full_log
                
                # 嘗試找最後的 Portfolio Value
                final_val = "N/A"
                for line in reversed(captured_lines):
                    if "Final Portfolio Value" in line:
                        final_val = line.strip()
                        break
                
                print(f"   🏁 測試結束。 Log 已存: {log_file}")
                if is_stopped:
                    print(f"   🛡️ 觸發硬止損 (Hard Stop): 是")
                else:
                    print(f"   ⚠️ 未觸發止損")
                
                if "Traceback" in full_log:
                     print(f"   ❌ 程式執行發生錯誤！請檢查 Log 檔")

        except Exception as e:
            print(f"   ❌ 執行腳本時發生異常: {e}")

    print(f"\n✨ 批量驗證完成！請檢查 {OUTPUT_DIR} 中的 Log 檔案。")

if __name__ == "__main__":
    main()