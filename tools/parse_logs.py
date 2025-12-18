import os
import re
import pandas as pd
import glob

# 設定 Log 檔案的路徑
LOG_DIR = 'backtest/candidates' 
OUTPUT_CSV = 'backtest/2025_validation_summary.csv'

def parse_log_file(filepath):
    """讀取 Log 檔案並抓取關鍵指標"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 使用正規表達式抓取數據
        # 1. 抓取 ROI (例如: Total ROI: -20.32%)
        roi_match = re.search(r"Total ROI:\s+([\d\.\-]+)%", content)
        roi = float(roi_match.group(1)) if roi_match else -999.0
        
        # 2. 抓取 Max Drawdown
        dd_match = re.search(r"Max Drawdown:\s+([\d\.\-]+)%", content)
        dd = float(dd_match.group(1)) if dd_match else 0.0
        
        # 3. 抓取是否觸發硬止損
        hard_stop = "HARD STOP" in content or "STOPPED" in content
        
        # 4. 從檔名解析出 2024 的 Rank 和 ROI
        filename = os.path.basename(filepath)
        
        # [修正] 這裡加上 \.yaml 讓它知道數字到這裡就結束了，不要多抓一個點
        rank_match = re.search(r"rank_(\d+)", filename)
        roi_2024_match = re.search(r"roi_([\d\.]+)\.yaml", filename)
        
        rank = int(rank_match.group(1)) if rank_match else 999
        
        # 額外防呆：如果還是抓到奇怪的字串，嘗試移除結尾的點
        if roi_2024_match:
            raw_roi_str = roi_2024_match.group(1).rstrip('.')
            roi_2024 = float(raw_roi_str)
        else:
            roi_2024 = 0.0
        
        return {
            "Rank_2024": rank,
            "ROI_2024": roi_2024,
            "ROI_2025": roi,
            "Drawdown_2025": dd,
            "Hard_Stop": hard_stop,
            "File": filename
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def main():
    # 搜尋所有 .log 檔案
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    data = []
    
    print(f"🔍 找到 {len(log_files)} 個 Log 檔案，開始分析...")
    
    for log_file in log_files:
        result = parse_log_file(log_file)
        if result:
            data.append(result)
            
    if not data:
        print("❌ 沒有讀取到任何有效數據")
        return

    # 轉成 DataFrame
    df = pd.DataFrame(data)
    
    # 計算「兩年總報酬」 (複利計算: (1 + 2024%) * (1 + 2025%) - 1)
    df['Two_Year_Total_Return'] = ((1 + df['ROI_2024']/100) * (1 + df['ROI_2025']/100) - 1) * 100
    
    # 依照「兩年總報酬」排序 (找出真正的穿越牛熊王者)
    df = df.sort_values(by='Two_Year_Total_Return', ascending=False)
    
    # 存檔
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n🏆 Top 10 最佳參數組合 (兼顧 2024 與 2025):")
    # 調整顯示格式，讓它對齊更好看
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df[['Rank_2024', 'ROI_2024', 'ROI_2025', 'Drawdown_2025', 'Two_Year_Total_Return']].head(10).to_string(index=False))
    print(f"\n✅ 完整報告已儲存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()