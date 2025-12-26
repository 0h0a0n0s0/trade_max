#!/usr/bin/env python3
"""
從 optimization_results_fee_002.csv 中提取前 200 名參數並生成完整的 YAML 配置文件

使用方法:
    python tools/extract_top200_params.py
"""
import pandas as pd
import yaml
from pathlib import Path
import sys

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def extract_top_params(
    csv_path: Path,
    base_config_path: Path,
    output_dir: Path,
    top_n: int = 200
):
    """
    從 CSV 中提取前 N 名參數並生成 YAML 文件
    
    Args:
        csv_path: 優化結果 CSV 文件路徑
        base_config_path: 基礎配置文件路徑
        output_dir: 輸出目錄
        top_n: 提取前 N 名
    """
    print(f"📋 開始提取前 {top_n} 名參數")
    print("=" * 60)
    
    # 讀取 CSV
    print(f"\n1. 讀取 CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   總試驗數: {len(df)}")
    
    # 只保留成功完成的試驗
    df_complete = df[df['state'] == 'COMPLETE'].copy()
    print(f"   成功完成: {len(df_complete)}")
    
    if len(df_complete) < top_n:
        print(f"   ⚠️  警告：成功完成的試驗數 ({len(df_complete)}) 少於要求的 {top_n}")
        top_n = len(df_complete)
    
    # 按 value (ROI) 排序，取前 N 名
    df_sorted = df_complete.sort_values('value', ascending=False).head(top_n)
    print(f"\n2. 選取前 {top_n} 名（按 2024 ROI 排序）")
    print(f"   ROI 範圍: {df_sorted['value'].min():.2f}% ~ {df_sorted['value'].max():.2f}%")
    
    # 讀取基礎配置
    print(f"\n3. 讀取基礎配置: {base_config_path}")
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # 創建輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n4. 輸出目錄: {output_dir}")
    
    # 參數欄位列表（需要從 CSV 覆蓋的參數）
    param_cols = [
        'small_gap', 'mid_mult', 'big_mult', 'size_pct_small',
        'bias_neutral_target', 'bias_rebalance_threshold_twd',
        'grid_aggression_multiplier', 'max_drawdown_stop_pct',
        'trend_ema_fast_bars', 'trend_ema_slow_bars'
    ]
    
    # 需要保留的固定設定（不從 CSV 覆蓋）
    preserve_keys = [
        'asset_pair', 'usdt_unit', 'twd_unit', 'taker_fee',
        'price_precision', 'qty_precision'
    ]
    
    print(f"\n5. 生成 YAML 文件...")
    generated_files = []
    
    for idx, row in df_sorted.iterrows():
        rank = idx + 1  # 排名從 1 開始
        trial_number = int(row['number'])
        roi_2024 = float(row['value'])
        
        # 複製基礎配置
        config = base_config.copy()
        
        # 用 CSV 中的參數覆蓋
        for param in param_cols:
            if param in row and pd.notna(row[param]):
                value = row[param]
                # 轉換為字符串（YAML 兼容性）
                if param in ['small_gap', 'size_pct_small', 'bias_neutral_target', 
                            'bias_rebalance_threshold_twd', 'grid_aggression_multiplier',
                            'max_drawdown_stop_pct']:
                    config[param] = str(value)
                else:
                    config[param] = value
        
        # 確保保留的設定不被覆蓋
        for key in preserve_keys:
            if key in base_config:
                config[key] = base_config[key]
        
        # 生成文件名
        filename = f"rank_{rank:03d}_trial_{trial_number}_roi_{roi_2024:.2f}.yaml"
        output_path = output_dir / filename
        
        # 寫入 YAML
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        generated_files.append({
            'rank': rank,
            'trial_number': trial_number,
            'roi_2024': roi_2024,
            'filename': filename,
            'path': str(output_path)
        })
        
        if rank % 50 == 0:
            print(f"   已生成 {rank}/{top_n} 個文件...")
    
    print(f"\n✅ 完成！共生成 {len(generated_files)} 個 YAML 文件")
    print(f"   輸出目錄: {output_dir}")
    
    # 生成索引文件（方便查閱）
    index_path = output_dir / "index.csv"
    index_df = pd.DataFrame(generated_files)
    index_df.to_csv(index_path, index=False, encoding='utf-8')
    print(f"   索引文件: {index_path}")
    
    return generated_files


def main():
    """主函數"""
    # 文件路徑
    csv_path = Path("backtest/optimization_results_fee_002.csv")
    base_config_path = Path("configs/config_rank77.yaml")
    output_dir = Path("backtest/candidates/top200")
    
    # 檢查文件是否存在
    if not csv_path.exists():
        print(f"❌ 錯誤：CSV 文件不存在: {csv_path}")
        sys.exit(1)
    
    if not base_config_path.exists():
        print(f"❌ 錯誤：基礎配置文件不存在: {base_config_path}")
        sys.exit(1)
    
    # 提取前 200 名
    try:
        generated_files = extract_top_params(
            csv_path=csv_path,
            base_config_path=base_config_path,
            output_dir=output_dir,
            top_n=200
        )
        
        print(f"\n📊 前 5 名參數：")
        for item in generated_files[:5]:
            print(f"   Rank {item['rank']}: Trial {item['trial_number']}, ROI 2024: {item['roi_2024']:.2f}%")
        
        print(f"\n✅ 提取完成！可以開始批量驗證 2025 數據")
        print(f"   執行: python tools/batch_validate_2025_top200.py")
        
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

