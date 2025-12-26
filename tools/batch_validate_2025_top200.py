#!/usr/bin/env python3
"""
批量驗證前 200 名參數在 2025 數據上的表現

使用方法:
    python tools/batch_validate_2025_top200.py
"""
import subprocess
import json
import re
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any, List
import time

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_backtest_result(output: str) -> Dict[str, Any]:
    """
    從回測輸出中解析 JSON 結果
    
    Args:
        output: 回測腳本的標準輸出
        
    Returns:
        解析後的結果字典，如果解析失敗返回 None
    """
    # 搜索 __BACKTEST_RESULT__: 模式
    for line in output.split('\n'):
        if '__BACKTEST_RESULT__:' in line:
            match = re.search(r'__BACKTEST_RESULT__:(.+)', line)
            if match:
                try:
                    result_json = json.loads(match.group(1))
                    return result_json
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  JSON 解析失敗: {e}")
                    return None
    return None


def extract_info_from_filename(filename: str) -> Dict[str, Any]:
    """
    從文件名中提取信息
    
    格式: rank_{rank}_trial_{trial_number}_roi_{roi_2024:.2f}.yaml
    
    Args:
        filename: 文件名
        
    Returns:
        包含 rank, trial_number, roi_2024 的字典
    """
    match = re.search(r'rank_(\d+)_trial_(\d+)_roi_([\d\.]+)\.yaml', filename)
    if match:
        return {
            'rank_2024': int(match.group(1)),
            'trial_number': int(match.group(2)),
            'roi_2024': float(match.group(3))
        }
    return {}


def run_backtest(config_path: Path, csv_path: Path, strategy_mode: str = 'pure_grid') -> Dict[str, Any]:
    """
    執行單次回測
    
    Args:
        config_path: 配置文件路徑
        csv_path: CSV 數據文件路徑
        strategy_mode: 策略模式
        
    Returns:
        回測結果字典
    """
    cmd = [
        'python',
        'core/backtester.py',
        '--csv', str(csv_path),
        '--config', str(config_path),
        '--strategy-mode', strategy_mode,
        '--init_usdt', '10000.0',
        '--init_twd', '300000.0'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300  # 5 分鐘超時
        )
        
        # 解析結果
        output = result.stdout + result.stderr
        backtest_result = parse_backtest_result(output)
        
        if backtest_result is None:
            return {
                'status': 'parse_error',
                'error': '無法解析回測結果'
            }
        
        return backtest_result
        
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'error': '回測超時（>5分鐘）'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def batch_validate(
    yaml_dir: Path,
    csv_path: Path,
    output_csv: Path,
    strategy_mode: str = 'pure_grid'
):
    """
    批量驗證 YAML 配置文件
    
    Args:
        yaml_dir: YAML 文件目錄
        csv_path: 回測數據 CSV 路徑
        output_csv: 輸出結果 CSV 路徑
        strategy_mode: 策略模式
    """
    print(f"📊 批量驗證前 200 名參數在 2025 數據上的表現")
    print("=" * 60)
    
    # 獲取所有 YAML 文件
    yaml_files = sorted(yaml_dir.glob("rank_*.yaml"))
    
    if not yaml_files:
        print(f"❌ 錯誤：在 {yaml_dir} 中找不到 YAML 文件")
        print(f"   請先執行: python tools/extract_top200_params.py")
        sys.exit(1)
    
    print(f"\n找到 {len(yaml_files)} 個 YAML 文件")
    print(f"回測數據: {csv_path}")
    print(f"策略模式: {strategy_mode}")
    print(f"\n開始批量回測...")
    print("-" * 60)
    
    results = []
    start_time = time.time()
    
    for idx, yaml_file in enumerate(yaml_files, 1):
        # 從文件名提取信息
        file_info = extract_info_from_filename(yaml_file.name)
        
        print(f"[{idx}/{len(yaml_files)}] 測試: {yaml_file.name}", end=" ... ")
        
        # 執行回測
        backtest_result = run_backtest(yaml_file, csv_path, strategy_mode)
        
        # 合併結果
        result = {
            'filename': yaml_file.name,
            **file_info,
            **backtest_result
        }
        
        # 顯示結果
        if backtest_result.get('status') == 'success':
            roi_2025 = backtest_result.get('roi_pct', 0.0)
            trades = backtest_result.get('trades', 0)
            print(f"✅ ROI: {roi_2025:.2f}% | Trades: {trades}")
        else:
            error = backtest_result.get('error', 'Unknown error')
            print(f"❌ 失敗: {error}")
        
        results.append(result)
        
        # 每 20 個保存一次（防止中斷丟失數據）
        if idx % 20 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_csv, index=False, encoding='utf-8')
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (len(yaml_files) - idx) * avg_time
            print(f"   進度: {idx}/{len(yaml_files)} | 已用時: {elapsed/60:.1f}分鐘 | 預計剩餘: {remaining/60:.1f}分鐘")
    
    # 保存最終結果
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"✅ 批量驗證完成！")
    print(f"   總文件數: {len(yaml_files)}")
    print(f"   成功回測: {len(df[df['status'] == 'success'])}")
    print(f"   失敗/錯誤: {len(df[df['status'] != 'success'])}")
    print(f"   總耗時: {total_time/60:.1f} 分鐘")
    print(f"   結果已保存至: {output_csv}")
    
    # 顯示前 10 名（按 2025 ROI）
    if len(df[df['status'] == 'success']) > 0:
        df_success = df[df['status'] == 'success'].copy()
        df_success = df_success.sort_values('roi_pct', ascending=False)
        
        print(f"\n📊 2025 數據上前 10 名參數：")
        print(f"{'Rank':<6} {'Trial':<8} {'ROI_2024':<12} {'ROI_2025':<12} {'Alpha_2025':<12} {'Trades':<8}")
        print("-" * 70)
        for idx, row in df_success.head(10).iterrows():
            print(f"{row.get('rank_2024', 'N/A'):<6} "
                  f"{row.get('trial_number', 'N/A'):<8} "
                  f"{row.get('roi_2024', 0):>10.2f}% "
                  f"{row.get('roi_pct', 0):>10.2f}% "
                  f"{row.get('alpha_pct', 0):>10.2f}% "
                  f"{row.get('trades', 0):<8}")


def main():
    """主函數"""
    # 文件路徑
    yaml_dir = Path("backtest/candidates/top200")
    csv_path = Path("data/btctwd_1m_2025.csv")
    output_csv = Path("backtest/validation_2025_top200.csv")
    
    # 檢查文件是否存在
    if not yaml_dir.exists():
        print(f"❌ 錯誤：YAML 目錄不存在: {yaml_dir}")
        print(f"   請先執行: python tools/extract_top200_params.py")
        sys.exit(1)
    
    if not csv_path.exists():
        print(f"❌ 錯誤：CSV 數據文件不存在: {csv_path}")
        sys.exit(1)
    
    # 執行批量驗證
    try:
        batch_validate(
            yaml_dir=yaml_dir,
            csv_path=csv_path,
            output_csv=output_csv,
            strategy_mode='pure_grid'
        )
        
        print(f"\n💡 下一步：分析結果")
        print(f"   查看: {output_csv}")
        print(f"   可以計算兩年總報酬來找出最佳參數組合")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用戶中斷")
        print(f"   已保存部分結果至: {output_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

