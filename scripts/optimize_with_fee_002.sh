#!/bin/bash
# 使用正確手續費（0.02%）重新優化參數
# 此腳本用於找到適用於實戰環境的最優參數

echo "🚀 開始使用 0.02% 手續費進行參數優化"
echo "============================================================"
echo ""
echo "配置信息："
echo "  - 基礎配置: configs/config_rank77.yaml (taker_fee: 0.0002)"
echo "  - 訓練數據: data/btctwd_1m_2024.csv"
echo "  - 策略模式: pure_grid"
echo "  - 優化試驗數: 500"
echo "  - 並行任務數: 4"
echo ""
echo "開始優化..."
echo ""

python core/optimizer.py \
    --csv data/btctwd_1m_2024.csv \
    --config configs/config_rank77.yaml \
    --strategy-mode pure_grid \
    --n-trials 500 \
    --n-jobs 4 \
    --output-yaml configs/best_params_fee_002.yaml \
    --output-csv backtest/optimization_results_fee_002.csv

echo ""
echo "============================================================"
echo "✅ 優化完成！"
echo "   最佳參數已保存至: configs/best_params_fee_002.yaml"
echo "   所有試驗結果已保存至: backtest/optimization_results_fee_002.csv"
echo ""
echo "📋 下一步：使用新參數進行驗證回測"
echo "   bash scripts/validate_new_params.sh"

