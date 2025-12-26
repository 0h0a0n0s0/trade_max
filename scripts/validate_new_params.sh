#!/bin/bash
# 驗證新優化參數在 2024 和 2025 數據上的表現

BEST_PARAMS="configs/best_params_fee_002.yaml"

if [ ! -f "$BEST_PARAMS" ]; then
    echo "❌ 錯誤：找不到最佳參數文件 $BEST_PARAMS"
    echo "   請先執行優化流程：bash scripts/optimize_with_fee_002.sh"
    exit 1
fi

echo "📊 驗證新優化參數"
echo "============================================================"
echo ""
echo "使用參數文件: $BEST_PARAMS"
echo ""

echo "1️⃣  回測 2024 數據..."
python core/backtester.py \
    --csv data/btctwd_1m_2024.csv \
    --config "$BEST_PARAMS" \
    --strategy-mode pure_grid \
    --init_usdt 10000.0 \
    --init_twd 300000.0

echo ""
echo "2️⃣  回測 2025 數據..."
python core/backtester.py \
    --csv data/btctwd_1m_2025.csv \
    --config "$BEST_PARAMS" \
    --strategy-mode pure_grid \
    --init_usdt 10000.0 \
    --init_twd 300000.0

echo ""
echo "============================================================"
echo "✅ 驗證完成！"
echo ""
echo "💡 如果結果滿意，可以將最佳參數合併到 config_rank77.yaml："
echo "   python scripts/merge_best_params.py $BEST_PARAMS configs/config_rank77.yaml"

