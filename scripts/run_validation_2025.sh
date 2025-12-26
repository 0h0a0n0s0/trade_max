#!/bin/bash
# 批量驗證前 200 名參數在 2025 數據上的表現

echo "🚀 開始批量驗證前 200 名參數在 2025 數據上的表現"
echo "============================================================"
echo ""
echo "此過程可能需要 1-2 小時"
echo "建議使用 screen 或 nohup 在後台運行"
echo ""
echo "開始執行..."
echo ""

python tools/batch_validate_2025_top200.py

echo ""
echo "============================================================"
echo "✅ 驗證完成！"
echo "   結果已保存至: backtest/validation_2025_top200.csv"
echo ""
echo "📊 下一步：分析結果，找出在 2024 和 2025 都表現良好的參數"
