# strategy_optimizer.py
"""
StrategyOptimizer - AI Agent 核心
負責監控市場狀態，動態調整策略參數
"""
from __future__ import annotations
from decimal import Decimal
from typing import Dict, List, Optional, Any, Deque, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque
import logging
import statistics

from strategy.base_strategy import BaseStrategy

log = logging.getLogger("StrategyOptimizer")


class ParameterHistory:
    """參數調整歷史記錄（用於獎勵反饋）"""
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
    
    def add_record(self, params: Dict[str, Any], timestamp: datetime, performance_before: Dict[str, Decimal]):
        """記錄參數調整"""
        self.records.append({
            'params': params.copy(),
            'timestamp': timestamp,
            'performance_before': {k: float(v) for k, v in performance_before.items()},
            'performance_after': None,  # 將在後續更新
            'reward': None  # 將由 RL 模型計算
        })
    
    def update_performance(self, params: Dict[str, Any], performance_after: Dict[str, Decimal], reward: Optional[float] = None):
        """更新參數調整後的績效"""
        for record in reversed(self.records):
            if record['params'] == params and record['performance_after'] is None:
                record['performance_after'] = {k: float(v) for k, v in performance_after.items()}
                if reward is not None:
                    record['reward'] = reward
                break


class StrategyOptimizer:
    """
    AI Agent - 策略優化器
    
    功能：
    1. 觀察市場狀態和策略表現
    2. 決定是否需要調整參數
    3. 計算新的參數值
    4. 注入參數到策略實例
    5. 追蹤參數調整的結果（用於未來 RL 訓練）
    """
    
    def __init__(self, strategy: BaseStrategy, config: Dict[str, Any]):
        """
        初始化優化器
        
        Args:
            strategy: 策略實例（將被調整參數）
            config: 優化器配置
        """
        self.strategy = strategy
        self.config = config
        
        # 優化參數
        self.optimization_enabled: bool = config.get("optimization_enabled", True)
        self.optimization_interval_sec: int = int(config.get("optimization_interval_sec", 3600))  # 預設每小時檢查一次
        self.min_performance_change_threshold: Decimal = Decimal(str(config.get("min_performance_change_threshold", "0.05")))  # 5% 變化才調整
        
        # 參數調整範圍限制
        self.param_bounds: Dict[str, Dict[str, float]] = config.get("param_bounds", {
            'small_gap': {'min': 0.01, 'max': 0.10},
            'ema_span_fast_bars': {'min': 5, 'max': 20},
            'ema_span_slow_bars': {'min': 30, 'max': 100},
            'bias_high': {'min': 0.5, 'max': 0.8},
            'bias_low': {'min': 0.2, 'max': 0.5}
        })
        
        # 狀態追蹤
        self.last_optimization_ts: Optional[datetime] = None
        self.parameter_history: ParameterHistory = ParameterHistory()
        self.performance_history: Deque[Dict[str, Decimal]] = deque(maxlen=100)  # 保留最近100次績效記錄
        
        # 市場狀態快照
        self.market_state_history: Deque[Dict[str, Any]] = deque(maxlen=50)
        
        log.info("StrategyOptimizer initialized.")
    
    def should_optimize(self, current_time: datetime) -> bool:
        """
        判斷是否應該進行優化
        
        Args:
            current_time: 當前時間
            
        Returns:
            bool: 是否應該優化
        """
        if not self.optimization_enabled:
            return False
        
        if self.last_optimization_ts is None:
            return True
        
        elapsed = (current_time - self.last_optimization_ts).total_seconds()
        return elapsed >= self.optimization_interval_sec
    
    def observe_market_state(self, current_price: Decimal, total_equity: Decimal, 
                            realized_pnl: Decimal, active_orders_count: int) -> Dict[str, Any]:
        """
        觀察當前市場狀態
        
        Args:
            current_price: 當前價格
            total_equity: 總權益
            realized_pnl: 已實現損益
            active_orders_count: 活躍訂單數
            
        Returns:
            市場狀態字典
        """
        # 獲取策略狀態向量
        strategy_state = self.strategy.get_state_vector()
        
        # 計算市場波動性（簡化：使用價格歷史）
        volatility = self._calculate_volatility()
        
        # 計算趨勢強度
        trend_strength = self._calculate_trend_strength()
        
        state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'current_price': float(current_price),
            'total_equity': float(total_equity),
            'realized_pnl': float(realized_pnl),
            'active_orders_count': active_orders_count,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'strategy_state': strategy_state,
            'performance_metrics': self._calculate_performance_metrics()
        }
        
        self.market_state_history.append(state)
        return state
    
    def optimize_parameters(self, market_state: Dict[str, Any], 
                           current_performance: Dict[str, Decimal]) -> Optional[Dict[str, Any]]:
        """
        計算新的最優參數
        
        注意：這是簡化版本，未來可以接入完整的 RL 模型
        
        Args:
            market_state: 當前市場狀態
            current_performance: 當前績效指標
            
        Returns:
            新的參數字典，如果不需要調整則返回 None
        """
        if not self.should_optimize(datetime.now(timezone.utc)):
            return None
        
        log.info("Starting parameter optimization...")
        
        # 記錄調整前的狀態
        performance_before = current_performance.copy()
        
        # 簡化版優化邏輯：基於規則的調整
        # 未來可以替換為 RL 模型
        new_params = self._rule_based_optimization(market_state, current_performance)
        
        if new_params:
            # 記錄參數調整
            self.parameter_history.add_record(
                new_params,
                datetime.now(timezone.utc),
                performance_before
            )
            
            self.last_optimization_ts = datetime.now(timezone.utc)
            log.info(f"Optimization complete. New parameters: {list(new_params.keys())}")
        
        return new_params
    
    def _rule_based_optimization(self, market_state: Dict[str, Any], 
                                performance: Dict[str, Decimal]) -> Optional[Dict[str, Any]]:
        """
        基於規則的參數優化（簡化版）
        
        未來可以替換為：
        - 強化學習模型
        - 遺傳算法
        - 貝葉斯優化
        """
        new_params = {}
        volatility = market_state.get('volatility', 0.0)
        trend_strength = market_state.get('trend_strength', 0.0)
        
        # 規則1：高波動性時，增大網格間距
        if volatility > 0.02:  # 波動性 > 2%
            current_gap = float(self.strategy.small_gap) if hasattr(self.strategy, 'small_gap') else 0.035
            new_gap = min(current_gap * 1.2, self.param_bounds['small_gap']['max'])
            if abs(new_gap - current_gap) > 0.001:
                new_params['small_gap'] = Decimal(str(new_gap))
        
        # 規則2：強趨勢時，調整 EMA 參數以更快響應
        if trend_strength > 0.7:
            if hasattr(self.strategy, 'ema_span_fast_bars'):
                current_fast = self.strategy.ema_span_fast_bars
                new_fast = max(int(current_fast * 0.9), int(self.param_bounds['ema_span_fast_bars']['min']))
                if new_fast != current_fast:
                    new_params['ema_span_fast_bars'] = new_fast
        
        # 規則3：根據績效調整偏置
        if 'roi' in performance and performance['roi'] < Decimal("-0.05"):  # ROI < -5%
            # 虧損時，調整偏置以減少風險
            if hasattr(self.strategy, 'bias_high'):
                current_bias_high = float(self.strategy.bias_high)
                new_bias_high = max(current_bias_high - 0.1, self.param_bounds['bias_high']['min'])
                if abs(new_bias_high - current_bias_high) > 0.01:
                    new_params['bias_high'] = Decimal(str(new_bias_high))
        
        return new_params if new_params else None
    
    def apply_parameters(self, new_params: Dict[str, Any]) -> bool:
        """
        將新參數應用到策略
        
        Args:
            new_params: 新參數字典
            
        Returns:
            bool: 是否成功應用
        """
        if not new_params:
            return False
        
        # 驗證參數範圍
        validated_params = self._validate_parameters(new_params)
        if not validated_params:
            log.warning("Parameter validation failed. Skipping update.")
            return False
        
        # 應用參數
        success = self.strategy.update_config(validated_params)
        
        if success:
            log.info(f"Successfully applied new parameters: {list(validated_params.keys())}")
        
        return success
    
    def _validate_parameters(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """驗證參數是否在允許範圍內"""
        validated = {}
        
        for key, value in params.items():
            if key not in self.param_bounds:
                log.warning(f"Unknown parameter: {key}. Skipping.")
                continue
            
            bounds = self.param_bounds[key]
            
            # 轉換為 float 進行比較
            if isinstance(value, Decimal):
                value_float = float(value)
            else:
                value_float = float(value)
            
            # 檢查範圍
            if 'min' in bounds and value_float < bounds['min']:
                log.warning(f"Parameter {key} ({value_float}) below minimum ({bounds['min']}). Clamping.")
                value_float = bounds['min']
            if 'max' in bounds and value_float > bounds['max']:
                log.warning(f"Parameter {key} ({value_float}) above maximum ({bounds['max']}). Clamping.")
                value_float = bounds['max']
            
            # 保持原始類型
            if isinstance(params[key], Decimal):
                validated[key] = Decimal(str(value_float))
            elif isinstance(params[key], int):
                validated[key] = int(value_float)
            else:
                validated[key] = value_float
        
        return validated if validated else None
    
    def _calculate_volatility(self) -> float:
        """計算市場波動性"""
        if not hasattr(self.strategy, 'price_history') or len(self.strategy.price_history) < 20:
            return 0.0
        
        prices = [float(p[1]) for p in list(self.strategy.price_history)[-20:]]
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        if not returns:
            return 0.0
        
        try:
            std_dev = statistics.stdev(returns) if len(returns) > 1 else 0.0
            return abs(std_dev)
        except:
            return 0.0
    
    def _calculate_trend_strength(self) -> float:
        """計算趨勢強度（0-1）"""
        indicators = self.strategy.calculate_indicators()
        ema_fast = indicators.get('ema_fast')
        ema_slow = indicators.get('ema_slow')
        
        if ema_fast is None or ema_slow is None:
            return 0.5  # 中性
        
        if ema_slow == 0:
            return 0.5
        
        # 計算快慢線差異的標準化值
        diff_ratio = abs(float(ema_fast - ema_slow) / float(ema_slow))
        # 歸一化到 0-1
        trend_strength = min(diff_ratio * 10, 1.0)  # 調整係數以獲得合理的範圍
        
        return trend_strength
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """計算績效指標"""
        if not self.performance_history:
            return {}
        
        recent_performance = list(self.performance_history)[-10:]  # 最近10次
        
        if not recent_performance:
            return {}
        
        # 計算平均 ROI
        rois = [float(p.get('roi', 0)) for p in recent_performance if 'roi' in p]
        avg_roi = statistics.mean(rois) if rois else 0.0
        
        # 計算夏普比率（簡化版）
        sharpe_ratio = 0.0
        if len(rois) > 1:
            try:
                mean_roi = statistics.mean(rois)
                std_roi = statistics.stdev(rois) if len(rois) > 1 else 0.0
                sharpe_ratio = mean_roi / std_roi if std_roi > 0 else 0.0
            except:
                pass
        
        return {
            'avg_roi': avg_roi,
            'sharpe_ratio': sharpe_ratio,
            'sample_count': len(recent_performance)
        }
    
    def record_performance(self, performance: Dict[str, Decimal]):
        """
        記錄策略績效（用於後續 RL 訓練）
        
        Args:
            performance: 績效字典，例如：
            {
                'roi': Decimal,
                'realized_pnl': Decimal,
                'max_drawdown': Decimal,
                ...
            }
        """
        self.performance_history.append(performance.copy())
        
        # 更新最近的參數調整記錄
        if self.parameter_history.records:
            latest_record = self.parameter_history.records[-1]
            if latest_record['performance_after'] is None:
                self.parameter_history.update_performance(
                    latest_record['params'],
                    performance
                )
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """獲取優化報告"""
        return {
            'optimization_enabled': self.optimization_enabled,
            'last_optimization_ts': self.last_optimization_ts.isoformat() if self.last_optimization_ts else None,
            'total_optimizations': len(self.parameter_history.records),
            'recent_performance': self._calculate_performance_metrics(),
            'market_state_snapshot': list(self.market_state_history)[-1] if self.market_state_history else None
        }

