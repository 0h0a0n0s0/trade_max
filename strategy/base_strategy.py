# base_strategy.py
"""
BaseStrategy 抽象基類
定義所有策略必須實現的介面，參考 Freqtrade 的 IStrategy 設計
"""
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional, Any, Deque, Tuple
from datetime import datetime
from collections import deque
import logging

log = logging.getLogger("BaseStrategy")


class BaseStrategy(ABC):
    """
    策略基類介面
    
    所有策略必須繼承此類並實現必要方法。
    策略層應該只包含純邏輯計算，不包含任何 API 調用。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略
        
        Args:
            config: 策略配置字典（從 YAML 載入）
        """
        self.config = config
        self.strategy_name: str = config.get("strategy_name", "Unknown")
        self.asset_pair: str = config.get("asset_pair", "usdttwd")
        
        # 價格歷史（由 BotEngine 注入）
        self.price_history: Deque[Tuple[int, Decimal]] = deque(maxlen=3100)
        
        # 策略狀態
        self.is_active: bool = True
        self.last_update_ts: Optional[datetime] = None
        
        log.info(f"Strategy '{self.strategy_name}' initialized.")
    
    @abstractmethod
    def calculate_indicators(self) -> Dict[str, Optional[Decimal]]:
        """
        計算技術指標
        
        Returns:
            包含各種指標值的字典，例如：
            {
                'ema_fast': Decimal,
                'ema_slow': Decimal,
                'atr': Decimal,
                'adx': Decimal,
                ...
            }
        """
        pass
    
    @abstractmethod
    def generate_signals(self, current_price: Decimal) -> Dict[str, Any]:
        """
        根據當前市場狀態生成交易信號
        
        Args:
            current_price: 當前市場價格
            
        Returns:
            信號字典，例如：
            {
                'action': 'buy' | 'sell' | 'hold',
                'price': Decimal,
                'quantity': Decimal,
                'layer_idx': Optional[int],
                'reason': str
            }
        """
        pass
    
    @abstractmethod
    def update_config(self, new_params: Dict[str, Any]) -> bool:
        """
        動態更新策略參數（由 StrategyOptimizer 調用）
        
        Args:
            new_params: 新的參數字典，例如：
            {
                'grid_gap': Decimal,
                'ema_fast_period': int,
                'bias_ratio': Decimal,
                ...
            }
            
        Returns:
            bool: 是否成功更新
        """
        pass
    
    def get_state_vector(self) -> Dict[str, Any]:
        """
        導出當前策略狀態向量（供 AI Agent 觀察）
        
        Returns:
            狀態字典，包含：
            - 當前參數值
            - 指標值
            - 策略狀態
            - 性能指標（如果可用）
        """
        indicators = self.calculate_indicators()
        return {
            'strategy_name': self.strategy_name,
            'is_active': self.is_active,
            'last_update_ts': self.last_update_ts.isoformat() if self.last_update_ts else None,
            'indicators': {k: float(v) if v else None for k, v in indicators.items()},
            'config_snapshot': self._get_config_snapshot()
        }
    
    def _get_config_snapshot(self) -> Dict[str, Any]:
        """
        獲取當前配置的快照（用於狀態觀察）
        子類應該覆蓋此方法以返回可調整的參數
        """
        return {}
    
    def validate_price_history(self, min_length: int = 10) -> bool:
        """
        驗證價格歷史是否足夠進行計算
        
        Args:
            min_length: 所需的最小歷史長度
            
        Returns:
            bool: 是否足夠
        """
        return len(self.price_history) >= min_length

