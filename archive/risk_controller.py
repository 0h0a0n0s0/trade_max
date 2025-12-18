# risk_controller.py
"""
Risk controller - V4 Final Version
"""
import asyncio
import os
import logging
import yaml
from telegram_alerter import alerter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional, Tuple
import aiohttp
from dotenv import load_dotenv

from max_async_api import max_api # 【V4】導入全局 API 客戶端實例

logger = logging.getLogger("RiskController")

class RiskController:
    def __init__(self, config_path: Optional[str] = None):
        if config_path: cfg_path = Path(config_path)
        else: cfg_path = Path(__file__).resolve().parent / "config_usdttwd.yaml"
        self.cfg: dict = {}
        try:
            if cfg_path.exists(): self.cfg = yaml.safe_load(cfg_path.read_text()) or {}
            logger.info(f"RiskController loaded configuration from: {cfg_path}")
        except Exception as e: logger.error(f"Failed to load config from {cfg_path}: {e}")
        
        self.usdt_unit: str = self.cfg.get("usdt_unit", "USDT").lower()
        self.twd_unit: str = self.cfg.get("twd_unit", "TWD").lower()
        try:
            self.max_net_open_usdt: Decimal = Decimal(str(self.cfg.get("max_net_open_usdt", "10000")))
            self.min_twd_balance_threshold: Decimal = Decimal(str(self.cfg.get("min_twd_balance_threshold", "5000")))
        except InvalidOperation:
            self.max_net_open_usdt, self.min_twd_balance_threshold = Decimal("10000"), Decimal("5000")
        
        self.initial_usdt_balance: Optional[Decimal] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> Decimal:
        async with self._lock:
            try:
                balance_data = await max_api.get_v2_balance(self.usdt_unit)
                if balance_data and 'balance' in balance_data:
                    # 【修正】正確計算總餘額
                    avail = Decimal(balance_data['balance'])
                    locked = Decimal(balance_data.get('locked', '0'))
                    total_balance = avail + locked
                    self.initial_usdt_balance = total_balance
                    logger.info(f"RiskController initialized with API total balance: {total_balance:.4f} USDT")
                else: 
                    raise ValueError(f"API returned invalid balance data: {balance_data}")
            except Exception as e:
                logger.error(f"RiskController failed to fetch initial balance: {e}", exc_info=True)
                est_bal = Decimal(self.cfg.get("initial_usdt_balance_estimate", "0"))
                self.initial_usdt_balance = est_bal
                logger.warning(f"Using estimated initial USDT balance: {est_bal:.4f}")
            
            return self.initial_usdt_balance

    async def get_current_balances(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        try:
            usdt_data = await max_api.get_v2_balance(self.usdt_unit)
            twd_data = await max_api.get_v2_balance(self.twd_unit)
            
            # 【修正】正確計算總餘額
            usdt_bal = Decimal(usdt_data['balance']) + Decimal(usdt_data.get('locked', '0')) if usdt_data and 'balance' in usdt_data else None
            twd_bal = Decimal(twd_data['balance']) + Decimal(twd_data.get('locked', '0')) if twd_data and 'balance' in twd_data else None
            
            return usdt_bal, twd_bal
        except Exception: 
            return None, None

    async def enforce_risk_limits(self) -> Tuple[bool, bool]:
        if self.initial_usdt_balance is None: return (True, True)
        async with self._lock:
            usdt, twd = await self.get_current_balances()
            if usdt is None or twd is None: return (False, False)
            
            net_change = usdt - self.initial_usdt_balance
            if abs(net_change) > self.max_net_open_usdt:
                # 這裡也可以新增一個警報
                logger.warning(f"USDT RISK LIMIT HIT! Abs change {abs(net_change):.2f} > Limit {self.max_net_open_usdt}.")
                return (True, True)

            if twd < self.min_twd_balance_threshold:
                logger.warning(f"TWD BALANCE ALERT! Current {twd:.2f} < Threshold {self.min_twd_balance_threshold}.")
                # --- 【↓↓↓ 在此處呼叫新的 alerter ↓↓↓】 ---
                # 關鍵字 'twd_balance_low' 用於冷卻機制，確保一小時內不重複發送
                msg = (f"*TWD 餘額過低!*\n\n"
                       f"當前餘額: `{twd:.2f} TWD`\n"
                       f"安全閾值: `{self.min_twd_balance_threshold} TWD`\n\n"
                       f"程式已暫停新的買入掛單。")
                await alerter.send_risk_alert(msg, alert_key='twd_balance_low')
                # --- 【↑↑↑ 修改結束 ↑↑↑】 ---
                return (True, False)
            
            return (False, False)