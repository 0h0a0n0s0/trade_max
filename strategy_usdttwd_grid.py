# strategy_usdttwd_grid.py
"""
核心策略 (基於 backtester_grid.py): V3 完整最終整合版
* 三層固定間隙網格 (small/mid/big gaps)。
* 每個網格層級的訂單數量基於當前總權益的百分比動態計算。
* EMA10h‑50h (分鐘線) 判斷趨勢，調整方向性倉位。
* 正確處理部分成交：只有當訂單完全成交後，才在另一側掛出新訂單。
* 啟動時清空舊掛單，避免孤兒訂單。
* 黑天鵝保護觸發後將永久停止，需要人工介入。
* 整合增強版風險控制器，監控TWD餘額。
* 【V3】與使用者提供的 db.py 和 db_schema.py 完全整合，無任何省略。
"""
from __future__ import annotations
import asyncio
import os
import time
import signal
import logging
import uuid
import yaml
import functools
from collections import deque
from sqlalchemy import func
from decimal import Decimal, getcontext
from datetime import datetime, timezone, timedelta, date as DateObject
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Deque, Any
import traceback

# --- .env 檔案載入 (最優先) ---
from dotenv import load_dotenv
def find_and_load_dotenv():
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        env_path = current_dir / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"INFO: Successfully loaded .env file from: {env_path}")
            return
        current_dir = current_dir.parent
    print("CRITICAL: .env file not found. Please ensure it exists.")
find_and_load_dotenv()

import pandas as pd
from max_async_api import max_api
from risk_controller import RiskController
from telegram_alerter import alerter
from db import db_session, check_db_connection, create_all_tables
from db_schema import (
    Strategy as DBStrategy, Order as DBOrder, TradeLog as DBTradeLog,
    BalanceSnapshot as DBBalanceSnapshot, DailyPNL as DBDailyPNL,
    MarketKline1m as DBMarketKline1m, OrderStatusEnum
)

# --- 設定 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger("三層固定間隙網格")
getcontext().prec = 28

# --- 全域變數 ---
CFG: Dict = {}
CFG_PATH: Path = Path(os.getenv("STRATEGY_CFG", Path(__file__).resolve().parent / "config_usdttwd.yaml"))
CFG_PRICE_PRECISION: Decimal = Decimal("0.001")
CFG_QTY_PRECISION: Decimal = Decimal("0.001")
STRATEGY_NAME: str = "Default_Grid_Strategy"
STRATEGY_DB_ID: Optional[int] = None
PRICE_HISTORY: deque = deque()
ACTIVE_ORDERS: Dict[str, Dict] = {}
USDT_BALANCE: Decimal = Decimal("0")
TWD_BALANCE: Decimal = Decimal("0")
AVAILABLE_USDT_BALANCE: Decimal = Decimal("0")
AVAILABLE_TWD_BALANCE: Decimal = Decimal("0")
TOTAL_EQUITY_TWD: Decimal = Decimal("0")
LAST_BALANCE_UPDATE_TS: Optional[datetime] = None
RISK_CTRL: Optional[RiskController] = None
STRATEGY_HALTED: bool = False
MAIN_LOOP: Optional[asyncio.Task] = None
PREVIOUS_EMA_TREND: Optional[str] = None
LAST_REPORT_HOUR: int = -1
LAST_TRADE_TS: Optional[datetime] = None
LAST_STRATEGIC_ACTION_TS: Optional[datetime] = None
LAST_BALANCE_LOG_TS: Optional[datetime] = None

class GridLayer:
    def __init__(self, idx: int, gap_abs: Decimal, size_pct: Decimal, levels_each_side: int):
        self.idx, self.gap_abs, self.size_pct, self.levels_each_side = idx, gap_abs, size_pct, levels_each_side
    def __repr__(self): return f"GridLayer(idx={self.idx}, gap={self.gap_abs}, pct={self.size_pct*100:.2f}%)"
GRID_LAYERS: List[GridLayer] = []

# --- 設定檔載入 ---
def load_cfg():
    global CFG, PRICE_HISTORY, GRID_LAYERS, STRATEGY_NAME, CFG_PRICE_PRECISION, CFG_QTY_PRECISION
    log.info(f"Loading configuration from: {CFG_PATH}")
    if not CFG_PATH.exists(): raise SystemExit(f"CRITICAL: Config file not found: {CFG_PATH}")
    CFG = yaml.safe_load(CFG_PATH.read_text()) or {}
    CFG_PRICE_PRECISION = Decimal(CFG.get("price_precision", "0.001"))
    CFG_QTY_PRECISION = Decimal(CFG.get("qty_precision", "0.001"))
    STRATEGY_NAME = CFG.get("strategy_name", STRATEGY_NAME)
    deque_size = int(CFG.get("price_data_deque_size", 3100))
    if PRICE_HISTORY.maxlen != deque_size: PRICE_HISTORY = deque(PRICE_HISTORY, maxlen=deque_size)
    GRID_LAYERS.clear()
    small_gap = Decimal(CFG["small_gap"])
    levels_each = int(CFG["levels_each"])
    GRID_LAYERS.extend([
        GridLayer(0, small_gap, Decimal(CFG["size_pct_small"]), levels_each),
        GridLayer(1, small_gap * int(CFG["mid_mult"]), Decimal(CFG["size_pct_mid"]), levels_each),
        GridLayer(2, small_gap * int(CFG["big_mult"]), Decimal(CFG["size_pct_big"]), levels_each)])
    log.info("Configuration loaded successfully.")

# --- 資料庫操作輔助函數 ---
async def run_db_sync(func, *args, **kwargs):
    """【最終校驗版】使用 functools.partial 來確保同步函數的參數被正確傳遞。"""
    loop = asyncio.get_event_loop()
    func_call = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)
    
def _db_get_or_create_strategy_sync(name: str, description: str, params: Dict) -> Optional[int]:
    """
    【V6 修正】修正 DetachedInstanceError
    在同一個會話中完成創建、提交和ID返回。
    """
    with db_session() as s:
        try:
            strategy = s.query(DBStrategy).filter_by(name=name).first()
            params_str = yaml.dump(params)

            if strategy:
                strategy.description = description
                strategy.params_json = params_str
                strategy.is_active = True
                log.info(f"DB: Updating existing strategy '{name}' (ID: {strategy.id}).")
            else:
                strategy = DBStrategy(name=name, description=description, params_json=params_str, is_active=True)
                s.add(strategy)
                log.info(f"DB: Creating new strategy '{name}'.")
            
            # 提交事務，讓資料庫生成 ID
            s.commit()
            
            # 在會話依然有效的狀態下，返回 ID
            strategy_id = strategy.id
            log.info(f"DB: Strategy '{name}' successfully processed. ID: {strategy_id}")
            return strategy_id

        except Exception as e:
            log.error(f"DB error for strategy '{name}': {e}", exc_info=True)
            s.rollback() # 確保在出錯時回滾
            return None

def _db_log_order_sync(order_data: Dict) -> Optional[str]:
    with db_session() as s:
        try:
            db_order = DBOrder(
                strategy_id=STRATEGY_DB_ID,
                client_oid=order_data["client_oid"],
                exchange_order_id=order_data.get("exchange_id"),
                asset_pair=CFG["asset_pair"],
                side=order_data["side"],
                order_type=order_data.get("order_type", "post_only"),
                price=order_data["price"],
                quantity=order_data["qty"],
                status=OrderStatusEnum.NEW,
                layer_idx=order_data.get("layer_idx")
            )
            s.add(db_order)
            log.debug(f"DB: Logged new order: client_oid={db_order.client_oid}")
            return db_order.client_oid
        except Exception as e:
            log.error(f"DB error logging order {order_data.get('client_oid')}: {e}", exc_info=True)
            return None

def _db_update_order_status_sync(update_data: dict) -> bool:
    """【重構版】接收一個字典來更新訂單狀態，以提高穩定性"""
    with db_session() as s:
        try:
            client_oid = update_data.get("client_oid")
            if not client_oid:
                return False
            
            order = s.query(DBOrder).filter_by(client_oid=client_oid).first()
            if order:
                # 從字典中取出數據並更新
                for key, value in update_data.items():
                    if hasattr(order, key) and value is not None:
                        setattr(order, key, value)
                log.debug(f"DB: Updated order {client_oid} with data: {update_data}")
                return True
            log.warning(f"DB: Order {client_oid} not found for status update.")
            return False
        except Exception as e:
            log.error(f"DB error updating order {update_data.get('client_oid')} status: {e}", exc_info=True)
            return False

def _db_log_trade_sync(trade_data: Dict) -> bool:
    with db_session() as s:
        try:
            order = s.query(DBOrder).filter_by(client_oid=trade_data["client_oid"]).first()
            if not order:
                log.error(f"DB: Cannot log trade, order with client_oid {trade_data['client_oid']} not found.")
                return False

            db_trade = DBTradeLog(
                order_id=order.id,
                asset_pair=CFG["asset_pair"],
                exchange_trade_id=trade_data["exchange_trade_id"],
                side=trade_data["side"],
                price=trade_data["price"],
                quantity=trade_data["quantity"],
                fee_amount=trade_data["fee_amount"],
                fee_currency=trade_data["fee_currency"],
                is_taker=trade_data.get("is_taker", False),
                executed_at=trade_data.get("executed_at_utc", datetime.now(timezone.utc))
            )
            s.add(db_trade)
            log.debug(f"DB: Logged trade for order {trade_data['client_oid']}")
            return True
        except Exception as e:
            log.error(f"DB error logging trade for order {trade_data.get('client_oid')}: {e}", exc_info=True)
            return False

def _db_log_balance_snapshot_sync() -> bool:
    with db_session() as s:
        try:
            ts = datetime.now(timezone.utc)
            s.add(DBBalanceSnapshot(snapshot_ts=ts, currency=CFG["usdt_unit"].upper(), total_balance=USDT_BALANCE, available_balance=USDT_BALANCE))
            s.add(DBBalanceSnapshot(snapshot_ts=ts, currency=CFG["twd_unit"].upper(), total_balance=TWD_BALANCE, available_balance=TWD_BALANCE))
            log.info("DB: Logged balance snapshot.")
            return True
        except Exception as e:
            log.error(f"DB error logging balance snapshot: {e}", exc_info=True)
            return False
            
async def _db_load_initial_price_history_async(limit: int) -> List[Tuple[int, Decimal]]:
    """
    【最終校準版】修正了遺漏的 API 呼叫，確保能獲取並儲存歷史價格。
    """
    with db_session() as s:
        kline_data_db = s.query(DBMarketKline1m.ts, DBMarketKline1m.close).filter(
            DBMarketKline1m.asset_pair == CFG["asset_pair"]
        ).order_by(DBMarketKline1m.ts.desc()).limit(limit).all()
    
    if not kline_data_db:
        log.info("DB has no K-line history, fetching from MAX API...")
        try:
            # 【關鍵修正】補回遺漏的 API 呼叫
            k_data_api = await max_api.get_v2_k_data(CFG["asset_pair"], limit, 1)
            
            if k_data_api:
                with db_session() as s:
                    for k in k_data_api:
                        try:
                            ts_dt = datetime.fromtimestamp(k[0], tz=timezone.utc)
                            open_p, high_p, low_p, close_p = (Decimal(str(p)) for p in k[1:5])
                            vol_asset = Decimal(str(k[5]))
                            vol_quote = vol_asset * close_p
                            _db_save_kline_sync(ts_dt, open_p, high_p, low_p, close_p, vol_asset, vol_quote)
                        except Exception:
                            pass # 忽略單筆儲存錯誤
                
                # 重新從 DB 查詢，以確保格式一致
                with db_session() as s:
                    kline_data_db = s.query(DBMarketKline1m.ts, DBMarketKline1m.close).filter(
                        DBMarketKline1m.asset_pair == CFG["asset_pair"]
                    ).order_by(DBMarketKline1m.ts.desc()).limit(limit).all()
            else:
                 log.warning("MAX API returned no K-line data.")

        except Exception as e:
            log.error(f"Failed to fetch K-line data from API: {e}", exc_info=True)
            return []

    history = [(int(row.ts.timestamp() * 1000), row.close) for row in reversed(kline_data_db or [])]
    log.info(f"DB: Loaded {len(history)} K-line records for initial price history.")
    return history

def _db_log_daily_pnl_sync(pnl_data: dict) -> bool:
    """【重構版】接收一個字典來記錄每日PNL，以提高穩定性"""
    with db_session() as s:
        try:
            today = datetime.now(timezone.utc).date()
            realized_pnl_twd = pnl_data.get("realized_pnl_twd", Decimal("0.0"))

            pnl_entry = s.query(DBDailyPNL).filter_by(trade_date=today, strategy_id=STRATEGY_DB_ID, asset_pair=CFG["asset_pair"]).first()
            if pnl_entry:
                pnl_entry.realized_pnl += realized_pnl_twd
                pnl_entry.net_pnl = pnl_entry.realized_pnl
            else:
                pnl_entry = DBDailyPNL(
                    trade_date=today,
                    strategy_id=STRATEGY_DB_ID,
                    asset_pair=CFG["asset_pair"],
                    realized_pnl=realized_pnl_twd,
                    net_pnl=realized_pnl_twd,
                    pnl_currency=CFG["twd_unit"].upper(),
                )
                s.add(pnl_entry)

            display_precision = CFG.get('price_precision', '0.001')
            log.info(f"DB: Logged/Updated daily PnL for {today}: {realized_pnl_twd:.{len(display_precision.split('.')[-1])}f} {CFG['twd_unit'].upper()}")
            return True
        except Exception as e:
            log.error(f"DB error logging daily PnL: {e}", exc_info=True)
            return False

def _db_get_pnl_summary_sync() -> Dict[str, Decimal]:
    """【新增】從資料庫查詢多個時間維度的已實現PNL總和"""
    with db_session() as s:
        today = datetime.now(timezone.utc).date()
        
        # 計算各個時間範圍的起始點
        seven_days_ago = today - timedelta(days=6)
        start_of_month = today.replace(day=1)
        start_of_half_year = today.replace(month=1, day=1) if today.month <= 6 else today.replace(month=7, day=1)
        start_of_year = today.replace(month=1, day=1)

        # 查詢函數
        def query_pnl(start_date, end_date):
            result = s.query(func.sum(DBDailyPNL.realized_pnl)).filter(
                DBDailyPNL.strategy_id == STRATEGY_DB_ID,
                DBDailyPNL.trade_date >= start_date,
                DBDailyPNL.trade_date <= end_date
            ).scalar()
            return result or Decimal("0.0")

        # 執行所有查詢
        pnl_today = query_pnl(today, today)
        pnl_7_days = query_pnl(seven_days_ago, today)
        pnl_month = query_pnl(start_of_month, today)
        pnl_half_year = query_pnl(start_of_half_year, today)
        pnl_year = query_pnl(start_of_year, today)
        
        # 獲取今日成交筆數
        pnl_entry_today = s.query(DBDailyPNL).filter_by(trade_date=today, strategy_id=STRATEGY_DB_ID).first()
        trades_count_today = pnl_entry_today.trades_count if pnl_entry_today else 0

        return {
            "today": pnl_today,
            "trades_today": trades_count_today,
            "last_7_days": pnl_7_days,
            "this_month": pnl_month,
            "this_half_year": pnl_half_year,
            "this_year": pnl_year,
        }

def _db_save_kline_sync(ts_dt: datetime, open_p: Decimal, high_p: Decimal, low_p: Decimal, close_p: Decimal, vol_asset: Decimal, vol_quote: Decimal):
    with db_session() as s:
        try:
            # Upsert logic: Check if exists, then update or insert
            kline = s.query(DBMarketKline1m).filter_by(ts=ts_dt, asset_pair=CFG["asset_pair"]).first()
            if kline: # Update existing
                kline.open = open_p
                kline.high = high_p
                kline.low = low_p
                kline.close = close_p
                kline.volume_asset = vol_asset
                kline.volume_quote = vol_quote
            else: # Insert new
                kline = DBMarketKline1m(
                    ts=ts_dt, asset_pair=CFG["asset_pair"],
                    open=open_p, high=high_p, low=low_p, close=close_p,
                    volume_asset=vol_asset, volume_quote=vol_quote
                )
                s.add(kline)
            # log.debug(f"DB: Saved 1m K-line for {ts_dt}") # Can be very verbose
        except Exception as e:
            log.error(f"DB error saving K-line for {ts_dt}: {e}", exc_info=True)


# --- Async Wrappers for DB Operations ---
async def run_db_sync(func, *args, **kwargs):
    """Helper to run synchronous DB functions in an executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)

# --- 工具函數 ---
def gen_client_oid(tag: str = "grid") -> str:
    return f"{tag}_{CFG.get('asset_pair','pair')}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"[:36]
def quantize_price(p: Decimal) -> Decimal: return p.quantize(CFG_PRICE_PRECISION, rounding=getcontext().rounding)
def quantize_qty(q: Decimal) -> Decimal: return q.quantize(CFG_QTY_PRECISION, rounding="ROUND_DOWN")

async def get_current_market_price() -> Optional[Decimal]:
    """
    【最終修正版】永遠優先從 API 獲取最新價格，失敗時才使用歷史數據作為備用。
    """
    try:
        # 優先嘗試 API 請求
        ticker = await max_api.get_v2_ticker(market=CFG["asset_pair"])
        if ticker and ticker.get("last"):
            return Decimal(str(ticker["last"]))
    except Exception as e:
        log.error(f"Error fetching ticker price: {e}")
    
    # 如果 API 失敗，且歷史紀錄存在，則使用最後一筆歷史價格
    if PRICE_HISTORY:
        log.warning("Falling back to last known price from history.")
        return PRICE_HISTORY[-1][1]
        
    # 如果連歷史紀錄都沒有，則返回 None
    return None

# --- 餘額與權益管理 ---
async def update_balances_from_api() -> None:
    """
    【智能日誌版】只在必要時（每小時或趨勢改變時）才打印詳細的餘額日誌。
    """
    global USDT_BALANCE, TWD_BALANCE, AVAILABLE_USDT_BALANCE, AVAILABLE_TWD_BALANCE
    global TOTAL_EQUITY_TWD, LAST_BALANCE_UPDATE_TS, PEAK_EQUITY_TWD, LAST_STRATEGIC_ACTION_TS
    global LAST_BALANCE_LOG_TS
        
    try:
        current_price = await get_current_market_price()
        if not current_price:
            log.warning("Could not fetch current price for equity calculation.")
            if PRICE_HISTORY: current_price = PRICE_HISTORY[-1][1]
            else: return

        usdt_balance_data = await max_api.get_v2_balance("usdt")
        twd_balance_data = await max_api.get_v2_balance("twd")
        if usdt_balance_data and 'balance' in usdt_balance_data and twd_balance_data and 'balance' in twd_balance_data:
            
            # --- 【↓↓↓ 核心修正：正確解讀 API 欄位 ↓↓↓】 ---
            usdt_avail = Decimal(str(usdt_balance_data.get("balance", "0")))
            usdt_locked = Decimal(str(usdt_balance_data.get("locked", "0")))
            twd_avail = Decimal(str(twd_balance_data.get("balance", "0")))
            twd_locked = Decimal(str(twd_balance_data.get("locked", "0")))
            
            # 正確計算總餘額與可用餘額
            USDT_BALANCE = usdt_avail + usdt_locked
            TWD_BALANCE = twd_avail + twd_locked
            AVAILABLE_USDT_BALANCE = usdt_avail
            AVAILABLE_TWD_BALANCE = twd_avail
            # --- 【↑↑↑ 修正結束 ↑↑↑】 ---

            TOTAL_EQUITY_TWD = TWD_BALANCE + USDT_BALANCE * current_price
            LAST_BALANCE_UPDATE_TS = datetime.now(timezone.utc)
            
            now = datetime.now(timezone.utc)
            target_usdt_ratio = get_ema_target_bias()
            current_trend_str = "看漲" if target_usdt_ratio == Decimal(CFG["bias_high"]) else "看跌" if target_usdt_ratio == Decimal(CFG["bias_low"]) else "中性"

            should_log = False
            # 1. 如果從未打印過，或距離上次打印已超過一小時
            if LAST_BALANCE_LOG_TS is None or (now - LAST_BALANCE_LOG_TS).total_seconds() >= 3600:
                should_log = True
            
            # 2. 或者，如果當前的趨勢判斷與上一次記錄的不同
            if current_trend_str != PREVIOUS_EMA_TREND:
                should_log = True
            
            if should_log:
                target_twd_ratio = Decimal("1.0") - target_usdt_ratio
                log_msg = (
                    f"Balances: USDT={USDT_BALANCE:<8.2f}(Avail:{AVAILABLE_USDT_BALANCE:<8.2f}) TWD={TWD_BALANCE:<8.2f}(Avail:{AVAILABLE_TWD_BALANCE:<8.2f}) | "
                    f"Equity: {TOTAL_EQUITY_TWD:.2f} TWD | "
                    f"EMA Target: {target_usdt_ratio:.0%} USDT / {target_twd_ratio:.0%} TWD"
                )
                log.info(log_msg)
                LAST_BALANCE_LOG_TS = now
    except Exception as e:
        log.error(f"Error updating balances: {e}", exc_info=True)

# --- 核心訂單邏輯 ---
async def place_grid_order(side: str, price: Decimal, qty: Decimal, layer_idx: Optional[int], tag: str = "grid") -> Optional[str]:
    global ACTIVE_ORDERS
    
    if RISK_CTRL:
        is_risk_hit, should_cancel_all = await RISK_CTRL.enforce_risk_limits()
        if is_risk_hit:
            if should_cancel_all:
                log.critical(f"Order placement HALTED: USDT risk limit exceeded.")
                return None
            elif side == "buy":
                log.warning(f"Order placement (BUY) HALTED: TWD balance risk limit hit.")
                return None
    
    client_oid = gen_client_oid(tag=f"{tag}{layer_idx if layer_idx is not None else ''}{side[0]}")
    price_q, qty_q = quantize_price(price), quantize_qty(qty)

    if qty_q <= 0 or price_q <= 0 or (price_q * qty_q) < Decimal(CFG.get("min_order_value_twd", "1.0")):
        log.warning(f"Order {client_oid} skipped. Calculated value {price_q * qty_q:.2f} TWD is below min_order_value_twd.")
        return None

    log.info(f"Attempting place: {client_oid} - {side.upper()} {qty_q} {CFG['usdt_unit']} @ {price_q} {CFG['twd_unit']}")
    try:
        # --- 【↓↓↓ 最終核心修正：使用正確的訂單類型，並移除無效參數 ↓↓↓】 ---
        # 網格訂單和偏好訂單都應為 'limit' (限價單)
        # MAX API v2 文件中沒有 'post_only' 參數，故移除
        response = await max_api.place_v2_order(
            market=CFG["asset_pair"], 
            side=side, 
            price=price_q, 
            volume=qty_q,
            client_oid=client_oid, 
            ord_type='limit' 
        )
        # --- 【↑↑↑ 修正結束 ↑↑↑】 ---

        if response and response.get("id"):
            order_data = {
                "client_oid": client_oid, "exchange_id": str(response["id"]), "price": price_q, "side": side,
                "qty": qty_q, "filled_qty": Decimal("0"), "layer_idx": layer_idx, "status": "open",
                "created_at_utc": datetime.now(timezone.utc), "order_type": 'limit'
            }
            ACTIVE_ORDERS[client_oid] = order_data
            await run_db_sync(_db_log_order_sync, order_data)
            log.info(f"Order placed: {client_oid}, Exchange ID: {response['id']}")
            return client_oid
        else:
            error_msg = response.get("error", {"message": "Unknown error"}) if response else {"message": "No response"}
            log.error(f"Failed to place order {client_oid}: {error_msg}")
            if "balance" in str(error_msg).lower(): await update_balances_from_api()
            return None
    except Exception as e:
        log.error(f"Exception placing order {client_oid}: {e}", exc_info=True)
        return None

async def cancel_all_market_orders(reason: str = "generic_sweep"):
    log.info(f"Sending command to cancel ALL orders for {CFG['asset_pair']} due to: {reason}")
    try:
        result = await max_api.cancel_all_v2_market_orders(market=CFG["asset_pair"])
        log.info(f"Exchange-level cancel-all command sent. Result: {result}")
        ACTIVE_ORDERS.clear()
    except Exception as e:
        log.error(f"Error during exchange-level mass cancel: {e}", exc_info=True)

async def handle_order_fill(fill_data: Dict):
    """【最終校準版】處理訂單成交，不再負責傳遞 peak_equity。"""
    global USDT_BALANCE, TWD_BALANCE, TOTAL_EQUITY_TWD, LAST_TRADE_TS
    client_oid = fill_data.get("client_oid")
    if not client_oid: return
    order = ACTIVE_ORDERS.get(client_oid)
    if not order: return
    if order['status'] in ['filled', 'cancelled']: return
    cummulative_qty = Decimal(str(fill_data.get("cummulative_quantity", "0")))
    final_status_str = fill_data.get("status")
    final_status = OrderStatusEnum(final_status_str) if final_status_str and final_status_str in OrderStatusEnum._value2member_map_ else OrderStatusEnum(order['status'])
    order['filled_qty'] = cummulative_qty
    order['status'] = final_status.value
    log.info(f"Order update: {client_oid}, Status: {final_status.value}, TotalFilled: {cummulative_qty}/{order.get('qty', 'N/A')}")
    db_update_payload = {"client_oid": client_oid, "status": final_status,"filled_quantity": cummulative_qty,"average_fill_price": Decimal(str(fill_data.get("avg_fill_price", order.get('price'))))}
    await run_db_sync(_db_update_order_status_sync, db_update_payload)
    if final_status == OrderStatusEnum.FILLED:
        LAST_TRADE_TS = datetime.now(timezone.utc)
        log.info(f"Order {client_oid} is fully filled. Processing balance update and placing replacement.")
        await update_balances_from_api() # <--- 直接呼叫，不傳遞參數
        
        layer_idx, side = order.get("layer_idx"), order.get("side")
        ACTIVE_ORDERS.pop(client_oid, None)

        if layer_idx is None and side == 'sell':
            pass
        
        if layer_idx is not None:
            layer = GRID_LAYERS[layer_idx]
            if side == "sell":
                realized_pnl = layer.gap_abs * cummulative_qty
                log.info(f"GRID PNL: Realized PNL of approx. {realized_pnl:.4f} TWD from trade {client_oid}")
                await run_db_sync(_db_log_daily_pnl_sync, {"realized_pnl_twd": realized_pnl})
            new_side = "sell" if side == "buy" else "buy"
            avg_fill_price = db_update_payload['average_fill_price']
            new_price = quantize_price(avg_fill_price + layer.gap_abs if new_side == 'sell' else avg_fill_price - layer.gap_abs)
            price_for_calc = await get_current_market_price() or new_price
            new_qty = quantize_qty((layer.size_pct * TOTAL_EQUITY_TWD) / price_for_calc)
            if new_qty > 0: await place_grid_order(new_side, new_price, new_qty, layer_idx, tag="gr_repl")
            else: log.warning(f"Calculated replacement qty for {client_oid} is zero, skipping.")


# --- 啟動與網格管理 ---
async def handle_orphan_orders_on_startup():
    log.info("Checking for existing open orders (orphans) on startup...")
    try:
        await cancel_all_market_orders(reason="startup_cleanup")
        await asyncio.sleep(3)
        log.info("Orphan order cleanup finished.")
    except Exception as e:
        log.error(f"Critical error handling orphan orders on startup: {e}.", exc_info=True)
        raise SystemExit("Failed to handle orphan orders. Halting.")

async def poll_order_updates():
    """【最終校準版】主動輪詢訂單狀態，不再負責傳遞 peak_equity。"""
    active_oids = list(ACTIVE_ORDERS.keys())
    if not active_oids: return

    for oid in active_oids:
        order_in_memory = ACTIVE_ORDERS.get(oid)
        if not order_in_memory or 'exchange_id' not in order_in_memory: continue
        try:
            exchange_id = int(order_in_memory['exchange_id'])
            order_status_from_api = await max_api.get_v2_order(exchange_id)
            if order_status_from_api:
                state = order_status_from_api.get("state")
                if state == 'done' and order_in_memory['status'] != 'filled':
                    log.info(f"Polled update: Order {oid} is fully filled.")
                    fill_payload = {"client_oid": oid,"status": "filled","cummulative_quantity": order_status_from_api.get("executed_volume"),"avg_fill_price": order_status_from_api.get("avg_price"),}
                    await handle_order_fill(fill_payload) # <--- 直接呼叫，不傳遞參數
                elif state in ['cancel', 'failed'] and oid in ACTIVE_ORDERS:
                    log.info(f"Polled update: Order {oid} is cancelled/failed. Removing from active tracking.")
                    ACTIVE_ORDERS.pop(oid, None)
                    await run_db_sync(_db_update_order_status_sync, {"client_oid": oid, "status": OrderStatusEnum.CANCELLED})
        except Exception as e:
            log.warning(f"Error polling order status for {oid}: {e}")
        await asyncio.sleep(0.2)

async def rebuild_grid_at_center(center_price: Decimal, full_rebuild: bool = True):
    global LAST_RECENTER_TS, TOTAL_EQUITY_TWD
    log.info(f"Attempting to rebuild grid around new center price: {center_price}")

    # --- 【↓↓↓ 新增：下單前預检 (Pre-flight Check) ↓↓↓】 ---
    if 'TOTAL_EQUITY_TWD' not in globals() or TOTAL_EQUITY_TWD is None or TOTAL_EQUITY_TWD <= 0:
        if not await update_balances_from_api() or TOTAL_EQUITY_TWD <= 0:
            log.error("Equity unavailable or zero. Aborting grid rebuild.")
            return

    price_for_calc = await get_current_market_price() or center_price
    if price_for_calc <= 0:
        log.error("Invalid price for quantity calculation. Aborting grid rebuild.")
        return

    # 找出最小的訂單百分比，用於計算最小的訂單數量
    try:
        min_size_pct = min(layer.size_pct for layer in GRID_LAYERS)
    except (ValueError, TypeError): # 處理 GRID_LAYERS 為空或格式錯誤的情況
        log.error("GRID_LAYERS is empty or invalid. Aborting grid rebuild.")
        return
        
    min_qty_usdt = quantize_qty(min_size_pct * TOTAL_EQUITY_TWD / price_for_calc)

    # 找出理論上價格最低的買單
    farthest_buy_price = center_price
    for layer in GRID_LAYERS:
        price = center_price - (layer.gap_abs * layer.levels_each_side)
        if price < farthest_buy_price:
            farthest_buy_price = price
    farthest_buy_price = quantize_price(farthest_buy_price)

    min_order_value_twd = Decimal(CFG.get("min_order_value_twd", "300.0"))
    
    # 計算最小訂單的理論價值
    smallest_order_value = min_qty_usdt * farthest_buy_price
    
    if smallest_order_value < min_order_value_twd:
        log.warning("Grid rebuild ABORTED. Calculated smallest order value "
                    f"({smallest_order_value:.2f} TWD) is below threshold ({min_order_value_twd} TWD).")
        log.warning("This is a protective measure to prevent emptying the order book due to insufficient funds.")
        log.warning("Please increase capital or adjust strategy parameters. The existing grid will remain active.")
        # 更新時間戳以避免在下一個間隔立即重試
        LAST_RECENTER_TS = datetime.now(timezone.utc)
        return
    # --- 【↑↑↑ 預检結束 ↑↑↑】 ---

    log.info(f"Pre-flight check passed. Proceeding with grid rebuild around {center_price}")
    if full_rebuild:
        await cancel_all_market_orders(reason="recenter_rebuild")
        await asyncio.sleep(2)

    tasks = []
    for layer in GRID_LAYERS:
        qty_usdt = quantize_qty(layer.size_pct * TOTAL_EQUITY_TWD / price_for_calc)
        if qty_usdt <= 0: continue
        for i in range(1, layer.levels_each_side + 1):
            buy_price = quantize_price(center_price - (layer.gap_abs * i))
            sell_price = quantize_price(center_price + (layer.gap_abs * i))
            if buy_price > 0: tasks.append(place_grid_order("buy", buy_price, qty_usdt, layer.idx))
            if sell_price > 0: tasks.append(place_grid_order("sell", sell_price, qty_usdt, layer.idx))

    await asyncio.gather(*tasks)
    # --- 【↓↓↓ 在網格重建成功後，發送通知 ↓↓↓】 ---
    log.info(f"Grid rebuild process completed. Attempted to place {len(tasks)} orders.")
    msg = (f"網格已圍繞中心價 `{center_price}` 重新建立。\n"
           f"共嘗試掛上 `{len(tasks)}` 筆新訂單。")
    # 'recenter' 作為 key，確保 15 分鐘內不重複發送
    await alerter.send_strategy_event(msg, alert_key='recenter')

    LAST_RECENTER_TS = datetime.now(timezone.utc)

def calculate_ema_from_history(span: int) -> Optional[Decimal]:
    if len(PRICE_HISTORY) < span / 10 and len(PRICE_HISTORY) < 10: return None
    prices = [p[1] for p in PRICE_HISTORY]
    series = pd.Series(prices, dtype=float)
    try:
        ema_val = series.ewm(span=span, adjust=False).mean().iloc[-1]
        return Decimal(str(ema_val))
    except Exception: return None

def get_ema_target_bias() -> Decimal:
    """
    【新增】根據EMA快慢線交叉，計算並返回目標USDT曝險比例。
    這是一個純計算函數，不執行任何交易。
    """
    ema_fast = calculate_ema_from_history(int(CFG["ema_span_fast_bars"]))
    ema_slow = calculate_ema_from_history(int(CFG["ema_span_slow_bars"]))

    # 如果無法計算EMA（例如歷史數據不足），則返回中性目標
    if ema_fast is None or ema_slow is None:
        return Decimal(CFG["bias_neutral_target"])

    if ema_fast > ema_slow:
        return Decimal(CFG["bias_high"])
    elif ema_fast < ema_slow:
        return Decimal(CFG["bias_low"])
    else: # 快慢線相等，趨勢中性
        return Decimal(CFG["bias_neutral_target"])

async def manage_directional_bias():
    """
    【最終穩健版】在可用餘額檢查時，加入了安全邊際，以應對時序競態問題。
    """
    global LAST_BIAS_REBALANCE_TS, PREVIOUS_EMA_TREND
    
    await update_balances_from_api() # 決策前，先更新情報
    
    target_ratio = get_ema_target_bias()
    
    # 判斷趨勢是否變更 (這部分邏輯不變)
    current_trend = "看漲" if target_ratio == Decimal(CFG["bias_high"]) else "看跌" if target_ratio == Decimal(CFG["bias_low"]) else "中性"
    if PREVIOUS_EMA_TREND is None:
        PREVIOUS_EMA_TREND = current_trend
    elif current_trend != PREVIOUS_EMA_TREND:
        log.info(f"EMA trend has changed from '{PREVIOUS_EMA_TREND}' to '{current_trend}'. Sending alert.")
        msg = (f"🧭 **趨勢變更: {current_trend}**\n\n"
               f"EMA 指標已發生變化。\n"
               f"目標 USDT 倉位比例已調整為: `{target_ratio:.0%}`")
        await alerter.send_strategy_event(msg, alert_key='trend_change')
        PREVIOUS_EMA_TREND = current_trend
    
    price = await get_current_market_price()
    if not price or price <= 0 or TOTAL_EQUITY_TWD <= 0: return
        
    current_ratio = (USDT_BALANCE * price) / TOTAL_EQUITY_TWD
    delta_value_target = (target_ratio - current_ratio) * TOTAL_EQUITY_TWD
    
    if abs(delta_value_target) > Decimal(CFG["bias_rebalance_threshold_twd"]):
        value_to_trade = delta_value_target * Decimal(CFG["bias_rebalance_fraction"])
        qty_to_trade = quantize_qty(value_to_trade / price)
        
        side = "buy" if qty_to_trade > 0 else "sell"
        qty_abs = abs(qty_to_trade)

        slip_price = price * (Decimal("1.001") if side == "buy" else Decimal("0.999"))
        order_value_twd = abs(qty_abs * slip_price)
        min_order_value = Decimal(CFG.get("min_order_value_twd", "300.0"))

        if order_value_twd < min_order_value:
            log.debug(f"Bias rebalance skipped. Calculated order value {order_value_twd:.2f} TWD is below threshold.")
            return
            
        # --- 【↓↓↓ 核心修正：引入安全邊際 ↓↓↓】 ---
        # 為了應對狀態延遲，我們在檢查時，人為地將所需金額提高 1% 作為安全邊際。
        SAFETY_MARGIN = Decimal("1.01") 
        
        if side == 'buy' and AVAILABLE_TWD_BALANCE < (order_value_twd * SAFETY_MARGIN):
            log.debug(f"Bias rebalance BUY skipped. Insufficient available TWD with safety margin. (Need: {order_value_twd * SAFETY_MARGIN:.2f}, Have: {AVAILABLE_TWD_BALANCE:.2f})")
            return
        if side == 'sell' and AVAILABLE_USDT_BALANCE < (qty_abs * SAFETY_MARGIN):
            log.debug(f"Bias rebalance SELL skipped. Insufficient available USDT with safety margin. (Need: {qty_abs * SAFETY_MARGIN}, Have: {AVAILABLE_USDT_BALANCE})")
            return
        # --- 【↑↑↑ 修正結束 ↑↑↑】 ---

        if qty_abs > 0:
            log.info(f"Bias rebalance: EMA trend suggests target {target_ratio:.0%}, trying to {side} {qty_abs} USDT.")
            await place_grid_order(side, slip_price, qty_abs, layer_idx=None, tag="bias_")
            LAST_BIAS_REBALANCE_TS = datetime.now(timezone.utc)

async def send_periodic_report():
    """【最終正確版】收集並發送包含多維度PNL的績效報告。"""
    try:
        # 1. 從新的輔助函數一次性獲取所有 PNL 匯總數據
        # 這個函數已經處理了 'today' 的定義和所有資料庫查詢
        pnl_summary = await run_db_sync(_db_get_pnl_summary_sync)

        # 2. 獲取當前持倉比例
        current_price = await get_current_market_price() or (PRICE_HISTORY[-1][1] if PRICE_HISTORY else Decimal("30.0"))
        if TOTAL_EQUITY_TWD > 0:
            current_usdt_ratio = (USDT_BALANCE * current_price) / TOTAL_EQUITY_TWD
        else:
            current_usdt_ratio = Decimal("0.0")

        # 3. 獲取當前趨勢判斷
        target_usdt_ratio = get_ema_target_bias()
        current_trend = "看漲" if target_usdt_ratio == Decimal(CFG["bias_high"]) else "看跌" if target_usdt_ratio == Decimal(CFG["bias_low"]) else "中性"

        # 4. 組裝訊息 (這裡直接使用 pnl_summary 的結果)
        report_text = (
            f"📊 **USDTTWD 網格策略績效報告**\n"
            f"_(截至 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})_\n\n"
            f"--- *績效 (TWD)* ---\n"
            f"**今日已實現利潤:** `{pnl_summary['today']:+.2f}`\n"
            f"**今日總成交筆數:** `{pnl_summary['trades_today']}`\n"
            f"**近七日已實現利潤:** `{pnl_summary['last_7_days']:+.2f}`\n"
            f"**當月已實現利潤:** `{pnl_summary['this_month']:+.2f}`\n"
            f"**當前半年已實現利潤:** `{pnl_summary['this_half_year']:+.2f}`\n"
            f"**當年已實現利潤:** `{pnl_summary['this_year']:+.2f}`\n\n"
            f"--- *狀態* ---\n"
            f"**帳戶總權益:** `{TOTAL_EQUITY_TWD:,.2f} TWD`\n"
            f"**當前持倉:** `{USDT_BALANCE:,.2f} USDT` / `{TWD_BALANCE:,.2f} TWD` ({current_usdt_ratio:.1%})\n"
            f"**在掛訂單數:** `{len(ACTIVE_ORDERS)}`\n"
            f"**當前趨勢判斷:** `{current_trend}` (目標 `{target_usdt_ratio:.0%} USDT`)"
        )

        await alerter.send_system_event(report_text)

    except Exception as e:
        log.error(f"Failed to send periodic report: {e}", exc_info=True)
        await alerter.send_critical_alert(f"❌ 產生績效報告時發生錯誤！\n\n原因: `{e}`", alert_key='report_fail')

async def check_black_swan_event():
    """
    檢查是否發生黑天鵝事件 (價格在短時間內劇烈波動)。
    """
    global STRATEGY_HALTED
    if not CFG.get("use_black_swan_protection"):
        return

    check_minutes = int(CFG["black_swan_check_minutes"])
    threshold_pct = Decimal(CFG["black_swan_threshold_pct"])
    
    # 需要足夠的歷史數據來比較
    if len(PRICE_HISTORY) < check_minutes * 5: # 假設每分鐘至少有5個價格點
        return

    now_ts = time.time()
    past_ts = now_ts - (check_minutes * 60)
    
    relevant_prices = [p[1] for p in PRICE_HISTORY if p[0]/1000 >= past_ts]
    if len(relevant_prices) < 2:
        return

    current_price = relevant_prices[-1]
    highest_price = max(relevant_prices)
    lowest_price = min(relevant_prices)

    # 檢查價格波動是否超過閾值
    if (highest_price - lowest_price) / lowest_price > threshold_pct:
        # --- 【↓↓↓ 在觸發黑天鵝事件時，呼叫 alerter ↓↓↓】 ---
        msg = (f"*USDTTWD 在 {check_minutes} 分鐘內波動超過 {threshold_pct:.1%}!*\n\n"
               f"價格區間: `{lowest_price}` - `{highest_price}`\n\n"
               f"策略已自動停止並撤銷所有訂單，請立即介入檢查！")
        await alerter.send_critical_alert(msg, alert_key='black_swan')

        log.critical("!!! BLACK SWAN EVENT DETECTED !!!")
        log.critical(f"Price fluctuated more than {threshold_pct:.2%} within {check_minutes} minutes.")
        log.critical("HALTING STRATEGY TO PREVENT FURTHER LOSSES. MANUAL INTERVENTION REQUIRED.")
        STRATEGY_HALTED = True
        # 觸發後永久停止 (根據設定檔註解)
        # 您可以選擇在此處呼叫 shutdown_strategy
        asyncio.create_task(shutdown_strategy(sig="BLACK_SWAN"))

async def strategy_main_loop():
    """【最終校驗版】不再管理 peak_equity，邏輯更清晰。"""
    global STRATEGY_DB_ID, RISK_CTRL, LAST_BALANCE_UPDATE_TS, LAST_RECENTER_TS
    global LAST_BIAS_REBALANCE_TS, LAST_DB_BALANCE_SNAPSHOT_TS, INITIAL_PRICE
    global LAST_REPORT_HOUR, LAST_TRADE_TS
    
    log.info("Strategy main loop starting...")
    await asyncio.sleep(1)
    
    db_strategy_id = await run_db_sync(_db_get_or_create_strategy_sync, STRATEGY_NAME, "USDTTWD Grid Strategy V13", CFG)
    if not db_strategy_id: raise SystemExit("Failed to get or create DB strategy entry. Halting.")
    STRATEGY_DB_ID = db_strategy_id
    
    RISK_CTRL = RiskController(config_path=str(CFG_PATH))
    await RISK_CTRL.initialize()
    
    await update_balances_from_api() # 首次呼叫，將在內部自動設定初始 Peak Equity

    initial_history = await _db_load_initial_price_history_async(PRICE_HISTORY.maxlen or 3100)
    if initial_history: PRICE_HISTORY.extend(initial_history)
    
    INITIAL_PRICE = await get_current_market_price() or Decimal(CFG.get("initial_price_estimate_for_equity", "30.0"))
    log.info(f"Initial market price for calculations: {INITIAL_PRICE}")

    await handle_orphan_orders_on_startup()
    await rebuild_grid_at_center(INITIAL_PRICE, full_rebuild=False)
    
    now_utc = datetime.now(timezone.utc)
    LAST_RECENTER_TS, LAST_BIAS_REBALANCE_TS, LAST_DB_BALANCE_SNAPSHOT_TS, LAST_TRADE_TS = now_utc, now_utc, now_utc, now_utc
    if LAST_BALANCE_UPDATE_TS is None: LAST_BALANCE_UPDATE_TS = now_utc

    log.info("Entering main strategy loop...")
    while not STRATEGY_HALTED:
        try:
            try:
                latest_price = await get_current_market_price()
                if latest_price: PRICE_HISTORY.append((int(time.time() * 1000), latest_price))
            except Exception as e: log.warning(f"Failed to update PRICE_HISTORY: {e}")
            
            now_utc = datetime.now(timezone.utc)
            await poll_order_updates()
            
            stagnation_alert_hours = int(CFG.get("stagnation_alert_hours", 12))
            stagnation_seconds = stagnation_alert_hours * 3600
            if LAST_TRADE_TS and (now_utc - LAST_TRADE_TS).total_seconds() > stagnation_seconds:
                msg = (f"*策略停滯警報!*\n\n"
                       f"距離上一筆成交已超過 `{stagnation_alert_hours}` 小時。\n\n"
                       f"市場價格可能已偏離網格有效區間，建議評估是否需要人工干預。")
                await alerter.send_strategy_event(msg, alert_key='stagnation_alert')
                LAST_TRADE_TS = now_utc

            if (now_utc - LAST_BALANCE_UPDATE_TS).total_seconds() >= int(CFG.get("api_balance_poll_interval_sec", 300)):
                await update_balances_from_api()
            
            if (now_utc - LAST_BIAS_REBALANCE_TS).total_seconds() >= int(CFG.get("bias_check_interval_sec", 60)):
                await manage_directional_bias()
            
            if (now_utc - LAST_RECENTER_TS).total_seconds() >= int(CFG.get("recenter_interval_minutes", 480)) * 60:
                price = await get_current_market_price()
                if price: await rebuild_grid_at_center(price, full_rebuild=True)

            if (now_utc - LAST_DB_BALANCE_SNAPSHOT_TS).total_seconds() >= int(CFG.get("db_snapshot_interval_sec", 3600)):
                await run_db_sync(_db_log_balance_snapshot_sync)
                LAST_DB_BALANCE_SNAPSHOT_TS = now_utc
            
            now = datetime.now()
            if now.hour in [0, 8, 18] and now.hour != LAST_REPORT_HOUR:
                log.info(f"Triggering periodic report for hour {now.hour}.")
                await send_periodic_report()
                LAST_REPORT_HOUR = now.hour
            elif now.hour not in [0, 8, 18]:
                LAST_REPORT_HOUR = -1 

            await check_black_swan_event()
            await asyncio.sleep(int(CFG.get("strategy_loop_interval_sec", 10)))
        except asyncio.CancelledError:
            log.info("Main strategy loop has been cancelled.")
            break
        except Exception as e:
            log.error(f"Unhandled error in main strategy loop: {e}", exc_info=True)
            log.info("Pausing for 30 seconds before retrying...")
            await asyncio.sleep(30)
    log.info("Strategy main loop has finished.")

# async def debug_peak_equity_logic():
#     """
#     【修正版的診斷工具】
#     修正了對 update_balances_from_api 的呼叫方式。
#     """
#     print("\n--- [開始] 權益計算邏輯診斷 ---")
    
#     # 診斷工具需要自己載入設定檔
#     load_cfg()
    
#     # 診斷工具需要自己管理 peak_equity 狀態
#     peak_equity = Decimal("0")
#     print(f"1. 初始狀態: peak_equity = {peak_equity}")

#     # 初始化 API client
#     await max_api.initialize()
#     print("\n2. 模擬第一次呼叫 update_balances_from_api (程式啟動時)...")
    
#     # 第一次呼叫，傳入初始的 peak_equity(0)
#     current_equity, peak_equity = await update_balances_from_api(peak_equity)
#     if current_equity > 0:
#         peak_equity = current_equity # 首次啟動，將當前權益設為峰值
#         print("   ✅ 第一次呼叫成功。")
#         print(f"   - update_balances_from_api 執行後的狀態:")
#         print(f"   - TOTAL_EQUITY_TWD = {TOTAL_EQUITY_TWD:,.2f}")
#         print(f"   - peak_equity (已更新) = {peak_equity:,.2f}")
#     else:
#         print("   ❌ 第一次呼叫失敗。")

#     print("\n3. 模擬第二次呼叫 update_balances_from_api (主迴圈正常運行時)...")
    
#     # 第二次呼叫，傳入更新後的 peak_equity
#     current_equity, peak_equity = await update_balances_from_api(peak_equity)
#     if current_equity > 0:
#         print("   ✅ 第二次呼叫成功。")
#         print(f"   - update_balances_from_api 執行後的狀態:")
#         print(f"   - TOTAL_EQUITY_TWD = {TOTAL_EQUITY_TWD:,.2f}")
#         print(f"   - peak_equity (可能已更新) = {peak_equity:,.2f}")
#     else:
#         print("   ❌ 第二次呼叫失敗。")
        
#     if peak_equity < 100000: # 檢查是否仍為舊的高點
#         print("\n✅ 診斷通過: peak_equity 已被成功重置為當前資金水平。")
#     else:
#         print("\n❌ 診斷失敗: peak_equity 仍然是舊的、過高的數值！")

#     print("\n--- [結束] 診斷完畢 ---")
#     await max_api.close()

async def main():
    try:
        load_cfg()
        await max_api.initialize()
        await alerter.send_system_event("✅ 交易機器人已成功啟動並初始化。")
        create_all_tables()
        if not await run_db_sync(check_db_connection):
             raise SystemExit("Database connection failed.")
        loop = asyncio.get_event_loop()
        for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown_strategy(s)))
        global MAIN_LOOP
        MAIN_LOOP = asyncio.create_task(strategy_main_loop())
        await MAIN_LOOP
    except SystemExit as e:
        log.warning(f"SystemExit: {e}")
        await alerter.send_critical_alert(f"❌ 機器人啟動失敗！\n\n原因: `{e}`", alert_key='startup_fail')
    except Exception as e:
        log.critical(f"Critical error during startup: {e}", exc_info=True)
        await alerter.send_critical_alert(f"❌ 機器人啟動時發生嚴重錯誤！\n\n原因: `{e}`", alert_key='startup_fail')
    finally:
        await shutdown_strategy()


async def shutdown_strategy(sig=None):
    global STRATEGY_HALTED
    if STRATEGY_HALTED: return
    STRATEGY_HALTED = True
    if sig:
        log.info(f"Received signal {sig}. Initiating shutdown...")
        await alerter.send_system_event(f"👋 收到訊號 {sig}，機器人已安全關閉。")
    if MAIN_LOOP and not MAIN_LOOP.done():
        MAIN_LOOP.cancel()
    log.info("Attempting to cancel all active orders...")
    try:
        if max_api and max_api._session and not max_api._session.closed:
            await max_api.cancel_all_v2_market_orders(CFG.get("asset_pair", "usdttwd"))
            log.info("Successfully sent cancel-all command.")
    except asyncio.CancelledError:
        log.warning("Cancel-all command was cancelled during shutdown.")
    except Exception as e:
        log.error(f"Final cancel orders failed during shutdown: {e}")
    if STRATEGY_DB_ID: pass
    await max_api.close()
    if sig is None: log.info("Graceful shutdown completed due to startup failure or normal exit.")


if __name__ == "__main__":
    asyncio.run(main())
    # print("--- 執行 Peak Equity 診斷模式 ---")
    # asyncio.run(debug_peak_equity_logic())    