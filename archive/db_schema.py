# db_schema.py
"""
資料庫模型定義 (使用 SQLAlchemy ORM) - V5 最終修正版

【V5 修正】
- 修正 Order 模型中的拼字錯誤 (Mched -> Mapped)。
"""

import enum
from datetime import datetime, date as DateObject
from decimal import Decimal
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# --- 通用基礎與混合類別 ---

class Base(DeclarativeBase):
    """乾淨的聲明性基礎類別，所有模型都應繼承它。"""
    pass

class IdMixin:
    """提供通用 id, created_at, updated_at 欄位的混合類別。"""
    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), server_default=sa.func.now())
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())

# --- Enum 定義 ---

class OrderStatusEnum(enum.Enum):
    NEW = "new"; OPEN = "open"; FILLED = "filled"; PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"; REJECTED = "rejected"; EXPIRED = "expired"; FAILED = "failed"

# --- 資料表模型定義 ---

class Strategy(IdMixin, Base):
    __tablename__ = 'strategies'
    name: Mapped[str] = mapped_column(sa.String(100), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(sa.Text)
    params_json: Mapped[Optional[str]] = mapped_column(sa.Text)
    is_active: Mapped[bool] = mapped_column(sa.Boolean, default=False, nullable=False)
    orders: Mapped[list["Order"]] = relationship("Order", back_populates="strategy")
    def __repr__(self): return f"<Strategy(id={self.id}, name='{self.name}', active={self.is_active})>"

class Order(IdMixin, Base):
    __tablename__ = 'orders'
    strategy_id: Mapped[Optional[int]] = mapped_column(sa.Integer, sa.ForeignKey('strategies.id'), index=True)
    client_oid: Mapped[str] = mapped_column(sa.String(64), unique=True, nullable=False, index=True)
    exchange_order_id: Mapped[Optional[str]] = mapped_column(sa.String(64), index=True)
    asset_pair: Mapped[str] = mapped_column(sa.String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    # 【V5 修正】修正拼字錯誤 Mched -> Mapped
    order_type: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    price: Mapped[Decimal] = mapped_column(sa.Numeric(20, 10))
    quantity: Mapped[Decimal] = mapped_column(sa.Numeric(20, 10))
    status: Mapped[OrderStatusEnum] = mapped_column(sa.Enum(OrderStatusEnum), nullable=False, default=OrderStatusEnum.NEW)
    filled_quantity: Mapped[Decimal] = mapped_column(sa.Numeric(20, 10), default=Decimal('0.0'))
    average_fill_price: Mapped[Optional[Decimal]] = mapped_column(sa.Numeric(20, 10))
    fee_amount: Mapped[Optional[Decimal]] = mapped_column(sa.Numeric(20, 10))
    fee_currency: Mapped[Optional[str]] = mapped_column(sa.String(10))
    layer_idx: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    strategy: Mapped["Strategy"] = relationship("Strategy", back_populates="orders")
    trades: Mapped[list["TradeLog"]] = relationship("TradeLog", back_populates="order")
    def __repr__(self): return f"<Order(id={self.id}, client_oid='{self.client_oid}', status='{self.status.value}')>"

class TradeLog(IdMixin, Base):
    __tablename__ = 'trade_logs'
    order_id: Mapped[int] = mapped_column(sa.Integer, sa.ForeignKey('orders.id'), nullable=False, index=True)
    asset_pair: Mapped[str] = mapped_column(sa.String(20), nullable=False, index=True)
    exchange_trade_id: Mapped[str] = mapped_column(sa.String(64), unique=True, nullable=False)
    side: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    price: Mapped[Decimal] = mapped_column(sa.Numeric(20, 10))
    quantity: Mapped[Decimal] = mapped_column(sa.Numeric(20, 10))
    fee_amount: Mapped[Decimal] = mapped_column(sa.Numeric(20, 10))
    fee_currency: Mapped[str] = mapped_column(sa.String(10))
    is_taker: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    executed_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    order: Mapped["Order"] = relationship("Order", back_populates="trades")
    def __repr__(self): return f"<TradeLog(id={self.id}, ex_trade_id='{self.exchange_trade_id}')>"

class BalanceSnapshot(IdMixin, Base):
    __tablename__ = 'balance_snapshots'
    snapshot_ts: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False, index=True)
    currency: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    total_balance: Mapped[Decimal] = mapped_column(sa.Numeric(30, 18))
    available_balance: Mapped[Decimal] = mapped_column(sa.Numeric(30, 18))
    __table_args__ = (sa.UniqueConstraint('snapshot_ts', 'currency', name='_snapshot_currency_uc'),)
    def __repr__(self): return f"<BalanceSnapshot(id={self.id}, currency='{self.currency}', balance={self.total_balance})>"

class DailyPNL(IdMixin, Base):
    __tablename__ = 'daily_pnl'
    trade_date: Mapped[DateObject] = mapped_column(sa.Date, nullable=False, index=True)
    strategy_id: Mapped[Optional[int]] = mapped_column(sa.Integer, sa.ForeignKey('strategies.id'), index=True)
    asset_pair: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(sa.Numeric(20, 8), default=Decimal('0.0'))
    unrealized_pnl: Mapped[Decimal] = mapped_column(sa.Numeric(20, 8), default=Decimal('0.0'))
    net_pnl: Mapped[Decimal] = mapped_column(sa.Numeric(20, 8), default=Decimal('0.0'))
    pnl_currency: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    trades_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    total_volume_quote: Mapped[Decimal] = mapped_column(sa.Numeric(30, 8), default=Decimal('0.0'))
    __table_args__ = (sa.UniqueConstraint('trade_date', 'strategy_id', 'asset_pair', name='_date_strategy_pair_uc'),)
    def __repr__(self): return f"<DailyPNL(id={self.id}, date='{self.trade_date}', net_pnl={self.net_pnl})>"

class MarketKline1m(Base):
    __tablename__ = 'market_kline_1m'
    ts: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), primary_key=True)
    asset_pair: Mapped[str] = mapped_column(sa.String(20), primary_key=True)
    open: Mapped[Decimal] = mapped_column(sa.Numeric(20, 10)); high: Mapped[Decimal] = mapped_column(sa.Numeric(20, 10))
    low: Mapped[Decimal] = mapped_column(sa.Numeric(20, 10)); close: Mapped[Decimal] = mapped_column(sa.Numeric(20, 10))
    volume_asset: Mapped[Decimal] = mapped_column(sa.Numeric(30, 10)); volume_quote: Mapped[Decimal] = mapped_column(sa.Numeric(30, 10))
    def __repr__(self): return f"<MarketKline1m(pair='{self.asset_pair}', ts='{self.ts}', close={self.close})>"