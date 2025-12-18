# db.py
"""
資料庫連線與會話管理 - V2 最終修正版

【V2 修正】
- 將 engine 和 SessionLocal 的定義移至此文件，作為資料庫操作的唯一核心。
- 提供 create_all_tables 函數作為建立資料表的唯一入口點。
"""

import os
import logging
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

# --- 設定 ---
log = logging.getLogger("db")
log.setLevel(logging.INFO)

# --- 連線設定 ---

# 尋找 .env 檔案
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

DB_URI = os.getenv("DB_URI")
if not DB_URI:
    default_db_path = Path(__file__).parent / "trading_strategy.db"
    DB_URI = f"sqlite+aiosqlite:///{default_db_path}"
    log.warning(f"DB_URI not found in environment. Using default SQLite DB at: {default_db_path}")

# 建立資料庫引擎
engine = create_engine(DB_URI, echo=False, pool_pre_ping=True)

# 建立 Session 工廠
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- 會話管理器 ---

@contextmanager
def db_session() -> Session:
    """
    提供一個事務性的資料庫會話 context manager。
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        log.error(f"Database session rollback due to exception: {e}", exc_info=False) # 設置為 False 避免過多日誌
        session.rollback()
        raise
    finally:
        session.close()

# --- 資料庫工具函數 ---

def check_db_connection() -> bool:
    """啟動時檢查資料庫連線是否正常"""
    try:
        with engine.connect() as connection:
            log.info(f"Successfully connected to the database: {engine.url.database}")
            return True
    except SQLAlchemyError as e:
        log.critical(f"Database connection failed: {e}", exc_info=True)
        return False

def create_all_tables():
    """
    根據 db_schema 中的定義，建立所有資料表。
    這是建立資料庫的唯一入口。
    """
    # 將 import 放在函數內部，避免在模組加載時就觸發
    from db_schema import Base
    log.info("Attempting to create tables on database...")
    try:
        Base.metadata.create_all(bind=engine)
        log.info("Tables created successfully (if they didn't exist already).")
    except Exception as e:
        log.error(f"Error creating tables: {e}", exc_info=True)
        raise

# --- 主執行區塊 ---

if __name__ == '__main__':
    # 如果直接執行此檔案 (python db.py)，則會建立所有資料表。
    print("Running database setup...")
    create_all_tables()
    print("Database setup finished.")