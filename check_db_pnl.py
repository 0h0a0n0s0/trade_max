# check_db_pnl.py
"""
一個獨立的診斷腳本，用於直接查詢並打印 daily_pnl 表中的所有數據，
以幫助我們判斷績效報告問題的根源。
"""
import os
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def check_pnl_data():
    # --- 連線設定 (與主程式一致) ---
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    DB_URI = os.getenv("DB_URI")
    if not DB_URI:
        default_db_path = Path(__file__).parent / "trading_strategy.db"
        DB_URI = f"sqlite:///{default_db_path}"
    
    engine = create_engine(DB_URI)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    print("\n--- 正在查詢 daily_pnl 表中的所有數據... ---\n")
    
    try:
        # 使用原生 SQL 查詢，避免任何 ORM 的潛在問題
        query = text("SELECT id, trade_date, strategy_id, asset_pair, realized_pnl FROM daily_pnl ORDER BY trade_date DESC")
        results = session.execute(query).fetchall()
        
        if not results:
            print("資料庫中的 daily_pnl 表是空的，或者查詢失敗。")
        else:
            print(f"{'ID':<5} {'Trade Date':<15} {'Strategy ID':<12} {'Asset Pair':<12} {'Realized PNL':<20}")
            print("-" * 70)
            for row in results:
                # 確保 PNL 以 Decimal 格式打印
                pnl = Decimal(str(row[4])).quantize(Decimal('0.0001'))
                print(f"{row[0]:<5} {str(row[1]):<15} {row[2]:<12} {row[3]:<12} {pnl:<20.4f}")
        
        print("\n--- 查詢完畢 ---")

    except Exception as e:
        print(f"\n查詢時發生錯誤: {e}")
        print("這通常意味著 'daily_pnl' 表不存在。請確認您已成功運行過主程式以建立資料表。")
    finally:
        session.close()

if __name__ == "__main__":
    check_pnl_data()