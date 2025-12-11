# test_telegram.py
import os
import asyncio
import aiohttp
from pathlib import Path
from dotenv import load_dotenv

async def send_test_message():
    """一個獨立的、專門用來測試 Telegram 發送功能的腳本"""
    
    # 載入 .env 檔案
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print("❌ 錯誤：在目前資料夾中找不到 .env 檔案！")
        return
    load_dotenv(dotenv_path=env_path)
    print(f"✅ 成功從 {env_path} 載入 .env 檔案。")
    
    # 讀取 Token 和 Chat ID
    token = os.getenv("TG_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")
    
    if not token or not chat_id:
        print("❌ 錯誤：在 .env 檔案中找不到 TG_TOKEN 或 TG_CHAT_ID。")
        return
        
    print(f"✅ 準備向 Chat ID: {chat_id} 發送測試訊息...")

    # 發送請求
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": "👋 這是一條來自 Python 測試腳本的訊息！如果看到，代表設定正確。"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, data=payload) as response:
                if response.status == 200:
                    print("🎉 成功！測試訊息已發送，請檢查您的 Telegram 群組。")
                else:
                    response_text = await response.text()
                    print(f"❌ 失敗！Telegram API 回傳錯誤 {response.status}:")
                    print(response_text)
    except Exception as e:
        print(f"❌ 發生網路或請求錯誤: {e}")

if __name__ == "__main__":
    asyncio.run(send_test_message())