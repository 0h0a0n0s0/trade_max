# telegram_alerter.py
"""
獨立的 Telegram 警報模組
- 統籌所有警報訊息的格式化與發送。
- 內建冷卻機制，避免對相同事件重複發送警報。
- 使用 aiohttp 進行非同步發送，不阻塞主策略。
"""
import os
import time
import logging
import aiohttp
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv

# 載入 .env 檔案（從項目根目錄）
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # 如果根目錄沒有，也嘗試從當前目錄載入（向後兼容）
    local_env_path = Path(__file__).parent / ".env"
    if local_env_path.exists():
        load_dotenv(dotenv_path=local_env_path)

# --- 設定 ---
log = logging.getLogger("TelegramAlerter")

class TelegramAlerter:
    def __init__(self):
        # 從 .env 讀取設定
        self.token = os.getenv("TG_TOKEN")
        self.chat_id = os.getenv("TG_CHAT_ID")
        
        # 警報冷卻機制，用於避免短時間內對同一事件重複發送
        # 結構: {'alert_key': last_sent_timestamp}
        self._last_alert_time: Dict[str, float] = {}
        
        if not self.token or not self.chat_id:
            log.warning("TG_TOKEN or TG_CHAT_ID not found in .env. Telegram alerts will be simulated in logs.")
            self.is_configured = False
        else:
            self.is_configured = True
            log.info(f"TelegramAlerter initialized for chat_id: {self.chat_id}")

    async def _send_message(self, text: str):
        """非同步發送訊息的核心函數"""
        if not self.is_configured:
            log.info(f"[TELEGRAM_SIMULATED] {text}")
            return

        api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
        
        try:
            # 使用 aiohttp 進行非同步請求
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(api_url, data=payload) as response:
                    if response.status == 200:
                        log.info("Telegram alert sent successfully.")
                    else:
                        response_text = await response.text()
                        log.error(f"Failed to send Telegram alert: {response.status} - {response_text}")
        except Exception as e:
            log.error(f"Exception while sending Telegram alert: {e}", exc_info=True)

    def _is_cooling_down(self, alert_key: str, cooldown_seconds: int) -> bool:
        """檢查特定警報是否在冷卻中"""
        last_time = self._last_alert_time.get(alert_key, 0)
        if time.time() - last_time < cooldown_seconds:
            # 仍在冷卻中，不發送
            return True
        # 已超過冷卻時間，可以發送
        self._last_alert_time[alert_key] = time.time()
        return False

    # --- 以下是各類警報的公開方法 ---

    async def send_system_event(self, message: str):
        """發送系統級事件（啟動/關閉）"""
        text = f"🤖 **系統事件** 🤖\n\n{message}"
        await self._send_message(text)

    async def send_critical_alert(self, message: str, alert_key: str):
        """發送嚴重警報（黑天鵝/嚴重錯誤），每小時只發一次"""
        if self._is_cooling_down(alert_key, 3600): # 1小時冷卻
            return
        text = f"🚨 **嚴重警報** 🚨\n\n{message}"
        await self._send_message(text)

    async def send_risk_alert(self, message: str, alert_key: str):
        """發送風險相關警報（資金水位），每小時只發一次"""
        if self._is_cooling_down(alert_key, 3600): # 1小時冷卻
            return
        text = f"⚠️ **風險警報** ⚠️\n\n{message}"
        await self._send_message(text)

    async def send_strategy_event(self, message: str, alert_key: str):
        """發送策略狀態變更事件（趨勢反轉/重置），每 15 分鐘只發一次"""
        if self._is_cooling_down(alert_key, 900): # 15分鐘冷卻
            return
        text = f"🧭 **策略事件** 🧭\n\n{message}"
        await self._send_message(text)

# 建立一個全局唯一的 alerter 實例，供其他檔案導入使用
alerter = TelegramAlerter()