# max_async_api.py
"""
Unified MAX exchange client - V5 Final Stable Version
"""
import base64
import aiohttp
import asyncio
import hmac
import hashlib
import json
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import logging

logger = logging.getLogger("max_async_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger("三層固定間隙網格")  # Use the module name as the logger name

class MaxAPI:
    def __init__(self):
        self._api_key: Optional[str] = os.getenv("MAX_API_KEY")
        self._api_secret_str: Optional[str] = os.getenv("MAX_API_SECRET")
        if not self._api_key or not self._api_secret_str:
            logger.critical("CRITICAL: MAX API Key or Secret is NOT loaded from .env.")
        else:
            logger.info("MAX API Key and Secret have been loaded from .env.")
       
        self._base_url_v2 = "https://max-api.maicoin.com"
        self._ws_url = "wss://max-stream.maicoin.com/ws"
        self._session: Optional[aiohttp.ClientSession] = None
        self._time_offset_ns: int = 0
        self._is_initialized = False

    async def _request_v3_ws_auth(self) -> Any:
        """
        專門用於呼叫 V3 WebSocket 認證端點的函數。
        V3 的簽名機制與 V2 不同。
        """
        if not self._session: raise RuntimeError("API client is not initialized.")
        if not self._api_key or not self._api_secret_str: raise RuntimeError("API key/secret is not configured.")

        # V3 認證：基於 timestamp 簽名
        timestamp = str(int(time.time()))
        # ws/auth 是 POST 請求但 body 為空，所以只簽署 timestamp
        signature = hmac.new(self._api_secret_str.encode('utf-8'), timestamp.encode('utf-8'), hashlib.sha256).hexdigest()

        headers = {
            'X-MAX-APIKEY': self._api_key,
            'X-MAX-TIMESTAMP': timestamp,
            'X-MAX-SIGNATURE': signature
        }
        
        url = f"{self._base_url_v2}/api/v3/ws/auth" # 使用 V3 路徑
        
        try:
            async with self._session.post(url, headers=headers) as r:
                text_response = await r.text()
                r.raise_for_status()
                return json.loads(text_response)
        except Exception as e:
            logger.error(f"V3 ws/auth request failed: {e}", exc_info=True)
            raise
 
    async def initialize(self):
        if self._is_initialized: return
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20, connect=10))
        await self._sync_time()
        self._is_initialized = True
        logger.info("MaxAPI client has been successfully initialized.")

    async def _sync_time(self):
        """
        【修正版】同步伺服器時間，但主要作為健康檢查。
        如果本地時間與伺服器時間差異過大，則發出嚴重警告。
        """
        if not self._session: return
        try:
            async with self._session.get(f"{self._base_url_v2}/api/v2/timestamp") as r:
                r.raise_for_status()
                server_time_s = await r.json()
                local_time_s = time.time()
                offset = local_time_s - server_time_s
                
                # 健康檢查：如果時間差異超過5秒，就發出嚴重警告
                if abs(offset) > 5:
                    logger.critical(
                        f"CRITICAL: System clock is out of sync with server by {offset:.2f} seconds. "
                        f"Please enable and check NTP service on your machine. Halting is recommended."
                    )
                else:
                    logger.info(f"Server time synchronized. Local clock offset: {offset:.4f} seconds (Healthy).")
        except Exception as e:
            logger.error(f"Failed to sync server time: {e}. Nonce generation will rely solely on local clock.", exc_info=True)


    def _get_synced_nonce(self) -> str:
        """
        【修正版】直接使用本地時間生成 nonce (Unix timestamp in milliseconds)。
        這是更穩定且常見的做法，前提是作業系統時間已同步(NTP)。
        """
        nonce = str(int(time.time() * 1000))
        logger.debug(f"Generated nonce: {nonce}")        
        return nonce

    async def _request_v2(self, method: str, api_path: str, params: Optional[Dict] = None, body: Optional[Dict] = None) -> Any:
        if not self._is_initialized or not self._session: raise RuntimeError("API client is not initialized.")
        if not self._api_key or not self._api_secret_str: raise RuntimeError("API key/secret is not configured.")

        nonce = self._get_synced_nonce()
        signature_path = f"/api/v2{api_path}"

        # --- 【↓↓↓ 統一處理所有參數的最終邏輯 ↓↓↓】 ---

        # 1. 將 GET 的 params 和 POST 的 body 都收集到一個字典中
        all_params = {}
        if params:
            all_params.update(params)
        if body:
            all_params.update(body)

        # 2. 準備用於 URL 查詢字串的最終參數，並【必須】包含 nonce
        query_params = all_params.copy()
        query_params['nonce'] = nonce

        # 3. 準備用於簽名的 payload，它也【必須】包含所有 URL 參數
        sig_payload_dict = {
            "path": signature_path,
        }
        sig_payload_dict.update(query_params)

        # --- 【↑↑↑ 邏輯結束 ↑↑↑】 ---

        # 將 payload 字典轉換為 JSON 字串，然後進行 Base64 編碼
        payload_json_str = json.dumps(sig_payload_dict, separators=(",", ":"))
        payload_b64 = base64.b64encode(payload_json_str.encode('utf-8')).decode('utf-8')
        
        # 使用 Base64 編碼後的字串進行簽名
        signature = hmac.new(self._api_secret_str.encode('utf-8'), payload_b64.encode('utf-8'), hashlib.sha256).hexdigest()
        
        # 在 Header 中使用 Base64 編碼後的 payload
        headers = { 
            "X-MAX-ACCESSKEY": self._api_key, 
            "X-MAX-PAYLOAD": payload_b64, 
            "X-MAX-SIGNATURE": signature 
        }
       
        url = f"{self._base_url_v2}{signature_path}"
       
        try:
            # 無論 GET 或 POST，所有參數都在 params 中發送，body 永遠為 None
            async with self._session.request(method.upper(), url, params=query_params, json=None, headers=headers) as r:
                text_response = await r.text()
                if not r.ok:
                    logger.error(f"API request failed: {r.status} {r.reason} for {method} {url} with params {query_params} - Response: {text_response[:300]}")
                r.raise_for_status()
                return json.loads(text_response) if text_response else None
        except Exception as e:
            logger.error(f"Generic error on API request for {method} {url}: {e}", exc_info=True)
            raise

    async def get_v2_ticker(self, market: str) -> Optional[Dict]:
        if not self._is_initialized or not self._session: raise RuntimeError("API client not initialized.")
        try:
            async with self._session.get(f"{self._base_url_v2}/api/v2/tickers/{market.lower()}") as r:
                r.raise_for_status(); return await r.json()
        except Exception as e:
            logger.error(f"Failed to get ticker for {market}: {e}"); return None
   
    async def get_v2_k_data(self, market: str, limit: int, period: int) -> Optional[List]:
        if not self._is_initialized or not self._session: raise RuntimeError("API client not initialized.")
        params = {"market": market.lower(), "limit": limit, "period": period}
        try:
            async with self._session.get(f"{self._base_url_v2}/api/v2/k", params=params) as r:
                r.raise_for_status(); return await r.json()
        except Exception as e:
            logger.error(f"Failed to get k-data for {market}: {e}"); return None

    async def get_ws_auth_token(self) -> Optional[str]:
        try:
            # --- 【↓↓↓ 將此處的呼叫從 _request_v2 改為 _request_v3_ws_auth ↓↓↓】 ---
            response = await self._request_v3_ws_auth()
            # --- 【↑↑↑ 修改結束 ↑↑↑】 ---
            if response and response.get("token"):
                logger.info("Successfully fetched WebSocket auth token (via V3 API)."); return response["token"]
            logger.error(f"WS auth token not found in V3 response: {response}"); return None
        except Exception as e:
            logger.error(f"Failed to get WS auth token: {e}"); return None

    async def get_v2_balance(self, currency: str, retries: int = 3) -> dict:
        for attempt in range(1, retries + 1):
            try:
                response = await self._request_v2("GET", f"/members/accounts/{currency.lower()}")
                return response
            except Exception as e:
                log.error(f"Attempt {attempt} failed to get balance: {e}")
                if attempt == retries:
                    raise
                await asyncio.sleep(1)
        return None

    async def place_v2_order(self, **kwargs):
        body = {k: str(v) if isinstance(v, Decimal) else v for k, v in kwargs.items()}
        return await self._request_v2("POST", "/orders", body=body)
   
    async def cancel_all_v2_market_orders(self, market: str):
        return await self._request_v2("POST", "/orders/clear", body={"market": market.lower()})

    async def get_v2_orders_by_market(self, market: str, state: str, limit: int = 100):
        params = {"market": market.lower(), "state": state, "limit": limit}
        return await self._request_v2("GET", "/orders", params=params)

    async def ws_subscribe(self, channels: List[Dict], message_queue: asyncio.Queue):
        if not self._is_initialized or not self._session: raise RuntimeError("API client not initialized.")
        retry_delay = 1
        while True:
            try:
                token = await self.get_ws_auth_token()
                if not token: raise RuntimeError("WS token fetch failed")
                
                async with self._session.ws_connect(self._ws_url, heartbeat=25) as ws:
                    logger.info("WebSocket connected. Authenticating...")
                    await ws.send_json({"action": "auth", "token": token, "id": "auth"})
                    auth_resp = await ws.receive_json(timeout=10)
                    if auth_resp.get("event") != "authenticated":
                        raise RuntimeError(f"WS auth failed: {auth_resp}")
                   
                    logger.info("WebSocket authenticated. Subscribing to channels...")
                    sub_payload = [{"channel": ch} for ch in channels]
                    await ws.send_json({"action": "subscribe", "subscriptions": sub_payload, "id": "sub"})
                   
                    retry_delay = 1
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await message_queue.put(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.warning(f"WebSocket connection closed or errored: {ws.exception()}")
                            break
            except Exception as e:
                logger.warning(f"WebSocket error: {e}. Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
    async def get_v2_order(self, order_id: int) -> Optional[Dict]:
        """
        【最終修正版】查詢單一訂單的詳細資訊。
        修正了端點路徑 (orders -> order)，並將 id 作為 params 傳遞。
        """
        return await self._request_v2("GET", "/order", params={"id": order_id})

        
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("MaxAPI client session closed.")
            self._is_initialized = False

max_api = MaxAPI()