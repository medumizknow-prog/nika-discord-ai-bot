import aiohttp, json, re
from typing import Any, Dict, List, Optional

class LMStudioClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
    async def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.4, max_tokens: int = 120, frequency_penalty: float = 0.2, presence_penalty: float = 0.1) -> Optional[str]:
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "frequency_penalty": frequency_penalty, "presence_penalty": presence_penalty, "stream": False}
        timeout = aiohttp.ClientTimeout(total=90)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/chat/completions", json=payload) as resp:
                raw = await resp.text()
                if resp.status != 200: raise RuntimeError(raw)
                data = await resp.json()
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                return content or None
    async def chat_json(self, messages: List[Dict[str, Any]], temperature: float = 0.1, max_tokens: int = 240) -> Dict[str, Any]:
        raw = await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        if not raw: return {"action": "ignore"}
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m: return {"action": "ignore"}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {"action": "ignore"}
        except Exception:
            return {"action": "ignore"}
