from __future__ import annotations

import re

from typing import Any, Dict

from .text_utils import extract_channel_tokens, normalize_compare_text


class DirectParser:
    def __init__(self, settings, store):
        self.settings = settings
        self.store = store

    def _wake(self, text: str) -> bool:
        return bool(re.search(r"\b(ника|nika|ник)\b", normalize_compare_text(text)))

    def _first_channel(self, raw: str) -> str:
        tokens = extract_channel_tokens(raw)
        return tokens[0].strip() if tokens else ""

    def _last_meta(self, message) -> Dict[str, Any]:
        if not getattr(self, "store", None) or not getattr(message, "guild", None):
            return {}
        try:
            meta = self.store.get_channel_meta(str(message.channel.id))
            return meta or {}
        except Exception:
            return {}

    def _meta_get(self, meta: Any, key: str, default: str = "") -> str:
        if meta is None: return default
        try:
            if isinstance(meta, dict): value = meta.get(key, default)
            elif hasattr(meta, "keys") and key in meta.keys(): value = meta[key]
            else: value = getattr(meta, key, default)
        except Exception: return default
        if value is None: return default
        return str(value)

    def _is_followup_read(self, low: str) -> bool:
        return any(p in low for p in [
            "continue", "more", "what else", "before that", "earlier", "а еще", "дальше", "что еще", "до этого", "раньше",
            "продолжай", "еще", "подробнее", "сводку", "что обсуждали", "что там было", "что происходило"
        ])

    def _is_followup_react(self, low: str) -> bool:
        return any(p in low for p in [
            "не на это", "не это", "не на то", "не на тот", "не на ту", "а не на это", "а на это",
            "а на то", "на предыдущее", "на прошлое", "на это сообщение", "на то сообщение"
        ])

    def _reaction_from_text(self, low: str) -> str:
        if any(w in low for w in ["клоун", "clown"]): return "🤡"
        if any(w in low for w in ["череп", "skull"]): return "💀"
        if any(w in low for w in ["огонь", "fire"]): return "🔥"
        if any(w in low for w in ["сердечк", "сердц", "heart", "love"]): return "❤️"
        if any(w in low for w in ["лайк", "thumb", "like", "палец", "thumbsup"]): return "👍"
        return ""

    def _read_limit_from_text(self, low: str) -> int:
        deep = ["подроб", "всё", "все", "полностью", "полный", "весь", "вся", "развернуто", "детально", "что обсуждали", "сводка", "что там было", "а еще", "что еще", "подробнее", "коротко", "что происходило"]
        summary = ["дай сводку", "короткую сводку", "сводку", "что обсуждали", "обсуждали", "что там было", "что там писали", "что писали", "о чем говорили", "расскажи", "поясни", "что происходило"]
        if any(p in low for p in deep): return 80
        if any(p in low for p in summary): return 50
        return 30

    def parse(self, message, text: str) -> Dict[str, Any] | None:
        raw = (text or "").strip()
        if not raw: return None
        low = raw.lower(); meta = self._last_meta(message)
        last_action = self._meta_get(meta, "last_action_type").strip().lower()
        last_channel = self._meta_get(meta, "last_target_channel_id").strip()
        last_limit = int(self._meta_get(meta, "last_read_limit") or 0)
        last_anchor = self._meta_get(meta, "last_read_anchor_message_id").strip()
        last_reaction = self._meta_get(meta, "last_reaction").strip()

        if low.startswith("запомни ") or low.startswith("remember "):
            payload = raw.split(" ", 1)[1].strip() if " " in raw else ""
            if payload: return {"action": "remember", "key": "note", "value": payload}

        reaction = self._reaction_from_text(low)
        if self._is_followup_react(low) and not reaction: reaction = last_reaction or self._reaction_from_text(low)
        if any(w in low for w in ["реакц", "поставь", "реагируй", "react"]):
            ch = self._first_channel(raw) or last_channel
            if reaction and ch: return {"action": "react", "channel": ch, "reaction": reaction}

        read_req = any(w in low for w in ["прочитай", "посмотри", "что в", "что на", "что обсуждали", "сводка", "что там было", "а еще", "что еще", "подробнее", "коротко", "что происходило", "continue", "more", "what else", "before that", "earlier", "дальше", "до этого", "раньше"])
        if read_req:
            ch = self._first_channel(raw) or last_channel
            if ch:
                limit = self._read_limit_from_text(low); before = ""
                if self._is_followup_read(low) and last_action == "read_channel" and last_channel == ch:
                    if last_anchor: before = last_anchor
                    limit = max(limit, last_limit or 30)
                return {"action": "read_channel", "channel": ch, "limit": limit, "before": before}
        if self._wake(raw): return None
        return None
