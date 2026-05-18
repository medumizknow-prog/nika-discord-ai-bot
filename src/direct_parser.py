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
        if meta is None:
            return default
        try:
            if isinstance(meta, dict):
                value = meta.get(key, default)
            elif hasattr(meta, "keys") and key in meta.keys():
                value = meta[key]
            else:
                value = getattr(meta, key, default)
        except Exception:
            return default
        if value is None:
            return default
        return str(value)

    def _is_followup_read(self, low: str) -> bool:
        return any(
            p in low
            for p in [
                "а еще",
                "а ещё",
                "еще",
                "ещё",
                "дальше",
                "продолжай",
                "что еще",
                "что ещё",
                "и еще",
                "и ещё",
                "еще раз",
                "ещё раз",
                "а что дальше",
                "что дальше",
                "подробнее",
                "короткую сводку",
                "сводку",
                "что обсуждали",
                "обсуждали",
                "что там писали",
                "что писали",
                "о чем говорили",
                "о чём говорили",
                "что происходило",
                "что было",
                "расскажи что там было",
                "расскажи что там",
            ]
        )

    def _is_followup_react(self, low: str) -> bool:
        return any(
            p in low
            for p in [
                "не на это",
                "не это",
                "не на то",
                "не на тот",
                "не на ту",
                "а не на это",
                "а на это",
                "а на то",
                "на предыдущее",
                "на прошлое",
                "на это сообщение",
                "на то сообщение",
            ]
        )

    def _reaction_from_text(self, low: str) -> str:
        if any(w in low for w in ["клоун", "clown"]):
            return "🤡"
        if any(w in low for w in ["череп", "skull"]):
            return "💀"
        if any(w in low for w in ["огонь", "fire"]):
            return "🔥"
        if any(w in low for w in ["сердечк", "сердц", "heart", "love"]):
            return "❤️"
        if any(w in low for w in ["лайк", "thumb", "like", "палец", "thumbsup"]):
            return "👍"
        return ""

    def _read_limit_from_text(self, low: str) -> int:
        deep_triggers = [
            "подроб",
            "всё",
            "все",
            "полностью",
            "полный",
            "весь",
            "вся",
            "все сообщения",
            "всё сообщение",
            "вся переписка",
            "всю переписку",
            "полную сводку",
            "развернуто",
            "развёрнуто",
            "детально",
            "досконально",
            "что обсуждали",
            "что происходило",
            "что там писали",
            "что писали",
        ]
        summary_triggers = [
            "дай сводку",
            "короткую сводку",
            "сводку",
            "что обсуждали",
            "обсуждали",
            "что там было",
            "что там писали",
            "что писали",
            "о чем говорили",
            "о чём говорили",
            "расскажи",
            "поясни",
        ]
        latest_triggers = [
            "последнее",
            "последнее сообщение",
            "что последнее",
            "last",
            "latest",
        ]
        if any(p in low for p in deep_triggers):
            return 80
        if any(p in low for p in summary_triggers):
            return 50
        if any(p in low for p in latest_triggers):
            return 1
        return 30

    def parse(self, message, text: str) -> Dict[str, Any] | None:
        raw = (text or "").strip()
        if not raw:
            return None

        low = raw.lower()
        meta = self._last_meta(message)

        last_action = (
            self._meta_get(meta, "last_action_type")
            or self._meta_get(meta, "last_action")
        ).strip().lower()
        last_channel = (
            self._meta_get(meta, "last_target_channel_id")
            or self._meta_get(meta, "last_channel_id")
        ).strip()
        last_read_limit = int(self._meta_get(meta, "last_read_limit") or 0)
        last_read_anchor = (
            self._meta_get(meta, "last_read_first_message_id")
            or self._meta_get(meta, "last_read_last_message_id")
        ).strip()
        last_reaction = self._meta_get(meta, "last_reaction").strip()

        if low.startswith("запомни ") or low.startswith("remember "):
            payload = raw.split(" ", 1)[1].strip() if " " in raw else ""
            if payload:
                return {"action": "remember", "key": "note", "value": payload}

        m = re.search(
            r"(?:напиши|отправь|скажи|сообщи)\s+(?:в|на)\s+(<#\d+>|#?[a-zA-Z0-9_\-]+)\s+(?:что\s+)?(.+)",
            raw,
            flags=re.I | re.S,
        )
        if m:
            return {"action": "send_message", "channel": m.group(1).strip(), "text": m.group(2).strip()}

        m = re.search(
            r"(?:напиши|отправь|скажи|сообщи)\s+(?:в|на)\s+(<#\d+>|#?[a-zA-Z0-9_\-]+)\s+(?:пользователю\s+)?(<@!?\d+>|@[\w.\-]+)\s+(.+)",
            raw,
            flags=re.I | re.S,
        )
        if m:
            return {"action": "ping_user", "channel": m.group(1).strip(), "user": m.group(2).strip(), "text": m.group(3).strip()}

        m = re.search(
            r"(?:позови|пингани|пинг)\s+(<@!?\d+>|@[\w.\-]+)(?:\s+в\s+(<#\d+>|#?[a-zA-Z0-9_\-]+))?",
            raw,
            flags=re.I,
        )
        if m:
            return {"action": "ping_user", "user": m.group(1).strip(), "channel": (m.group(2) or "").strip(), "text": "тебя зовут сюда"}

        # Reaction commands and follow-up reactions.
        reaction = self._reaction_from_text(low)
        if self._is_followup_react(low) and not reaction:
            reaction = last_reaction or self._reaction_from_text(low)
        if any(w in low for w in ["реакц", "поставь", "реагируй", "react", "сердечк", "сердц", "heart", "лайк", "like", "thumb", "палец", "клоун", "огонь", "череп"]):
            ch = self._first_channel(raw) or last_channel
            if not ch and last_channel:
                ch = last_channel
            if reaction and ch:
                return {"action": "react", "channel": ch, "reaction": reaction}
            if self._is_followup_react(low) and last_channel and last_reaction:
                return {"action": "react", "channel": last_channel, "reaction": last_reaction}

        # Read-channel commands.
        read_requested = any(
            w in low
            for w in [
                "прочитай",
                "посмотри",
                "что в",
                "что на",
                "что последнее",
                "что было",
                "последнее сообщение",
                "последнее было",
                "что обсуждали",
                "обсуждали",
                "что там писали",
                "что писали",
                "дай сводку",
                "короткую сводку",
                "сводку",
                "о чем говорили",
                "о чём говорили",
                "расскажи",
                "что происходило",
                "продолжай",
                "а еще",
                "а ещё",
                "еще",
                "ещё",
                "дальше",
                "подробнее",
            ]
        )
        if read_requested:
            ch = self._first_channel(raw) or last_channel
            if ch:
                limit = self._read_limit_from_text(low)
                before = ""
                # Follow-up continuation: use the previous read anchor and go older.
                if self._is_followup_read(low) and last_action == "read_channel" and last_channel:
                    if last_read_anchor and (not self._first_channel(raw) or ch == last_channel):
                        before = last_read_anchor
                    limit = max(30, last_read_limit or 30, limit)
                return {
                    "action": "read_channel",
                    "channel": ch,
                    "limit": limit,
                    "before": before,
                }

        if self._wake(raw):
            return None

        return None
