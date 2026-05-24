from __future__ import annotations

import asyncio
import json
import re
import time
import datetime
from typing import Any, Dict, List

import discord

from .prompts import (
    ACTION_SYSTEM_PROMPT,
    AUTONOMY_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    COMPOSE_READ_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    USER_CARD_SYSTEM_PROMPT,
    build_system_prompt,
)
from .text_utils import (
    attachment_summary,
    clean_response,
    is_short_text,
    is_too_similar,
    normalize_compare_text,
    sanitize_summary_text,
    strip_output_labels,
)

NEGATIVE_WORDS = ["туп", "идиот", "долба", "блять", "еб", "нах", "сука", "пизд", "хуй", "дурак"]
POSITIVE_WORDS = ["спасибо", "класс", "круто", "норм", "люблю", "уважаю", "молодец", "приятно", "умница"]
PLAYFUL_WORDS = ["ахах", "хаха", "лол", "ржу", "угар", "мем"]


class AgentPlanner:
    def __init__(self, settings, store, llm_client, direct_parser):
        self.settings = settings
        self.store = store
        self.llm = llm_client
        self.direct_parser = direct_parser
        self._runtime: Dict[str, Dict[str, object]] = {}

    def _state(self, channel_id: str):
        return self._runtime.setdefault(channel_id, {"recent": [], "anchors": []})

    def _meta(self, channel_id: str) -> Dict[str, Any]:
        meta = self.store.get_channel_meta(channel_id) or {}
        return meta

    def _parse_db_timestamp(self, ts_str: str | None) -> float:
        if not ts_str: return 0.0
        try:
            ts_str = ts_str.replace("Z", "+00:00")
            if " " in ts_str and "T" not in ts_str:
                return datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
            return datetime.datetime.fromisoformat(ts_str).timestamp()
        except Exception:
            return 0.0

    def _is_bad_summary(self, text: str) -> bool:
        low = (text or "").lower()
        bad_fragments = [
            "не могу посмотреть", "не может посмотреть", "не могу прочитать", "не может прочитать",
            "не могу видеть", "не может видеть", "не вижу канал", "не видит канал",
            "cannot read", "cannot view", "can't read", "can't view", "не знаю что там было"
        ]
        return any(frag in low for frag in bad_fragments)

    def _fallback_summary_from_recent(self, recent) -> str:
        lines = []
        for row in recent[-5:]:
            role = (row.get("role") or "").strip().lower()
            content = (row.get("content") or "").strip()
            if not content: continue
            if role == "assistant": content = strip_output_labels(content)
            if content: lines.append(content)
        if not lines: return ""
        if len(lines) == 1: return lines[0][:180]
        return " / ".join(lines[-3:])[:240]

    def _trim_recent_for_current_user(self, recent, user_text: str):
        if not recent: return recent
        norm = normalize_compare_text(user_text)
        if not norm: return recent
        last = recent[-1]
        if (last.get("role") or "").lower() != "user": return recent
        content = (last.get("content") or "").strip()
        variants = [content]
        if ":" in content: variants.append(content.split(":", 1)[1].strip())
        for variant in variants:
            if normalize_compare_text(variant) == norm: return recent[:-1]
        return recent

    def _prompt_user_text(self, message: discord.Message, user_text: str) -> str:
        user_text = (user_text or "").strip()
        if not user_text: return user_text
        return f"{message.author.display_name}: {user_text}"

    def _last_assistant_reply(self, recent) -> str:
        for row in reversed(recent or []):
            if (row.get("role") or "").lower() == "assistant":
                return (row.get("content") or "").strip()
        return ""

    def _looks_like_echo(self, candidate: str, user_text: str, recent) -> bool:
        if not candidate: return True
        low = normalize_compare_text(candidate)
        bad_set = {"", "мм", "ммм", "м", "а", "э", "о", "ок", "ok", "поняла", "понял", "ясно", "угу", "ага", "хз", "ну", "мм?"}
        if not low or low in bad_set: return True
        if len(low) < 2: return True
        if is_too_similar(candidate, user_text, threshold=0.80): return True
        for row in (recent or []):
            if is_too_similar(candidate, row.get("content") or "", threshold=0.85): return True
        last_asst = self._last_assistant_reply(recent)
        if last_asst and is_too_similar(candidate, last_asst, threshold=0.85): return True
        return False

    def _read_followup(self, low: str) -> bool:
        return any(p in low for p in [
            "а еще", "еще", "дальше", "продолжай", "что еще", "подробнее", "сводку", "что обсуждали",
            "что там было", "что происходило", "continue", "more", "what else", "before that", "earlier", "до этого", "раньше"
        ])

    def _read_limit_from_text(self, low: str) -> int:
        if any(p in low for p in ["подроб", "полностью", "весь", "вся", "развернуто", "что обсуждали", "сводка", "что происходило"]):
            return 80
        if any(p in low for p in ["дай сводку", "короткую сводку", "сводку", "расскажи", "поясни"]):
            return 50
        return 30

    def _extract_user_hints(self, user_text: str) -> List[str]:
        low = (user_text or "").lower()
        hints: List[str] = []
        if "работаю" in low or "работает" in low or "работал" in low: hints.append("связан с работой/занятостью")
        if "учусь" in low or "учится" in low or "студент" in low: hints.append("учится или связан с учебой")
        if "играю" in low or "играет" in low or "геймер" in low: hints.append("любит игры")
        if "люблю" in low or "нравится" in low:
            snippet = user_text.strip()
            if len(snippet) > 90: snippet = snippet[:90].rstrip() + "..."
            hints.append(f"любит/предпочитает: {snippet}")
        return hints

    async def refresh_user_card(self, message: discord.Message, user_text: str) -> None:
        if not message.guild or message.author.bot: return
        uid = str(message.author.id)
        cid = str(message.channel.id)
        self.store.ensure_profile(uid, username=message.author.name, display_name=message.author.display_name)
        card = self.store.get_user_card(uid) or {}
        hints = self._extract_user_hints(user_text)
        for hint in hints: self.store.append_user_card_note(uid, hint)
        seen = int(card.get("messages_seen") or 0)
        explicit = any(p in user_text.lower() for p in ["меня зовут", "я работаю", "я учусь", "я люблю", "мне нравится"])
        if explicit or seen < 3 or (seen % 10 == 0):
            recent_user = self.store.get_recent_user_history(uid, channel_id=cid, limit=10)
            current_card = self.store.build_profile_card(subject_id=uid, subject_name=message.author.display_name, max_facts=10)
            payload = f"Текущая карточка:\n{current_card}\n\nНовые сообщения:\n" + "\n".join(f"- {r['content']}" for r in recent_user)
            try:
                raw = await self.llm.chat_json([{"role": "system", "content": USER_CARD_SYSTEM_PROMPT}, {"role": "user", "content": payload}], temperature=0.2)
                if raw: self.store.update_user_card(uid, **{k: raw.get(k) for k in ["summary", "interests", "communication_style", "traits", "relationship", "relationship_trend", "opinion", "topics", "activity_level", "behaviors", "notes"] if raw.get(k)})
            except Exception: pass

    def detect_emotion(self, text: str) -> str:
        low = (text or "").lower()
        if any(w in low for w in NEGATIVE_WORDS): return "angry"
        if any(w in low for w in POSITIVE_WORDS): return "happy"
        if any(w in low for w in PLAYFUL_WORDS): return "hype"
        return "neutral"

    def update_mood(self, channel_id: str, user_text: str):
        meta = self._meta(channel_id)
        mood, energy, sass = meta.get("mood") or "calm", float(meta.get("energy") or 0.5), float(meta.get("sass") or 0.5)
        emo = self.detect_emotion(user_text)
        if emo == "angry": mood, sass, energy = "annoyed", min(1.0, sass + 0.1), min(1.0, energy + 0.05)
        elif emo == "happy": mood, energy = "friendly", min(1.0, energy + 0.05)
        self.store.update_channel_meta(channel_id, mood=mood, energy=energy, sass=sass)

    async def decide(self, message: discord.Message, user_text: str) -> Dict[str, Any]:
        cid = str(message.channel.id)
        self.update_mood(cid, user_text)
        asyncio.create_task(self.refresh_user_card(message, user_text))
        direct = self.direct_parser.parse(message, user_text)
        if direct: return direct
        meta = self._meta(cid)
        low = user_text.lower().strip()
        if self._read_followup(low) and meta.get("last_action_type") == "read_channel":
            return {"action": "read_channel", "channel": meta.get("last_target_channel_id"), "limit": self._read_limit_from_text(low), "before": meta.get("last_read_anchor_message_id") if any(k in low for k in ["раньше", "before", "earlier"]) else ""}
        for attempt in range(2):
            msgs = self.build_chat_messages(message, user_text)
            raw = await self.llm.chat(msgs, temperature=0.4 + (attempt * 0.1), max_tokens=250)
            cleaned = clean_response(raw)
            if cleaned and not self.is_duplicate_response(cid, cleaned) and not self._looks_like_echo(cleaned, user_text, self.store.get_recent_history(cid, 5)):
                return {"action": "reply", "text": cleaned}
        return {"action": "ignore"}

    async def run_autonomy(self, message: discord.Message) -> Dict[str, Any]:
        if not self.settings.autonomy_enabled or message.author.bot: return {"action": "ignore"}
        cid = str(message.channel.id); meta = self._meta(cid); now = time.time()
        if now - self._parse_db_timestamp(meta.get("last_autonomy_at")) < 120: return {"action": "ignore"}
        recent = self.store.get_recent_history_rows(cid, 20)
        active = [r for r in recent if (now - self._parse_db_timestamp(r.get("created_at"))) < 600]
        if len(active) < 5 or len({r.get("speaker_id") for r in active if (r.get("role") or "").lower() == "user"}) < 2: return {"action": "ignore"}
        last_int, last_emo = self._parse_db_timestamp(meta.get("last_interjection_at")), self._parse_db_timestamp(meta.get("last_emoji_at"))
        try:
            raw = await self.llm.chat_json([{"role": "system", "content": AUTONOMY_SYSTEM_PROMPT}, {"role": "user", "content": f"Чат:\n" + "\n".join(f"{r['speaker_name']}: {r['content']}" for r in active[-10:])}], temperature=0.4)
            action = (raw.get("action") or "ignore").lower()
            if action in ["short_interject", "contextual_reply", "reply"] and now - last_int >= 1200:
                text = clean_response(raw.get("text") or "")
                if text and not self._looks_like_echo(text, "", self.store.get_recent_history(cid, 5)): return {"action": "reply", "text": text}
            if action == "react" and now - last_emo >= 300:
                if raw.get("reaction"): return {"action": "react", "reaction": raw.get("reaction")}
        except Exception: pass
        return {"action": "ignore"}

    def is_duplicate_response(self, channel_id: str, text: str) -> bool:
        st = self._state(channel_id); norm = normalize_compare_text(text)
        if norm and norm in st["recent"]: return True
        if norm: st["recent"].append(norm);
        if len(st["recent"]) > 5: st["recent"].pop(0)
        return False

    async def summarize_channel(self, channel_id: str, force: bool = False):
        meta = self.store.get_channel_meta(channel_id) or {}; now = time.time()
        if not force and (now - self._parse_db_timestamp(meta.get("summary_timestamp")) < 1200) and (int(meta.get("message_count") or 0) - int(meta.get("last_summary_count") or 0) < 10): return
        recent = self.store.get_recent_history(channel_id, 40)
        raw = await self.llm.chat([{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(f"{m['role']}: {m['content']}" for m in recent)}], temperature=0.2)
        summary = sanitize_summary_text(raw)
        if summary: self.store.update_channel_meta(channel_id, summary=summary, summary_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), last_summary_count=int(meta.get("message_count") or 0))

    def should_summarize(self, channel_id: str) -> bool:
        meta = self._meta(channel_id)
        return (int(meta.get("message_count") or 0) - int(meta.get("last_summary_count") or 0)) >= self.settings.summary_trigger_messages

    def build_snapshot(self, message: discord.Message, user_text: str) -> str:
        cid = str(message.channel.id); meta = self._meta(cid)
        lines = [f"Mood: {meta.get('mood')}, Energy: {meta.get('energy')}, Sass: {meta.get('sass')}", f"Author Card:\n{self.store.build_profile_card(subject_id=str(message.author.id), subject_name=message.author.display_name)}"]
        if meta.get("last_read_summary"): lines += ["Last Read:\n", meta["last_read_summary"]]
        if self.store.get_summary(cid): lines += ["Channel Summary:\n", self.store.get_summary(cid)]
        return "\n".join(lines)[: self.settings.max_context_chars]

    def build_chat_messages(self, message: discord.Message, user_text: str):
        cid = str(message.channel.id); meta = self._meta(cid)
        msgs = [{"role": "system", "content": build_system_prompt(meta.get("mood") or "calm", float(meta.get("energy") or 0.5), float(meta.get("sass") or 0.5))}, {"role": "system", "content": CHAT_SYSTEM_PROMPT}, {"role": "system", "content": self.build_snapshot(message, user_text)}]
        msgs.extend(self._trim_recent_for_current_user(self.store.get_recent_history(cid, self.settings.max_recent_turns), user_text))
        msgs.append({"role": "user", "content": self._prompt_user_text(message, user_text)})
        return msgs

    async def compose_read_response(self, message: discord.Message, user_text: str, action: Dict[str, Any], observation: Dict[str, Any]) -> str:
        raw = await self.llm.chat([{"role": "system", "content": COMPOSE_READ_PROMPT}, {"role": "user", "content": f"Запрос: {user_text}\n\nДанные:\n{observation.get('text')}"}], temperature=0.3)
        return clean_response(raw) or "там пусто"
