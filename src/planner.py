from __future__ import annotations

import asyncio
import json
import re
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
    is_degenerate_response,
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

    def _meta_get(self, meta: Dict[str, Any], key: str, default: Any = "") -> Any:
        if not meta:
            return default
        return meta.get(key, default)

    def _is_bad_summary(self, text: str) -> bool:
        low = (text or "").lower()
        bad_fragments = [
            "не могу посмотреть",
            "не может посмотреть",
            "не могу прочитать",
            "не может прочитать",
            "не могу видеть",
            "не может видеть",
            "не вижу канал",
            "не видит канал",
            "cannot read",
            "cannot view",
            "can't read",
            "can't view",
            "не знаю что там было",
            "не могу читать конкретные сообщения",
        ]
        return any(frag in low for frag in bad_fragments)

    def _fallback_summary_from_recent(self, recent) -> str:
        lines = []
        for row in recent[-5:]:
            role = (row.get("role") or "").strip().lower()
            content = (row.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                content = strip_output_labels(content)
            if content:
                lines.append(content)
        if not lines:
            return ""
        if len(lines) == 1:
            return lines[0][:180]
        return " / ".join(lines[-3:])[:240]

    def _trim_recent_for_current_user(self, recent, user_text: str):
        if not recent:
            return recent
        norm = normalize_compare_text(user_text)
        if not norm:
            return recent
        last = recent[-1]
        if (last.get("role") or "").lower() != "user":
            return recent
        content = (last.get("content") or "").strip()
        variants = [content]
        if ":" in content:
            variants.append(content.split(":", 1)[1].strip())
        for variant in variants:
            if normalize_compare_text(variant) == norm:
                return recent[:-1]
        return recent

    def _prompt_user_text(self, message: discord.Message, user_text: str) -> str:
        user_text = (user_text or "").strip()
        if not user_text:
            return user_text
        return f"{message.author.display_name}: {user_text}"

    def _last_assistant_reply(self, recent) -> str:
        for row in reversed(recent or []):
            if (row.get("role") or "").lower() == "assistant":
                return (row.get("content") or "").strip()
        return ""

    def _looks_like_echo(self, candidate: str, user_text: str, recent) -> bool:
        if is_degenerate_response(candidate):
            return True

        if is_too_similar(candidate, user_text, threshold=0.80):
            return True

        # Check against recent history to prevent repetition
        for row in (recent or []):
            if is_too_similar(candidate, row.get("content") or "", threshold=0.85):
                return True

        last_assistant = self._last_assistant_reply(recent)
        if last_assistant and is_too_similar(candidate, last_assistant, threshold=0.85):
            return True

        return False

    def _read_followup(self, low: str) -> bool:
        return any(
            p in low
            for p in [
                "а еще", "а ещё", "еще", "ещё", "дальше", "продолжай", "что еще", "что ещё",
                "и еще", "и ещё", "еще раз", "ещё раз", "а что дальше", "что дальше",
                "подробнее", "короткую сводку", "сводку", "что обсуждали", "обсуждали",
                "что там писали", "что писали", "о чем говорили", "о чём говорили",
                "что происходило", "что было", "расскажи что там было", "расскажи что там",
                "continue", "more", "what else", "before that", "earlier", "до этого", "раньше",
            ]
        )

    def _read_limit_from_text(self, low: str) -> int:
        deep_triggers = [
            "подроб", "полностью", "полный", "весь", "вся", "все сообщения", "всё сообщение",
            "вся переписка", "всю переписку", "полную сводку", "развернуто", "развёрнуто", "детально",
            "досконально", "что обсуждали", "что происходило", "что там писали", "что писали",
            "сводка", "а еще", "что еще", "подробнее",
        ]
        summary_triggers = [
            "дай сводку", "короткую сводку", "сводку", "что обсуждали", "обсуждали", "что там было",
            "что там писали", "что писали", "о чем говорили", "о чём говорили", "расскажи", "поясни",
            "что происходило", "коротко",
        ]
        latest_triggers = ["последнее", "последнее сообщение", "что последнее", "last", "latest"]

        if any(p in low for p in deep_triggers):
            return 80
        if any(p in low for p in summary_triggers):
            return 50
        if any(p in low for p in latest_triggers):
            return 1
        return 30

    def _extract_user_hints(self, user_text: str) -> List[str]:
        low = (user_text or "").lower()
        hints: List[str] = []
        if "работаю" in low or "работает" in low or "работал" in low:
            hints.append("связан с работой/занятостью")
        if "учусь" in low or "учится" in low or "студент" in low:
            hints.append("учится или связан с учебой")
        if "играю" in low or "играет" in low or "геймер" in low:
            hints.append("любит игры")
        return hints

    async def refresh_user_card(self, message: discord.Message, user_text: str) -> None:
        if not message.guild or message.author.bot:
            return

        uid = str(message.author.id)
        cid = str(message.channel.id)
        self.store.ensure_user_card(uid, username=message.author.name, display_name=message.author.display_name)
        card = self.store.get_user_card(uid) or {}
        hints = self._extract_user_hints(user_text)
        for hint in hints:
            self.store.append_user_card_note(uid, hint)

        seen = int(card.get("messages_seen") or 0)
        should_refresh = (seen % 10 == 0) or "меня зовут" in (user_text or "").lower()
        if not should_refresh:
            return

        recent_user = self.store.get_recent_user_history(uid, channel_id=cid, limit=10)
        user_lines = [row.get("content", "") for row in recent_user if row.get("content")]

        current_card_text = self.store.build_profile_card(subject_id=uid, subject_name=message.author.display_name)

        payload = (
            f"Текущая карточка:\n{current_card_text}\n\n"
            f"Последние сообщения:\n" + "\n".join(user_lines) +
            "\n\nОбнови карточку. Не выдумывай факты."
        )

        try:
            raw = await self.llm.chat_json(
                [
                    {"role": "system", "content": USER_CARD_SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                temperature=0.2,
            )
            if raw:
                self.store.update_user_card(uid, **raw)
        except Exception:
            pass

    def detect_emotion(self, text: str) -> str:
        low = (text or "").lower()
        if any(w in low for w in NEGATIVE_WORDS): return "angry"
        if any(w in low for w in POSITIVE_WORDS): return "happy"
        if any(w in low for w in PLAYFUL_WORDS): return "hype"
        return "neutral"

    def update_mood(self, channel_id: str, user_text: str):
        meta = self._meta(channel_id)
        energy = float(meta.get("energy") or 0.5)
        sass = float(meta.get("sass") or 0.5)
        emo = self.detect_emotion(user_text)
        if emo == "angry":
            sass = min(1.0, sass + 0.1)
        elif emo == "happy":
            energy = min(1.0, energy + 0.05)
        self.store.update_channel_meta(channel_id, energy=energy, sass=sass)

    def build_snapshot(self, message: discord.Message, user_text: str) -> str:
        cid = str(message.channel.id)
        meta = self._meta(cid)
        author_id = str(message.author.id)

        user_card = self.store.build_profile_card(subject_id=author_id, subject_name=message.author.display_name)

        lines = [
            f"Настроение канала: {meta.get('mood', 'calm')}",
            f"Энергия: {meta.get('energy', 0.5):.2f}, Sass: {meta.get('sass', 0.5):.2f}",
            f"Память об авторе ({message.author.display_name}):\n{user_card}",
        ]

        last_read = meta.get("last_read_summary")
        if last_read:
            lines.append(f"Недавно прочитано в другом канале: {last_read}")

        summary = self.store.get_summary(cid)
        if summary:
            lines.append(f"Сводка текущего канала: {summary}")

        return "\n".join(lines)[: self.settings.max_context_chars]

    async def decide(self, message: discord.Message, user_text: str) -> Dict[str, Any]:
        cid = str(message.channel.id)
        uid = str(message.author.id)
        self.update_mood(cid, user_text)
        asyncio.create_task(self.refresh_user_card(message, user_text))

        low = (user_text or "").lower().strip()
        meta = self._meta(cid)

        # Pagination check
        if self._read_followup(low):
            last_action = meta.get("last_action_type")
            last_channel = meta.get("last_target_channel_id")
            if last_action == "read_channel" and last_channel:
                anchor = meta.get("last_read_anchor_message_id")
                return {
                    "action": "read_channel",
                    "channel": last_channel,
                    "limit": self._read_limit_from_text(low),
                    "before": anchor,
                }

        # Regular decision
        for attempt in range(2):
            system = build_system_prompt(meta.get("mood", "calm"), float(meta.get("energy", 0.5)), float(meta.get("sass", 0.5)))
            msgs = [
                {"role": "system", "content": system},
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "system", "content": self.build_snapshot(message, user_text)},
            ]
            recent = self.store.get_recent_history(cid, self.settings.max_recent_turns)
            msgs.extend(recent)
            msgs.append({"role": "user", "content": self._prompt_user_text(message, user_text)})

            raw = await self.llm.chat(msgs, temperature=0.3 + attempt * 0.2)
            if not raw: continue

            cleaned = clean_response(raw)
            if cleaned and not self._looks_like_echo(cleaned, user_text, recent):
                return {"action": "reply", "text": cleaned}

        return {"action": "ignore"}

    async def summarize_channel(self, channel_id: str, force: bool = False):
        meta = self.store.get_channel_meta(channel_id) or {}
        import time
        now = time.time()

        # Cache for 20 minutes
        last_ts_str = meta.get("summary_timestamp")
        last_ts = 0.0
        if last_ts_str:
            try:
                import datetime
                last_ts = datetime.datetime.fromisoformat(last_ts_str.replace("Z", "+00:00")).timestamp()
            except: pass

        current_count = int(meta.get("message_count") or 0)
        last_summary_count = int(meta.get("last_summary_count") or 0)

        # Invalidate if 20 mins passed OR >10 new messages
        if not force and last_ts > 0:
            if (now - last_ts < 1200) and (current_count - last_summary_count < 10):
                return

        recent = self.store.get_recent_history(channel_id, 40)
        if not recent: return

        text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

        raw = await self.llm.chat([
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ], temperature=0.2)

        if raw and not self._is_bad_summary(raw):
            import datetime
            self.store.update_channel_meta(channel_id,
                summary=raw,
                summary_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                last_summary_count=current_count
            )

    def should_summarize(self, channel_id: str) -> bool:
        meta = self.store.get_channel_meta(channel_id) or {}
        diff = int(meta.get("message_count") or 0) - int(meta.get("last_summary_count") or 0)
        return diff >= 10

    async def compose_read_response(self, message: discord.Message, user_text: str, action: Dict[str, Any], observation: Dict[str, Any]) -> str:
        text = observation.get("text")
        if not text: return "там пусто"

        raw = await self.llm.chat([
            {"role": "system", "content": COMPOSE_READ_PROMPT},
            {"role": "user", "content": f"User: {user_text}\nObservation: {text}"}
        ], temperature=0.4)
        return clean_response(raw) or "ну такое"

    async def run_autonomy(self, message: discord.Message) -> Dict[str, Any]:
        if message.author.bot: return {"action": "ignore"}

        cid = str(message.channel.id)
        meta = self.store.get_channel_meta(cid) or {}

        import time
        now = time.time()

        def parse_ts(ts_str):
            if not ts_str: return 0.0
            try:
                import datetime
                return datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
            except: return 0.0

        last_global = parse_ts(meta.get("last_autonomy_at"))
        if now - last_global < 120: return {"action": "ignore"} # 2 min global

        recent_rows = self.store.get_recent_history_rows(cid, 10)
        if len(recent_rows) < 5: return {"action": "ignore"}

        # Check 10 min window and distinct users
        active_msgs = []
        users = set()
        for r in reversed(recent_rows):
            ts = parse_ts(r.get("created_at")) or now
            if now - ts > 600: break
            active_msgs.append(r)
            if (r.get("role") or "").lower() == "user":
                users.add(r.get("speaker_id"))

        if len(active_msgs) < 5 or len(users) < 2: return {"action": "ignore"}

        # Prompt autonomy
        last_interject = parse_ts(meta.get("last_interjection_at"))
        last_emoji = parse_ts(meta.get("last_emoji_at"))

        history_text = "\n".join([f"{r['speaker_name']}: {r['content']}" for r in reversed(active_msgs)])

        raw = await self.llm.chat_json([
            {"role": "system", "content": AUTONOMY_SYSTEM_PROMPT},
            {"role": "user", "content": f"History:\n{history_text}\nCooldowns: interject={int(now-last_interject)}s, emoji={int(now-last_emoji)}s"}
        ], temperature=0.5)

        action = raw.get("action", "ignore")

        if action in {"short_interject", "contextual_reply"}:
            if now - last_interject < 1200: return {"action": "ignore"} # 20 min
            return {"action": "reply", "text": raw.get("text")}

        if action == "react":
            if now - last_emoji < 300: return {"action": "ignore"} # 5 min
            return {"action": "react", "reaction": raw.get("reaction")}

        return {"action": "ignore"}
