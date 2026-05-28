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
    normalize_compare_text,
    sanitize_summary_text,
    strip_output_labels,
)

# Constants for emotion detection and social engagement
NEGATIVE_WORDS = ["туп", "идиот", "долба", "блять", "еб", "нах", "сука", "пизд", "хуй", "дурак", "бесит", "злюсь"]
POSITIVE_WORDS = ["спасибо", "класс", "круто", "норм", "люблю", "уважаю", "молодец", "приятно", "умница", "рад", "супер"]
PLAYFUL_WORDS = ["ахах", "хаха", "лол", "ржу", "угар", "мем", "кайф", "ура", "топ"]


class AgentPlanner:
    def __init__(self, settings, store, llm_client, direct_parser):
        self.settings = settings
        self.store = store
        self.llm = llm_client
        self.direct_parser = direct_parser
        self._runtime: Dict[str, Dict[str, object]] = {}

    def _state(self, channel_id: str):
        """Maintains in-memory state for duplicate prevention and easter eggs."""
        return self._runtime.setdefault(channel_id, {"recent": [], "anchors": []})

    def _meta(self, channel_id: str) -> Dict[str, Any]:
        meta = self.store.get_channel_meta(channel_id) or {}
        if isinstance(meta, dict):
            return meta
        return {}

    def _is_bad_summary(self, text: str) -> bool:
        """Heuristic to detect if LLM failed to read history properly."""
        low = (text or "").lower()
        bad_fragments = [
            "не могу посмотреть", "не может посмотреть",
            "не могу прочитать", "не может прочитать",
            "не могу видеть", "не может видеть",
            "не вижу канал", "не видит канал",
            "cannot read", "cannot view", "can't read", "can't view",
            "не знаю что там было", "не могу читать конкретные сообщения",
            "ошибка доступа", "нет доступа"
        ]
        return any(frag in low for frag in bad_fragments)

    def _fallback_summary_from_recent(self, recent) -> str:
        """Simple summary generator if LLM fails."""
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
        """Prevents the bot from seeing the current user message twice in history."""
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
        """Duplicate / Degenerate guard: prevents echoes and repetitive garbage."""
        if not candidate:
            return True
        low = normalize_compare_text(candidate)

        # Prevent filler garbage
        if not low or low in {
            "мм", "мм?", "м?", "а?", "э?", "эх?", "поняла", "понял",
            "ок", "ok", "ясно", "пон", "ладно", "угу"
        }:
            return True

        # Prevent very short responses
        if len(low) < 2:
            return True

        # Exact or high similarity echo of current user
        if is_too_similar(candidate, user_text, threshold=0.75):
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
        """Detects if the user wants to continue reading or get more details."""
        triggers = [
            "а еще", "еще", "дальше", "продолжай", "что еще", "и еще", "еще раз",
            "а что дальше", "что дальше", "подробнее", "короткую сводку", "сводку",
            "что обсуждали", "что там писали", "что писали", "о чем говорили",
            "что происходило", "что было", "расскажи что там было", "расскажи что там",
            "continue", "more", "what else", "before that", "earlier", "до этого", "раньше"
        ]
        return any(p in low for p in triggers)

    def _read_limit_from_text(self, low: str) -> int:
        """Dynamic limit based on user keywords."""
        deep_triggers = ["подроб", "полностью", "полный", "весь", "вся", "все сообщения", "вся переписка", "развернуто", "детально"]
        summary_triggers = ["сводка", "что обсуждали", "обсуждали", "что там было", "что там писали", "о чем говорили", "что происходило"]
        latest_triggers = ["последнее", "last", "latest"]

        if any(p in low for p in deep_triggers): return 80
        if any(p in low for p in summary_triggers): return 50
        if any(p in low for p in latest_triggers): return 1
        return 30

    def _extract_user_hints(self, user_text: str) -> List[str]:
        """Simple rule-based fact extraction for user cards."""
        low = (user_text or "").lower()
        hints: List[str] = []
        if any(w in low for w in ["работаю", "работает", "работал"]): hints.append("связан с работой/занятостью")
        if any(w in low for w in ["учусь", "учится", "студент"]): hints.append("учится или связан с учебой")
        if any(w in low for w in ["играю", "играет", "геймер"]): hints.append("любит игры")
        if any(w in low for w in ["люблю", "нравится"]):
            snippet = user_text.strip()
            if len(snippet) > 90: snippet = snippet[:90].rstrip() + "..."
            hints.append(f"любит/предпочитает: {snippet}")
        return hints

    async def refresh_user_card(self, message: discord.Message, user_text: str) -> None:
        """Updates persistent user card in SQLite using LLM analysis."""
        if not message.guild or message.author.bot:
            return

        uid = str(message.author.id)
        cid = str(message.channel.id)
        self.store.ensure_profile(uid, username=message.author.name, display_name=message.author.display_name)
        card = self.store.get_user_card(uid) or {}

        # Quick rule-based hints
        for hint in self._extract_user_hints(user_text):
            self.store.append_user_card_note(uid, hint)
            self.store.append_profile_note(uid, hint)

        seen = int(card.get("messages_seen") or 0)
        # Refresh logic: explicit mentions, new user, or every 8 messages
        explicit = any(p in (user_text or "").lower() for p in ["меня зовут", "зови меня", "я работаю", "я учусь", "я играю", "я люблю", "я из", "я живу"])
        if not (explicit or seen < 3 or (seen % 8 == 0)):
            return

        recent_user = self.store.get_recent_user_history(uid, channel_id=cid, limit=8)
        recent_channel = self.store.get_recent_history_rows(cid, limit=12)
        if not recent_user: return

        current_profile = self.store.build_profile_card(subject_id=uid, subject_name=message.author.display_name, max_facts=4)
        user_lines = [r.get("content") for r in recent_user if r.get("content")]
        context_lines = [f"{r.get('speaker_name')}: {r.get('content')}" for r in recent_channel if r.get("content")]

        payload = (
            f"Текущая карточка:\n{current_profile or 'нет'}\n\n"
            f"Последние сообщения пользователя:\n" + "\n".join(f"- {x}" for x in user_lines) +
            f"\n\nКонтекст канала:\n" + "\n".join(f"- {x}" for x in context_lines) +
            "\n\nОбнови карточку по правилам. Не выдумывай лишнего."
        )

        try:
            raw = await self.llm.chat_json([{"role": "system", "content": USER_CARD_SYSTEM_PROMPT}, {"role": "user", "content": payload}], temperature=0.2)
            if not raw: return

            # Sanitize and update
            updates = {k: sanitize_summary_text(str(v)) for k, v in raw.items() if k in {
                "summary", "interests", "communication_style", "traits", "relationship",
                "relationship_trend", "opinion", "topics", "activity_level", "behaviors"
            }}
            updates["username"] = message.author.name
            updates["display_name"] = message.author.display_name

            if raw.get("notes"):
                existing = (self.store.get_user_card(uid) or {}).get("notes") or ""
                updates["notes"] = f"{existing}; {raw['notes']}".strip("; ").strip()

            self.store.update_user_card(uid, **updates)

            # Sync back to legacy profiles table for compatibility
            if updates.get("summary"): self.store.append_profile_note(uid, f"card: {updates['summary']}")
            if updates.get("traits"): self.store.upsert_profile_fields(uid, traits=updates["traits"])
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
        mood = meta.get("mood") or "calm"
        energy = float(meta.get("energy") or 0.5)
        sass = float(meta.get("sass") or 0.5)
        emo = self.detect_emotion(user_text)

        if emo == "angry": mood, sass, energy = "annoyed", min(1.0, sass + 0.12), min(1.0, energy + 0.03)
        elif emo == "happy": mood, energy = "friendly", min(1.0, energy + 0.04)
        elif emo == "hype": mood, energy = "playful", min(1.0, energy + 0.06)

        self.store.update_channel_meta(channel_id, mood=mood, energy=energy, sass=sass)
        return mood

    def update_affinity_from_message(self, author_id: str, user_text: str):
        low = user_text.lower()
        delta = 0
        if any(w in low for w in POSITIVE_WORDS): delta += 1
        if any(w in low for w in NEGATIVE_WORDS): delta -= 1
        if "запомни" in low: delta += 1
        if delta: self.store.adjust_affinity(author_id, delta)

    def learn_feedback(self, message: discord.Message, user_text: str):
        low = (user_text or "").lower().strip()
        uid, cid = str(message.author.id), str(message.channel.id)
        if any(w in low for w in ["молодец", "круто", "топ"]): self.store.adjust_affinity(uid, 1)
        if any(w in low for w in ["плохо", "тупо", "кринж", "ошибка"]): self.store.adjust_affinity(uid, -1)
        if "короче" in low or "кратко" in low: self.store.set_user_pref(uid, "length", "short")
        if "подробно" in low or "развернуто" in low: self.store.set_user_pref(uid, "length", "long")

    def build_snapshot(self, message: discord.Message, user_text: str) -> str:
        cid = str(message.channel.id)
        meta = self._meta(cid)
        author_id, author_name = str(message.author.id), message.author.display_name

        lines = [
            "КОНТЕКСТ ДЛЯ Nika:",
            f"Настроение канала: {meta.get('mood', 'calm')}, Энергия: {float(meta.get('energy', 0.5)):.2f}, Sass: {float(meta.get('sass', 0.5)):.2f}",
            "ПАМЯТЬ ОБ АВТОРЕ:", self.store.build_profile_card(subject_id=author_id, subject_name=author_name, max_facts=6)
        ]

        last_summary = meta.get("last_read_summary") or ""
        if last_summary: lines += ["ПОСЛЕДНЯЯ ВЫЖИМКА ПРОЧИТАННОГО:", last_summary]

        summary = self.store.get_summary(cid)
        if summary: lines += ["СВОДКА КАНАЛА:", summary]

        return "\n".join(lines)[: self.settings.max_context_chars]

    def build_chat_messages(self, message: discord.Message, user_text: str):
        cid = str(message.channel.id)
        meta = self._meta(cid)
        system = build_system_prompt(meta.get("mood", "calm"), float(meta.get("energy", 0.5)), float(meta.get("sass", 0.5)))
        msgs = [{"role": "system", "content": system}, {"role": "system", "content": CHAT_SYSTEM_PROMPT}, {"role": "system", "content": self.build_snapshot(message, user_text)}]
        recent = self.store.get_recent_history(cid, self.settings.max_recent_turns)
        msgs.extend(self._trim_recent_for_current_user(recent, user_text))
        msgs.append({"role": "user", "content": self._prompt_user_text(message, user_text)})
        return msgs

    async def decide(self, message: discord.Message, user_text: str) -> Dict[str, Any]:
        """Main entry point for reactive decisions with duplicate guard."""
        cid, author_id = str(message.channel.id), str(message.author.id)
        self.update_affinity_from_message(author_id, user_text)
        self.update_mood(cid, user_text)
        asyncio.create_task(self.refresh_user_card(message, user_text))

        # 1. Check direct parser (explicit commands)
        direct = self.direct_parser.parse(message, user_text)
        if direct: return direct

        # 2. Check for pagination followup
        low, meta = user_text.lower().strip(), self._meta(cid)
        if self._read_followup(low) and meta.get("last_action_type") == "read_channel":
            return {"action": "read_channel", "channel": meta.get("last_target_channel_id"), "limit": self._read_limit_from_text(low), "before": meta.get("last_read_first_message_id")}

        # 3. LLM Decision with Bounded Retries (Duplicate Guard)
        for attempt in range(2):
            msgs = self.build_chat_messages(message, user_text)
            raw = await self.llm.chat(msgs, temperature=0.3 + (attempt * 0.1))
            if not raw: continue

            cleaned = clean_response(raw)
            if not cleaned or self.is_duplicate_response(cid, cleaned) or self._looks_like_echo(cleaned, user_text, msgs):
                continue

            # If JSON returned, parse action
            if cleaned.startswith("{"):
                try:
                    action = self._action_from_llm(json.loads(cleaned))
                    if action.get("action") == "reply" and self._looks_like_echo(action.get("text", ""), user_text, msgs): continue
                    return action
                except: pass

            return {"action": "reply", "text": cleaned}

        return {"action": "ignore"}

    async def summarize_channel(self, channel_id: str, force: bool = False):
        """Maintains a rolling summary of the channel, cached for 20 minutes."""
        meta = self.store.get_channel_meta(channel_id) or {}
        import time, datetime
        now = time.time()

        last_ts = self._parse_db_timestamp(meta.get("summary_timestamp"))
        last_count = int(meta.get("last_summary_count") or 0)
        curr_count = int(meta.get("message_count") or 0)

        # Cache check: 20m or 10 new messages
        if not force and last_ts > 0 and (now - last_ts < 1200) and (curr_count - last_count < 10):
            return

        recent = self.store.get_recent_history(channel_id, 30)
        if not recent: return

        recent_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        raw = await self.llm.chat([{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}, {"role": "user", "content": f"Текущая сводка: {meta.get('summary', 'нет')}\n\nНовые сообщения:\n{recent_text}"}], temperature=0.2)

        summary = sanitize_summary_text(raw) or self._fallback_summary_from_recent(recent)
        if summary and not self._is_bad_summary(summary):
            self.store.update_channel_meta(channel_id, summary=summary[:self.settings.max_summary_chars], summary_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), last_summary_count=curr_count)

    def should_summarize(self, channel_id: str) -> bool:
        meta = self.store.get_channel_meta(channel_id) or {}
        return (int(meta.get("message_count") or 0) - int(meta.get("last_summary_count") or 0)) >= self.settings.summary_trigger_messages

    def _parse_db_timestamp(self, ts_str: str) -> float:
        if not ts_str: return 0.0
        import datetime
        for fmt in ["%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"]:
            try: return datetime.datetime.strptime(ts_str.split('.')[0] if ' ' in ts_str else ts_str, fmt.replace('.%f', '')).timestamp()
            except: continue
        try: return datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except: return 0.0

    async def run_autonomy(self, message: discord.Message) -> Dict[str, Any]:
        """Autonomous watcher: observes chat and interjects based on heuristics."""
        if not self.settings.autonomy_enabled or message.author.bot: return {"action": "ignore"}

        cid = str(message.channel.id)
        meta = self.store.get_channel_meta(cid) or {}
        import time
        now = time.time()

        # Cooldowns: global 2m, interjection 20m, emoji 5m
        if now - self._parse_db_timestamp(meta.get("last_autonomy_at")) < 120: return {"action": "ignore"}

        recent_rows = self.store.get_recent_history_rows(cid, 20)
        active_msgs = [r for r in recent_rows if now - self._parse_db_timestamp(r.get("created_at")) < 600]

        # Rule: >5 msgs, >2 distinct users in 10 min
        if len(active_msgs) < 5 or len({r.get("speaker_id") for r in active_msgs if r.get("role") == "user"}) < 2:
            return {"action": "ignore"}

        # Build participants cards for LLM
        participants = [self.store.build_profile_card(subject_id=r.get("speaker_id"), max_facts=2) for r in active_msgs[:3]]
        recent_text = "\n".join(f"{r.get('speaker_name')}: {r.get('content')}" for r in active_msgs[-10:])

        try:
            raw = await self.llm.chat_json([
                {"role": "system", "content": AUTONOMY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Чат:\n{recent_text}\n\nУчастники:\n" + "\n".join(participants)}
            ], temperature=0.3)

            action = (raw.get("action") or "ignore").lower()
            if action in {"short_interject", "contextual_reply", "reply"}:
                if now - self._parse_db_timestamp(meta.get("last_interjection_at")) < 1200: return {"action": "ignore"}
                text = raw.get("text") or raw.get("message")
                if text and not self._looks_like_echo(text, "", []): return {"action": "reply", "text": text}
            elif action == "react":
                if now - self._parse_db_timestamp(meta.get("last_emoji_at")) < 300: return {"action": "ignore"}
                return {"action": "react", "reaction": raw.get("reaction")}
        except: pass
        return {"action": "ignore"}

    def _action_from_llm(self, obj: dict) -> dict:
        action = (obj.get("action") or "ignore").strip().lower()
        if action == "reply": return {"action": "reply", "text": (obj.get("text") or "").strip()}
        if action == "read_channel": return {"action": "read_channel", "channel": str(obj.get("channel") or ""), "limit": int(obj.get("limit") or 30), "before": str(obj.get("before") or "")}
        return {"action": "ignore"}

    def is_duplicate_response(self, channel_id: str, text: str) -> bool:
        st = self._state(channel_id)
        norm = normalize_compare_text(text)
        if norm and norm in st["recent"]: return True
        if norm:
            st["recent"].append(norm)
            if len(st["recent"]) > 4: st["recent"].pop(0)
        return False

    async def compose_read_response(self, message: discord.Message, user_text: str, action: Dict[str, Any], observation: Dict[str, Any]) -> str:
        text = (observation.get("text") or "").strip()
        if not text: return "там пусто"
        raw = await self.llm.chat([{"role": "system", "content": COMPOSE_READ_PROMPT}, {"role": "user", "content": f"Запрос: {user_text}\nНаблюдение:\n{text}"}], temperature=0.3)
        return sanitize_summary_text(raw) or "ну вот как-то так"
