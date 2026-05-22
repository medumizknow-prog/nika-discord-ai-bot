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
        if isinstance(meta, dict):
            return meta
        try:
            if hasattr(meta, "keys"):
                return {k: meta[k] for k in meta.keys()}
        except Exception:
            pass
        return {}

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
        if not candidate:
            return True
        low = normalize_compare_text(candidate)
        # Block common fillers and empty/very short responses
        if not low or low in {"мм", "мм?", "м?", "а?", "э?", "эх?", "поняла", "понял", "ок", "ok"}:
            return True

        if len(low) < 2:
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
                "continue",
                "more",
                "what else",
                "before that",
                "earlier",
                "до этого",
                "раньше",
            ]
        )

    def _read_limit_from_text(self, low: str) -> int:
        deep_triggers = [
            "подроб",
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
            "сводка",
            "а еще",
            "что еще",
            "подробнее",
            "коротко",
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
            "что происходило",
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

    def _extract_user_hints(self, user_text: str) -> List[str]:
        low = (user_text or "").lower()
        hints: List[str] = []
        if "работаю" in low or "работает" in low or "работал" in low:
            hints.append("связан с работой/занятостью")
        if "учусь" in low or "учится" in low or "студент" in low:
            hints.append("учится или связан с учебой")
        if "играю" in low or "играет" in low or "геймер" in low:
            hints.append("любит игры")
        if "люблю" in low or "нравится" in low:
            snippet = user_text.strip()
            if len(snippet) > 90:
                snippet = snippet[:90].rstrip() + "..."
            hints.append(f"любит/предпочитает: {snippet}")
        if "из" in low and ("я из" in low or "живу" in low or "родом" in low):
            snippet = user_text.strip()
            if len(snippet) > 90:
                snippet = snippet[:90].rstrip() + "..."
            hints.append(f"контекст о месте/происхождении: {snippet}")
        if "мне нравится" in low or "мне не нравится" in low:
            snippet = user_text.strip()
            if len(snippet) > 90:
                snippet = snippet[:90].rstrip() + "..."
            hints.append(f"личная позиция: {snippet}")
        return hints

    async def refresh_user_card(self, message: discord.Message, user_text: str) -> None:
        if not message.guild or message.author.bot:
            return

        uid = str(message.author.id)
        cid = str(message.channel.id)
        self.store.ensure_profile(uid, username=message.author.name, display_name=message.author.display_name)
        card = self.store.get_user_card(uid) or {}
        hints = self._extract_user_hints(user_text)
        for hint in hints:
            self.store.append_user_card_note(uid, hint)
            self.store.append_profile_note(uid, hint)

        seen = int(card.get("messages_seen") or 0)
        explicit = any(
            p in (user_text or "").lower()
            for p in [
                "меня зовут",
                "зови меня",
                "я работаю",
                "я учусь",
                "я играю",
                "я люблю",
                "мне нравится",
                "я из",
                "я живу",
                "я занимаюсь",
                "я увлекаюсь",
            ]
        )
        should_refresh = explicit or seen < 3 or (seen % 8 == 0)
        if not should_refresh:
            return

        recent_user = self.store.get_recent_user_history(uid, channel_id=cid, limit=8)
        recent_channel = self.store.get_recent_history_rows(cid, limit=12)
        if not recent_user and not recent_channel:
            return

        current_profile = self.store.build_profile_card(subject_id=uid, subject_name=message.author.display_name, max_facts=4)
        user_lines = []
        for row in recent_user[-8:]:
            content = (row.get("content") or "").strip()
            if content:
                user_lines.append(content)
        context_lines = []
        for row in recent_channel[-10:]:
            role = (row.get("role") or "").strip().lower()
            speaker = (row.get("speaker_name") or "").strip()
            content = (row.get("content") or "").strip()
            if not content:
                continue
            context_lines.append(f"{role} / {speaker}: {content}")

        payload = (
            f"Текущая карточка:\n{current_profile or 'нет'}\n\n"
            f"Последние сообщения пользователя:\n" + "\n".join(f"- {x}" for x in user_lines) +
            f"\n\nКонтекст канала:\n" + "\n".join(f"- {x}" for x in context_lines) +
            "\n\nОбнови карточку по правилам. Не выдумывай лишнего."
        )

        try:
            raw = await self.llm.chat_json(
                [
                    {"role": "system", "content": USER_CARD_SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                temperature=0.2,
                max_tokens=220,
            )
        except Exception:
            return

        if not isinstance(raw, dict) or not raw:
            return

        summary = sanitize_summary_text(raw.get("summary") or "")
        interests = sanitize_summary_text(raw.get("interests") or "")
        communication_style = sanitize_summary_text(raw.get("communication_style") or "")
        traits = sanitize_summary_text(raw.get("traits") or "")
        relationship = sanitize_summary_text(raw.get("relationship") or "")
        relationship_trend = sanitize_summary_text(raw.get("relationship_trend") or "")
        opinion = sanitize_summary_text(raw.get("opinion") or "")
        topics = sanitize_summary_text(raw.get("topics") or "")
        activity_level = sanitize_summary_text(raw.get("activity_level") or "")
        behaviors = sanitize_summary_text(raw.get("behaviors") or "")
        notes = sanitize_summary_text(raw.get("notes") or "")
        confidence_raw = raw.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else 0.0
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        updates: Dict[str, Any] = {
            "username": message.author.name,
            "display_name": message.author.display_name,
            "messages_seen": seen,
        }
        if summary:
            updates["summary"] = summary
        if interests:
            updates["interests"] = interests
        if communication_style:
            updates["communication_style"] = communication_style
        if traits:
            updates["traits"] = traits
        if relationship:
            updates["relationship"] = relationship
        if relationship_trend:
            updates["relationship_trend"] = relationship_trend
        if opinion:
            updates["opinion"] = opinion
        if topics:
            updates["topics"] = topics
        if activity_level:
            updates["activity_level"] = activity_level
        if behaviors:
            updates["behaviors"] = behaviors
        if notes:
            existing_note = (self.store.get_user_card(uid) or {}).get("notes") or ""
            merged_notes = f"{existing_note}; {notes}".strip("; ").strip()
            updates["notes"] = merged_notes
        if confidence:
            if not notes:
                updates["notes"] = f"confidence={confidence:.2f}"

        self.store.update_user_card(uid, **updates)
        if summary:
            self.store.append_profile_note(uid, f"card: {summary}")
        if traits:
            self.store.upsert_profile_fields(uid, traits=traits)
        if relationship:
            self.store.upsert_profile_fields(uid, relationship=relationship)
        if notes:
            self.store.append_profile_note(uid, notes)

    def detect_emotion(self, text: str) -> str:
        low = (text or "").lower()
        if any(w in low for w in NEGATIVE_WORDS):
            return "angry"
        if any(w in low for w in POSITIVE_WORDS):
            return "happy"
        if any(w in low for w in PLAYFUL_WORDS):
            return "hype"
        if any(w in low for w in ["бесит", "злюсь", "раздраж", "ненавиж", "тупо", "достал"]):
            return "angry"
        if any(w in low for w in ["рад", "супер", "кайф", "ура", "топ"]):
            return "happy"
        return "neutral"

    def update_mood(self, channel_id: str, user_text: str, author_id: str = ""):
        meta = self._meta(channel_id)
        mood = meta.get("mood") or "calm"
        energy = float(meta.get("energy") or 0.5)
        sass = float(meta.get("sass") or 0.5)
        emo = self.detect_emotion(user_text)
        if emo == "angry":
            mood, sass, energy = "annoyed", min(1.0, sass + 0.12), min(1.0, energy + 0.03)
        elif emo == "happy":
            mood, energy = "friendly", min(1.0, energy + 0.04)
        elif emo == "hype":
            mood, energy = "playful", min(1.0, energy + 0.06)
        elif is_short_text(user_text):
            mood = "chaotic"
        self.store.update_channel_meta(channel_id, mood=mood, energy=energy, sass=sass)
        return mood

    def update_affinity_from_message(self, author_id: str, user_text: str):
        low = user_text.lower()
        delta = 0
        if any(w in low for w in POSITIVE_WORDS):
            delta += 1
        if any(w in low for w in NEGATIVE_WORDS):
            delta -= 1
        if "запомни" in low:
            delta += 1
        if delta:
            self.store.adjust_affinity(author_id, delta)

    def learn_feedback(self, message: discord.Message, user_text: str):
        low = (user_text or "").lower().strip()
        if not low:
            return
        uid = str(message.author.id)
        cid = str(message.channel.id)
        pos = any(w in low for w in ["молодец", "умница", "круто", "класс", "хорошо", "супер", "топ", "норм"])
        neg = any(w in low for w in ["не так", "плохо", "криво", "тупо", "кринж", "не нравится", "формально", "ошибка", "не то"])
        if pos or neg:
            rating = 1 if pos and not neg else -1 if neg and not pos else 0
            if rating:
                self.store.add_feedback(uid, cid, "chat_feedback", rating, "tone", low, user_text)
                self.store.adjust_affinity(uid, rating)
        if "короче" in low or "кратко" in low:
            self.store.set_user_pref(uid, "length", "short", weight=0.2)
        if "подробно" in low or "развернуто" in low:
            self.store.set_user_pref(uid, "length", "long", weight=0.2)
        if "жёстче" in low or "жестче" in low or "дерзко" in low:
            self.store.set_user_pref(uid, "tone", "spicy", weight=0.2)

    def maybe_learn_explicit_memory(self, author: discord.Member, user_text: str):
        low = user_text.lower().strip()
        if low.startswith("запомни ") or low.startswith("remember "):
            payload = user_text.split(" ", 1)[1].strip() if " " in user_text else ""
            if payload:
                self.store.upsert_fact(str(author.id), author.display_name, "note", payload, confidence=0.95, source="explicit")
                self.store.append_profile_note(str(author.id), payload)
                self.store.adjust_affinity(str(author.id), 1)
        if "меня зовут" in low or "зови меня" in low:
            m = re.search(r"(?:меня зовут|зови меня)\s+(.+)$", user_text, flags=re.I)
            if m:
                val = m.group(1).strip(" .!?")
                if val:
                    self.store.upsert_fact(str(author.id), author.display_name, "preferred_name", val, confidence=0.9, source="explicit")
                    self.store.upsert_profile_fields(str(author.id), preferred_name=val)

    def build_snapshot(self, message: discord.Message, user_text: str) -> str:
        cid = str(message.channel.id)
        meta = self._meta(cid)
        mood = meta.get("mood") or "calm"
        energy = float(meta.get("energy") or 0.5)
        sass = float(meta.get("sass") or 0.5)
        author_id = str(message.author.id)
        author_name = message.author.display_name

        user_card = self.store.build_profile_card(subject_id=author_id, subject_name=author_name, max_facts=6)
        prefs = self.store.get_user_prefs(author_id, limit=5)
        episodes = self.store.get_recent_episodes(channel_id=cid, user_id=author_id, limit=4)
        recent_rows = self.store.get_recent_history_rows(cid, limit=8)

        mentions = []
        seen = set()
        for member in message.mentions:
            if str(member.id) == author_id:
                continue
            if member.display_name.lower() in seen:
                continue
            seen.add(member.display_name.lower())
            card = self.store.build_profile_card(subject_id=str(member.id), subject_name=member.display_name, max_facts=4)
            if card:
                mentions.append(card)

        for row in recent_rows:
            if (row.get("role") or "").lower() != "user":
                continue
            sid = (row.get("speaker_id") or "").strip()
            sname = (row.get("speaker_name") or "").strip()
            key = sid or sname.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            card = self.store.build_profile_card(subject_id=sid, subject_name=sname, max_facts=3)
            if card:
                mentions.append(card)

        for name in self.store.build_known_names():
            n = name.lower().strip()
            if not n or n == author_name.lower() or n in seen:
                continue
            if n in user_text.lower():
                seen.add(n)
                card = self.store.build_profile_card(subject_name=name, max_facts=4)
                if card:
                    mentions.append(card)

        lines = [
            "КОНТЕКСТ ДЛЯ Nika:",
            f"Текущее настроение канала: {mood}",
            f"Энергия канала: {energy:.2f}",
            f"Sass: {sass:.2f}",
            f"Эмоция сообщения: {self.detect_emotion(user_text)}",
            "ФАКТЫ О ВОЗМОЖНОСТЯХ:",
            "- direct parser handles explicit commands",
            "- can read/write/ping/react/remember",
            "- read_channel returns actual channel history when available",
            "- if the channel is lively, autonomy may interject briefly",
        ]

        last_read_summary = meta.get("last_read_summary") or ""
        if last_read_summary:
            lines += ["ПОСЛЕДНЯЯ ВЫЖИМКА ПРОЧИТАННОГО КАНАЛА:", last_read_summary]

        if prefs:
            lines.append("ПРЕДПОЧТЕНИЯ АВТОРА:")
            lines.extend([f"{p['key']}: {p['value']} ({float(p['weight'] or 0):+.2f})" for p in prefs])

        if user_card:
            lines += ["ПАМЯТЬ ОБ АВТОРЕ:", user_card]

        if episodes:
            lines.append("ПОСЛЕДНИЕ СОБЫТИЯ:")
            lines.extend([f"- {e['summary']}" for e in episodes])

        if mentions:
            lines.append("ПАМЯТЬ ОБ УПОМИНУТЫХ:")
            lines.extend(mentions)

        summary = self.store.get_summary(cid)
        if summary:
            lines += ["СВОДКА КАНАЛА:", summary]

        att = attachment_summary(message.attachments)
        if att:
            lines += ["ВЛОЖЕНИЯ:", att]

        snap = "\n".join(lines).strip()
        return snap[: self.settings.max_context_chars]

    def build_chat_messages(self, message: discord.Message, user_text: str):
        cid = str(message.channel.id)
        meta = self._meta(cid)
        system = build_system_prompt(
            meta.get("mood") or "calm",
            float(meta.get("energy") or 0.5),
            float(meta.get("sass") or 0.5),
            capability_lines="- personality chat mode",
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "system", "content": self.build_snapshot(message, user_text)},
        ]
        summary = self.store.get_summary(cid)
        if summary:
            msgs.append({"role": "system", "content": f"Сводка канала: {summary}"})
        recent = self.store.get_recent_history(cid, 5 if is_short_text(user_text) else self.settings.max_recent_turns)
        recent = self._trim_recent_for_current_user(recent, user_text)
        msgs.extend(recent)
        msgs.append({"role": "user", "content": self._prompt_user_text(message, user_text)})
        return msgs

    def build_action_messages(self, message: discord.Message, user_text: str):
        cid = str(message.channel.id)
        meta = self._meta(cid)
        system = build_system_prompt(
            meta.get("mood") or "calm",
            float(meta.get("energy") or 0.5),
            float(meta.get("sass") or 0.5),
            capability_lines="- command parser mode",
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "system", "content": self.build_snapshot(message, user_text)},
        ]
        summary = self.store.get_summary(cid)
        if summary:
            msgs.append({"role": "system", "content": f"Сводка канала: {summary}"})
        recent = self.store.get_recent_history(cid, 5 if is_short_text(user_text) else self.settings.max_recent_turns)
        recent = self._trim_recent_for_current_user(recent, user_text)
        msgs.extend(recent)
        msgs.append({"role": "user", "content": self._prompt_user_text(message, user_text)})
        return msgs

    def is_duplicate_response(self, channel_id: str, text: str) -> bool:
        st = self._state(channel_id)
        norm = normalize_compare_text(text)
        if norm and norm in st["recent"]:
            return True
        if norm:
            st["recent"].append(norm)
            if len(st["recent"]) > 4:
                st["recent"].pop(0)
        return False

    def anchor_hit(self, channel_id: str, text: str) -> bool:
        st = self._state(channel_id)
        low = (text or "").lower()
        hits = [a for a in ["вирус", "шашлык", "коты", "красный мозг", "шарик", "пинг-понг", "косметик"] if a in low]
        if hits:
            st["anchors"].extend(hits)
            counts = {}
            for a in st["anchors"]:
                counts[a] = counts.get(a, 0) + 1
            return any(v >= 3 for v in counts.values())
        return False

    def _action_from_llm(self, obj: dict) -> dict:
        action = (obj.get("action") or "ignore").strip().lower()
        if action == "reply":
            return {"action": "reply", "text": (obj.get("text") or "").strip()}
        if action == "remember":
            return {"action": "remember", "key": (obj.get("key") or "note").strip(), "value": (obj.get("value") or obj.get("text") or "").strip()}
        if action == "read_channel":
            return {
                "action": "read_channel",
                "channel": (obj.get("channel") or "").strip(),
                "limit": int(obj.get("limit") or 30),
                "before": (obj.get("before") or "").strip(),
            }
        if action == "send_message":
            return {"action": "send_message", "channel": (obj.get("channel") or "").strip(), "text": (obj.get("text") or "").strip()}
        if action == "ping_user":
            return {"action": "ping_user", "channel": (obj.get("channel") or "").strip(), "user": (obj.get("user") or "").strip(), "text": (obj.get("text") or "").strip()}
        if action == "react":
            return {"action": "react", "channel": (obj.get("channel") or "").strip(), "reaction": (obj.get("reaction") or "").strip()}
        if action == "post_thought":
            return {"action": "post_thought", "text": (obj.get("text") or "").strip()}
        return {"action": "ignore"}

    async def decide(self, message: discord.Message, user_text: str) -> Dict[str, Any]:
        cid = str(message.channel.id)
        author_id = str(message.author.id)
        self.store.ensure_profile(author_id, username=message.author.name, display_name=message.author.display_name)
        self.update_affinity_from_message(author_id, user_text)
        self.update_mood(cid, user_text, author_id=author_id)
        self.maybe_learn_explicit_memory(message.author, user_text)
        asyncio.create_task(self.refresh_user_card(message, user_text))

        direct = self.direct_parser.parse(message, user_text)
        if direct:
            return direct

        low = (user_text or "").lower().strip()
        meta = self._meta(cid)
        last_action = (meta.get("last_action_type") or "").strip().lower()
        last_channel = (meta.get("last_target_channel_id") or "").strip()
        last_read_summary = (meta.get("last_read_summary") or "").strip()
        last_read_limit = int(meta.get("last_read_limit") or 0)
        last_read_anchor = (meta.get("last_read_first_message_id") or "").strip()

        continuation_phrases = ["continue", "more", "what else", "before that", "earlier", "а еще", "дальше", "что еще", "до этого", "раньше"]

        if any(p in low for p in continuation_phrases) and last_action == "read_channel" and last_channel:
             return {
                "action": "read_channel",
                "channel": last_channel,
                "limit": self._read_limit_from_text(low),
                "before": last_read_anchor,
            }

        if self._read_followup(low) and last_action == "read_channel" and last_channel:
            if any(k in low for k in ["сводк", "коротк", "обсуждали", "что там было", "что происходило", "что писали", "о чем говорили", "о чём говорили"]) and last_read_summary:
                if not any(k in low for k in continuation_phrases):
                    return {"action": "reply", "text": last_read_summary}

            return {
                "action": "read_channel",
                "channel": last_channel,
                "limit": self._read_limit_from_text(low),
                "before": last_read_anchor,
            }

        # Bounded retries for duplicate/echo prevention
        for attempt in range(2):
            msgs = self.build_chat_messages(message, user_text)
            raw = await self.llm.chat(msgs, temperature=0.3 + (attempt * 0.1), max_tokens=220, frequency_penalty=0.5, presence_penalty=0.3)
            if not raw:
                continue

            parsed = None
            if raw.strip().startswith("{"):
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None
            if not parsed:
                m = re.search(r"\{.*\}", raw, flags=re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = None

            if not parsed:
                cleaned = clean_response(raw)
                if cleaned and not self.is_duplicate_response(cid, cleaned) and not self.anchor_hit(cid, cleaned) and not self._looks_like_echo(cleaned, user_text, self.store.get_recent_history(cid, 5)):
                    return {"action": "reply", "text": cleaned}
                continue

            action = self._action_from_llm(parsed)
            if action.get("action") == "reply":
                txt = action.get("text") or ""
                if not txt or self.is_duplicate_response(cid, txt) or self.anchor_hit(cid, txt) or self._looks_like_echo(txt, user_text, self.store.get_recent_history(cid, 5)):
                    continue
            return action

        return {"action": "ignore"}

    async def compose_read_response(self, message: discord.Message, user_text: str, action: Dict[str, Any], observation: Dict[str, Any]) -> str:
        text = (observation.get("text") or "").strip()
        if not text:
            return "там пусто"
        channel_name = observation.get("channel_name") or (action.get("channel") or "канал")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        if not lines:
            return "там пусто"

        raw = await self.llm.chat(
            [
                {"role": "system", "content": COMPOSE_READ_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Запрос пользователя: {user_text}\n"
                        f"Канал: {channel_name}\n"
                        f"Наблюдение инструмента:\n{text}\n\n"
                        "Сделай короткий живой ответ по-русски. "
                        "Если это сводка — кратко опиши суть. Не пиши, что не можешь читать канал."
                    ),
                },
            ],
            temperature=0.35,
            max_tokens=120,
        )
        cleaned = sanitize_summary_text(raw or "") or clean_response(raw or "")
        if cleaned and not self._looks_like_echo(cleaned, user_text, []):
            return cleaned

        if len(lines) == 1:
            return f"Последнее в {channel_name}: {lines[0]}"
        if len(lines) <= 3:
            return f"В {channel_name} обсуждали: " + "; ".join(lines)
        return f"В {channel_name} обсуждали: " + "; ".join(lines[-3:])

    async def summarize_channel(self, channel_id: str, force: bool = False):
        meta = self.store.get_channel_meta(channel_id) or {}

        import time
        now = time.time()

        last_summary_ts = self._parse_db_timestamp(meta.get("summary_timestamp"))
        last_summary_count = int(meta.get("last_summary_count") or 0)
        current_count = int(meta.get("message_count") or 0)

        # Cache for 20 minutes unless significant activity (>10 new messages)
        if not force and last_summary_ts > 0:
            if now - last_summary_ts < 1200 and (current_count - last_summary_count) < 10:
                return

        recent = self.store.get_recent_history(channel_id, 30)
        recent_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        cur = (meta.get("summary") or "").strip()
        if not recent_text.strip():
            return

        raw = await self.llm.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Текущая сводка:\n{cur or 'нет'}\n\nПоследний диалог:\n{recent_text}\n\nНовая короткая сводка:"},
            ],
            temperature=0.2,
            max_tokens=250,
            frequency_penalty=0.1,
            presence_penalty=0.0,
        )

        summary = sanitize_summary_text(raw or "") or clean_response(raw or "")
        if not summary or self._is_bad_summary(summary):
            summary = self._fallback_summary_from_recent(recent)

        if summary and not self._is_bad_summary(summary):
            import datetime
            self.store.update_channel_meta(
                channel_id,
                summary=summary[: self.settings.max_summary_chars],
                summary_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                last_summary_count=current_count
            )

    def should_summarize(self, channel_id: str) -> bool:
        meta = self.store.get_channel_meta(channel_id) or {}
        return (int(meta.get("message_count") or 0) - int(meta.get("last_summary_count") or 0)) >= self.settings.summary_trigger_messages

    def _parse_db_timestamp(self, ts_str: str) -> float:
        if not ts_str:
            return 0.0
        import datetime
        try:
            return datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except Exception:
            try:
                return datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
            except Exception:
                return 0.0

    def maybe_autonomy_prompt(self, message: discord.Message):
        if not self.settings.autonomy_enabled or message.author.bot or isinstance(message.channel, discord.DMChannel):
            return None

        cid = str(message.channel.id)
        meta = self.store.get_channel_meta(cid) or {}

        # Anti-spam cooldowns
        import time
        now = time.time()

        last_autonomy_at = self._parse_db_timestamp(meta.get("last_autonomy_at"))
        last_interjection_at = self._parse_db_timestamp(meta.get("last_interjection_at"))
        last_emoji_at = self._parse_db_timestamp(meta.get("last_emoji_at"))

        if now - last_autonomy_at < 120: # 2 min global
            return None

        recent_rows = self.store.get_recent_history_rows(cid, 20)
        if not recent_rows:
            return None

        # Detect active conversation: >=5 messages, >=2 distinct users, within last 10 minutes
        active_msgs = []
        for r in reversed(recent_rows):
            ts = self._parse_db_timestamp(r.get("created_at") or "")
            if ts == 0.0:
                ts = now
            if now - ts > 600: # 10 min
                break
            active_msgs.append(r)

        if len(active_msgs) < 5:
            return None

        participants_ids = {r.get("speaker_id") for r in active_msgs if (r.get("role") or "").lower() == "user"}
        if len(participants_ids) < 2:
            return None

        # Build participants info
        participants = []
        seen = set()
        for r in active_msgs:
            if (r.get("role") or "").lower() != "user":
                continue
            sid = (r.get("speaker_id") or "").strip()
            sname = (r.get("speaker_name") or "").strip()
            key = sid or sname.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            participants.append(self.store.build_profile_card(subject_id=sid, subject_name=sname, max_facts=3))

        recent_text = "\n".join(
            f"{(r.get('speaker_name') or 'user')}: {(r.get('content') or '').strip()}"
            for r in reversed(active_msgs[-10:])
            if (r.get("content") or "").strip()
        )

        payload = [
            {"role": "system", "content": AUTONOMY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Последние сообщения канала:\n{recent_text}\n\n"
                    f"Участники:\n" + ("\n".join(f"- {p}" for p in participants) if participants else "- нет карточек") + "\n\n"
                    f"Cooldowns: interjection={int(now - last_interjection_at)}s, emoji={int(now - last_emoji_at)}s (min 1200s and 300s)\n"
                    "Выбери действие. Если interjection или reply — они должны быть очень короткими."
                ),
            },
        ]
        return payload, last_interjection_at, last_emoji_at

    async def run_autonomy(self, message: discord.Message) -> Dict[str, Any]:
        res = self.maybe_autonomy_prompt(message)
        if not res:
            return {"action": "ignore"}

        prompt, last_interjection_at, last_emoji_at = res
        import time
        now = time.time()

        raw = await self.llm.chat_json(prompt, temperature=0.3, max_tokens=140)
        action = (raw.get("action") or "ignore").strip().lower()

        # Probabilistic decide is mostly handled by LLM, but we enforce hard cooldowns here
        if action in {"short_interject", "contextual_reply", "reply"}:
            if now - last_interjection_at < 1200: # 20 min
                return {"action": "ignore"}
        elif action == "react":
            if now - last_emoji_at < 300: # 5 min
                return {"action": "ignore"}

        if action in {"reply", "short_interject", "contextual_reply"}:
            text = (raw.get("text") or raw.get("message") or "").strip()
            if text and not self._looks_like_echo(text, "", self.store.get_recent_history(str(message.channel.id), 5)):
                return {"action": "reply", "text": text}
            return {"action": "ignore"}

        if action == "react" and (raw.get("reaction") or "").strip():
            return {"action": "react", "reaction": raw.get("reaction").strip()}

        return {"action": "ignore"}
