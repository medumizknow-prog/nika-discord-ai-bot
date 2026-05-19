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
            "не могу посмотреть", "не может посмотреть", "не могу прочитать",
            "не может прочитать", "не могу видеть", "не может видеть",
            "не вижу канал", "не видит канал", "cannot read", "cannot view",
            "can't read", "can't view", "не знаю что там было",
            "не могу читать конкретные сообщения",
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
            if normalize_compare_text(variant) == norm:
                return recent[:-1]
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
        if not low or low in {"мм", "мм?", "м?", "а?", "э?", "эх?", "поняла", "понял", "ок", "ok"}:
            return True
        if len(low) < 2: return True

        # Guard thresholds: 0.86 for direct echo, 0.92 for repetition (restored)
        if is_too_similar(candidate, user_text, threshold=0.86):
            return True

        for row in (recent or []):
            if is_too_similar(candidate, row.get("content") or "", threshold=0.92):
                return True

        last_assistant = self._last_assistant_reply(recent)
        if last_assistant and is_too_similar(candidate, last_assistant, threshold=0.92):
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
                "кто начал", "после этого", "до конфликта", "кто ответил", "что ответил"
            ]
        )

    def _read_limit_from_text(self, low: str) -> int:
        if any(p in low for p in ["подроб", "полностью", "весь", "вся", "вся переписка", "развернуто", "детально"]):
            return 80
        if any(p in low for p in ["сводка", "что обсуждали", "что происходило", "о чем говорили"]):
            return 50
        if any(p in low for p in ["последнее", "last", "latest"]):
            return 1
        return 30

    async def refresh_user_card(self, message: discord.Message, user_text: str) -> None:
        if not message.guild or message.author.bot: return
        uid = str(message.author.id)
        cid = str(message.channel.id)
        self.store.ensure_profile(uid, username=message.author.name, display_name=message.author.display_name)
        card = self.store.get_user_card(uid) or {}

        seen = int(card.get("messages_seen") or 0)
        should_refresh = seen < 3 or (seen % 8 == 0)
        if not should_refresh: return

        recent_user = self.store.get_recent_user_history(uid, channel_id=cid, limit=8)
        recent_channel = self.store.get_recent_history_rows(cid, limit=12)
        if not recent_user and not recent_channel: return

        current_profile = self.store.build_profile_card(subject_id=uid, subject_name=message.author.display_name, max_facts=4)
        user_lines = [row.get("content", "").strip() for row in recent_user if row.get("content")]
        context_lines = [f"{row.get('role')} / {row.get('speaker_name')}: {row.get('content')}" for row in recent_channel if row.get("content")]

        payload = (
            f"Текущая карточка:\n{current_profile or 'нет'}\n\n"
            f"Последние сообщения пользователя:\n" + "\n".join(f"- {x}" for x in user_lines) +
            f"\n\nКонтекст канала:\n" + "\n".join(f"- {x}" for x in context_lines)
        )

        try:
            raw = await self.llm.chat_json(
                [{"role": "system", "content": USER_CARD_SYSTEM_PROMPT}, {"role": "user", "content": payload}],
                temperature=0.2, max_tokens=400
            )
        except Exception: return
        if not isinstance(raw, dict) or not raw: return

        updates: Dict[str, Any] = {
            "username": message.author.name,
            "display_name": message.author.display_name,
            "messages_seen": seen,
        }
        fields = [
            "summary", "personality_traits", "humor_style", "toxicity_level", "friendliness",
            "relationship", "relationship_trend", "recurring_topics", "inside_jokes",
            "nicknames", "bot_opinion", "confidence_score"
        ]
        for f in fields:
            val = raw.get(f)
            if val: updates[f] = sanitize_summary_text(str(val))

        self.store.update_user_card(uid, **updates)

    def detect_emotion(self, text: str) -> str:
        low = (text or "").lower()
        if any(w in low for w in NEGATIVE_WORDS): return "angry"
        if any(w in low for w in POSITIVE_WORDS): return "happy"
        if any(w in low for w in PLAYFUL_WORDS): return "hype"
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
        self.store.update_channel_meta(channel_id, mood=mood, energy=energy, sass=sass)
        return mood

    def update_affinity_from_message(self, author_id: str, user_text: str):
        low = user_text.lower()
        delta = 0
        if any(w in low for w in POSITIVE_WORDS): delta += 1
        if any(w in low for w in NEGATIVE_WORDS): delta -= 1
        if delta: self.store.adjust_affinity(author_id, delta)

    def learn_feedback(self, message: discord.Message, user_text: str):
        low = (user_text or "").lower().strip()
        if not low: return
        uid, cid = str(message.author.id), str(message.channel.id)
        pos = any(w in low for w in ["молодец", "умница", "круто", "класс", "хорошо", "топ"])
        neg = any(w in low for w in ["плохо", "криво", "тупо", "кринж", "ошибка"])
        if pos or neg:
            rating = 1 if pos and not neg else -1 if neg and not pos else 0
            if rating:
                self.store.add_feedback(uid, cid, "chat_feedback", rating, "tone", low, user_text)
                self.store.adjust_affinity(uid, rating)

    def maybe_learn_explicit_memory(self, author: discord.Member, user_text: str):
        low = user_text.lower().strip()
        if low.startswith("запомни ") or low.startswith("remember "):
            payload = user_text.split(" ", 1)[1].strip() if " " in user_text else ""
            if payload:
                self.store.upsert_fact(str(author.id), author.display_name, "note", payload, confidence=0.95, source="explicit")
                self.store.adjust_affinity(str(author.id), 1)

    def build_snapshot(self, message: discord.Message, user_text: str) -> str:
        cid = str(message.channel.id)
        meta = self._meta(cid)
        author_id, author_name = str(message.author.id), message.author.display_name
        user_card = self.store.build_profile_card(subject_id=author_id, subject_name=author_name, max_facts=6)

        lines = [
            "КОНТЕКСТ ДЛЯ Nika:",
            f"Текущее настроение канала: {meta.get('mood', 'calm')}",
            f"Энергия: {float(meta.get('energy', 0.5)):.2f}, Sass: {float(meta.get('sass', 0.5)):.2f}",
            f"Эмоция сообщения: {self.detect_emotion(user_text)}",
            f"ПАМЯТЬ ОБ АВТОРЕ:\n{user_card}",
        ]

        summary = self.store.get_summary(cid)
        if summary: lines += ["СВОДКА КАНАЛА:", summary]

        att = attachment_summary(message.attachments)
        if att: lines += ["ВЛОЖЕНИЯ:", att]

        return "\n".join(lines)[:self.settings.max_context_chars]

    def build_chat_messages(self, message: discord.Message, user_text: str):
        cid = str(message.channel.id)
        meta = self._meta(cid)
        system = build_system_prompt(meta.get("mood", "calm"), float(meta.get("energy", 0.5)), float(meta.get("sass", 0.5)), capability_lines="- personality chat mode")
        msgs = [
            {"role": "system", "content": system},
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "system", "content": self.build_snapshot(message, user_text)},
        ]
        recent = self.store.get_recent_history(cid, self.settings.max_recent_turns)
        msgs.extend(self._trim_recent_for_current_user(recent, user_text))
        msgs.append({"role": "user", "content": self._prompt_user_text(message, user_text)})
        return msgs

    def build_action_messages(self, message: discord.Message, user_text: str):
        cid = str(message.channel.id)
        meta = self._meta(cid)
        system = build_system_prompt(meta.get("mood", "calm"), float(meta.get("energy", 0.5)), float(meta.get("sass", 0.5)), capability_lines="- command parser mode")
        msgs = [
            {"role": "system", "content": system},
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "system", "content": self.build_snapshot(message, user_text)},
        ]
        recent = self.store.get_recent_history(cid, self.settings.max_recent_turns)
        msgs.extend(self._trim_recent_for_current_user(recent, user_text))
        msgs.append({"role": "user", "content": self._prompt_user_text(message, user_text)})
        return msgs

    def is_duplicate_response(self, channel_id: str, text: str) -> bool:
        st = self._state(channel_id)
        norm = normalize_compare_text(text)
        if norm and norm in st["recent"]: return True
        if norm:
            st["recent"].append(norm)
            if len(st["recent"]) > 4: st["recent"].pop(0)
        return False

    def _action_from_llm(self, obj: dict) -> dict:
        action = (obj.get("action") or "ignore").strip().lower()
        if action in {"reply", "short_interject", "contextual_reply", "sarcastic_comment", "playful_question", "meme_reply"}:
            return {"action": action, "text": (obj.get("text") or obj.get("message") or "").strip()}
        if action == "remember":
            return {"action": "remember", "key": "note", "value": (obj.get("value") or obj.get("text") or "").strip()}
        if action == "read_channel":
            return {"action": "read_channel", "channel": (obj.get("channel") or "").strip(), "limit": int(obj.get("limit") or 30), "before": (obj.get("before") or "").strip()}
        if action == "send_message":
            return {"action": "send_message", "channel": (obj.get("channel") or "").strip(), "text": (obj.get("text") or "").strip()}
        if action == "ping_user":
            return {"action": "ping_user", "channel": (obj.get("channel") or "").strip(), "user": (obj.get("user") or "").strip(), "text": (obj.get("text") or "").strip()}
        if action == "react":
            return {"action": "react", "channel": (obj.get("channel") or "").strip(), "reaction": (obj.get("reaction") or "").strip()}
        return {"action": "ignore"}

    async def decide(self, message: discord.Message, user_text: str) -> Dict[str, Any]:
        cid, author_id = str(message.channel.id), str(message.author.id)
        self.store.ensure_profile(author_id, username=message.author.name, display_name=message.author.display_name)
        self.update_affinity_from_message(author_id, user_text)
        self.update_mood(cid, user_text, author_id=author_id)
        self.maybe_learn_explicit_memory(message.author, user_text)
        asyncio.create_task(self.refresh_user_card(message, user_text))

        low = (user_text or "").lower().strip()
        direct = self.direct_parser.parse(message, user_text)
        if direct:
            if direct.get("action") == "reply" and any(k in low for k in ["не сюда", "в другой", "перепутал"]):
                 return {"action": "ignore"}
            return direct

        meta = self._meta(cid)
        last_action, last_channel = (meta.get("last_action_type") or "").strip().lower(), (meta.get("last_target_channel_id") or "").strip()
        last_read_summary, last_read_anchor = (meta.get("last_read_summary") or "").strip(), (meta.get("last_read_first_message_id") or "").strip()

        if self._read_followup(low) and last_action == "read_channel" and last_channel:
            structured = any(k in low for k in ["кто начал", "конфликт", "кто ответил"])
            if any(k in low for k in ["сводк", "коротк", "обсуждали"]) and last_read_summary and not structured:
                return {"action": "reply", "text": last_read_summary}
            return {"action": "read_channel", "channel": last_channel, "limit": 80 if structured else self._read_limit_from_text(low), "before": last_read_anchor if any(k in low for k in ["до этого", "раньше", "before", "earlier"]) else ""}

        for attempt in range(2):
            msgs = self.build_chat_messages(message, user_text)
            raw_llm = await self.llm.chat(msgs, temperature=0.3 + (attempt * 0.1), frequency_penalty=0.5, presence_penalty=0.3)
            if not raw_llm: continue

            parsed = None
            try:
                parsed = json.loads(raw_llm)
            except Exception:
                m = re.search(r"\{.*\}", raw_llm, flags=re.S)
                if m:
                    try: parsed = json.loads(m.group(0))
                    except Exception: parsed = None

            if not parsed:
                cleaned = clean_response(raw_llm)
                if cleaned and not self.is_duplicate_response(cid, cleaned) and not self.anchor_hit(cid, cleaned) and not self._looks_like_echo(cleaned, user_text, self.store.get_recent_history(cid, 5)):
                    return {"action": "reply", "text": cleaned}
                continue

            action = self._action_from_llm(parsed)
            if action.get("action") in {"reply", "short_interject", "contextual_reply"}:
                txt = action.get("text") or ""
                if not txt or self.is_duplicate_response(cid, txt) or self.anchor_hit(cid, txt) or self._looks_like_echo(txt, user_text, self.store.get_recent_history(cid, 5)):
                    continue
            return action
        return {"action": "ignore"}

    def anchor_hit(self, channel_id: str, text: str) -> bool:
        st = self._state(channel_id)
        low = (text or "").lower()
        hits = [a for a in ["вирус", "шашлык", "коты", "красный мозг", "шарик", "пинг-понг"] if a in low]
        if hits:
            st["anchors"].extend(hits)
            counts = {a: st["anchors"].count(a) for a in set(st["anchors"])}
            return any(v >= 3 for v in counts.values())
        return False

    async def compose_read_response(self, message: discord.Message, user_text: str, action: Dict[str, Any], observation: Dict[str, Any]) -> str:
        text = (observation.get("text") or "").strip()
        if not text: return "там пусто"
        cid, name = observation.get("channel_id") or "", observation.get("channel_name") or (action.get("channel") or "канал")
        meta = self._meta(cid) if cid else {}
        ctx = f"Темы: {meta.get('summary_topics')}\nУчастники: {meta.get('summary_participants')}\nАтмосфера: {meta.get('summary_mood')}\n" if meta.get("summary_topics") else ""
        raw = await self.llm.chat([{"role": "system", "content": COMPOSE_READ_PROMPT}, {"role": "user", "content": f"Запрос: {user_text}\nКанал: {name}\nКонтекст: {ctx}\nНаблюдение:\n{text}"}], temperature=0.35, max_tokens=250)
        cleaned = sanitize_summary_text(raw or "") or clean_response(raw or "")
        if cleaned and not self._looks_like_echo(cleaned, user_text, []): return cleaned
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) == 1: return f"Последнее в {name}: {lines[0]}"
        return f"В {name} обсуждали: " + "; ".join(lines[-3:])

    async def summarize_channel(self, channel_id: str, force: bool = False):
        meta = self.store.get_channel_meta(channel_id) or {}
        now = time.time()
        last_summary_ts = self._parse_db_timestamp(meta.get("summary_timestamp"))
        last_count, current_count = int(meta.get("last_summary_count") or 0), int(meta.get("message_count") or 0)
        if not force and last_summary_ts > 0 and (now - last_summary_ts < 1200) and (current_count - last_count < 10): return

        recent = self.store.get_recent_history(channel_id, 50)
        recent_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        if not recent_text.strip(): return
        try:
            raw = await self.llm.chat_json([{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}, {"role": "user", "content": f"История:\n{recent_text}"}], temperature=0.2, max_tokens=500)
        except Exception: return
        if not raw or not isinstance(raw, dict): return

        brief = sanitize_summary_text(raw.get("brief_summary") or "")
        if not brief or self._is_bad_summary(brief): brief = self._fallback_summary_from_recent(recent)
        self.store.update_channel_meta(
            channel_id, summary=brief[:self.settings.max_summary_chars], summary_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), last_summary_count=current_count,
            summary_participants=", ".join(raw.get("participants") or []), summary_topics=", ".join(raw.get("topics") or []), summary_mood=str(raw.get("mood") or ""),
            summary_jokes=", ".join(raw.get("jokes") or []), summary_conflicts=", ".join(raw.get("conflicts") or []), summary_events=", ".join(raw.get("events") or []), summary_unresolved=", ".join(raw.get("unresolved") or [])
        )

    def should_summarize(self, channel_id: str) -> bool:
        meta = self._meta(channel_id)
        return (int(meta.get("message_count") or 0) - int(meta.get("last_summary_count") or 0)) >= self.settings.summary_trigger_messages

    def _parse_db_timestamp(self, ts_str: str) -> float:
        if not ts_str: return 0.0
        try: return datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except Exception:
            try: return datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
            except Exception: return 0.0

    def maybe_autonomy_prompt(self, message: discord.Message):
        if not self.settings.autonomy_enabled or message.author.bot or isinstance(message.channel, discord.DMChannel): return None
        cid = str(message.channel.id)
        meta = self._meta(cid)
        now = time.time()
        last_auto = self._parse_db_timestamp(meta.get("last_autonomy_at"))
        if now - last_auto < self.settings.autonomy_global_cooldown: return None

        recent_rows = self.store.get_recent_history_rows(cid, 30)
        if not recent_rows: return None
        active_msgs = []
        for r in reversed(recent_rows):
            ts = self._parse_db_timestamp(r.get("created_at") or "")
            if ts == 0.0: ts = now
            if now - ts > self.settings.autonomy_active_window_sec: break
            active_msgs.append(r)

        if len(active_msgs) < self.settings.autonomy_min_messages: return None
        participants_ids = {r.get("speaker_id") for r in active_msgs if (r.get("role") or "").lower() == "user"}
        if len(participants_ids) < self.settings.autonomy_min_users: return None

        participants = []
        seen = set()
        for r in reversed(active_msgs):
            if (r.get("role") or "").lower() != "user": continue
            key = (r.get("speaker_id") or "").strip() or (r.get("speaker_name") or "").strip().lower()
            if not key or key in seen: continue
            seen.add(key)
            participants.append(self.store.build_profile_card(subject_id=r.get("speaker_id"), subject_name=r.get("speaker_name"), max_facts=3))

        recent_text = "\n".join(f"{(r.get('speaker_name') or 'user')}: {(r.get('content') or '').strip()}" for r in reversed(active_msgs[-10:]))
        last_interjection = self._parse_db_timestamp(meta.get("last_interjection_at"))
        last_emoji = self._parse_db_timestamp(meta.get("last_emoji_at"))

        payload = [{"role": "system", "content": AUTONOMY_SYSTEM_PROMPT}, {"role": "user", "content": f"Последние сообщения:\n{recent_text}\n\nУчастники:\n" + ("\n".join(f"- {p}" for p in participants) if participants else "- нет") + f"\n\nCooldowns: interjection={int(now - last_interjection)}s, emoji={int(now - last_emoji)}s (min {self.settings.autonomy_interjection_cooldown}s and {self.settings.autonomy_emoji_cooldown}s)"}]
        return payload, last_interjection, last_emoji

    async def run_autonomy(self, message: discord.Message) -> Dict[str, Any]:
        res = self.maybe_autonomy_prompt(message)
        if not res: return {"action": "ignore"}
        prompt, last_interjection_at, last_emoji_at = res
        now = time.time()
        try:
            raw = await self.llm.chat_json(prompt, temperature=0.4, max_tokens=250)
        except Exception: return {"action": "ignore"}
        action = (raw.get("action") or "ignore").strip().lower()
        if self.settings.autonomy_debug: print(f"[AUTONOMY DEBUG] Action: {action}, Reasoning: {raw.get('reasoning')}")

        if action in {"short_interject", "contextual_reply", "reply", "sarcastic_comment", "playful_question", "meme_reply"}:
            if now - last_interjection_at < self.settings.autonomy_interjection_cooldown: return {"action": "ignore"}
            text = (raw.get("text") or raw.get("message") or "").strip()
            if text and not self._looks_like_echo(text, "", self.store.get_recent_history(str(message.channel.id), 5)):
                return {"action": action, "text": text}
        elif action == "react":
            if now - last_emoji_at < self.settings.autonomy_emoji_cooldown: return {"action": "ignore"}
            if (raw.get("reaction") or "").strip(): return {"action": "react", "reaction": raw.get("reaction").strip()}
        return {"action": "ignore"}
