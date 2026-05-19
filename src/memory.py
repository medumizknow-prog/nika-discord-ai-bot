from __future__ import annotations

import sqlite3
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .text_utils import clean_response, sanitize_summary_text, strip_output_labels


@dataclass
class ProfileCard:
    user_id: str
    username: str
    display_name: str
    preferred_name: str = ""
    relationship: str = ""
    traits: str = ""
    notes: str = ""
    affinity: int = 0
    last_seen: str = ""


class MemoryStore:
    def __init__(self, db_file: str):
        self.db = sqlite3.connect(db_file)
        self.db.row_factory = sqlite3.Row
        self.cur = self.db.cursor()
        self._migrate()

    def _row_to_dict(self, row):
        if row is None: return None
        try:
            if hasattr(row, "keys"): return {k: row[k] for k in row.keys()}
        except Exception: pass
        try: return dict(row)
        except Exception:
            try: return {k: row[k] for k in row.keys()}
            except Exception: return None

    def _get_version(self) -> int:
        try:
            row = self.cur.execute("SELECT version FROM schema_version").fetchone()
            return int(row["version"]) if row else 0
        except sqlite3.OperationalError: return 0

    def _set_version(self, version: int):
        self.cur.execute("DELETE FROM schema_version")
        self.cur.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        self.db.commit()

    def _migrate(self):
        self.cur.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
        version = self._get_version()
        if version < 1:
            self._migrate_v1()
            version = 1
            self._set_version(version)
        if version < 2:
            self._migrate_v2()
            version = 2
            self._set_version(version)

    def _migrate_v1(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            guild_id TEXT DEFAULT '',
            role TEXT NOT NULL,
            speaker_id TEXT DEFAULT '',
            speaker_name TEXT NOT NULL,
            content TEXT NOT NULL,
            attachments TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS user_cards (
            user_id TEXT PRIMARY KEY,
            username TEXT DEFAULT '',
            display_name TEXT DEFAULT '',
            summary TEXT DEFAULT '',
            traits TEXT DEFAULT '',
            relationship TEXT DEFAULT '',
            relationship_trend TEXT DEFAULT '',
            opinion TEXT DEFAULT '',
            topics TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            affinity INTEGER DEFAULT 0,
            messages_seen INTEGER DEFAULT 0,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id TEXT DEFAULT '',
            subject_name TEXT DEFAULT '',
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 0.7,
            source TEXT DEFAULT 'chat',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS prefs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            weight REAL DEFAULT 0.0,
            source TEXT DEFAULT 'feedback',
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, key, value)
        )""")
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT DEFAULT '',
            guild_id TEXT DEFAULT '',
            user_id TEXT DEFAULT '',
            user_name TEXT DEFAULT '',
            episode_type TEXT DEFAULT 'message',
            summary TEXT NOT NULL,
            importance REAL DEFAULT 0.5,
            emotion TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            channel_id TEXT DEFAULT '',
            action TEXT DEFAULT '',
            rating INTEGER DEFAULT 0,
            aspect TEXT DEFAULT '',
            note TEXT DEFAULT '',
            source_text TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS channel_meta (
            channel_id TEXT PRIMARY KEY,
            summary TEXT DEFAULT '',
            summary_timestamp DATETIME,
            message_count INTEGER DEFAULT 0,
            last_summary_count INTEGER DEFAULT 0,
            mood TEXT DEFAULT 'calm',
            energy REAL DEFAULT 0.5,
            sass REAL DEFAULT 0.5,
            last_target_channel_id TEXT DEFAULT '',
            last_target_user_id TEXT DEFAULT '',
            last_action_type TEXT DEFAULT '',
            last_reaction TEXT DEFAULT '',
            last_bot_post_count INTEGER DEFAULT 0,
            last_read_limit INTEGER DEFAULT 0,
            last_read_anchor_message_id TEXT DEFAULT '',
            last_read_first_message_id TEXT DEFAULT '',
            last_read_last_message_id TEXT DEFAULT '',
            last_read_summary TEXT DEFAULT '',
            last_autonomy_count INTEGER DEFAULT 0,
            last_autonomy_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_interjection_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_emoji_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_interjection_type TEXT DEFAULT '',
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            username TEXT DEFAULT '',
            display_name TEXT DEFAULT '',
            preferred_name TEXT DEFAULT '',
            relationship TEXT DEFAULT '',
            traits TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            affinity INTEGER DEFAULT 0,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        self.db.commit()

    def _migrate_v2(self):
        cols = {r["name"] for r in self.cur.execute("PRAGMA table_info(user_cards)").fetchall()}
        for col in ["personality_traits", "humor_style", "toxicity_level", "friendliness", "inside_jokes", "nicknames", "bot_opinion", "confidence_score", "recurring_topics"]:
            if col not in cols: self.cur.execute(f"ALTER TABLE user_cards ADD COLUMN {col} TEXT DEFAULT ''")
        cols_meta = {r["name"] for r in self.cur.execute("PRAGMA table_info(channel_meta)").fetchall()}
        for col in ["summary_participants", "summary_topics", "summary_mood", "summary_jokes", "summary_conflicts", "summary_events", "summary_unresolved"]:
            if col not in cols_meta: self.cur.execute(f"ALTER TABLE channel_meta ADD COLUMN {col} TEXT DEFAULT ''")
        self.db.commit()

    def ensure_profile(self, user_id: str, username: str = "", display_name: str = "") -> None:
        self.cur.execute("INSERT INTO profiles (user_id, username, display_name, last_seen) VALUES (?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT(user_id) DO UPDATE SET username=excluded.username, display_name=excluded.display_name, last_seen=CURRENT_TIMESTAMP", (user_id, username or "", display_name or ""))
        self.db.commit()

    def get_profile(self, user_id: str) -> Optional[ProfileCard]:
        row = self.cur.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        if not row: return None
        return ProfileCard(user_id=row["user_id"], username=row["username"] or "", display_name=row["display_name"] or "", preferred_name=row["preferred_name"] or "", relationship=row["relationship"] or "", traits=row["traits"] or "", notes=row["notes"] or "", affinity=int(row["affinity"] or 0), last_seen=row["last_seen"] or "")

    def upsert_profile_fields(self, user_id: str, **fields: Any) -> None:
        if not fields: return
        allowed = {"username", "display_name", "preferred_name", "relationship", "traits", "notes", "affinity", "last_seen"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates: return
        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [user_id]
        self.cur.execute(f"UPDATE profiles SET {sets}, last_seen = CURRENT_TIMESTAMP WHERE user_id = ?", values)
        self.db.commit()

    def adjust_affinity(self, user_id: str, delta: int) -> None:
        self.cur.execute("UPDATE profiles SET affinity = COALESCE(affinity, 0) + ?, last_seen = CURRENT_TIMESTAMP WHERE user_id = ?", (delta, user_id))
        self.db.commit()

    def ensure_user_card(self, user_id: str, username: str = "", display_name: str = "") -> None:
        self.cur.execute("INSERT INTO user_cards (user_id, username, display_name, last_seen) VALUES (?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT(user_id) DO UPDATE SET username=excluded.username, display_name=excluded.display_name, last_seen=CURRENT_TIMESTAMP", (user_id, username or "", display_name or ""))
        self.db.commit()

    def bump_user_card(self, user_id: str, username: str = "", display_name: str = "") -> Dict[str, Any]:
        self.ensure_user_card(user_id, username=username, display_name=display_name)
        self.cur.execute("UPDATE user_cards SET username = COALESCE(?, username), display_name = COALESCE(?, display_name), messages_seen = COALESCE(messages_seen, 0) + 1, last_seen = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?", (username or None, display_name or None, user_id))
        self.db.commit()
        return self.get_user_card(user_id) or {}

    def get_user_card(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._row_to_dict(self.cur.execute("SELECT * FROM user_cards WHERE user_id = ?", (user_id,)).fetchone())

    def find_user_card_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self._row_to_dict(self.cur.execute("SELECT * FROM user_cards WHERE LOWER(display_name) = LOWER(?) OR LOWER(username) = LOWER(?) LIMIT 1", (name, name)).fetchone())

    def update_user_card(self, user_id: str, **fields: Any) -> None:
        allowed = {"username", "display_name", "summary", "traits", "relationship", "relationship_trend", "opinion", "topics", "notes", "affinity", "messages_seen", "last_seen", "personality_traits", "humor_style", "toxicity_level", "friendliness", "inside_jokes", "nicknames", "bot_opinion", "confidence_score", "recurring_topics"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates: return
        self.ensure_user_card(user_id, username=updates.get("username", ""), display_name=updates.get("display_name", ""))
        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [user_id]
        self.cur.execute(f"UPDATE user_cards SET {sets}, updated_at = CURRENT_TIMESTAMP, last_seen = CURRENT_TIMESTAMP WHERE user_id = ?", values)
        self.db.commit()

    def append_user_card_note(self, user_id: str, note: str) -> None:
        note = (note or "").strip()
        if not note: return
        card = self.get_user_card(user_id) or {}
        existing = (card.get("notes") or "").strip()
        if note.lower() in existing.lower(): return
        self.update_user_card(user_id, notes=f"{existing}; {note}".strip("; ").strip())

    def upsert_fact(self, subject_id: str, subject_name: str, key: str, value: str, confidence: float = 0.7, source: str = "chat") -> None:
        key, value = (key or "").strip(), (value or "").strip()
        if not key or not value: return
        existing = self.cur.execute("SELECT id FROM facts WHERE subject_id = ? AND subject_name = ? AND key = ? AND value = ?", (subject_id or "", subject_name or "", key, value)).fetchone()
        if existing: self.cur.execute("UPDATE facts SET confidence = ?, source = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (confidence, source, existing["id"]))
        else: self.cur.execute("INSERT INTO facts (subject_id, subject_name, key, value, confidence, source) VALUES (?, ?, ?, ?, ?, ?)", (subject_id or "", subject_name or "", key, value, confidence, source))
        self.db.commit()

    def add_episode(self, channel_id, guild_id, user_id, user_name, summary, episode_type="message", importance=0.5, emotion="") -> None:
        if not summary or not summary.strip(): return
        self.cur.execute("INSERT INTO episodes (channel_id, guild_id, user_id, user_name, episode_type, summary, importance, emotion) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (channel_id or "", guild_id or "", user_id or "", user_name or "", episode_type, summary.strip(), importance, emotion))
        self.db.commit()

    def get_recent_episodes(self, channel_id="", user_id="", limit=8):
        clauses, params = [], []
        if channel_id: clauses.append("channel_id = ?"); params.append(channel_id)
        if user_id: clauses.append("user_id = ?"); params.append(user_id)
        if not clauses: return []
        return self.cur.execute(f"SELECT * FROM episodes WHERE {' OR '.join(clauses)} ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()

    def get_facts(self, subject_id="", subject_name="", limit=6):
        clauses, params = [], []
        if subject_id: clauses.append("subject_id = ?"); params.append(subject_id)
        if subject_name: clauses.append("LOWER(subject_name) = LOWER(?)"); params.append(subject_name)
        if not clauses: return []
        return self.cur.execute(f"SELECT * FROM facts WHERE {' OR '.join(clauses)} ORDER BY confidence DESC, updated_at DESC LIMIT ?", (*params, limit)).fetchall()

    def build_known_names(self) -> List[str]:
        names = set()
        for table in ["profiles", "user_cards"]:
            for row in self.cur.execute(f"SELECT display_name, username FROM {table}").fetchall():
                for v in [row["display_name"], row["username"]]:
                    if v: names.add(v.strip())
        return sorted(n for n in names if n)

    def add_message(self, channel_id, guild_id, role, speaker_id, speaker_name, content, attachments="") -> None:
        text = strip_output_labels((content or "").strip())
        if role == "assistant":
            cl = clean_response(text)
            if cl: text = cl
        self.cur.execute("INSERT INTO history (channel_id, guild_id, role, speaker_id, speaker_name, content, attachments) VALUES (?, ?, ?, ?, ?, ?, ?)", (channel_id, guild_id, role, speaker_id or "", speaker_name or "", text or "", attachments or ""))
        self.cur.execute("INSERT INTO channel_meta (channel_id, message_count, updated_at) VALUES (?, 1, CURRENT_TIMESTAMP) ON CONFLICT(channel_id) DO UPDATE SET message_count = COALESCE(message_count, 0) + 1, updated_at = CURRENT_TIMESTAMP", (channel_id,))
        self.db.commit()

    def get_recent_history_rows(self, channel_id, limit=8):
        return [self._row_to_dict(r) for r in self.cur.execute("SELECT role, speaker_id, speaker_name, content, attachments, id, created_at FROM history WHERE channel_id = ? ORDER BY id DESC LIMIT ?", (channel_id, limit)).fetchall()[::-1]]

    def get_recent_user_history(self, user_id, channel_id="", limit=8):
        sql = "SELECT role, speaker_id, speaker_name, content, attachments, channel_id, id FROM history WHERE speaker_id = ?"
        params = [user_id]
        if channel_id: sql += " AND channel_id = ?"; params.append(channel_id)
        return [self._row_to_dict(r) for r in self.cur.execute(sql + " ORDER BY id DESC LIMIT ?", tuple(params)).fetchall()[::-1]]

    def get_recent_history(self, channel_id, limit=8):
        rows = self.get_recent_history_rows(channel_id, limit=limit)
        out = []
        for r in rows:
            content, atts = (r["content"] or "").strip(), (r.get("attachments") or "").strip()
            if atts: content = f"{content}\n[attachments]\n{atts}" if content else f"[attachments]\n{atts}"
            role, speaker = (r.get("role") or "user").strip().lower(), (r.get("speaker_name") or "").strip()
            if role == "assistant": out.append({"role": "assistant", "content": content})
            elif role == "system": out.append({"role": "system", "content": content})
            elif speaker and content: out.append({"role": "user", "content": f"{speaker}: {content}"})
            else: out.append({"role": "user", "content": content})
        return out

    def build_profile_card(self, subject_id="", subject_name="", max_facts=4) -> str:
        profile = self.get_profile(subject_id) if subject_id else None
        if profile is None and subject_name:
            row = self.cur.execute("SELECT * FROM profiles WHERE LOWER(display_name) = LOWER(?) OR LOWER(preferred_name) = LOWER(?) OR LOWER(username) = LOWER(?) LIMIT 1", (subject_name, subject_name, subject_name)).fetchone()
            if row: profile = ProfileCard(user_id=row["user_id"], username=row["username"] or "", display_name=row["display_name"] or "", preferred_name=row["preferred_name"] or "", relationship=row["relationship"] or "", traits=row["traits"] or "", notes=row["notes"] or "", affinity=int(row["affinity"] or 0), last_seen=row["last_seen"] or "")
        user_card = self.get_user_card(subject_id) if subject_id else (self.find_user_card_by_name(subject_name) if subject_name else None)
        facts = self.get_facts(subject_id=subject_id, subject_name=subject_name, limit=max_facts)
        episodes = self.get_recent_episodes(user_id=subject_id, limit=3) if subject_id else []

        lines = []
        label = subject_name or (profile.display_name if profile else subject_id) or ""
        if profile:
            lines.append(f"Имя: {profile.display_name or profile.username or label}")
            if profile.preferred_name: lines.append(f"Как звать: {profile.preferred_name}")
            lines.append(f"Affinity: {profile.affinity}")
            if profile.traits: lines.append(f"Черты: {profile.traits}")
            if profile.notes: lines.append(f"Заметки: {profile.notes}")
        elif label: lines.append(f"Имя: {label}")
        if user_card:
            for f in ["summary", "traits", "relationship", "relationship_trend", "personality_traits", "humor_style", "toxicity_level", "friendliness", "inside_jokes", "nicknames", "bot_opinion", "confidence_score"]:
                if user_card.get(f): lines.append(f"{f.capitalize()}: {user_card[f]}")
        for fact in facts: lines.append(f"{fact['key']}: {fact['value']}")
        for ep in episodes: lines.append(f"Событие: {ep['summary']}")
        return "\n".join(lines).strip()

    def get_user_prefs(self, user_id, limit=8):
        return self.cur.execute("SELECT * FROM prefs WHERE user_id = ? ORDER BY weight DESC LIMIT ?", (user_id, limit)).fetchall()

    def add_feedback(self, uid, cid, action, rating, aspect, note, text):
        self.cur.execute("INSERT INTO feedback (user_id, channel_id, action, rating, aspect, note, source_text) VALUES (?, ?, ?, ?, ?, ?, ?)", (uid, cid, action, rating, aspect, note, text))
        self.db.commit()

    def get_channel_meta(self, cid):
        row = self.cur.execute("SELECT * FROM channel_meta WHERE channel_id = ?", (cid,)).fetchone()
        if not row:
            self.cur.execute("INSERT INTO channel_meta (channel_id) VALUES (?) ON CONFLICT(channel_id) DO NOTHING", (cid,))
            self.db.commit()
            row = self.cur.execute("SELECT * FROM channel_meta WHERE channel_id = ?", (cid,)).fetchone()
        return self._row_to_dict(row)

    def update_channel_meta(self, cid, **fields):
        allowed = {"summary", "summary_timestamp", "message_count", "last_summary_count", "mood", "energy", "sass", "last_target_channel_id", "last_target_user_id", "last_action_type", "last_reaction", "last_bot_post_count", "last_read_limit", "last_read_anchor_message_id", "last_read_first_message_id", "last_read_last_message_id", "last_read_summary", "last_autonomy_count", "last_autonomy_at", "last_interjection_at", "last_emoji_at", "last_interjection_type", "updated_at", "summary_participants", "summary_topics", "summary_mood", "summary_jokes", "summary_conflicts", "summary_events", "summary_unresolved"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates: return
        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        self.cur.execute(f"UPDATE channel_meta SET {sets}, updated_at = CURRENT_TIMESTAMP WHERE channel_id = ?", list(updates.values()) + [cid])
        self.db.commit()

    def get_summary(self, cid):
        row = self.get_channel_meta(cid) or {}
        return sanitize_summary_text(strip_output_labels((row.get("summary") or "").strip())) or ""

    def record_read_state(self, cid, target_channel_id="", limit=0, anchor_message_id="", first_message_id="", last_message_id=""):
        self.update_channel_meta(cid, last_action_type="read_channel", last_target_channel_id=target_channel_id, last_read_limit=int(limit), last_read_anchor_message_id=anchor_message_id, last_read_first_message_id=first_message_id, last_read_last_message_id=last_message_id)

    def record_autonomy_state(self, cid, action_type="", count=0, interjection_type=""):
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        updates = {"last_autonomy_count": int(count), "last_autonomy_at": now_ts, "last_interjection_type": interjection_type or ""}
        if action_type: updates["last_action_type"] = action_type
        if interjection_type in {"reply", "short_interject", "contextual_reply", "sarcastic_comment", "playful_question", "meme_reply"}: updates["last_interjection_at"] = now_ts
        elif interjection_type == "react": updates["last_emoji_at"] = now_ts
        self.update_channel_meta(cid, **updates)
