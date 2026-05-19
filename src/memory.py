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
        if row is None:
            return None
        try:
            if hasattr(row, "keys"):
                return {k: row[k] for k in row.keys()}
        except Exception:
            pass
        try:
            return dict(row)
        except Exception:
            try:
                return {k: row[k] for k in row.keys()}
            except Exception:
                return None

    def _get_version(self) -> int:
        try:
            row = self.cur.execute("SELECT version FROM schema_version").fetchone()
            return int(row["version"]) if row else 0
        except sqlite3.OperationalError:
            return 0

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
        )
        """)
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS user_cards (
            user_id TEXT PRIMARY KEY,
            username TEXT DEFAULT '',
            display_name TEXT DEFAULT '',
            summary TEXT DEFAULT '',
            interests TEXT DEFAULT '',
            communication_style TEXT DEFAULT '',
            traits TEXT DEFAULT '',
            relationship TEXT DEFAULT '',
            relationship_trend TEXT DEFAULT '',
            opinion TEXT DEFAULT '',
            topics TEXT DEFAULT '',
            activity_level TEXT DEFAULT '',
            behaviors TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            affinity INTEGER DEFAULT 0,
            messages_seen INTEGER DEFAULT 0,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
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
        )
        """)
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
        )
        """)
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
        )
        """)
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
        )
        """)
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
        )
        """)
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
        )
        """)
        self.db.commit()

    def _migrate_v2(self):
        cols = {r["name"] for r in self.cur.execute("PRAGMA table_info(user_cards)").fetchall()}
        for col in [
            "personality_traits", "humor_style", "toxicity_level", "friendliness",
            "inside_jokes", "nicknames", "bot_opinion", "confidence_score"
        ]:
            if col not in cols:
                self.cur.execute(f"ALTER TABLE user_cards ADD COLUMN {col} TEXT DEFAULT ''")

        cols = {r["name"] for r in self.cur.execute("PRAGMA table_info(channel_meta)").fetchall()}
        for col in [
            "summary_participants", "summary_topics", "summary_mood",
            "summary_jokes", "summary_conflicts", "summary_events", "summary_unresolved"
        ]:
            if col not in cols:
                self.cur.execute(f"ALTER TABLE channel_meta ADD COLUMN {col} TEXT DEFAULT ''")
        self.db.commit()

    def ensure_profile(self, user_id: str, username: str = "", display_name: str = "") -> None:
        self.cur.execute(
            """
            INSERT INTO profiles (user_id, username, display_name, last_seen)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                username=excluded.username,
                display_name=excluded.display_name,
                last_seen=CURRENT_TIMESTAMP
            """,
            (user_id, username or "", display_name or ""),
        )
        self.db.commit()

    def get_profile(self, user_id: str) -> Optional[ProfileCard]:
        row = self.cur.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return ProfileCard(
            user_id=row["user_id"],
            username=row["username"] or "",
            display_name=row["display_name"] or "",
            preferred_name=row["preferred_name"] or "",
            relationship=row["relationship"] or "",
            traits=row["traits"] or "",
            notes=row["notes"] or "",
            affinity=int(row["affinity"] or 0),
            last_seen=row["last_seen"] or "",
        )

    def upsert_profile_fields(self, user_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {"username", "display_name", "preferred_name", "relationship", "traits", "notes", "affinity", "last_seen"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values())
        values.append(user_id)
        self.cur.execute(f"UPDATE profiles SET {sets}, last_seen = CURRENT_TIMESTAMP WHERE user_id = ?", values)
        self.db.commit()

    def append_profile_note(self, user_id: str, note: str) -> None:
        profile = self.get_profile(user_id)
        if not profile:
            return
        note = (note or "").strip()
        if not note:
            return
        existing = profile.notes.strip()
        if note.lower() in existing.lower():
            return
        new_notes = f"{existing}; {note}".strip("; ").strip()
        self.upsert_profile_fields(user_id, notes=new_notes)

    def adjust_affinity(self, user_id: str, delta: int) -> None:
        self.cur.execute(
            """
            UPDATE profiles
            SET affinity = COALESCE(affinity, 0) + ?,
                last_seen = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """,
            (delta, user_id),
        )
        self.db.commit()

    def ensure_user_card(self, user_id: str, username: str = "", display_name: str = "") -> None:
        self.cur.execute(
            """
            INSERT INTO user_cards (user_id, username, display_name, last_seen)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                username=excluded.username,
                display_name=excluded.display_name,
                last_seen=CURRENT_TIMESTAMP
            """,
            (user_id, username or "", display_name or ""),
        )
        self.db.commit()

    def bump_user_card(self, user_id: str, username: str = "", display_name: str = "") -> Dict[str, Any]:
        self.ensure_user_card(user_id, username=username, display_name=display_name)
        self.cur.execute(
            """
            UPDATE user_cards
            SET username = COALESCE(?, username),
                display_name = COALESCE(?, display_name),
                messages_seen = COALESCE(messages_seen, 0) + 1,
                last_seen = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """,
            (username or None, display_name or None, user_id),
        )
        self.db.commit()
        return self.get_user_card(user_id) or {}

    def get_user_card(self, user_id: str) -> Optional[Dict[str, Any]]:
        row = self.cur.execute("SELECT * FROM user_cards WHERE user_id = ?", (user_id,)).fetchone()
        return self._row_to_dict(row)

    def find_user_card_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        name = (name or "").strip()
        if not name:
            return None
        row = self.cur.execute(
            """
            SELECT * FROM user_cards
            WHERE LOWER(display_name) = LOWER(?)
               OR LOWER(username) = LOWER(?)
            LIMIT 1
            """,
            (name, name),
        ).fetchone()
        return self._row_to_dict(row)

    def update_user_card(self, user_id: str, **fields: Any) -> None:
        allowed = {
            "username",
            "display_name",
            "summary",
            "interests",
            "communication_style",
            "traits",
            "relationship",
            "relationship_trend",
            "opinion",
            "topics",
            "activity_level",
            "behaviors",
            "notes",
            "affinity",
            "messages_seen",
            "last_seen",
            "personality_traits",
            "humor_style",
            "toxicity_level",
            "friendliness",
            "inside_jokes",
            "nicknames",
            "bot_opinion",
            "confidence_score"
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        self.ensure_user_card(user_id, username=updates.get("username", ""), display_name=updates.get("display_name", ""))
        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values())
        values.append(user_id)
        self.cur.execute(
            f"UPDATE user_cards SET {sets}, updated_at = CURRENT_TIMESTAMP, last_seen = CURRENT_TIMESTAMP WHERE user_id = ?",
            values,
        )
        self.db.commit()

    def append_user_card_note(self, user_id: str, note: str) -> None:
        note = (note or "").strip()
        if not note:
            return
        card = self.get_user_card(user_id) or {}
        existing = (card.get("notes") or "").strip()
        if note.lower() in existing.lower():
            return
        new_notes = f"{existing}; {note}".strip("; ").strip()
        self.update_user_card(user_id, notes=new_notes)

    def upsert_fact(self, subject_id: str, subject_name: str, key: str, value: str, confidence: float = 0.7, source: str = "chat") -> None:
        subject_id = subject_id or ""
        subject_name = subject_name or ""
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return
        existing = self.cur.execute(
            """
            SELECT id FROM facts
            WHERE subject_id = ? AND subject_name = ? AND key = ? AND value = ?
            """,
            (subject_id, subject_name, key, value),
        ).fetchone()
        if existing:
            self.cur.execute(
                """
                UPDATE facts
                SET confidence = ?,
                    source = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (confidence, source, existing["id"]),
            )
        else:
            self.cur.execute(
                """
                INSERT INTO facts (subject_id, subject_name, key, value, confidence, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (subject_id, subject_name, key, value, confidence, source),
            )
        self.db.commit()

    def add_episode(
        self,
        channel_id: str,
        guild_id: str,
        user_id: str,
        user_name: str,
        summary: str,
        episode_type: str = "message",
        importance: float = 0.5,
        emotion: str = "",
    ) -> None:
        summary = (summary or "").strip()
        if not summary:
            return
        self.cur.execute(
            """
            INSERT INTO episodes (channel_id, guild_id, user_id, user_name, episode_type, summary, importance, emotion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (channel_id or "", guild_id or "", user_id or "", user_name or "", episode_type or "message", summary, importance, emotion),
        )
        self.db.commit()

    def get_recent_episodes(self, channel_id: str = "", user_id: str = "", limit: int = 8):
        clauses = []
        params: List[Any] = []
        if channel_id:
            clauses.append("channel_id = ?")
            params.append(channel_id)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if not clauses:
            return []
        where = " OR ".join(clauses)
        return self.cur.execute(
            f"""
            SELECT * FROM episodes
            WHERE {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()

    def get_facts(self, subject_id: str = "", subject_name: str = "", limit: int = 6):
        subject_id = subject_id or ""
        subject_name = subject_name or ""
        if not subject_id and not subject_name:
            return []
        clauses = []
        params: List[Any] = []
        if subject_id:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        if subject_name:
            clauses.append("LOWER(subject_name) = LOWER(?)")
            params.append(subject_name)
        where = " OR ".join(clauses)
        rows = self.cur.execute(
            f"""
            SELECT * FROM facts
            WHERE {where}
            ORDER BY confidence DESC, updated_at DESC, id DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return rows

    def get_user_entity_names(self) -> List[str]:
        names = set()

        rows = self.cur.execute("SELECT display_name, preferred_name, username FROM profiles").fetchall()
        for row in rows:
            for v in (row["display_name"], row["preferred_name"], row["username"]):
                if v:
                    names.add(v.strip())

        rows = self.cur.execute("SELECT display_name, username FROM user_cards").fetchall()
        for row in rows:
            for v in (row["display_name"], row["username"]):
                if v:
                    names.add(v.strip())

        rows = self.cur.execute("SELECT DISTINCT subject_name FROM facts WHERE subject_name != ''").fetchall()
        for row in rows:
            names.add(row["subject_name"].strip())

        rows = self.cur.execute("SELECT DISTINCT user_name FROM episodes WHERE user_name != ''").fetchall()
        for row in rows:
            names.add(row["user_name"].strip())

        return sorted(n for n in names if n)

    def add_message(self, channel_id: str, guild_id: str, role: str, speaker_id: str, speaker_name: str, content: str, attachments: str = "") -> None:
        role = (role or "user").strip().lower()
        text = strip_output_labels((content or "").strip())
        if role == "assistant":
            cleaned = clean_response(text)
            if cleaned:
                text = cleaned
        self.cur.execute(
            """
            INSERT INTO history (channel_id, guild_id, role, speaker_id, speaker_name, content, attachments)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (channel_id, guild_id, role, speaker_id or "", speaker_name or "", text or "", attachments or ""),
        )
        self.cur.execute(
            """
            INSERT INTO channel_meta (channel_id, message_count, updated_at)
            VALUES (?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(channel_id) DO UPDATE SET
                message_count = COALESCE(message_count, 0) + 1,
                updated_at = CURRENT_TIMESTAMP
            """,
            (channel_id,),
        )
        self.db.commit()

    def get_recent_history_rows(self, channel_id: str, limit: int = 8):
        rows = self.cur.execute(
            """
            SELECT role, speaker_id, speaker_name, content, attachments, id, created_at
            FROM history
            WHERE channel_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (channel_id, limit),
        ).fetchall()[::-1]
        return [self._row_to_dict(r) for r in rows]

    def get_recent_user_history(self, user_id: str, channel_id: str = "", limit: int = 8):
        if not user_id:
            return []
        sql = """
            SELECT role, speaker_id, speaker_name, content, attachments, channel_id, id
            FROM history
            WHERE speaker_id = ?
        """
        params: List[Any] = [user_id]
        if channel_id:
            sql += " AND channel_id = ?"
            params.append(channel_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.cur.execute(sql, tuple(params)).fetchall()[::-1]
        return [self._row_to_dict(r) for r in rows]

    def get_recent_history(self, channel_id: str, limit: int = 8):
        rows = self.get_recent_history_rows(channel_id, limit=limit)
        out = []
        for r in rows:
            content = (r["content"] or "").strip()
            attachments = (r.get("attachments") or "").strip()
            if attachments:
                content = f"{content}\n[attachments]\n{attachments}" if content else f"[attachments]\n{attachments}"
            role = (r.get("role") or "user").strip().lower()
            speaker = (r.get("speaker_name") or "").strip()
            if role == "assistant":
                out.append({"role": "assistant", "content": content})
            elif role == "system":
                out.append({"role": "system", "content": content})
            else:
                if speaker and content:
                    out.append({"role": "user", "content": f"{speaker}: {content}"})
                else:
                    out.append({"role": "user", "content": content})
        return out

    def build_profile_card(self, subject_id: str = "", subject_name: str = "", max_facts: int = 4) -> str:
        profile = self.get_profile(subject_id) if subject_id else None
        user_card = self.get_user_card(subject_id) if subject_id else None

        if profile is None and subject_name:
            row = self.cur.execute(
                """
                SELECT * FROM profiles
                WHERE LOWER(display_name) = LOWER(?)
                   OR LOWER(preferred_name) = LOWER(?)
                   OR LOWER(username) = LOWER(?)
                LIMIT 1
                """,
                (subject_name, subject_name, subject_name),
            ).fetchone()
            if row:
                profile = ProfileCard(
                    user_id=row["user_id"],
                    username=row["username"] or "",
                    display_name=row["display_name"] or "",
                    preferred_name=row["preferred_name"] or "",
                    relationship=row["relationship"] or "",
                    traits=row["traits"] or "",
                    notes=row["notes"] or "",
                    affinity=int(row["affinity"] or 0),
                    last_seen=row["last_seen"] or "",
                )

        if user_card is None and subject_name:
            user_card = self.find_user_card_by_name(subject_name)

        facts = self.get_facts(subject_id=subject_id, subject_name=subject_name, limit=max_facts)
        prefs = self.get_user_prefs(subject_id, limit=5) if subject_id else []
        episodes = self.get_recent_episodes(user_id=subject_id, limit=3) if subject_id else []

        lines: List[str] = []
        label = subject_name or (profile.display_name if profile else subject_id) or ""
        if profile:
            lines.append(f"Имя: {profile.display_name or profile.username or label}")
            if profile.preferred_name:
                lines.append(f"Как звать: {profile.preferred_name}")
            if profile.relationship:
                lines.append(f"Связь: {profile.relationship}")
            lines.append(f"Affinity: {profile.affinity}")
            if profile.traits:
                lines.append(f"Черты: {profile.traits}")
            if profile.notes:
                lines.append(f"Заметки: {profile.notes}")
        elif label:
            lines.append(f"Имя: {label}")

        if user_card:
            if user_card.get("summary"):
                lines.append(f"Карточка: {user_card['summary']}")
            if user_card.get("interests"):
                lines.append(f"Интересы: {user_card['interests']}")
            if user_card.get("communication_style"):
                lines.append(f"Стиль общения: {user_card['communication_style']}")
            if user_card.get("traits"):
                lines.append(f"Карточка-черты: {user_card['traits']}")
            if user_card.get("relationship"):
                lines.append(f"Карточка-связь: {user_card['relationship']}")
            if user_card.get("relationship_trend"):
                lines.append(f"Тренд отношений: {user_card['relationship_trend']}")
            if user_card.get("opinion"):
                lines.append(f"Мнение Nika: {user_card['opinion']}")
            if user_card.get("topics"):
                lines.append(f"Темы: {user_card['topics']}")
            if user_card.get("activity_level"):
                lines.append(f"Активность: {user_card['activity_level']}")
            if user_card.get("behaviors"):
                lines.append(f"Поведение: {user_card['behaviors']}")
            if user_card.get("notes"):
                lines.append(f"Карточка-заметки: {user_card['notes']}")
            if user_card.get("messages_seen") is not None:
                lines.append(f"Сообщений: {int(user_card['messages_seen'] or 0)}")

            # Social intelligence fields
            if user_card.get("personality_traits"):
                lines.append(f"Личность: {user_card['personality_traits']}")
            if user_card.get("humor_style"):
                lines.append(f"Юмор: {user_card['humor_style']}")
            if user_card.get("toxicity_level"):
                lines.append(f"Токсичность: {user_card['toxicity_level']}")
            if user_card.get("friendliness"):
                lines.append(f"Дружелюбность: {user_card['friendliness']}")
            if user_card.get("inside_jokes"):
                lines.append(f"Локальные мемы: {user_card['inside_jokes']}")
            if user_card.get("nicknames"):
                lines.append(f"Никнеймы: {user_card['nicknames']}")
            if user_card.get("bot_opinion"):
                lines.append(f"Мнение Nika (соц): {user_card['bot_opinion']}")
            if user_card.get("confidence_score"):
                lines.append(f"Уверенность (соц): {user_card['confidence_score']}")

        for pref in prefs:
            lines.append(f"Предпочтение: {pref['key']}={pref['value']} ({float(pref['weight'] or 0):+.2f})")
        for fact in facts:
            lines.append(f"{fact['key']}: {fact['value']}")
        for ep in episodes:
            lines.append(f"Событие: {ep['summary']}")
        return "\n".join(lines).strip()

    def build_known_names(self) -> List[str]:
        return self.get_user_entity_names()

    def set_user_pref(self, user_id: str, key: str, value: str, weight: float = 0.1, source: str = "feedback"):
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return
        row = self.cur.execute(
            "SELECT id, weight FROM prefs WHERE user_id = ? AND key = ? AND value = ?",
            (user_id, key, value),
        ).fetchone()
        if row:
            self.cur.execute(
                "UPDATE prefs SET weight = ?, source = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (float(row["weight"] or 0.0) + weight, source, row["id"]),
            )
        else:
            self.cur.execute(
                "INSERT INTO prefs (user_id, key, value, weight, source) VALUES (?, ?, ?, ?, ?)",
                (user_id, key, value, weight, source),
            )
        self.db.commit()

    def get_user_prefs(self, user_id: str, limit: int = 8):
        return self.cur.execute(
            "SELECT * FROM prefs WHERE user_id = ? ORDER BY weight DESC, updated_at DESC, id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()

    def add_feedback(self, user_id: str, channel_id: str, action: str, rating: int, aspect: str, note: str, source_text: str):
        self.cur.execute(
            "INSERT INTO feedback (user_id, channel_id, action, rating, aspect, note, source_text) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, channel_id, action, rating, aspect, note, source_text),
        )
        self.db.commit()

    def get_feedback_stats(self, user_id: str, limit: int = 30) -> Dict[str, float]:
        rows = self.cur.execute(
            "SELECT rating FROM feedback WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        if not rows:
            return {"avg": 0.0, "count": 0}
        vals = [int(r["rating"] or 0) for r in rows]
        return {"avg": sum(vals) / max(1, len(vals)), "count": len(vals)}

    def get_channel_meta(self, channel_id: str):
        row = self.cur.execute("SELECT * FROM channel_meta WHERE channel_id = ?", (channel_id,)).fetchone()
        if row:
            return self._row_to_dict(row)
        self.cur.execute("INSERT INTO channel_meta (channel_id) VALUES (?) ON CONFLICT(channel_id) DO NOTHING", (channel_id,))
        self.db.commit()
        row = self.cur.execute("SELECT * FROM channel_meta WHERE channel_id = ?", (channel_id,)).fetchone()
        return self._row_to_dict(row)

    def update_channel_meta(self, channel_id: str, **fields: Any) -> None:
        allowed = {
            "summary",
            "summary_timestamp",
            "message_count",
            "last_summary_count",
            "mood",
            "energy",
            "sass",
            "last_target_channel_id",
            "last_target_user_id",
            "last_action_type",
            "last_reaction",
            "last_bot_post_count",
            "last_read_limit",
            "last_read_anchor_message_id",
            "last_read_first_message_id",
            "last_read_last_message_id",
            "last_read_summary",
            "last_autonomy_count",
            "last_autonomy_at",
            "last_interjection_at",
            "last_emoji_at",
            "last_interjection_type",
            "updated_at",
            "summary_participants",
            "summary_topics",
            "summary_mood",
            "summary_jokes",
            "summary_conflicts",
            "summary_events",
            "summary_unresolved"
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        sets = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values())
        values.append(channel_id)
        self.cur.execute(f"UPDATE channel_meta SET {sets}, updated_at = CURRENT_TIMESTAMP WHERE channel_id = ?", values)
        self.db.commit()

    def set_last_target(self, channel_id: str, target_channel_id: str = "", target_user_id: str = ""):
        self.update_channel_meta(channel_id, last_target_channel_id=target_channel_id or "", last_target_user_id=target_user_id or "")

    def set_summary(self, channel_id: str, summary: str):
        cleaned = sanitize_summary_text(summary or "")
        self.update_channel_meta(channel_id, summary=cleaned)

    def get_summary(self, channel_id: str) -> str:
        row = self.get_channel_meta(channel_id) or {}
        summary = strip_output_labels((row.get("summary") or "").strip())
        return sanitize_summary_text(summary) or ""

    def record_read_state(
        self,
        channel_id: str,
        *,
        target_channel_id: str = "",
        limit: int = 0,
        anchor_message_id: str = "",
        first_message_id: str = "",
        last_message_id: str = "",
    ) -> None:
        self.update_channel_meta(
            channel_id,
            last_action_type="read_channel",
            last_target_channel_id=target_channel_id or "",
            last_read_limit=int(limit or 0),
            last_read_anchor_message_id=anchor_message_id or "",
            last_read_first_message_id=first_message_id or "",
            last_read_last_message_id=last_message_id or "",
        )

    def record_autonomy_state(
        self,
        channel_id: str,
        *,
        action_type: str = "",
        count: int = 0,
        interjection_type: str = "",
    ) -> None:
        import datetime
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        updates = {
            "last_autonomy_count": int(count or 0),
            "last_autonomy_at": now_ts,
            "last_interjection_type": interjection_type or "",
        }
        if action_type:
            updates["last_action_type"] = action_type
        if interjection_type in {"reply", "short_interject", "contextual_reply", "sarcastic_comment", "playful_question", "meme_reply"}:
            updates["last_interjection_at"] = now_ts
        elif interjection_type == "react":
            updates["last_emoji_at"] = now_ts

        self.update_channel_meta(channel_id, **updates)
