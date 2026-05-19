from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import aiohttp
import discord


class ActionExecutor:
    def __init__(self, settings, store):
        self.settings = settings
        self.store = store

    async def safe_reply(self, message: discord.Message, text: str, retries: int = 3):
        for attempt in range(retries):
            try:
                return await message.reply(text)
            except (discord.errors.DiscordServerError, aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                raise
            except discord.errors.HTTPException as e:
                if getattr(e, "status", None) in (500, 502, 503, 504) and attempt < retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                raise

    async def safe_send(self, channel, text: str, retries: int = 3):
        for attempt in range(retries):
            try:
                return await channel.send(text)
            except (discord.errors.DiscordServerError, aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                raise
            except discord.errors.HTTPException as e:
                if getattr(e, "status", None) in (500, 502, 503, 504) and attempt < retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                raise

    def _resolve_channel(self, guild: discord.Guild, token: str) -> Optional[discord.TextChannel]:
        token = (token or "").strip()
        if not token:
            return None
        if token.startswith("<#") and token.endswith(">") and token[2:-1].isdigit():
            ch = guild.get_channel(int(token[2:-1]))
            return ch if isinstance(ch, discord.TextChannel) else None
        if token.startswith("#") and token[1:].isdigit():
            ch = guild.get_channel(int(token[1:]))
            return ch if isinstance(ch, discord.TextChannel) else None
        if token.isdigit():
            ch = guild.get_channel(int(token))
            return ch if isinstance(ch, discord.TextChannel) else None
        if token.startswith("#"):
            token = token[1:]
        low = token.lower().strip()
        for ch in guild.text_channels:
            if ch.name.lower() == low or low in ch.name.lower():
                return ch
        return None

    def _resolve_channel_by_name(self, guild: discord.Guild, token: str) -> Optional[discord.TextChannel]:
        return self._resolve_channel(guild, token)

    async def _resolve_member(self, guild: discord.Guild, token: str) -> Optional[discord.Member]:
        token = (token or "").strip()
        if not token:
            return None
        if token.startswith("<@") and token.endswith(">"):
            inner = token[2:-1]
            if inner.startswith("!"):
                inner = inner[1:]
            if inner.isdigit():
                mid = int(inner)
                member = guild.get_member(mid)
                if member:
                    return member
                try:
                    return await guild.fetch_member(mid)
                except Exception:
                    return None
        token = token.replace("@", "").replace("<", "").replace(">", "").replace("!", "").strip().lower()
        for member in guild.members:
            if member.display_name.lower() == token or member.name.lower() == token or str(member.id) == token:
                return member
        for member in guild.members:
            if token in member.display_name.lower() or token in member.name.lower():
                return member
        rows = self.store.cur.execute(
            "SELECT user_id FROM user_cards WHERE LOWER(display_name) LIKE LOWER(?) OR LOWER(username) LIKE LOWER(?) LIMIT 10",
            (f"%{token}%", f"%{token}%"),
        ).fetchall()
        for row in rows:
            try:
                member = guild.get_member(int(row["user_id"]))
                if member:
                    return member
                member = await guild.fetch_member(int(row["user_id"]))
                if member:
                    return member
            except Exception:
                continue
        return None

    async def execute(self, message: discord.Message, action: Dict[str, Any]) -> Dict[str, Any]:
        act = (action.get("action") or "").strip().lower()
        if act in {"ignore", ""}:
            return {"kind": "ignore"}

        if act == "reply":
            return {"kind": "reply", "text": (action.get("text") or "").strip()}

        if act == "remember":
            key = (action.get("key") or "note").strip()
            value = (action.get("value") or action.get("text") or "").strip()
            if not value:
                return {"kind": "error", "text": "пустота, запоминать нечего"}
            self.store.upsert_fact(str(message.author.id), message.author.display_name, key, value, confidence=0.95, source="explicit")
            self.store.append_user_card_note(str(message.author.id), f"{key}: {value}")
            self.store.adjust_affinity(str(message.author.id), 1)
            return {"kind": "status", "text": f"ок, запомнила {key}"}

        if act == "read_channel":
            if not message.guild:
                return {"kind": "error", "text": "я не в лс"}

            channel_token = (action.get("channel") or "").strip()
            channel = self._resolve_channel(message.guild, channel_token) if channel_token else None

            if not channel:
                # Try to use last channel if none provided
                last_meta = self.store.get_channel_meta(str(message.channel.id)) or {}
                last_channel_id = last_meta.get("last_target_channel_id")
                if last_channel_id:
                    channel = message.guild.get_channel(int(last_channel_id))

            if not channel:
                return {"kind": "error", "text": "канал не найден"}

            limit = max(1, min(int(action.get("limit") or 30), 80))
            before = (action.get("before") or "").strip()
            before_obj = discord.Object(id=int(before)) if before.isdigit() else None

            items = []
            history_kwargs = {"limit": limit}
            if before_obj:
                history_kwargs["before"] = before_obj

            async for msg in channel.history(**history_kwargs):
                author = msg.author.display_name
                body = (msg.content or "").strip() or "[file/empty]"
                items.append({"id": str(msg.id), "author": author, "text": body})

            if not items:
                return {"kind": "observation", "text": "", "channel_name": channel.name}

            # Items are fetched from newest to oldest.
            # The "anchor" for further pagination is the oldest message in this batch.
            anchor_id = items[-1]["id"]

            self.store.record_read_state(
                str(message.channel.id),
                target_channel_id=str(channel.id),
                limit=limit,
                anchor_message_id=anchor_id
            )

            items.reverse() # Show oldest first for natural reading
            return {
                "kind": "observation",
                "channel_id": str(channel.id),
                "channel_name": channel.name,
                "text": "\n".join(f"{i['author']}: {i['text']}" for i in items),
                "limit": limit
            }

        if act == "send_message":
            if not message.guild: return {"kind": "error", "text": "я не в лс"}
            channel = self._resolve_channel(message.guild, (action.get("channel") or "").strip())
            text = (action.get("text") or "").strip()
            if not channel or not text: return {"kind": "error", "text": "что-то пошло не так"}
            await self.safe_send(channel, text)
            return {"kind": "status", "text": f"скинула в {channel.mention}"}

        if act == "react":
            reaction = (action.get("reaction") or "").strip()
            if not reaction: return {"kind": "error", "text": "эмодзи где"}

            # Simple alias
            aliases = {"clown": "🤡", "fire": "🔥", "skull": "💀", "heart": "❤️", "thumb": "👍"}
            reaction = aliases.get(reaction.lower(), reaction)

            # React to the message before this one if it's a command
            target = message
            async for m in message.channel.history(limit=2):
                if m.id != message.id:
                    target = m
                    break

            try:
                await target.add_reaction(reaction)
                return {"kind": "status", "text": f"ткнула {reaction}"}
            except:
                return {"kind": "error", "text": "не могу"}

        return {"kind": "ignore"}
