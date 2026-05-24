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
        if not token: return None
        if token.startswith("<#") and token.endswith(">") and token[2:-1].isdigit():
            ch = guild.get_channel(int(token[2:-1]))
            return ch if isinstance(ch, discord.TextChannel) else None
        if token.startswith("#") and token[1:].isdigit():
            ch = guild.get_channel(int(token[1:]))
            return ch if isinstance(ch, discord.TextChannel) else None
        if token.isdigit():
            ch = guild.get_channel(int(token))
            return ch if isinstance(ch, discord.TextChannel) else None
        if token.startswith("#"): token = token[1:]
        low = token.lower().strip()
        for ch in guild.text_channels:
            if ch.name.lower() == low or low in ch.name.lower():
                return ch
        return None

    async def _resolve_member(self, guild: discord.Guild, token: str) -> Optional[discord.Member]:
        token = (token or "").strip()
        if not token: return None
        if token.startswith("<@") and token.endswith(">"):
            inner = token[2:-1]
            if inner.startswith("!"): inner = inner[1:]
            if inner.isdigit():
                mid = int(inner)
                member = guild.get_member(mid)
                if member: return member
                try: return await guild.fetch_member(mid)
                except Exception: return None
        token = token.replace("@", "").replace("<", "").replace(">", "").replace("!", "").strip().lower()
        for member in guild.members:
            if member.display_name.lower() == token or member.name.lower() == token or str(member.id) == token:
                return member
        for member in guild.members:
            if token in member.display_name.lower() or token in member.name.lower():
                return member
        return None

    async def execute(self, message: discord.Message, action: Dict[str, Any]) -> Dict[str, Any]:
        act = (action.get("action") or "").strip().lower()
        if act in {"ignore", ""}: return {"kind": "ignore"}
        if act == "reply": return {"kind": "reply", "text": (action.get("text") or "").strip()}

        if act == "remember":
            key, value = (action.get("key") or "note").strip(), (action.get("value") or action.get("text") or "").strip()
            if not value: return {"kind": "error", "text": "нечего запоминать"}
            self.store.upsert_fact(str(message.author.id), message.author.display_name, key, value, confidence=0.95, source="explicit")
            return {"kind": "status", "text": f"запомнила: {key} — {value}"}

        if act == "read_channel":
            if not message.guild: return {"kind": "error", "text": "только на сервере"}
            ch_token = (action.get("channel") or "").strip()
            channel = self._resolve_channel(message.guild, ch_token)
            if not channel:
                meta = self.store.get_channel_meta(str(message.channel.id)) or {}
                channel = self._resolve_channel(message.guild, meta.get("last_target_channel_id", ""))
            if not channel: return {"kind": "error", "text": "канал не найден"}

            limit = max(1, min(int(action.get("limit") or 30), 80))
            before = (action.get("before") or "").strip()
            items = []
            kwargs = {"limit": limit, "oldest_first": False}
            if before.isdigit(): kwargs["before"] = discord.Object(id=int(before))

            async for msg in channel.history(**kwargs):
                if msg.id == message.id: continue
                items.append({"id": str(msg.id), "author": msg.author.display_name, "text": (msg.content or "").strip() or "[media]"})

            if not items:
                self.store.record_read_state(str(message.channel.id), target_channel_id=str(channel.id), limit=limit)
                return {"kind": "observation", "text": "", "channel_name": channel.name}

            items.reverse()
            first_id, last_id = items[0]["id"], items[-1]["id"]
            self.store.record_read_state(str(message.channel.id), target_channel_id=str(channel.id), limit=limit, anchor_message_id=first_id, first_message_id=first_id, last_message_id=last_id)

            return {
                "kind": "observation", "channel_id": str(channel.id), "channel_name": channel.name, "text": "\n".join(f"{i['author']}: {i['text']}" for i in items),
                "anchor_message_id": first_id, "first_message_id": first_id, "last_message_id": last_id, "limit": limit
            }

        if act == "send_message":
            if not message.guild: return {"kind": "error", "text": "только на сервере"}
            channel = self._resolve_channel(message.guild, action.get("channel"))
            text = (action.get("text") or "").strip()
            if not channel or not text: return {"kind": "error", "text": "мало данных"}
            await self.safe_send(channel, text)
            self.store.update_channel_meta(str(message.channel.id), last_action_type="send_message", last_target_channel_id=str(channel.id))
            return {"kind": "status", "text": f"отправила в {channel.mention}", "sent_text": text, "channel_id": str(channel.id)}

        if act == "ping_user":
            if not message.guild: return {"kind": "error", "text": "только на сервере"}
            member = await self._resolve_member(message.guild, action.get("user"))
            if not member: return {"kind": "error", "text": "юзер не найден"}
            channel = self._resolve_channel(message.guild, action.get("channel")) or message.channel
            text = (action.get("text") or "").strip() or "тебя зовут сюда"
            await self.safe_send(channel, f"{member.mention} {text}")
            return {"kind": "status", "text": f"позвала {member.display_name}"}

        if act == "react":
            reaction = (action.get("reaction") or "").strip()
            if not reaction: return {"kind": "error", "text": "нет эмодзи"}
            target_msg = message
            async for msg in message.channel.history(limit=5):
                if msg.id != message.id:
                    target_msg = msg
                    break
            try:
                await target_msg.add_reaction(reaction)
                self.store.update_channel_meta(str(message.channel.id), last_action_type="react", last_reaction=reaction)
                return {"kind": "status", "text": f"поставила {reaction}"}
            except Exception as e: return {"kind": "error", "text": f"ошибка: {e}"}

        return {"kind": "ignore"}
