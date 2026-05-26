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
            "SELECT user_id FROM profiles WHERE LOWER(display_name) LIKE LOWER(?) OR LOWER(preferred_name) LIKE LOWER(?) OR LOWER(username) LIKE LOWER(?) LIMIT 10",
            (f"%{token}%", f"%{token}%", f"%{token}%"),
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
                return {"kind": "error", "text": "Окей, но запоминать нечего."}
            self.store.upsert_fact(str(message.author.id), message.author.display_name, key, value, confidence=0.95, source="explicit")
            self.store.append_profile_note(str(message.author.id), f"{key}: {value}")
            self.store.adjust_affinity(str(message.author.id), 1)
            self.store.update_channel_meta(str(message.channel.id), last_action_type="remember")
            self.store.add_episode(
                str(message.channel.id),
                str(message.guild.id) if message.guild else "",
                str(message.author.id),
                message.author.display_name,
                f"Запомнила: {key} — {value}",
                episode_type="memory",
                importance=0.8,
                emotion="neutral",
            )
            return {"kind": "status", "text": f"Окей, запомнила: {key} — {value}"}

        if act == "read_channel":
            if not message.guild:
                return {"kind": "error", "text": "Это работает только на сервере."}

            channel_token = (action.get("channel") or "").strip()
            channel = self._resolve_channel(message.guild, channel_token) if channel_token else None
            if not channel and self.store.get_channel_meta(str(message.channel.id)):
                last_meta = self.store.get_channel_meta(str(message.channel.id)) or {}
                channel = self._resolve_channel(message.guild, (last_meta.get("last_target_channel_id") or "").strip())

            if not channel:
                return {"kind": "error", "text": "Канал не найден."}

            perms = channel.permissions_for(message.guild.me) if message.guild.me else None
            if perms and (not perms.view_channel or not perms.read_message_history):
                return {"kind": "error", "text": "У меня нет прав читать тот канал."}

            limit = max(1, min(int(action.get("limit") or 30), 80))
            before = (action.get("before") or "").strip()
            before_obj = discord.Object(id=int(before)) if before.isdigit() else None

            # Use a slightly larger scan window to avoid missing the tail around anchors.
            scan_limit = min(100, max(limit + 10, 20))
            items = []
            first_message_id = ""
            last_message_id = ""
            history_kwargs = {"limit": scan_limit, "oldest_first": False}
            if before_obj:
                history_kwargs["before"] = before_obj

            async for msg in channel.history(**history_kwargs):
                if msg.id == message.id:
                    continue
                author = msg.author.display_name if hasattr(msg.author, "display_name") else msg.author.name
                body = (msg.content or "").strip() or "[без текста]"
                items.append(
                    {
                        "id": str(msg.id),
                        "author": author,
                        "text": body,
                    }
                )
                if len(items) >= limit:
                    break

            if not items:
                self.store.record_read_state(
                    str(message.channel.id),
                    target_channel_id=str(channel.id),
                    limit=limit,
                    first_message_id="",
                    last_message_id="",
                )
                return {
                    "kind": "observation",
                    "channel_id": str(channel.id),
                    "channel_name": channel.name,
                    "text": "",
                    "first_message_id": "",
                    "last_message_id": "",
                    "limit": limit,
                    "before": before,
                }

            items.reverse()
            first_message_id = items[0]["id"]
            last_message_id = items[-1]["id"]

            self.store.record_read_state(
                str(message.channel.id),
                target_channel_id=str(channel.id),
                limit=limit,
                anchor_message_id=first_message_id, # Oldest of current batch is anchor for "earlier"
                first_message_id=first_message_id,
                last_message_id=last_message_id,
            )

            # Also store summary state
            self.store.update_channel_meta(
                str(message.channel.id),
                last_action_type="read_channel",
                last_target_channel_id=str(channel.id),
                last_read_limit=limit,
                last_read_anchor_message_id=first_message_id
            )

            return {
                "kind": "observation",
                "channel_id": str(channel.id),
                "channel_name": channel.name,
                "text": "\n".join(f"{item['author']}: {item['text']}" for item in items),
                "anchor_message_id": first_message_id,
                "first_message_id": first_message_id,
                "last_message_id": last_message_id,
                "limit": limit,
                "before": before,
            }

        if act == "send_message":
            if not message.guild:
                return {"kind": "error", "text": "Это работает только на сервере."}
            channel = self._resolve_channel(message.guild, (action.get("channel") or "").strip())
            text = (action.get("text") or "").strip()
            if not channel or not text:
                return {"kind": "error", "text": "Не хватает данных для отправки."}
            perms = channel.permissions_for(message.guild.me) if message.guild.me else None
            if perms and not perms.send_messages:
                return {"kind": "error", "text": "У меня нет прав писать в тот канал."}
            await self.safe_send(channel, text)
            self.store.set_last_target(str(message.channel.id), target_channel_id=str(channel.id))
            self.store.update_channel_meta(str(message.channel.id), last_action_type="send_message")
            self.store.add_episode(
                str(message.channel.id),
                str(message.guild.id) if message.guild else "",
                str(message.author.id),
                message.author.display_name,
                f"Отправила сообщение в {channel.name}: {text}",
                episode_type="action",
                importance=0.7,
                emotion="neutral",
            )
            return {
                "kind": "status",
                "text": f"отправила в {channel.mention}",
                "channel_id": str(channel.id),
                "channel_name": channel.name,
                "sent_text": text,
            }

        if act == "ping_user":
            if not message.guild:
                return {"kind": "error", "text": "Это работает только на сервере."}
            member = await self._resolve_member(message.guild, (action.get("user") or "").strip())
            if not member:
                return {"kind": "error", "text": "Юзер не найден."}
            channel_token = (action.get("channel") or "").strip()
            channel = self._resolve_channel(message.guild, channel_token) if channel_token else (message.channel if isinstance(message.channel, discord.TextChannel) else None)
            if not channel:
                return {"kind": "error", "text": "Не могу определить канал."}
            perms = channel.permissions_for(message.guild.me) if message.guild.me else None
            if perms and not perms.send_messages:
                return {"kind": "error", "text": "У меня нет прав писать в тот канал."}
            text = (action.get("text") or "").strip() or "тебя зовут сюда"
            final_text = f"{member.mention} {text}"
            await self.safe_send(channel, final_text)
            self.store.set_last_target(str(message.channel.id), target_channel_id=str(channel.id), target_user_id=str(member.id))
            self.store.update_channel_meta(str(message.channel.id), last_action_type="ping_user")
            self.store.add_episode(
                str(message.channel.id),
                str(message.guild.id) if message.guild else "",
                str(message.author.id),
                message.author.display_name,
                f"Пинганула {member.display_name} в {channel.name}",
                episode_type="action",
                importance=0.7,
                emotion="neutral",
            )
            return {
                "kind": "status",
                "text": f"позвала {member.display_name}",
                "channel_id": str(channel.id),
                "channel_name": channel.name,
                "sent_text": final_text,
            }

        if act == "react":
            reaction = (action.get("reaction") or "").strip()
            channel_token = (action.get("channel") or "").strip()
            if not reaction:
                return {"kind": "error", "text": "Реакция не указана."}
            aliases = {
                "clown": "🤡",
                "clown emoji": "🤡",
                "clown face": "🤡",
                "emoji clown": "🤡",
                "fire": "🔥",
                "skull": "💀",
                "heart": "❤️",
                "thumbsup": "👍",
                "thumbs up": "👍",
            }
            reaction = aliases.get(reaction.lower(), reaction)
            target_message = message
            target_channel_id = str(message.channel.id)
            search_channel = None
            if channel_token and message.guild:
                search_channel = self._resolve_channel(message.guild, channel_token)
                if not search_channel:
                    return {"kind": "error", "text": "Канал не найден."}
            elif message.guild and isinstance(message.channel, discord.TextChannel):
                search_channel = message.channel

            if search_channel is not None:
                target_channel_id = str(search_channel.id)
                async for msg in search_channel.history(limit=20, oldest_first=False):
                    if msg.id == message.id:
                        continue
                    target_message = msg
                    break

            try:
                await target_message.add_reaction(reaction)
                self.store.set_last_target(str(message.channel.id), target_channel_id=target_channel_id)
                self.store.update_channel_meta(
                    str(message.channel.id),
                    last_action_type="react",
                    last_reaction=reaction,
                    last_target_channel_id=target_channel_id,
                )
                self.store.add_episode(
                    str(message.channel.id),
                    str(message.guild.id) if message.guild else "",
                    str(message.author.id),
                    message.author.display_name,
                    f"Поставила реакцию {reaction}",
                    episode_type="action",
                    importance=0.6,
                    emotion="neutral",
                )
                return {"kind": "status", "text": f"добавила {reaction}"}
            except Exception as e:
                return {"kind": "error", "text": f"не смогла поставить реакцию: {e}"}

        if act == "post_thought":
            if not message.guild:
                return {"kind": "error", "text": "Это работает только на сервере."}
            text = (action.get("text") or "").strip()
            if not text:
                return {"kind": "error", "text": "Мысль пустая."}
            channel = self._resolve_channel_by_name(message.guild, self.settings.thought_channel_name)
            if not channel:
                return {"kind": "error", "text": "Не нашла свой канал."}
            perms = channel.permissions_for(message.guild.me) if message.guild.me else None
            if perms and not perms.send_messages:
                return {"kind": "error", "text": "Не могу писать в свой канал."}
            await self.safe_send(channel, text)
            current = self.store.get_channel_meta(str(message.channel.id)) or {}
            self.store.update_channel_meta(str(message.channel.id), last_bot_post_count=int(current.get("last_bot_post_count") or 0) + 1)
            self.store.update_channel_meta(str(message.channel.id), last_action_type="post_thought")
            self.store.add_episode(
                str(message.channel.id),
                str(message.guild.id) if message.guild else "",
                str(message.author.id),
                message.author.display_name,
                f"Подумала и написала в свой канал: {text}",
                episode_type="thought",
                importance=0.5,
                emotion="neutral",
            )
            return {
                "kind": "status",
                "text": f"мысль отправлена в {channel.mention}",
                "channel_id": str(channel.id),
                "channel_name": channel.name,
                "sent_text": text,
            }

        return {"kind": "ignore"}
