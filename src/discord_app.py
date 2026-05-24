from __future__ import annotations

import asyncio
import discord

from .perception import analyze_trigger, attachment_snapshot
from .text_utils import attachment_summary, clean_response, strip_output_labels


class NikaDiscordClient(discord.Client):
    def __init__(self, settings, store, planner, executor):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True
        intents.messages = True
        super().__init__(intents=intents)
        self.settings = settings
        self.store = store
        self.planner = planner
        self.executor = executor

    def clean_input(self, message: discord.Message) -> str:
        text = message.content or ""
        if self.user:
            text = text.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
        if text.lower().startswith(f"{self.settings.bot_name.lower()} "):
            text = text[len(self.settings.bot_name) + 1 :]
        return text.strip()

    def is_called(self, message: discord.Message) -> bool:
        if not message.content: return False
        if self.user and self.user in message.mentions: return True
        text = message.content.lower()
        return any(word in text for word in ["ника", "nika", "ник"])

    async def on_ready(self):
        print(f"Online as {self.user}")

    async def on_message(self, message: discord.Message):
        if message.author.bot: return

        channel_id, guild_id = str(message.channel.id), str(message.guild.id) if message.guild else ""
        raw_text = (message.content or "").strip()
        cleaned = self.clean_input(message)
        planner_text = cleaned or raw_text
        att_text = attachment_summary(message.attachments)

        self.store.add_message(channel_id, guild_id, "user", str(message.author.id), message.author.display_name, planner_text, attachments=att_text)

        engage, _ = analyze_trigger(message, self.user, self.settings.bot_name, self.settings.bot_aliases, self.settings.watch_channel_name)
        called = self.is_called(message)

        # Decide if we should summarize
        if self.planner.should_summarize(channel_id):
            asyncio.create_task(self.planner.summarize_channel(channel_id))

        if not (engage or called):
            # Try autonomy
            auto = await self.planner.run_autonomy(message)
            if auto.get("action") in {"reply", "react"}:
                try:
                    result = await self.executor.execute(message, auto)
                    if result.get("kind") == "reply" and result.get("text"):
                        reply_text = strip_output_labels(clean_response(result["text"]) or result["text"])
                        if reply_text:
                            await self.executor.safe_reply(message, reply_text)
                            self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id), self.settings.bot_name, reply_text)

                    self.store.record_autonomy_state(channel_id, interjection_type=auto.get("action"), count=self.store.get_channel_meta(channel_id).get("message_count", 0))
                except Exception as e: print(f"[AUTONOMY ERROR] {e}")
            return

        # Direct response or action
        async with message.channel.typing():
            try:
                action = await self.planner.decide(message, planner_text)
                result = await self.executor.execute(message, action)
            except Exception as e:
                await self.executor.safe_reply(message, f"ошибка: {e}")
                return

        kind = result.get("kind")
        if kind == "reply":
            reply_text = strip_output_labels(clean_response(result.get("text", "")) or result.get("text", "").strip())
            if reply_text:
                await self.executor.safe_reply(message, reply_text)
                self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id), self.settings.bot_name, reply_text)
        elif kind == "observation":
            composed = await self.planner.compose_read_response(message, planner_text, action, result)
            if composed:
                await self.executor.safe_reply(message, composed)
                self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id), self.settings.bot_name, composed)
                self.store.update_channel_meta(channel_id, last_read_summary=composed)
        elif kind == "status" and result.get("text"):
            await self.executor.safe_reply(message, result["text"])
            self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id), self.settings.bot_name, result["text"])
