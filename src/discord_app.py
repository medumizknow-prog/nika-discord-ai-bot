from __future__ import annotations

import asyncio

import discord

from .perception import analyze_trigger, attachment_snapshot
from .text_utils import attachment_summary, clean_response, is_short_text, normalize_compare_text, strip_output_labels


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
        if not message.content:
            return False
        if self.user and self.user in message.mentions:
            return True
        text = message.content.lower()
        wake_words = ["ника", "nika", "ник"]
        return any(word in text for word in wake_words)

    async def on_ready(self):
        print(f"Online as {self.user}")
        for guild in self.guilds:
            for member in guild.members:
                if not member.bot:
                    self.store.ensure_user_card(str(member.id), username=member.name, display_name=member.display_name)

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        channel_id = str(message.channel.id)
        guild_id = str(message.guild.id) if message.guild else ""
        raw_text = (message.content or "").strip()
        cleaned = self.clean_input(message)
        planner_text = cleaned or raw_text
        att_text = attachment_summary(message.attachments)

        # Basic storage
        self.store.bump_user_card(str(message.author.id), username=message.author.name, display_name=message.author.display_name)
        self.store.add_message(channel_id, guild_id, "user", str(message.author.id), message.author.display_name, planner_text, attachments=att_text)

        # Trigger check
        called = self.is_called(message)
        engage, _ = analyze_trigger(
            message,
            self.user,
            self.settings.bot_name,
            self.settings.bot_aliases,
            self.settings.watch_channel_name,
        )

        # Direct parser (quick commands)
        preview_action = self.planner.direct_parser.parse(message, planner_text)

        should_respond = engage or called or bool(preview_action)

        if not should_respond:
            # Try autonomy
            auto = await self.planner.run_autonomy(message)
            if auto.get("action") in {"reply", "react"}:
                try:
                    result = await self.executor.execute(message, auto)

                    # Record autonomy state
                    self.store.record_autonomy_state(
                        channel_id,
                        interjection_type=auto.get("action") or "",
                        action_type=result.get("kind") or ""
                    )

                    if auto.get("action") == "reply" and result.get("kind") == "reply" and result.get("text"):
                        reply_text = result.get("text")
                        await self.executor.safe_reply(message, reply_text)
                        self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id) if self.user else "", self.settings.bot_name, reply_text)
                except Exception as e:
                    print(f"[AUTONOMY ERROR] {e}")

            # Maintenance summary
            if self.planner.should_summarize(channel_id):
                asyncio.create_task(self.planner.summarize_channel(channel_id))
            return

        # Regular response flow
        async with message.channel.typing():
            try:
                action = preview_action or await self.planner.decide(message, planner_text)
                result = await self.executor.execute(message, action)
            except Exception as e:
                print(f"[ERROR] {e}")
                return

        kind = result.get("kind")
        if kind == "reply":
            reply_text = result.get("text")
            if reply_text:
                await self.executor.safe_reply(message, reply_text)
                self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id) if self.user else "", self.settings.bot_name, reply_text)

        elif kind == "observation":
            composed = await self.planner.compose_read_response(message, planner_text, action, result)
            if composed:
                await self.executor.safe_reply(message, composed)
                self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id) if self.user else "", self.settings.bot_name, composed)
                # Cache the summary
                self.store.update_channel_meta(channel_id, last_read_summary=composed)

        elif kind == "status":
            status_text = result.get("text")
            if status_text:
                await self.executor.safe_reply(message, status_text)
                self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id) if self.user else "", self.settings.bot_name, status_text)

        # Always update summary if needed
        if self.planner.should_summarize(channel_id):
            asyncio.create_task(self.planner.summarize_channel(channel_id))
