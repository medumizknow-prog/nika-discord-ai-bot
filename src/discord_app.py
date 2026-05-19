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
                    self.store.ensure_profile(str(member.id), username=member.name, display_name=member.display_name)
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

        self.store.ensure_profile(str(message.author.id), username=message.author.name, display_name=message.author.display_name)
        self.store.ensure_user_card(str(message.author.id), username=message.author.name, display_name=message.author.display_name)
        self.store.bump_user_card(str(message.author.id), username=message.author.name, display_name=message.author.display_name)
        self.store.add_message(channel_id, guild_id, "user", str(message.author.id), message.author.display_name, planner_text, attachments=att_text)

        # Lightweight memory refresh for user cards runs in the background.
        try:
            asyncio.create_task(self.planner.refresh_user_card(message, planner_text))
        except Exception:
            pass

        self.planner.learn_feedback(message, planner_text)

        engage, _ = analyze_trigger(
            message,
            self.user,
            self.settings.bot_name,
            self.settings.bot_aliases,
            self.settings.watch_channel_name,
        )
        if self.planner.should_summarize(channel_id):
            asyncio.create_task(self.planner.summarize_channel(channel_id))

        called = self.is_called(message)
        preview_action = self.planner.direct_parser.parse(message, planner_text)
        engage = engage or called or bool(preview_action)

        if not engage:
            auto = await self.planner.run_autonomy(message)
            auto_action = auto.get("action")
            if auto_action in {"react", "reply", "short_interject", "contextual_reply", "sarcastic_comment", "playful_question", "meme_reply"}:
                try:
                    result = await self.executor.execute(message, auto)

                    # Visible interjection for reply-style actions
                    if auto_action in {"reply", "short_interject", "contextual_reply", "sarcastic_comment", "playful_question", "meme_reply"} and result.get("kind") == "reply" and result.get("text"):
                        reply_text = strip_output_labels(clean_response(result.get("text", "")) or result.get("text", "").strip())
                        if reply_text:
                            await self.executor.safe_reply(message, reply_text)
                            self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id) if self.user else "", self.settings.bot_name, reply_text)

                    elif result.get("kind") == "status" and result.get("text") and auto_action == "react":
                        # We don't log a message for a reaction status, just execute it
                        pass

                    # Mark cooldown after autonomous intervention.
                    current_meta = self.store.get_channel_meta(channel_id) or {}
                    msg_count = 0
                    if isinstance(current_meta, dict):
                        msg_count = int(current_meta.get("message_count") or 0)
                    else:
                        try: msg_count = int(current_meta["message_count"] or 0)
                        except Exception: pass

                    self.store.record_autonomy_state(
                        channel_id,
                        count=msg_count,
                        interjection_type=auto_action or "",
                    )
                except Exception as e:
                    print(f"[AUTONOMY ERROR] {e}")
            return

        async with message.channel.typing():
            try:
                action = preview_action or await self.planner.decide(message, planner_text)
                result = await self.executor.execute(message, action)
            except Exception as e:
                await self.executor.safe_reply(message, f"AI error: {e}")
                return

        kind = result.get("kind")

        if kind == "reply":
            reply_text = strip_output_labels(clean_response(result.get("text", "")) or result.get("text", "").strip())
            if reply_text and normalize_compare_text(reply_text) == normalize_compare_text(planner_text):
                reply_text = "Поняла."
            if reply_text:
                await self.executor.safe_reply(message, reply_text)
                self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id) if self.user else "", self.settings.bot_name, reply_text)

        elif kind == "observation":
            composed_raw = await self.planner.compose_read_response(message, planner_text, action, result)
            composed = strip_output_labels(clean_response(composed_raw) or composed_raw)
            await self.executor.safe_reply(message, composed)
            self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id) if self.user else "", self.settings.bot_name, composed)

            # Cache the summary for follow-ups and future snapshot context.
            self.store.update_channel_meta(
                channel_id,
                last_read_summary=composed,
                last_action_type="read_channel",
                last_target_channel_id=str(result.get("channel_id") or ""),
                last_read_limit=int(result.get("limit") or 0),
                last_read_first_message_id=str(result.get("first_message_id") or ""),
                last_read_last_message_id=str(result.get("last_message_id") or ""),
            )

        elif kind == "status":
            status_text = strip_output_labels(clean_response(result.get("text", "")) or result.get("text", "").strip())
            if status_text:
                await self.executor.safe_reply(message, status_text)
                self.store.add_message(channel_id, guild_id, "assistant", str(self.user.id) if self.user else "", self.settings.bot_name, status_text)
            if result.get("channel_id") and result.get("sent_text") and message.guild:
                self.store.add_message(
                    str(result["channel_id"]),
                    guild_id,
                    "assistant",
                    str(self.user.id) if self.user else "",
                    self.settings.bot_name,
                    strip_output_labels(clean_response(result["sent_text"]) or result["sent_text"]),
                )

        elif kind == "error":
            await self.executor.safe_reply(message, strip_output_labels(clean_response(result.get("text", "ошибка")) or result.get("text", "ошибка").strip()))

        # Keep the channel summary fresh.
        if self.planner.should_summarize(channel_id):
            asyncio.create_task(self.planner.summarize_channel(channel_id))
