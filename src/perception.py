from __future__ import annotations
from dataclasses import dataclass
import discord, re
WORD_BOUNDARY_TEMPLATE = r"(?<!\w){name}(?!\w)"
@dataclass
class Observation:
    channel_id: str
    guild_id: str
    channel_name: str
    author_id: str
    author_name: str
    content: str
    is_direct: bool
    trigger_reason: str
    attachments: str = ""

def _alias_hit(content: str, alias: str) -> bool:
    alias=(alias or '').strip()
    if not alias: return False
    return re.search(WORD_BOUNDARY_TEMPLATE.format(name=re.escape(alias.lower())), content.lower()) is not None

def analyze_trigger(message: discord.Message, bot_user: discord.ClientUser | None, bot_name: str, aliases: list[str], watch_channel_name: str):
    content = message.content or ''
    channel_name = getattr(message.channel, 'name', '').lower()
    if isinstance(message.channel, discord.DMChannel): return True, 'dm'
    if bot_user and bot_user in message.mentions: return True, 'mention'
    for alias in aliases or []:
        if _alias_hit(content, alias): return True, f'alias:{alias}'
    if channel_name == watch_channel_name.lower(): return True, 'watch_channel'
    if content.lower().startswith(f"{bot_name.lower()} "): return True, 'bot_name_prefix'
    return False, 'passive_only'

def attachment_snapshot(message: discord.Message) -> str:
    parts=[]
    for att in getattr(message, 'attachments', []) or []:
        ctype = getattr(att, 'content_type', '') or ''
        parts.append(f"[image] {att.filename} {att.url}" if ctype.startswith('image/') else f"[file] {att.filename} {att.url}")
    return '\n'.join(parts).strip()
