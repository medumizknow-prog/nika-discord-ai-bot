from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
load_dotenv()

def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()

def _csv(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]

@dataclass(frozen=True)
class Settings:
    discord_token: str
    lmstudio_url: str
    model: str
    db_file: str
    bot_name: str = "Nika"
    bot_aliases: list[str] = field(default_factory=list)
    watch_channel_name: str = "nika"
    thought_channel_name: str = "nika"
    max_recent_turns: int = 8
    summary_trigger_messages: int = 18
    max_context_chars: int = 4200
    max_summary_chars: int = 900
    autonomy_enabled: bool = True

settings = Settings(
    discord_token=_env("DISCORD_TOKEN"),
    lmstudio_url=_env("LMSTUDIO_URL", "http://127.0.0.1:1234/v1").rstrip("/"),
    model=_env("MODEL", "qwen2.5-14b-instruct"),
    db_file=_env("DB_FILE", "nika_v9.db"),
    bot_name=_env("BOT_NAME", "Nika") or "Nika",
    bot_aliases=sorted(set([_env("BOT_NAME", "Nika") or "Nika", *_csv(_env("BOT_ALIASES", "Nika,Ника,ник"))])),
    watch_channel_name=_env("WATCH_CHANNEL_NAME", "nika") or "nika",
    thought_channel_name=_env("THOUGHT_CHANNEL_NAME", "nika") or "nika",
    autonomy_enabled=_env("AUTONOMY_ENABLED", "true").lower() in {"1","true","yes","on"},
)
if not settings.discord_token:
    raise RuntimeError("DISCORD_TOKEN is missing")
