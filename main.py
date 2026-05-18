from __future__ import annotations

import time
import discord

from src.config import settings
from src.direct_parser import DirectParser
from src.discord_app import NikaDiscordClient
from src.executor import ActionExecutor
from src.llm import LMStudioClient
from src.memory import MemoryStore
from src.planner import AgentPlanner


def build_bot() -> NikaDiscordClient:
    store = MemoryStore(settings.db_file)
    llm = LMStudioClient(settings.lmstudio_url, settings.model)
    parser = DirectParser(settings, store)
    planner = AgentPlanner(settings, store, llm, parser)
    executor = ActionExecutor(settings, store)
    return NikaDiscordClient(settings, store, planner, executor)


def main() -> None:
    while True:
        bot = build_bot()
        try:
            bot.run(settings.discord_token)
            return
        except discord.LoginFailure:
            print("BAD DISCORD TOKEN")
            return
        except KeyboardInterrupt:
            return
        except Exception as exc:
            print(f"BOT CRASHED: {exc}")
            print("restart in 10 sec...")
            time.sleep(10)


if __name__ == "__main__":
    main()
