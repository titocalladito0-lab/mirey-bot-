import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    discord_bot_token: str = os.getenv("DISCORD_BOT_TOKEN", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    owner_discord_id: int = int(os.getenv("OWNER_DISCORD_ID", "0"))
    db_path: str = os.getenv("DB_PATH", "db.sqlite")
    people_path: str = os.getenv("PEOPLE_PATH", "personas/people.yaml")
    context_token_budget: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "4096"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    reply_to_rejected_dm: bool = os.getenv("REPLY_TO_REJECTED_DM", "true").lower() == "true"

settings = Settings()