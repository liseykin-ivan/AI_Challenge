"""Загрузка и проверка настроек из .env."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).with_name(".env")
DEFAULT_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_TIMEOUT = 60.0


class ConfigError(Exception):
    """Ошибка настроек; текст сразу понятен пользователю."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    timeout: float


def load_settings() -> Settings:
    load_dotenv(ENV_FILE)

    api_key = os.getenv("OPENCODE_API_KEY", "").strip()
    if not api_key or api_key == "your-key-here":
        raise ConfigError(
            "не задан OPENCODE_API_KEY. Скопируйте .env.example в .env и впишите ключ."
        )

    try:
        timeout = float(os.getenv("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        raise ConfigError("LLM_TIMEOUT в .env должен быть числом (секунды).")

    return Settings(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL,
        model=os.getenv("LLM_MODEL") or DEFAULT_MODEL,
        timeout=timeout,
    )
