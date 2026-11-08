"""CLI: python main.py "Привет, кто ты?" [--model <id>]"""
import argparse
import sys

from config import ConfigError, load_settings
from llm_client import LLMClient, LLMError

SYSTEM_PROMPT = "Ты — полезный ассистент. Отвечай кратко и по делу, на языке пользователя."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Задать вопрос LLM через OpenCode Go.")
    parser.add_argument("prompt", nargs="?", help="текст запроса; без него спросим в консоли")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"Ошибка: {message}", file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()

    try:
        settings = load_settings()
    except ConfigError as e:
        return fail(str(e))

    try:
        prompt = args.prompt or input("Ваш запрос: ")
    except (EOFError, KeyboardInterrupt):
        return fail("ввод прерван.")
    if not prompt.strip():
        return fail("пустой запрос.")

    client = LLMClient(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=args.model or settings.model,
        timeout=settings.timeout,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        result = client.chat(messages)
    except LLMError as e:
        return fail(str(e))

    usage = result.usage
    print(result.text)
    print()
    print(
        f"{result.model} | токены: {usage.get('prompt_tokens', '?')}/"
        f"{usage.get('completion_tokens', '?')} | {result.latency:.2f} с"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
