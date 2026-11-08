"""Клиент OpenAI-совместимого Chat Completions API на голом httpx."""
import time
import uuid
from dataclasses import dataclass

import httpx

USER_AGENT = "ai-challenge-cli/0.1"


class LLMError(Exception):
    """Ошибка обращения к модели; текст сразу понятен пользователю."""


@dataclass
class ChatResult:
    text: str
    model: str
    usage: dict
    latency: float  # секунды


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0,
                 session_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # OpenCode Go требует стабильный ID диалога: по нему он маршрутизирует и кэширует запросы.
        self.session_id = session_id or uuid.uuid4().hex

    def chat(self, messages: list[dict]) -> ChatResult:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": USER_AGENT,
            "x-opencode-session": self.session_id,
        }
        payload = {"model": self.model, "messages": messages}

        start = time.perf_counter()
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
        except httpx.TimeoutException:
            raise LLMError(f"модель не ответила за {self.timeout:g} с (таймаут).")
        except httpx.RequestError as e:
            raise LLMError(f"нет связи с {self.base_url}. Проверьте интернет и LLM_BASE_URL. ({e})")
        latency = time.perf_counter() - start

        if not response.is_success:
            raise LLMError(self._http_error(response))

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError(f"неожиданный ответ сервера: {response.text[:200]!r}")

        return ChatResult(
            text=text or "",
            model=data.get("model") or self.model,
            usage=data.get("usage") or {},
            latency=latency,
        )

    def _http_error(self, response: httpx.Response) -> str:
        code = response.status_code
        if code in (401, 403):
            message = "ключ не принят. Проверьте OPENCODE_API_KEY в .env."
        elif code == 400:
            message = f"запрос отклонён. Часто причина — модель «{self.model}» недоступна для вашего аккаунта."
        elif code == 404:
            message = f"не найдено: проверьте модель «{self.model}» и LLM_BASE_URL."
        elif code == 429:
            message = "превышен лимит запросов. Подождите и повторите."
        elif code >= 500:
            message = "сбой на стороне сервера. Попробуйте позже."
        else:
            message = "сервер отклонил запрос."
        return f"HTTP {code}: {message} Ответ сервера: {_server_detail(response)}"


def _server_detail(response: httpx.Response) -> str:
    """Достаёт текст ошибки из JSON вида {"error": {"message": ...}} или из тела."""
    try:
        error = response.json().get("error")
        if isinstance(error, dict):
            error = error.get("message")
        if error:
            return str(error)
    except (ValueError, AttributeError):
        pass
    return response.text[:200] or "(пусто)"
