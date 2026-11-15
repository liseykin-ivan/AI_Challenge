"""Клиент OpenAI-совместимого Chat Completions API на голом httpx.

Временные сбои — таймаут, обрыв связи, HTTP 408/409/425/429 и любой 5xx — повторяются: до `retries` раз
(LLM_RETRIES в .env, по умолчанию 2), с паузой 1, 2, 4… с ±25% (или сколько просит заголовок Retry-After).
Ошибки запроса (400, 401, 404, переполнение окна) не повторяются: повтор дал бы тот же ответ.
"""
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import httpx

USER_AGENT = "ai-challenge-cli/0.1"
DEFAULT_RETRIES = 2
RETRY_STATUSES = {408, 409, 425, 429}  # плюс все 5xx: временные сбои, повтор может пройти (как в SDK OpenAI)
MAX_DELAY = 30.0  # потолок паузы между попытками, секунды


class LLMError(Exception):
    """Ошибка обращения к модели; текст сразу понятен пользователю.
    status — HTTP-код ответа (None, если до сервера не достучались), detail — текст ошибки от сервера,
    retryable — сбой временный, повтор может пройти; attempts — сколько попыток было сделано."""

    def __init__(self, message: str, status: int | None = None, detail: str = "", *, retryable: bool = False,
                 retry_after: float | None = None, attempts: int = 1):
        super().__init__(message)
        self.status = status
        self.detail = detail
        self.retryable = retryable
        self.retry_after = retry_after  # секунды из заголовка Retry-After, если сервер его прислал
        self.attempts = attempts


# По этим словам в тексте ошибки сервера узнаём, что запрос не поместился в контекстное окно модели.
CONTEXT_OVERFLOW_HINTS = ("context length", "context_length", "context window", "maximum context",
                          "too many tokens", "too long", "token limit", "exceeds")


@dataclass
class ChatResult:
    text: str
    model: str
    usage: dict
    latency: float  # секунды
    finish_reason: str | None = None  # "stop" | "length" | ... — почему модель остановилась
    reasoning: str = ""  # скрытое рассуждение reasoning-моделей (reasoning_content), если провайдер его отдаёт
    tool_calls: list[dict] = field(default_factory=list)  # запросы модели на вызов инструментов (если передавали tools)
    message: dict = field(default_factory=dict)  # сообщение ассистента как пришло от сервера — для истории диалога агента
    extra: dict = field(default_factory=dict)  # прочие поля ответа и заголовки провайдера (id, opencode-*, x-ratelimit-*)
    attempts: int = 1  # сколько попыток понадобилось (1 — с первого раза)
    retry_errors: list[str] = field(default_factory=list)  # что случилось с неудачными попытками


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0,
                 session_id: str | None = None, retries: int | None = None, backoff: float = 1.0,
                 on_retry: Callable[[dict], None] | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # OpenCode Go требует стабильный ID диалога: по нему он маршрутизирует и кэширует запросы.
        self.session_id = session_id or uuid.uuid4().hex
        self.retries = env_retries() if retries is None else max(retries, 0)
        self.backoff = backoff
        self.on_retry = on_retry  # колбэк перед паузой: {"attempt", "retries", "delay", "error"}
        self.stats = {"requests": 0, "retries": 0, "failed": 0}  # по всем вызовам chat() этого клиента

    def chat(self, messages: list[dict], **params) -> ChatResult:
        """params — дополнительные поля запроса: max_tokens, stop, response_format, temperature, tools...
        Временный сбой повторяется до self.retries раз; если не помогло — LLMError с числом попыток."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": USER_AGENT,
            "x-opencode-session": self.session_id,
        }
        payload = {"model": self.model, "messages": messages}
        payload.update({k: v for k, v in params.items() if v is not None})

        self.stats["requests"] += 1
        errors: list[str] = []
        for attempt in range(1, self.retries + 2):
            try:
                result = self._post(url, headers, payload)
            except LLMError as e:
                if not e.retryable or attempt > self.retries:
                    self.stats["failed"] += 1
                    if attempt == 1:
                        raise
                    raise LLMError(f"{e} Попыток: {attempt}, все неудачные.", status=e.status, detail=e.detail,
                                   retryable=e.retryable, attempts=attempt) from e
                delay = self._delay(attempt, e.retry_after)
                errors.append(str(e))
                self.stats["retries"] += 1
                if self.on_retry:
                    self.on_retry({"attempt": attempt, "retries": self.retries, "delay": delay, "error": e})
                time.sleep(delay)
                continue
            result.attempts, result.retry_errors = attempt, errors
            return result
        raise AssertionError("unreachable")

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(max(retry_after, 0.0), MAX_DELAY)
        base = self.backoff * 2 ** (attempt - 1)
        return min(base * random.uniform(0.75, 1.25), MAX_DELAY)

    def _post(self, url: str, headers: dict, payload: dict) -> ChatResult:
        """Одна попытка запроса."""
        start = time.perf_counter()
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
        except httpx.TimeoutException:
            raise LLMError(f"модель не ответила за {self.timeout:g} с (таймаут).", retryable=True)
        except httpx.UnsupportedProtocol as e:
            raise LLMError(f"неверный адрес {self.base_url}: проверьте LLM_BASE_URL. ({e})")
        except httpx.RequestError as e:
            raise LLMError(f"нет связи с {self.base_url}. Проверьте интернет и LLM_BASE_URL. ({e})", retryable=True)
        latency = time.perf_counter() - start

        if not response.is_success:
            detail = _server_detail(response)
            code = response.status_code
            raise LLMError(self._http_error(response), status=code, detail=detail,
                           retryable=(code in RETRY_STATUSES or code >= 500) and not is_context_overflow(detail),
                           retry_after=_retry_after(response))

        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content")
            tool_calls = message.get("tool_calls") or []
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError(f"неожиданный ответ сервера: {response.text[:200]!r}")

        return ChatResult(
            text=text or "",
            model=data.get("model") or self.model,
            usage=data.get("usage") or {},
            latency=latency,
            finish_reason=choice.get("finish_reason"),
            reasoning=reasoning if isinstance(reasoning, str) else str(reasoning),
            tool_calls=tool_calls,
            message=message,
            extra=_extra(data, response.headers),
        )

    def _http_error(self, response: httpx.Response) -> str:
        code = response.status_code
        detail = _server_detail(response)
        if code in (400, 413, 422) and is_context_overflow(detail):
            message = "запрос не помещается в контекстное окно модели (слишком длинная история или сообщение)."
        elif code == 413:
            message = "запрос слишком большой: сервер не принял тело запроса."
        elif code in (401, 403):
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
        return f"HTTP {code}: {message} Ответ сервера: {detail}"


def env_retries() -> int:
    """LLM_RETRIES из окружения (config.load_settings уже проверил и загрузил .env); по умолчанию 2."""
    try:
        return max(int(os.getenv("LLM_RETRIES") or DEFAULT_RETRIES), 0)
    except ValueError:
        return DEFAULT_RETRIES


def _retry_after(response: httpx.Response) -> float | None:
    """Retry-After в секундах (формат с датой не разбираем — тогда обычная пауза)."""
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return None


def is_context_overflow(text: str) -> bool:
    text = (text or "").lower()
    return any(hint in text for hint in CONTEXT_OVERFLOW_HINTS)


def _extra(data: dict, headers: httpx.Headers) -> dict:
    """Всё из ответа, кроме choices и usage, плюс заголовки провайдера про лимиты и учёт запроса."""
    extra = {k: v for k, v in data.items() if k not in ("choices", "usage")}
    marked = {k: v for k, v in headers.items() if "opencode" in k.lower() or "ratelimit" in k.lower()}
    if marked:
        extra["headers"] = marked
    return extra


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
