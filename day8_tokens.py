"""День 8. Работа с токенами.

Агент из Дня 7 (`PersistentAgent`: инструменты + история в JSON), который считает токены:
    • текущего запроса — сообщения пользователя и всего запроса к модели целиком;
    • всей истории диалога — сколько её хранится и сколько реально ушло в модель (с учётом кэша);
    • ответа модели — с разбивкой на рассуждение (reasoning) и текст.
Плюс стоимость каждого хода и диалога целиком и заполнение контекстного окна модели.

Откуда числа:
    • точные — из поля `usage` ответа API (prompt_tokens, completion_tokens, кэш, reasoning);
    • до запроса — оценка без токенизатора модели: по составу символов (кириллица, латиница, цифры…),
      которая после каждого ответа калибруется по точному prompt_tokens. Оценка нужна, чтобы решить,
      поместится ли запрос в окно, ещё до отправки.

Переполнение окна (`--overflow`):
    none     — отправлять всё как есть (как агент Дня 7): что сломается, решает API;
    truncate — скользящее окно: старые ходы целиком не отправляются, пока запрос + запас на ответ
               не поместятся в окно. Агент продолжает работать, но забывает начало диалога.
`--window N` эмулирует модель с окном N токенов по тем же правилам, что у настоящего API: запрос больше
окна отклоняется, а на ответ остаётся не больше «окно − запрос» (max_tokens). Так переполнение видно
за доли цента; настоящее окно deepseek-v4.1-flash — 1 048 576 токенов (сценарий real в --demo).

    python day8_tokens.py                                 # чат со счётчиком токенов (сессия tokens)
    python day8_tokens.py --window 6000                   # чат с эмуляцией маленького окна
    python day8_tokens.py --window 6000 --overflow truncate
    python day8_tokens.py --demo                          # 4 сценария + отчёт в results/
    python day8_tokens.py --demo --scenarios short,long   # только выбранные сценарии

Команды чата: /doc [N] — отправить синтетический отчёт склада (~900 токенов), /tokens, /window <N|real>,
/overflow <none|truncate>, /history, /tools, /reset, /exit.
"""
import argparse
import json
import math
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from config import ConfigError, load_settings
from day6_agent import TOOLS, AgentReply, print_event
from day7_memory import (MEMORY_DIR, ROOT, HistoryStore, PersistentAgent, RestoreInfo, Saved, StoreError,
                         print_history, print_restore)
from llm_client import LLMClient, LLMError, is_context_overflow

RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------- модель: окно и цены


@dataclass(frozen=True)
class Price:
    """Доллары за 1 млн токенов."""
    input: float
    output: float
    cached: float  # вход, который провайдер взял из кэша префиксов


# OpenCode Go, https://opencode.ai/docs/go/ (проверено 15.11.2026). Пик — будни 01:00–04:00 и 06:00–10:00 UTC.
PRICES = {
    "deepseek-v4.1-flash": {"off-peak": Price(0.15, 0.60, 0.003), "peak": Price(0.30, 1.20, 0.006)},
}
# Документация пишет «1M»; точное число — из ответа API на запрос больше окна (прогон 15.11.2026):
# «maximum context length is 1048576 tokens». Ответ — до 384K.
CONTEXT_WINDOWS = {"deepseek-v4.1-flash": 1_048_576}
MAX_OUTPUT = 384_000


def is_peak(moment: datetime) -> bool:
    utc = moment.astimezone(timezone.utc)
    return utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10)


def price_for(model: str, moment: datetime | None = None) -> tuple[Price, str] | None:
    tiers = PRICES.get(model)
    if not tiers:
        return None
    tier = "peak" if is_peak(moment or datetime.now(timezone.utc)) else "off-peak"
    return tiers[tier], tier


@dataclass
class Usage:
    """Поле usage ответа API в одном виде для разных провайдеров."""
    prompt: int = 0
    completion: int = 0
    cached: int = 0  # из prompt: сколько взято из кэша
    reasoning: int = 0  # из completion: сколько ушло на рассуждение

    @classmethod
    def parse(cls, usage: dict) -> "Usage":
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        return cls(prompt=int(usage.get("prompt_tokens") or 0),
                   completion=int(usage.get("completion_tokens") or 0),
                   cached=int(usage.get("prompt_cache_hit_tokens")  # DeepSeek
                              or prompt_details.get("cached_tokens") or 0),  # OpenAI-формат
                   reasoning=int(completion_details.get("reasoning_tokens") or 0))

    def cost(self, price: Price | None) -> float | None:
        if price is None:
            return None
        fresh = max(self.prompt - self.cached, 0)
        return (fresh * price.input + self.cached * price.cached + self.completion * price.output) / 1_000_000


# ---------------------------------------------------------------- оценка токенов до запроса


class TokenEstimator:
    """Оценка числа токенов без токенизатора модели, с калибровкой по точным prompt_tokens из API.

    Токенизатор DeepSeek — BPE: английское слово обычно 1 токен на ~4 символа, русское — на ~3,
    цифры идут группами до трёх, знаки препинания — отдельными токенами. Сырая оценка считает символы
    этих классов; `factor` — поправка, которую подбирает калибровка (сглаженное отношение факт/оценка).
    """
    PER_MESSAGE = 4  # служебные токены шаблона чата на каждое сообщение (роль, разделители)
    PER_REQUEST = 3

    def __init__(self):
        self.factor = 1.0
        self.samples = 0
        self.last_error: float | None = None  # ошибка последней оценки, доля: (оценка − факт) / факт

    @staticmethod
    def raw_text(text: str | None) -> float:
        if not text:
            return 0.0
        cyr = latin = digits = cjk = other = 0
        for ch in text:
            if ch.isspace():
                continue  # пробел почти всегда склеивается со следующим словом в один токен
            code = ord(ch)
            if 0x400 <= code <= 0x4FF:
                cyr += 1
            elif ch.isascii() and ch.isalpha():
                latin += 1
            elif ch.isdigit():
                digits += 1
            elif code >= 0x2E80:
                cjk += 1
            else:
                other += 1
        return cyr / 3.0 + latin / 4.0 + digits / 2.5 + cjk * 0.8 + other * 0.7

    def raw_message(self, m: dict, current_turn: bool = True) -> float:
        total = self.PER_MESSAGE + self.raw_text(m.get("content") if isinstance(m.get("content"), str) else
                                                 json.dumps(m.get("content"), ensure_ascii=False))
        if m.get("tool_calls"):
            total += self.raw_text(json.dumps(m["tool_calls"], ensure_ascii=False))
        # По документации DeepSeek рассуждения прошлых ходов в контекст не попадают; внутри хода с инструментами — попадают.
        if current_turn and m.get("reasoning_content"):
            total += self.raw_text(m["reasoning_content"])
        return total

    def raw_messages(self, messages: list[dict], tools: list[dict] | None = None) -> float:
        last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
        total = self.PER_REQUEST + sum(self.raw_message(m, i > last_user) for i, m in enumerate(messages))
        if tools:
            total += self.raw_text(json.dumps(tools, ensure_ascii=False))
        return total

    def apply(self, raw: float) -> int:
        return max(1, round(raw * self.factor)) if raw else 0

    def text(self, text: str) -> int:
        return self.apply(self.raw_text(text))

    def calibrate(self, raw: float, actual: int) -> None:
        if raw <= 0 or actual <= 0:
            return
        self.last_error = (self.apply(raw) - actual) / actual
        ratio = min(max(actual / raw, 0.3), 3.0)
        self.factor = ratio if self.samples == 0 else 0.5 * self.factor + 0.5 * ratio
        self.samples += 1


# ---------------------------------------------------------------- статистика


@dataclass
class CallStats:
    """Один запрос к модели внутри хода."""
    step: int
    estimate: int  # оценка входа до отправки
    prompt: int  # факт: токенов во входе (весь контекст)
    completion: int  # факт: токенов в ответе
    cached: int
    reasoning: int
    cost: float | None
    latency: float
    finish_reason: str | None
    sent_messages: int  # сколько сообщений ушло в запрос
    max_tokens: int | None  # потолок ответа (при эмуляции окна)
    tool_calls: int = 0
    usage_raw: dict = field(default_factory=dict)

    @property
    def provider_cost(self) -> float | None:
        """Стоимость, которую сообщил сам провайдер (OpenCode: usage.opencode.list_cost), — для сверки."""
        value = (self.usage_raw.get("opencode") or {}).get("list_cost")
        return float(value) if isinstance(value, (int, float)) else None


@dataclass
class TurnStats:
    """Один ход пользователя: всё, что по нему посчитано."""
    n: int
    user: str
    user_tokens: int  # ≈ токенов в сообщении пользователя
    history_before: int  # ≈ токенов во всей сохранённой истории до хода (без инструментов)
    history_after: int = 0  # ≈ то же после хода
    window: int = 0
    emulated: bool = False
    policy: str = "none"
    tier: str = ""
    calls: list[CallStats] = field(default_factory=list)
    dropped: int = 0  # сообщений истории, не отправленных в модель (truncate)
    status: str = "ok"  # ok | cut (ответ обрезан по max_tokens) | overflow | error
    error: str = ""
    answer: str = ""
    factor: float = 1.0  # поправка оценки после хода — сохраняется, чтобы калибровка пережила перезапуск
    last_estimate: int = 0  # оценка последнего запроса хода — по ней строится полоска окна, если ответа нет

    @property
    def prompt(self) -> int:
        return sum(c.prompt for c in self.calls)

    @property
    def completion(self) -> int:
        return sum(c.completion for c in self.calls)

    @property
    def cached(self) -> int:
        return sum(c.cached for c in self.calls)

    @property
    def reasoning(self) -> int:
        return sum(c.reasoning for c in self.calls)

    @property
    def cost(self) -> float | None:
        costs = [c.cost for c in self.calls]
        return None if any(c is None for c in costs) else sum(costs)

    @property
    def provider_cost(self) -> float | None:
        costs = [c.provider_cost for c in self.calls]
        return None if not costs or any(c is None for c in costs) else sum(costs)

    @property
    def context(self) -> int:
        """Самый большой запрос хода — им и определяется, влезает ли ход в окно."""
        return max((c.prompt + c.completion for c in self.calls), default=0)

    @classmethod
    def from_dict(cls, data: dict) -> "TurnStats":
        calls = [CallStats(**c) for c in data.get("calls") or []]
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "calls"}
        return cls(**known, calls=calls)


def clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def clip_user(text: str, limit: int = 400) -> str:
    """Текст запроса для статистики и отчёта: журнал на миллион токенов в JSON-отчёт не нужен."""
    return text if len(text) <= limit else f"{text[:limit]}… (+{fmt(len(text) - limit)} символов)"


def fmt(n: int | float | None) -> str:
    if n is None:
        return "?"
    return f"{round(n):,}".replace(",", " ")


def money(x: float | None) -> str:
    if x is None:
        return "—"
    return f"${x:.6f}" if x < 0.01 else f"${x:.4f}"


def bar(part: float, total: float, width: int = 20) -> str:
    if total <= 0:
        return ""
    filled = min(width, round(width * part / total))
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------- агент со счётчиком токенов


class ContextOverflow(LLMError):
    """Запрос не помещается в контекстное окно (решено до отправки — при эмуляции окна или truncate)."""


class TokenAgent(PersistentAgent):
    """PersistentAgent из Дня 7, который считает токены каждого запроса и следит за контекстным окном."""

    def __init__(self, client: LLMClient, tools, store: HistoryStore, *, window: int | None = None,
                 emulated: bool = False, policy: str = "none", reserve: int = 1024, **kwargs):
        self.estimator = TokenEstimator()
        self.ledger: list[TurnStats] = []
        self.window = window or CONTEXT_WINDOWS.get(client.model, 0)
        self.emulated = emulated
        self.policy = policy
        self.reserve = reserve  # truncate: сколько места оставить под ответ
        self._turn: TurnStats | None = None
        super().__init__(client, tools, store, **kwargs)

    # --- восстановление и сброс

    def restore(self, saved: Saved) -> RestoreInfo:
        info = super().restore(saved)
        self.ledger = [TurnStats.from_dict(t["tokens"]) for t in self.turns if isinstance(t.get("tokens"), dict)]
        if self.ledger:
            self.estimator.factor, self.estimator.samples = self.ledger[-1].factor, 1
        return info

    def reset(self) -> None:
        self.ledger = []
        super().reset()

    # --- оценки

    def history_tokens(self) -> int:
        """≈ токенов во всей сохранённой истории (то, что уйдёт в модель без обрезки, без схем инструментов)."""
        return self.estimator.apply(self.estimator.raw_messages(self.history))

    def totals(self) -> dict:
        costs = [t.cost for t in self.ledger]
        return {"turns": len(self.ledger), "prompt": sum(t.prompt for t in self.ledger),
                "completion": sum(t.completion for t in self.ledger), "cached": sum(t.cached for t in self.ledger),
                "cost": None if any(c is None for c in costs) else sum(costs)}

    # --- ход

    def run(self, user_text: str) -> AgentReply:
        priced = price_for(self.client.model)
        self._turn = TurnStats(n=len(self.ledger) + 1, user=clip_user(user_text), user_tokens=self.estimator.text(user_text),
                               history_before=self.history_tokens(), window=self.window, emulated=self.emulated,
                               policy=self.policy, tier=priced[1] if priced else "")
        turn = self._turn
        try:
            reply = super().run(user_text)  # PersistentAgent: цикл с инструментами, откат при ошибке, сохранение
        except LLMError as e:
            overflow = isinstance(e, ContextOverflow) or is_context_overflow(getattr(e, "detail", "") or str(e))
            turn.status, turn.error = ("overflow" if overflow else "error"), str(e)
            turn.history_after = self.history_tokens()
            self.ledger.append(turn)
            raise
        turn.answer = reply.text
        turn.factor = round(self.estimator.factor, 4)
        turn.history_after = self.history_tokens()
        if turn.calls and turn.calls[-1].finish_reason == "length":
            turn.status = "cut"
        self.ledger.append(turn)
        self.turns[-1]["tokens"] = asdict(turn)
        self.save()
        return reply

    def _select(self, schemas: list[dict]) -> tuple[list[dict], int]:
        """Что отправить в модель: (сообщения, сколько сообщений истории в запрос не попало).
        Точка расширения: День 9 подставляет сюда summary вместо старой части истории."""
        return self._fit(schemas) if self.policy == "truncate" else (self.history, 0)

    def _call_model(self, schemas: list[dict]):
        est = self.estimator
        messages, dropped = self._select(schemas)
        raw = est.raw_messages(messages, schemas)
        estimate = est.apply(raw)
        self._turn.dropped = max(self._turn.dropped, dropped)
        if dropped and self.policy == "truncate":
            self.on_event("truncate", {"dropped": dropped, "sent": len(messages), "total": len(self.history),
                                       "estimate": estimate, "window": self.window})

        max_tokens = None
        if self.emulated:
            room = self.window - estimate
            if room <= 0:
                self._turn.last_estimate = estimate
                raise ContextOverflow(
                    f"запрос ≈{fmt(estimate)} токенов не помещается в окно {fmt(self.window)} (эмуляция). "
                    f"Настоящий API в этом случае отвечает HTTP 400 — см. сценарий real в --demo.", status=400)
            max_tokens = min(room, MAX_OUTPUT)  # как у настоящей модели: вход + ответ ≤ окно

        self._turn.last_estimate = estimate
        result = self.client.chat(messages, tools=schemas, max_tokens=max_tokens)
        usage = Usage.parse(result.usage)
        est.calibrate(raw, usage.prompt)
        priced = price_for(self.client.model)
        call = CallStats(step=len(self._turn.calls) + 1, estimate=estimate, prompt=usage.prompt,
                         completion=usage.completion, cached=usage.cached, reasoning=usage.reasoning,
                         cost=usage.cost(priced[0] if priced else None), latency=result.latency,
                         finish_reason=result.finish_reason, sent_messages=len(messages), max_tokens=max_tokens,
                         tool_calls=len(result.tool_calls), usage_raw=result.usage)
        self._turn.calls.append(call)
        self.on_event("tokens", {"call": call, "window": self.window, "text": result.text})
        if self.emulated and usage.prompt > self.window:  # оценка ошиблась: по факту вход больше окна
            raise ContextOverflow(f"по факту вход {fmt(usage.prompt)} токенов больше окна {fmt(self.window)} "
                                  f"(оценка была {fmt(estimate)}): настоящий API такой запрос отклонил бы.", status=400)
        return result

    def _fit(self, schemas: list[dict]) -> tuple[list[dict], int]:
        """truncate: самое длинное окончание истории (с границы хода), которое влезает в окно с запасом на ответ.
        Системный промпт отправляется всегда, текущий ход — всегда целиком."""
        est = self.estimator
        system, rest = self.history[:1], self.history[1:]
        starts = [i for i, m in enumerate(rest) if m.get("role") == "user"]
        last_user = starts[-1] if starts else 0
        per_message = [est.raw_message(m, i > last_user) for i, m in enumerate(rest)]
        base = est.PER_REQUEST + sum(est.raw_message(m) for m in system) + est.raw_text(
            json.dumps(schemas, ensure_ascii=False))
        reserve = min(self.reserve, self.window // 4)  # в маленьком окне запас — не больше четверти окна
        budget = self.window - reserve
        for cut in starts:  # от самого раннего хода к текущему
            if est.apply(base + sum(per_message[cut:])) <= budget:
                return system + rest[cut:], cut
        estimate = est.apply(base + sum(per_message[last_user:]))
        self._turn.last_estimate = estimate
        raise ContextOverflow(f"даже один текущий ход (≈{fmt(estimate)} токенов) не помещается в окно "
                              f"{fmt(self.window)} с запасом {fmt(reserve)} на ответ.", status=400)


# ---------------------------------------------------------------- вывод


def on_event(kind: str, data: dict) -> None:
    if kind == "tokens":
        c: CallStats = data["call"]
        what = f"просит инструменты: {c.tool_calls}" if c.tool_calls else "финальный ответ"
        err = (c.estimate - c.prompt) / c.prompt * 100 if c.prompt else 0
        cache = f", кэш {fmt(c.cached)}" if c.cached else ""
        think = f" (рассуждение {fmt(c.reasoning)})" if c.reasoning else ""
        cap = f" из max_tokens {fmt(c.max_tokens)}" if c.max_tokens else ""
        print(f"  · шаг {c.step}: модель → {what} | вход {fmt(c.prompt)} (оценка {fmt(c.estimate)}, {err:+.0f}%{cache})"
              f" → выход {fmt(c.completion)}{think}{cap} | {c.latency:.2f} с | {money(c.cost)}")
        if c.finish_reason == "length":
            print("    ⚠️  ответ обрезан: finish_reason=length — модель упёрлась в max_tokens")
        if c.tool_calls and (data.get("text") or "").strip():
            print(f"    💭 {data['text'].strip()}")
    elif kind == "truncate":
        print(f"  ✂️  в окно {fmt(data['window'])} не помещается вся история: старые {fmt(data['dropped'])} сообщений "
              f"не отправлены, ушло {data['sent']} из {data['total']} (≈{fmt(data['estimate'])} токенов)")
    elif kind == "llm":
        pass  # строку шага печатает событие tokens — в нём больше подробностей
    elif kind == "reset":
        archived = data.get("archived")
        note = f" Прошлая история сохранена в {archived.name}." if archived else ""
        print(f"История очищена.{note}\n")
    else:
        print_event(kind, data)


def print_turn(agent: TokenAgent, turn: TurnStats) -> None:
    w = turn.window
    mode = f"окно {fmt(w)}" + (" (эмуляция)" if turn.emulated else "") + f", переполнение: {turn.policy}"
    print(f"📊 Токены · ход {turn.n} · {mode}")
    print(f"   ваш запрос:       ≈{fmt(turn.user_tokens)}")
    if turn.status in ("overflow", "error") and turn.last_estimate:
        sent = "не отправлен" if turn.emulated and not turn.calls else "отклонён API"
        print(f"   запрос к модели:  ≈{fmt(turn.last_estimate)} по оценке — {sent}, ход в историю не попал")
    if turn.calls:
        last = turn.calls[-1]
        cached = f" (из кэша {fmt(turn.cached)})" if turn.cached else ""
        print(f"   запрос к модели:  {fmt(last.prompt)} — весь контекст последнего шага; за ход вход {fmt(turn.prompt)}"
              f"{cached}, запросов {len(turn.calls)}")
        text_tokens = max(turn.completion - turn.reasoning, 0)
        split = f" (рассуждение {fmt(turn.reasoning)}, текст {fmt(text_tokens)})" if turn.reasoning else ""
        print(f"   ответ модели:     {fmt(turn.completion)}{split}, finish_reason={last.finish_reason}")
    dropped = f", в модель не ушло {turn.dropped} старых сообщений" if turn.dropped else ""
    print(f"   история диалога:  ≈{fmt(turn.history_after)} в памяти (сообщения без схем инструментов){dropped}")
    if w:
        used = turn.context or turn.last_estimate
        print(f"   окно:             {bar(used, w)} {used / w * 100:.1f}% ({fmt(used)} из {fmt(w)})")
    t = agent.totals()
    turn_cost = money(turn.cost) if turn.calls else "0 (ответа нет)"
    if turn.provider_cost is not None:
        turn_cost += f" (OpenCode: {money(turn.provider_cost)})"
    print(f"   стоимость:        ход {turn_cost} · диалог {money(t['cost'])} "
          f"({t['turns']} ход., вход {fmt(t['prompt'])}, выход {fmt(t['completion'])})"
          + (f", тариф {turn.tier}" if turn.tier else ""))


def print_ledger(agent: TokenAgent) -> None:
    if not agent.ledger:
        print("  (ходов ещё не было)")
        return
    peak = max((t.context or t.last_estimate) for t in agent.ledger) or 1
    print(f"  {'ход':>3} {'запрос':>7} {'контекст':>9} {'ответ':>6} {'$ хода':>10} {'$ итого':>10}  рост контекста")
    total = 0.0
    for t in agent.ledger:
        total += t.cost or 0
        status = {"ok": "", "cut": " ⚠️ обрезан", "overflow": " ❌ переполнение", "error": " ❌ ошибка"}[t.status]
        used = t.context or t.last_estimate
        cost = money(t.cost) if t.calls else "—"
        print(f"  {t.n:>3} {fmt(t.user_tokens):>7} {fmt(used):>9} {fmt(t.completion):>6} {cost:>10} "
              f"{money(total):>10}  {bar(used, peak, 24)}{status}")


# ---------------------------------------------------------------- чат


def chat(agent: TokenAgent, real_window: int) -> int:
    print(f"Агент на {agent.client.model}. Окно {fmt(agent.window)}{' (эмуляция)' if agent.emulated else ''}, "
          f"переполнение: {agent.policy}. Команды: /doc [N] — отправить отчёт склада, /tokens, /window <N|real>, "
          f"/overflow <none|truncate>, /history, /tools, /reset, /exit\n")
    next_doc = 1
    while True:
        try:
            text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return 0
        command, _, arg = text.partition(" ")
        try:
            if command == "/reset":
                agent.reset()
                continue
            if command == "/history":
                print_history(agent.history)
                print(f"  ≈{fmt(agent.history_tokens())} токенов в истории\n")
                continue
            if command == "/tools":
                for t in agent.tools.values():
                    print(f"  {t.name}: {t.description}")
                print()
                continue
            if command == "/tokens":
                print_ledger(agent)
                print()
                continue
            if command == "/window":
                if arg.strip() == "real":
                    agent.window, agent.emulated = real_window, False
                elif arg.strip().isdigit() and int(arg) > 0:
                    agent.window, agent.emulated = int(arg), True
                else:
                    print("  /window <число токенов> — эмулировать окно; /window real — настоящее окно модели\n")
                    continue
                print(f"  Окно: {fmt(agent.window)}{' (эмуляция)' if agent.emulated else ''}. "
                      f"История сейчас ≈{fmt(agent.history_tokens())} токенов.\n")
                continue
            if command == "/doc":
                if arg.strip() and not arg.strip().isdigit():
                    print("  /doc [N] — отправить отчёт склада №N (без номера — следующий по порядку)\n")
                    continue
                k = int(arg) if arg.strip() else next_doc
                next_doc = k + 1
                text = report_prompt(k)
                print(f"  → отправляю отчёт склада №{k}: {len(text.splitlines())} строк, ≈{fmt(agent.estimator.text(text))} "
                      f"токенов. История сейчас ≈{fmt(agent.history_tokens())}.")
            if command == "/overflow":
                if arg.strip() not in ("none", "truncate"):
                    print("  /overflow none | truncate\n")
                    continue
                agent.policy = arg.strip()
                print(f"  При переполнении: {agent.policy}\n")
                continue
            if command.startswith("/") and command != "/doc":
                print("  Неизвестная команда. Есть: /doc, /tokens, /window, /overflow, /history, /tools, /reset, /exit\n")
                continue
            reply = agent.run(text)
        except LLMError as e:
            print(f"❌ Ошибка: {e}", file=sys.stderr)
            if agent.ledger and agent.ledger[-1].status == "overflow":
                print_turn(agent, agent.ledger[-1])
                print("   (ход отклонён и в историю не попал)")
            print()
            continue
        except StoreError as e:
            print(f"⚠️  {e}\n", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("\n(ход прерван — в историю он не попал)\n")
            continue
        print(f"\nАгент: {reply.text.strip() or '(пустой ответ)'}\n")
        print_turn(agent, agent.ledger[-1])
        print()


# ---------------------------------------------------------------- сценарии для сравнения


FACT = ("Запомни, это понадобится в конце: номер моего заказа — 7341-ZX, кодовое слово — «бергамот». "
        "Ответь одним словом: «Запомнил».")
FINAL = "Напомни, какой у меня номер заказа и кодовое слово? Ответь по истории диалога, инструменты не вызывай."

GOODS = [("кабель медный ВВГ 3×2,5", "м"), ("автоматический выключатель 16 А", "шт"), ("розетка двойная", "шт"),
         ("гофротруба 20 мм", "м"), ("светильник LED 36 Вт", "шт"), ("щит распределительный на 24 модуля", "шт"),
         ("провод ПуГВ 1×6", "м"), ("УЗО 40 А 30 мА", "шт"), ("клеммник WAGO на 3 провода", "уп"),
         ("кабель-канал 40×25", "м"), ("выключатель одноклавишный", "шт"), ("дифавтомат 25 А", "шт"),
         ("лампа E27 12 Вт", "шт"), ("стабилизатор напряжения 5 кВт", "шт"), ("изолента ПВХ 19 мм", "шт"),
         ("датчик движения", "шт"), ("удлинитель 5 м на 4 гнезда", "шт"), ("счётчик электроэнергии", "шт")]
SUPPLIERS = ["«Северная линия»", "«ЭлектроСнаб»", "«Вольт-Трейд»", "«Ампер и сыновья»", "«СветоПро»"]


def warehouse_report(k: int, rows: int = 22) -> str:
    """Синтетический отчёт склада: ~900 токенов текста, который надо прочитать, чтобы ответить."""
    rnd = random.Random(k)
    lines = [f"Отчёт склада №{k} за {k + 3} ноября 2026 г. Ответственный: смена {rnd.choice('АБВГ')}."]
    for i, (name, unit) in enumerate(rnd.sample(GOODS, k=min(rows, len(GOODS))), 1):
        qty, price = rnd.randint(3, 400), rnd.uniform(15, 9000)
        lines.append(f"{i:02d}. {name} — {qty} {unit} по {price:.2f} ₽, поставщик {rnd.choice(SUPPLIERS)}, "
                     f"ячейка {rnd.choice('АБВ')}-{rnd.randint(1, 40)}, приход {rnd.randint(1, 14)}.11.")
    return "\n".join(lines)


def report_prompt(k: int) -> str:
    return f"{warehouse_report(k)}\n\nОтветь одной строкой: какая позиция в отчёте №{k} самая дорогая за единицу?"


def log_dump(lines: int, offset: int = 0) -> str:
    """ASCII-журнал датчиков: предсказуемое число токенов на строку — для запроса больше окна."""
    out = []
    for i in range(offset, offset + lines):
        out.append(f"{i:07d} 2026-11-15T{(i // 3600) % 24:02d}:{(i // 60) % 60:02d}:{i % 60:02d} "
                   f"sensor-{i % 97:02d} temp={20 + (i * 7) % 150 / 10:.1f} hum={30 + i % 50} status=ok")
    return "\n".join(out)


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    about: str
    window: int | None = None  # None — настоящее окно модели; число — эмуляция
    policy: str = "none"


SCENARIOS = {
    "short": [Scenario("short", "короткий диалог", "3 хода, два из них с инструментами")],
    "long": [Scenario("long", "длинный диалог", "факт в начале, 10 отчётов склада по ~900 токенов, вопрос о факте")],
    "overflow": [
        Scenario("overflow_none", "переполнение окна, без защиты",
                 "тот же длинный диалог + просьба о длинном ответе, окно 6 000 (эмуляция)",
                 window=6000, policy="none"),
        Scenario("overflow_truncate", "переполнение окна, скользящее окно",
                 "тот же диалог в окне 6 000 (эмуляция), старые ходы не отправляются", window=6000,
                 policy="truncate"),
    ],
    "real": [Scenario("real", "запрос больше настоящего окна",
                      "журнал на ~1,05 окна в окно 1 048 576 deepseek-v4.1-flash")],
}

SHORT_STEPS = ["Привет! Кто ты и что умеешь? Ответь в двух предложениях.",
               "Который час сейчас в Токио?",
               "Спасибо! А сколько рублей стоят 20 евро?"]
LONG_STEPS = [FACT] + [report_prompt(k) for k in range(1, 11)] + [FINAL]
# В сценариях переполнения перед вопросом о факте — просьба о длинном ответе: проверяет, хватит ли места под ответ.
LIST_STEP = ("Перечисли все позиции из последнего отчёта склада в нашем диалоге с ценой за единицу — "
             "по одной на строку, без комментариев.")
OVERFLOW_STEPS = LONG_STEPS[:-1] + [LIST_STEP, FINAL]


def remembers_fact(text: str) -> bool:
    return "7341" in text and "бергамот" in text.lower()


def run_steps(agent: TokenAgent, steps: list[str], total: int | None = None) -> None:
    for i, prompt in enumerate(steps, len(agent.ledger) + 1):
        first_line = prompt.splitlines()[0]
        print(f"\n--- ход {i}/{total or len(steps)}: {clip(first_line, 110)}"
              + (f" (+{len(prompt.splitlines()) - 1} строк)" if "\n" in prompt else ""))
        try:
            reply = agent.run(prompt)
            print(f"Агент: {clip(reply.text, 300) if reply.text.strip() else '(пустой ответ)'}")
        except LLMError as e:
            print(f"❌ {e}")
        print_turn(agent, agent.ledger[-1])


def make_agent(client: LLMClient, sc: Scenario, stamp: str, max_steps: int, reserve: int) -> TokenAgent:
    store = HistoryStore(MEMORY_DIR / f"day8_{sc.id}_{stamp}.json")
    return TokenAgent(client, TOOLS, store, window=sc.window, emulated=sc.window is not None, policy=sc.policy,
                      reserve=reserve, max_steps=max_steps, on_event=on_event)


def check(sc: Scenario, agent: TokenAgent) -> tuple[bool, str]:
    ledger = agent.ledger
    overflows = [t.n for t in ledger if t.status == "overflow"]
    final = ledger[-1].answer if ledger else ""
    if sc.id == "short":
        used = [c for t in ledger for c in t.calls if c.tool_calls]
        ok = all(t.status == "ok" for t in ledger) and len(used) >= 2
        return ok, "все ходы без ошибок, инструменты вызваны" if ok else "были ошибки или инструменты не вызваны"
    if sc.id == "long":
        grows = all(b.calls[0].prompt > a.calls[0].prompt for a, b in zip(ledger, ledger[1:]) if a.calls and b.calls)
        ok = not overflows and remembers_fact(final) and grows
        return ok, ("контекст растёт каждый ход, факт из хода 1 помнит" if ok else
                    f"переполнений {len(overflows)}, рост {'да' if grows else 'нет'}, "
                    f"факт {'помнит' if remembers_fact(final) else 'не помнит'}")
    if sc.id == "overflow_none":
        cut = [t.n for t in ledger if t.status == "cut"]
        ok = bool(overflows)
        return ok, (f"отклонены ходы {', '.join(map(str, overflows))}" if overflows else "переполнения не было") + (
            f"; ответ обрезан в ходах {', '.join(map(str, cut))}" if cut else "") + (
            f"; на вопрос о факте {'ответил' if remembers_fact(final) else 'не ответил'}")
    if sc.id == "overflow_truncate":
        dropped = max((t.dropped for t in ledger), default=0)
        forgot = not remembers_fact(final)
        ok = not overflows and dropped > 0 and forgot
        return ok, (f"переполнений нет, до {dropped} старых сообщений не отправлялись, "
                    f"факт из хода 1 {'забыт (ожидаемо)' if forgot else 'всё ещё помнит'}")
    if sc.id == "real":
        big = next((t for t in ledger if t.n == 3), None)
        alive = len(ledger) >= 4 and ledger[3].status == "ok"
        if big is None:
            return False, "сценарий не дошёл до большого запроса"
        if big.status == "overflow":
            return alive, f"API отклонил запрос больше окна; следующий ход {'прошёл' if alive else 'не прошёл'}"
        return False, f"большой запрос не отклонён: статус {big.status}, вход {fmt(big.prompt)}"
    return False, "?"


def run_real(client: LLMClient, sc: Scenario, stamp: str, max_steps: int, reserve: int,
             settings_timeout: float) -> TokenAgent:
    """Диалог, в котором третье сообщение больше настоящего окна модели. Размер подбирается по факту:
    сначала маленький журнал, по его prompt_tokens считается, сколько строк нужно на ~1,05 окна."""
    big_client = LLMClient(client.base_url, client.api_key, client.model, max(settings_timeout, 300.0))
    agent = make_agent(big_client, sc, stamp, max_steps, reserve)
    probe_lines = 1000
    steps = ["Привет! Ответь одним словом: «готов».",
             f"Вот журнал датчиков:\n{log_dump(probe_lines)}\n\nОтветь одним словом: «получил»."]
    run_steps(agent, steps, total=4)
    if len(agent.ledger) < 2 or agent.ledger[-1].status != "ok":
        return agent
    before, after = agent.ledger[0].calls[-1], agent.ledger[1].calls[-1]
    per_line = (after.prompt - before.prompt) / probe_lines
    target = int(agent.window * 1.05)
    lines = math.ceil((target - after.prompt) / per_line)
    megabytes = lines * (len(log_dump(100, probe_lines)) + 100) / 100 / 1e6
    priced = price_for(client.model)
    cost = f" Если сервер всё же примет запрос, он обойдётся примерно в {money(target * priced[0].input / 1e6)}." \
        if priced else ""
    print(f"\nПо факту строка журнала ≈ {per_line:.1f} токена. Для ~{fmt(target)} токенов нужно {fmt(lines)} строк "
          f"(~{megabytes:.1f} МБ текста).{cost}")
    run_steps(agent, [f"Вот продолжение журнала:\n{log_dump(lines, probe_lines)}\n\nОтветь одним словом: «получил».",
                      "Ты ещё здесь? Ответь одним словом."], total=4)
    return agent


def demo(client: LLMClient, which: list[str], max_steps: int, reserve: int, settings_timeout: float) -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    for key in which:
        for sc in SCENARIOS[key]:
            print(f"\n{'=' * 78}\n=== {sc.title}: {sc.about}\n{'=' * 78}")
            if sc.id == "real":
                agent = run_real(client, sc, stamp, max_steps, reserve, settings_timeout)
            else:
                agent = make_agent(client, sc, stamp, max_steps, reserve)
                run_steps(agent, {"short": SHORT_STEPS, "long": LONG_STEPS}.get(sc.id, OVERFLOW_STEPS))
            ok, note = check(sc, agent)
            print(f"\nИтог сценария: {'✅' if ok else '❌'} {note}")
            print_ledger(agent)
            results.append({"scenario": asdict(sc), "ok": ok, "note": note, "window": agent.window,
                            "totals": agent.totals(), "estimator_factor": round(agent.estimator.factor, 4),
                            "turns": [asdict(t) | {"cost": t.cost, "context": t.context} for t in agent.ledger],
                            "history_file": agent.store.path.name})
    passed = sum(r["ok"] for r in results)
    print(f"\nИтог: {passed}/{len(results)} сценариев прошли проверку.")
    save_report(results, client.model, stamp)
    return 0 if passed == len(results) else 2


# ---------------------------------------------------------------- отчёт


def save_report(results: list[dict], model: str, stamp: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path, md_path = RESULTS_DIR / f"day8_{stamp}.json", RESULTS_DIR / f"day8_{stamp}.md"
    json_path.write_text(json.dumps({"model": model, "prices": {k: {t: asdict(p) for t, p in v.items()}
                                                                for k, v in PRICES.items()},
                                     "scenarios": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    status_icon = {"ok": "✅", "cut": "⚠️ обрезан", "overflow": "❌ переполнение", "error": "❌ ошибка"}
    lines = [f"# День 8 · Работа с токенами — прогон {stamp}", "",
             f"Модель: `{model}`. Цены OpenCode Go за 1 млн токенов: вне пика $0.15 вход / $0.60 выход / "
             f"$0.003 кэш, в пик вдвое дороже. «Запрос» и «история» до отправки — оценка (≈), "
             f"вход/выход/кэш — точные числа из `usage`.", "",
             "## Сравнение", "",
             "| Сценарий | Окно | Ходов | Контекст: первый → последний ход | Вход всего | Выход всего | Кэш | "
             "Стоимость | Проверка |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        turns, t = r["turns"], r["totals"]
        ctx = [x["context"] for x in turns if x["context"]]
        window = f"{fmt(r['window'])}{' (эмул.)' if r['scenario']['window'] else ''}, {r['scenario']['policy']}"
        lines.append(f"| {r['scenario']['title']} | {window} | {len(turns)} | "
                     f"{fmt(ctx[0]) if ctx else '—'} → {fmt(ctx[-1]) if ctx else '—'} | {fmt(t['prompt'])} | "
                     f"{fmt(t['completion'])} | {fmt(t['cached'])} | {money(t['cost'])} | "
                     f"{'✅' if r['ok'] else '❌'} {r['note']} |")
    for r in results:
        sc, turns = r["scenario"], r["turns"]
        peak = max((x["context"] or x["last_estimate"]) for x in turns) if turns else 1
        lines += ["", f"## {sc['title']}", "", f"{sc['about']}. Файл истории: `memory/{r['history_file']}`.", "",
                  "| Ход | Запрос ≈ | История ≈ | Вход (кэш) | Ответ (рассужд.) | Окно | $ хода | $ итого | "
                  "Рост контекста | Статус |", "|---|---|---|---|---|---|---|---|---|---|"]
        total = 0.0
        for x in turns:
            total += x["cost"] or 0
            used = x["context"] or x["last_estimate"]
            prompt = sum(c["prompt"] for c in x["calls"])
            cached = sum(c["cached"] for c in x["calls"])
            completion = sum(c["completion"] for c in x["calls"])
            reasoning = sum(c["reasoning"] for c in x["calls"])
            dropped = f", −{x['dropped']} сообщ." if x["dropped"] else ""
            sent = f"{fmt(prompt)} ({fmt(cached)})" if x["calls"] else f"≈{fmt(used)} — не принят"
            answer = f"{fmt(completion)} ({fmt(reasoning)})" if x["calls"] else "—"
            lines.append(f"| {x['n']} | {fmt(x['user_tokens'])} | {fmt(x['history_after'])}{dropped} | "
                         f"{sent} | {answer} | "
                         f"{used / r['window'] * 100 if r['window'] else 0:.1f}% | "
                         f"{money(x['cost']) if x['calls'] else '—'} | {money(total)} | "
                         f"`{bar(used, peak, 20)}` | {status_icon[x['status']]} |")
        errors = [x for x in turns if x["error"]]
        if errors:
            lines += [""] + [f"- ход {x['n']}: {x['error']}" for x in errors]
        last = turns[-1] if turns else None
        if last and last["answer"]:
            lines += ["", f"**Последний ответ:** {last['answer'].strip()}"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Отчёт: {md_path.relative_to(ROOT)}, {json_path.name}")


# ---------------------------------------------------------------- запуск


def main() -> int:
    parser = argparse.ArgumentParser(description="День 8: агент, который считает токены и следит за окном.")
    parser.add_argument("prompt", nargs="?", help="один ход и выход; без него — чат в консоли")
    parser.add_argument("--session", default="tokens", help="имя сессии: файл memory/<имя>.json")
    parser.add_argument("--new", action="store_true", help="начать новый диалог (старая история уходит в архив)")
    parser.add_argument("--window", type=int, help="эмулировать окно в N токенов (по умолчанию — настоящее окно)")
    parser.add_argument("--overflow", choices=("none", "truncate"), default="none",
                        help="что делать при переполнении окна")
    parser.add_argument("--reserve", type=int, default=1024,
                        help="truncate: запас под ответ, токенов (не больше четверти окна)")
    parser.add_argument("--demo", action="store_true", help="сравнение сценариев + отчёт в results/")
    parser.add_argument("--scenarios", default="short,long,overflow,real",
                        help="для --demo: через запятую из short, long, overflow, real")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--max-steps", type=int, default=8, help="предел запросов к модели за один ход")
    args = parser.parse_args()

    if not re.fullmatch(r"[\w.-]+", args.session):
        print("Ошибка: имя сессии — буквы, цифры, точка, дефис, подчёркивание.", file=sys.stderr)
        return 1
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    model = args.model or settings.model
    real_window = CONTEXT_WINDOWS.get(model, 0)

    if args.demo:
        which = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        unknown = [s for s in which if s not in SCENARIOS]
        if unknown or not which:
            print(f"Ошибка: неизвестные сценарии {', '.join(unknown)}. Есть: {', '.join(SCENARIOS)}.", file=sys.stderr)
            return 1
        client = LLMClient(settings.base_url, settings.api_key, model, settings.timeout)
        return demo(client, which, args.max_steps, args.reserve, settings.timeout)

    store = HistoryStore(MEMORY_DIR / f"{args.session}.json")
    try:
        if args.new and (archived := store.archive()):
            print(f"Прошлая история сохранена в {archived.name}.")
        saved = store.load()
    except StoreError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    client = LLMClient(settings.base_url, settings.api_key, model, settings.timeout,
                       session_id=(saved.session_id or None) if saved else None)
    agent = TokenAgent(client, TOOLS, store, window=args.window or real_window, emulated=args.window is not None,
                       policy=args.overflow, reserve=args.reserve, max_steps=args.max_steps, on_event=on_event)
    info = agent.restore(saved) if saved else RestoreInfo(1, 0, 0, 0, "")
    print_restore(info, store, agent)
    if agent.ledger:
        t = agent.totals()
        print(f"📊 До перезапуска: {t['turns']} ход., вход {fmt(t['prompt'])}, выход {fmt(t['completion'])}, "
              f"{money(t['cost'])}. История ≈{fmt(agent.history_tokens())} токенов.")
    print()

    if args.prompt:
        try:
            reply = agent.run(args.prompt)
        except (LLMError, StoreError) as e:
            print(f"❌ Ошибка: {e}", file=sys.stderr)
            if agent.ledger:
                print_turn(agent, agent.ledger[-1])
            return 1
        print(f"\nАгент: {reply.text.strip()}\n")
        print_turn(agent, agent.ledger[-1])
        return 0
    return chat(agent, real_window)


if __name__ == "__main__":
    from run_timer import run_timed
    sys.exit(run_timed(main))
