"""День 10. Управление контекстом: сравнение стратегий.

Агент из Дня 8 (`TokenAgent`: инструменты, история в JSON, счётчик токенов и стоимости) с переключаемой
стратегией того, что уходит в модель:
    sliding — Sliding Window: системный промпт + последние N сообщений, всё, что старше, отбрасывается;
    facts   — Sticky Facts: отдельный блок «факты» (ключ → значение). После каждого сообщения пользователя его
              обновляет служебный запрос к модели; в запрос уходят факты + последние N сообщений;
    summary — summary из Дня 9: старая часть истории каждые `--chunk` сообщений сжимается в резюме,
              в запрос уходят summary + несжатая часть;
    full    — эталон: вся история в каждом запросе (как в Днях 7–8).
Полная история всегда хранится на диске: стратегия решает только, что отправить в модель. Переключение
(/strategy) ничего не теряет: состояние стратегии догоняется лениво — факты досчитываются по пропущенным
сообщениям пользователя, summary сжимает то, что накопилось.

Ветвление диалога: /checkpoint <имя> — снимок (история + факты + summary); /branch <имя> [checkpoint] — новая
ветка от снимка (без checkpoint — от текущего места); /switch <ветка> — переключиться. Ветки независимы:
у каждой своя история и свои факты. Всё лежит в memory/day10/<сессия>/.

    python day10_strategies.py                            # чат (сессия strategies, стратегия facts)
    python day10_strategies.py --strategy sliding --keep 4
    python day10_strategies.py --demo                     # сценарий Дня 9 на всех стратегиях + ветвление
    python day10_strategies.py --demo --modes sliding,facts --no-branches

Команды чата: /strategy [имя], /play [N], /probe, /facts, /summary, /context, /compress,
/checkpoint <имя>, /branch <имя> [checkpoint], /switch <ветка>, /branches, /tokens, /history, /reset, /exit.
"""
import argparse
import ast
import inspect
import json
import os
import re
import sys
import textwrap
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from config import ConfigError, load_settings
from day6_agent import TOOLS
from day7_memory import MEMORY_DIR, ROOT, HistoryStore, RestoreInfo, Saved, StoreError, now_iso, print_history, \
    print_restore
from day8_tokens import TokenAgent, TokenEstimator, TurnStats, Usage, clip, fmt, money, price_for, print_ledger
from day9_compression import (AGENT_PROMPT, DETAILS, DIALOGUE, PROBE_SUFFIX, SUMMARIZER_PROMPTS, SUMMARY_HEADER,
                              Detail, matches, normalize, render, watch_retries)
from day9_compression import on_event as day9_event
from llm_client import LLMClient, LLMError

RESULTS_DIR = ROOT / "results"
DAY_DIR = MEMORY_DIR / "day10"
STRATEGIES = ("sliding", "facts", "summary", "full")
TITLES = {"sliding": "Sliding Window", "facts": "Sticky Facts", "summary": "Summary (День 9)", "full": "вся история"}
NAME_RE = re.compile(r"[\w-]+")

FACTS_PROMPT = (
    "Ты ведёшь блок «факты» — память ассистента о диалоге в виде пар ключ → значение. Старых сообщений ассистент "
    "не видит: всё важное о пользователе и его задаче он знает только из фактов.\n"
    "Тебе дают текущие факты и новые сообщения пользователя. Верни JSON вида "
    '{"set": {"ключ": "значение"}, "delete": ["ключ"]}:\n'
    "- set — новые факты и факты, значение которых изменилось; delete — факты, которые пользователь отменил;\n"
    "- что хранить: имена и роли людей, названия, цель, числа с единицами, суммы с валютой, даты и сроки, "
    "ограничения и требования, предпочтения, принятые решения и от чего отказались, договорённости;\n"
    "- значения пиши точно, как у пользователя, и коротко (до 20 слов). Если значение изменилось — новое, "
    "а в скобках старое: «2,7 млн ₽ (было 2,4 млн ₽)»;\n"
    "- ключи короткие, по-русски, в нижнем регистре через подчёркивание: «имя_пользователя», «бюджет», "
    "«срок_релиза». Если факт уже есть — обнови тот же ключ, не заводи новый;\n"
    "- вопросы пользователя и просьбы что-то объяснить — не факты. Ничего не придумывай;\n"
    '- если новых фактов нет — верни {"set": {}, "delete": []}.\n'
    "Верни только JSON.")

FACTS_HEADER = ("\n\n## Факты из разговора\n"
                "Старые сообщения диалога тебе не отправлены — всё важное из них собрано здесь. "
                "Это пользователь сообщил тебе раньше:\n")

DENIAL = re.compile(r"не (упомина|называ|говори|обсужда|сообща|указыва|озвучива|уточня)|нет (информации|данных|сведений)"
                    r"|не знаю|не помню|в (нашем )?(разговоре|диалоге) не|не могу (сказать|знать)|мне неизвестн")


# ---------------------------------------------------------------- стратегии


@dataclass
class HelperCall:
    """Служебный запрос стратегии к модели (обновление фактов или сжатие) и во что он обошёлся."""
    strategy: str
    before_turn: int  # перед каким ходом пользователя (или перед вопросами — номер следующего хода)
    messages: int  # сколько сообщений обработано
    prompt: int
    completion: int
    cached: int
    cost: float | None
    latency: float
    at: str = ""
    note: str = ""


def tail_start(history: list[dict], end: int, keep: int) -> int:
    """Начало «хвоста» из последних keep сообщений history[:end]. Граница — начало хода (сообщение пользователя),
    чтобы вызов инструмента не оторвался от результата; поэтому в хвосте бывает и чуть больше keep сообщений."""
    if keep <= 0:
        return end
    starts = [i for i in range(1, end) if history[i].get("role") == "user" and end - i >= keep]
    return starts[-1] if starts else 1


class Strategy:
    """Что отправить в модель. Историю стратегия не меняет: только выбирает из неё и дополняет системный промпт."""
    name = ""

    def __init__(self, keep_last: int):
        self.keep_last = max(keep_last, 0)
        self.calls: list[HelperCall] = []

    def prepare(self, agent: "StrategyAgent", user_text: str | None, force: bool = False) -> None:
        """Перед ходом (user_text — новое сообщение пользователя) или перед контрольными вопросами (None)."""

    def build(self, agent: "StrategyAgent", end: int) -> tuple[list[dict], int]:
        """(сообщения для запроса, сколько сообщений истории в него не попало как есть). end — где начинается
        текущий ход (до него — прошлое, после — текущий ход, который уходит всегда)."""
        raise NotImplementedError

    def rollback(self) -> None:
        """Ход не прошёл и откачен из истории — забыть то, что стратегия успела из него извлечь."""

    def clear(self) -> None:
        self.calls = []

    def state(self) -> dict:
        return {"calls": [asdict(c) for c in self.calls]}

    def load(self, data: dict, history: list[dict]) -> str:
        """Восстановить состояние из файла; вернуть предупреждение, если оно не сходится с историей."""
        self.clear()
        known = HelperCall.__dataclass_fields__
        self.calls = [HelperCall(**{k: v for k, v in c.items() if k in known}) for c in data.get("calls") or []]
        return ""


class FullHistory(Strategy):
    """Эталон для сравнения: вся история как есть."""
    name = "full"

    def build(self, agent, end):
        return agent.history, 0


class SlidingWindow(Strategy):
    """Последние N сообщений (от начала хода) и текущий ход; всё, что старше, — отбрасывается."""
    name = "sliding"

    def build(self, agent, end):
        start = tail_start(agent.history, end, self.keep_last)
        return agent.history[:1] + agent.history[start:], start - 1


def _merge_json(base: dict, extra: dict) -> dict:
    """Слияние двух объектов ответа: вложенные словари — по ключам, списки — сложением, остальное — позже побеждает."""
    out = dict(base)
    for key, value in extra.items():
        old = out.get(key)
        if isinstance(old, dict) and isinstance(value, dict):
            out[key] = _merge_json(old, value)
        elif isinstance(old, list) and isinstance(value, list):
            out[key] = old + value
        else:
            out[key] = value
    return out


def extract_json_object(text: str) -> dict:
    """Первый JSON-объект из ответа модели. Соседние объекты (ответ пришёл частями) сливаются,
    текст после них (пояснение, второй ответ через прозу) игнорируется."""
    raw = text or ""
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start >= 0:
        try:
            data, end = decoder.raw_decode(raw, start)
        except ValueError:
            start = raw.find("{", start + 1)
            continue
        if not isinstance(data, dict):
            raise ValueError("ожидался объект")
        result = data
        while True:
            nxt = end
            while nxt < len(raw) and raw[nxt].isspace():
                nxt += 1
            if nxt >= len(raw) or raw[nxt] != "{":
                break
            try:
                extra, end = decoder.raw_decode(raw, nxt)
            except ValueError:
                break
            if not isinstance(extra, dict):
                break
            result = _merge_json(result, extra)
        return result
    raise ValueError("нет JSON")


class StickyFacts(Strategy):
    """Блок «факты» (ключ → значение) + последние N сообщений. Факты обновляются после каждого сообщения
    пользователя отдельным запросом: модель возвращает только изменения, слияние делает код — так модель
    не может случайно «забыть» старый факт, переписывая весь блок."""
    name = "facts"

    def __init__(self, keep_last: int):
        super().__init__(keep_last)
        self.facts: dict[str, str] = {}
        self.upto = 1  # сообщения пользователя из history[1:upto] уже учтены в фактах
        self._before: tuple[dict, int] | None = None

    def prepare(self, agent, user_text, force=False):
        end = agent.turn_end()
        # пропущенные сообщения (стратегия была выключена) + новое — одним запросом
        pending = [m.get("content") or "" for m in agent.history[self.upto:end] if m.get("role") == "user"]
        if user_text is not None:
            pending.append(user_text)
        self._before = (dict(self.facts), self.upto)
        if not pending:
            return
        numbered = "\n\n".join(f"{i}. {t}" for i, t in enumerate(pending, 1))
        current = json.dumps(self.facts, ensure_ascii=False, indent=1) if self.facts else "{}"
        request = [{"role": "system", "content": FACTS_PROMPT},
                   {"role": "user", "content": f"Текущие факты:\n{current}\n\nНовые сообщения пользователя:\n\n{numbered}"}]
        text, call = agent.helper_chat(self, request, len(pending))
        try:
            update, delete = self.parse(text)
        except ValueError as e:
            raise LLMError(f"ответ с фактами не разобран ({e}): {clip(text, 120)!r}")
        added = [k for k in update if k not in self.facts]
        changed = [k for k in update if k in self.facts and self.facts[k] != update[k]]
        removed = [k for k in delete if k in self.facts]
        self.facts.update(update)
        for k in removed:
            del self.facts[k]
        self.upto = end + (1 if user_text is not None else 0)
        call.note = (f"+{len(added)} ~{len(changed)} −{len(removed)}, всего {len(self.facts)}"
                     + (f" ({', '.join(added + changed + removed)})" if added or changed or removed else ""))
        agent.on_event("helper", {"call": call})

    @staticmethod
    def parse(text: str) -> tuple[dict[str, str], list[str]]:
        data = extract_json_object(text)
        if "set" not in data and "delete" not in data:  # модель вернула плоский словарь фактов
            data = {"set": data}
        update, delete = data.get("set") or {}, data.get("delete") or []
        if not isinstance(update, dict) or not isinstance(delete, list):
            raise ValueError("set — объект, delete — список")
        value = lambda v: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)  # noqa: E731
        return ({str(k).strip(): value(v).strip() for k, v in update.items() if str(k).strip() and v not in (None, "")},
                [str(k).strip() for k in delete])

    def text(self) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in self.facts.items())

    def build(self, agent, end):
        start = tail_start(agent.history, end, self.keep_last)
        system = agent.history[:1]
        if self.facts:
            system = [dict(system[0], content=(system[0].get("content") or "") + FACTS_HEADER + self.text())]
        return system + agent.history[start:], start - 1

    def rollback(self):
        if self._before:
            self.facts, self.upto = self._before

    def clear(self):
        super().clear()
        self.facts, self.upto, self._before = {}, 1, None

    def state(self):
        return {**super().state(), "upto": self.upto, "facts": self.facts}

    def load(self, data, history):
        super().load(data, history)
        upto = int(data.get("upto") or 1)
        if not 1 <= upto <= max(len(history), 1):
            return f"факты не сходятся с историей (учтено до сообщения {upto}, в истории {len(history)}) — пересчитаю."
        facts = data.get("facts") or {}
        self.facts, self.upto = {str(k): str(v) for k, v in facts.items()}, upto
        return ""


class SummaryStrategy(Strategy):
    """Summary из Дня 9: когда за последними N сообщениями накопится chunk несжатых, они вместе с прежним summary
    уходят сжимателю (промпт «факты» Дня 9). В запрос — summary в системном промпте + несжатая часть истории."""
    name = "summary"

    def __init__(self, keep_last: int, chunk: int):
        super().__init__(keep_last)
        self.chunk = max(chunk, 1)
        self.summary = ""
        self.upto = 1  # history[1:upto] заменены summary

    def cut(self, history: list[dict], end: int) -> int | None:
        """Самое позднее начало хода, после которого до end остаётся не меньше keep_last сообщений."""
        cuts = [i for i in range(self.upto + 1, end) if history[i].get("role") == "user" and end - i >= self.keep_last]
        return cuts[-1] if cuts else None

    def pending(self, agent) -> int:
        cut = self.cut(agent.history, agent.turn_end())
        return cut - self.upto if cut else 0

    def prepare(self, agent, user_text, force=False):
        cut = self.cut(agent.history, agent.turn_end())
        if cut is None or (not force and cut - self.upto < self.chunk):
            return
        start, part = self.upto, agent.history[self.upto:cut]
        request = [{"role": "system", "content": SUMMARIZER_PROMPTS["facts"]},
                   {"role": "user", "content": f"Прежнее резюме:\n{self.summary or '(пока нет — это начало диалога)'}"
                                               f"\n\nНовые сообщения диалога:\n\n{render(part)}"}]
        text, call = agent.helper_chat(self, request, len(part))
        if not text.strip():
            raise LLMError("сжиматель вернул пустой ответ")
        self.summary, self.upto = text.strip(), cut
        call.note = f"сообщения {start}–{cut - 1} → summary ≈{fmt(agent.estimator.text(self.summary))} токенов"
        agent.on_event("helper", {"call": call})

    def build(self, agent, end):
        if not self.summary:
            return agent.history, 0
        system = dict(agent.history[0], content=(agent.history[0].get("content") or "") + SUMMARY_HEADER + self.summary)
        return [system] + agent.history[self.upto:], self.upto - 1

    def clear(self):
        super().clear()
        self.summary, self.upto = "", 1

    def state(self):
        return {**super().state(), "upto": self.upto, "summary": self.summary}

    def load(self, data, history):
        super().load(data, history)
        upto, n = int(data.get("upto") or 1), len(history)
        if not (1 <= upto <= n and (upto == n or history[upto].get("role") == "user")):
            return f"summary не сходится с историей (сжато до сообщения {upto}, в истории {n}) — сожму заново."
        self.summary, self.upto = data.get("summary") or "", upto
        return ""


# ---------------------------------------------------------------- агент


def state_path_for(history_path: Path) -> Path:
    return history_path.with_name(history_path.stem + ".state.json")


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        raise StoreError(f"не удалось сохранить {path}: {e}")


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("ожидался JSON-объект")
        return data
    except (OSError, ValueError) as e:
        raise StoreError(f"не удалось прочитать {path}: {e}")


class StrategyAgent(TokenAgent):
    """TokenAgent из Дня 8 с переключаемой стратегией управления контекстом."""

    def __init__(self, client: LLMClient, tools, store: HistoryStore, *, strategy: str = "facts", keep_last: int = 6,
                 chunk: int = 10, helper_client: LLMClient | None = None, **kwargs):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy: {', '.join(STRATEGIES)}")
        self.keep_last, self.chunk = max(keep_last, 0), max(chunk, 1)
        self.strategies: dict[str, Strategy] = {"sliding": SlidingWindow(keep_last), "facts": StickyFacts(keep_last),
                                                "summary": SummaryStrategy(keep_last, chunk),
                                                "full": FullHistory(keep_last)}
        self.strategy_name = strategy
        self.helper_client = helper_client or client
        self.state_path = state_path_for(store.path)
        self.notes: list[str] = []
        self.full_estimate = 0  # ≈ токенов, если бы последний запрос ушёл со всей историей
        self._turn_start: int | None = None
        kwargs.setdefault("system_prompt", AGENT_PROMPT)
        super().__init__(client, tools, store, **kwargs)

    @property
    def strategy(self) -> Strategy:
        return self.strategies[self.strategy_name]

    def switch(self, name: str) -> None:
        if name not in self.strategies:
            raise ValueError(f"нет стратегии «{name}». Есть: {', '.join(STRATEGIES)}")
        old, self.strategy_name = self.strategy_name, name
        self.save_state()
        self.on_event("switch", {"from": old, "to": name})

    # --- что уходит в модель

    def turn_end(self) -> int:
        """Где кончается прошлое: начало текущего хода, а вне хода — конец истории."""
        return len(self.history) if self._turn_start is None else self._turn_start

    def context(self) -> list[dict]:
        return self.strategy.build(self, self.turn_end())[0]

    def _select(self, schemas):
        self.full_estimate = self.estimator.apply(self.estimator.raw_messages(self.history, schemas))
        return self.strategy.build(self, self.turn_end())

    def helper_chat(self, strategy: Strategy, request: list[dict], processed: int) -> tuple[str, HelperCall]:
        """Служебный запрос стратегии (факты, сжатие): отдельный клиент, temperature 0, учёт токенов и стоимости."""
        result = self.helper_client.chat(request, temperature=0)
        usage = Usage.parse(result.usage)
        priced = price_for(self.helper_client.model)
        call = HelperCall(strategy=strategy.name, before_turn=len(self.ledger) + 1, messages=processed,
                          prompt=usage.prompt, completion=usage.completion, cached=usage.cached,
                          cost=usage.cost(priced[0] if priced else None), latency=result.latency, at=now_iso())
        strategy.calls.append(call)
        return result.text or "", call

    def prepare(self, force: bool = False) -> None:
        """Подготовить контекст вне хода — перед контрольными вопросами (как перед следующим ходом)."""
        self.strategy.prepare(self, None, force=force)
        self.save_state()

    def run(self, user_text: str):
        self._turn_start = len(self.history)
        try:
            try:
                self.strategy.prepare(self, user_text)
            except LLMError as e:  # без свежих фактов / summary агент всё равно ответит, а догонит в следующий ход
                self.on_event("helper_error", {"strategy": self.strategy_name, "error": str(e)})
            try:
                return super().run(user_text)
            except BaseException:
                self.strategy.rollback()
                self.save_state()
                raise
        finally:
            self._turn_start = None

    def helper_totals(self, name: str | None = None) -> dict:
        calls = [c for s in self.strategies.values() for c in s.calls if name is None or c.strategy == name]
        costs = [c.cost for c in calls]
        return {"count": len(calls), "prompt": sum(c.prompt for c in calls),
                "completion": sum(c.completion for c in calls), "cached": sum(c.cached for c in calls),
                "cost": None if any(c is None for c in costs) else sum(costs)}

    # --- хранение: состояние стратегий — рядом с историей, в <имя>.state.json

    def state_payload(self) -> dict:
        return {"format": 1, "strategy": self.strategy_name, "keep_last": self.keep_last, "chunk": self.chunk,
                "updated_at": now_iso(), "strategies": {n: s.state() for n, s in self.strategies.items()}}

    def save_state(self) -> None:
        write_json(self.state_path, self.state_payload())

    def save(self) -> None:
        super().save()
        self.save_state()

    def load_state(self) -> None:
        if not self.state_path.exists():
            return
        data = read_json(self.state_path)
        for name, s in self.strategies.items():
            note = s.load((data.get("strategies") or {}).get(name) or {}, self.history)
            if note:
                self.notes.append(f"{TITLES[name]}: {note}")
        if data.get("strategy") in self.strategies:
            self.strategy_name = data["strategy"]

    def restore(self, saved: Saved) -> RestoreInfo:
        info = super().restore(saved)
        self.load_state()
        return info

    def attach(self, store: HistoryStore) -> RestoreInfo:
        """Переключиться на другой файл истории (ветку) без архивации; текущая стратегия сохраняется."""
        current = self.strategy_name
        self.store, self.state_path = store, state_path_for(store.path)
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.turns, self.ledger, self.notes, self.created_at = [], [], [], now_iso()
        for s in self.strategies.values():
            s.clear()
        saved = store.load()
        self.client.session_id = (saved.session_id if saved else "") or uuid.uuid4().hex
        info = self.restore(saved) if saved else RestoreInfo(1, 0, 0, 0, "")
        self.strategy_name = current
        return info

    def reset(self) -> None:
        for s in self.strategies.values():
            s.clear()
        if getattr(self, "_ready", False) and self.state_path.exists():
            try:
                os.replace(self.state_path, self.state_path.with_name(
                    f"{self.store.path.stem}_{datetime.now():%Y%m%d_%H%M%S}.state.json"))
            except OSError as e:
                raise StoreError(f"не удалось переместить {self.state_path} в архив: {e}")
        super().reset()


# ---------------------------------------------------------------- ветки и checkpoints


class Workspace:
    """Сессия Дня 10 на диске, memory/day10/<сессия>/:
        branches.json           — текущая ветка, ветки (от какой ветки и какого checkpoint отходят) и checkpoints;
        <ветка>.json            — история ветки в формате Дня 7;
        <ветка>.state.json      — состояние стратегий ветки (факты, summary, служебные запросы);
        checkpoints/<имя>.json  — снимок: история и состояние стратегий в момент checkpoint."""

    def __init__(self, root: Path):
        self.root = root
        self.meta_path = root / "branches.json"
        self.meta = self._load()

    def _load(self) -> dict:
        if not self.meta_path.exists():
            return {"format": 1, "current": "main", "checkpoints": {},
                    "branches": {"main": {"parent": None, "checkpoint": None, "turn": 0, "created_at": now_iso()}}}
        data = read_json(self.meta_path)
        if not isinstance(data.get("branches"), dict) or data.get("current") not in data["branches"]:
            raise StoreError(f"{self.meta_path}: нет списка веток или текущей ветки. Начните заново с --new.")
        data.setdefault("checkpoints", {})
        return data

    def save(self) -> None:
        write_json(self.meta_path, self.meta)

    @property
    def current(self) -> str:
        return self.meta["current"]

    def store(self, branch: str) -> HistoryStore:
        return HistoryStore(self.root / f"{branch}.json")

    def checkpoint_path(self, name: str) -> Path:
        return self.root / "checkpoints" / f"{name}.json"

    @staticmethod
    def _check_name(name: str, taken, what: str) -> None:
        if not NAME_RE.fullmatch(name or "") or name == "branches":
            raise StoreError(f"имя {what} — буквы, цифры, дефис и подчёркивание.")
        if name in taken:
            raise StoreError(f"{what} «{name}» уже есть.")

    def checkpoint(self, agent: StrategyAgent, name: str) -> dict:
        self._check_name(name, self.meta["checkpoints"], "checkpoint")
        agent.save()
        snap = {"format": 1, "name": name, "branch": self.current, "at": now_iso(), "turns": len(agent.ledger),
                "messages": len(agent.history), "strategy": agent.strategy_name,
                "history": read_json(agent.store.path), "state": agent.state_payload()}
        write_json(self.checkpoint_path(name), snap)
        entry = {k: snap[k] for k in ("branch", "at", "turns", "messages", "strategy")}
        self.meta["checkpoints"][name] = entry
        self.save()
        return entry

    def branch(self, agent: StrategyAgent, name: str, checkpoint: str | None = None) -> RestoreInfo:
        """Новая ветка от checkpoint (или от текущего места) — и сразу переключение на неё."""
        self._check_name(name, self.meta["branches"], "ветка")
        if checkpoint:
            if checkpoint not in self.meta["checkpoints"]:
                raise StoreError(f"нет checkpoint «{checkpoint}». Есть: {', '.join(self.meta['checkpoints']) or '—'}.")
            snap = read_json(self.checkpoint_path(checkpoint))
        else:
            agent.save()
            snap = {"branch": self.current, "turns": len(agent.ledger), "history": read_json(agent.store.path),
                    "state": agent.state_payload()}
        h = snap["history"]
        # своя сессия у провайдера: ветка — отдельный диалог
        self.store(name).save(Saved(session_id=uuid.uuid4().hex, model=h.get("model") or "",
                                    created_at=h.get("created_at") or now_iso(), updated_at=now_iso(),
                                    messages=h.get("messages") or [], turns=h.get("turns") or []))
        write_json(state_path_for(self.store(name).path), snap["state"])
        self.meta["branches"][name] = {"parent": snap["branch"], "checkpoint": checkpoint, "turn": snap["turns"],
                                       "created_at": now_iso()}
        self.save()
        return self.switch(agent, name)

    def switch(self, agent: StrategyAgent, name: str) -> RestoreInfo:
        if name not in self.meta["branches"]:
            raise StoreError(f"нет ветки «{name}». Есть: {', '.join(self.meta['branches'])}.")
        agent.save()
        info = agent.attach(self.store(name))
        self.meta["current"] = name
        self.save()
        return info

    def turns(self, branch: str) -> int:
        path = self.store(branch).path
        if not path.exists():
            return 0
        try:
            return sum(m.get("role") == "user" for m in read_json(path).get("messages") or [])
        except StoreError:
            return 0

    def archive(self) -> Path | None:
        if not self.root.exists():
            return None
        target = self.root.with_name(f"{self.root.name}_{datetime.now():%Y%m%d_%H%M%S}")
        try:
            os.replace(self.root, target)
        except OSError as e:
            raise StoreError(f"не удалось переместить {self.root} в архив: {e}")
        return target


# ---------------------------------------------------------------- вывод


def on_event(kind: str, data: dict) -> None:
    if kind == "helper":
        c: HelperCall = data["call"]
        icon, what = ("📌", "факты") if c.strategy == "facts" else ("🗜 ", "сжатие")
        print(f"  {icon} {what}: {c.note} | вход {fmt(c.prompt)}, выход {fmt(c.completion)}, {c.latency:.1f} с, "
              f"{money(c.cost)}")
    elif kind == "helper_error":
        print(f"  ⚠️  служебный запрос ({TITLES[data['strategy']]}) не прошёл: {data['error']} — "
              f"отвечу с прежним контекстом, догоню в следующий ход")
    elif kind == "switch":
        print(f"  🔀 стратегия: {TITLES[data['from']]} → {TITLES[data['to']]}")
    else:
        day9_event(kind, data)


def quiet_event(kind: str, data: dict) -> None:
    """Для --demo: служебные запросы, инструменты и повторы; строку хода печатает print_turn_line."""
    if kind in ("tokens", "llm", "truncate"):
        return
    on_event(kind, data)


def describe(agent: StrategyAgent, sent: int, dropped: int) -> str:
    """Из чего состоял запрос: sent — сообщений в запросе (с системным), dropped — сколько не ушло как есть."""
    s, n = agent.strategy, max(sent - 1, 0)
    if s.name == "sliding":
        return f"system + последние {n} сообщ.; {dropped} старых отброшено"
    if s.name == "facts":
        facts = f"факты ({len(s.facts)} шт., ≈{fmt(agent.estimator.text(s.text()))})" if s.facts else "фактов пока нет"
        return f"system + {facts} + {n} сообщ.; {dropped} старых не отправлено"
    if s.name == "summary" and s.summary:
        return f"system + summary (≈{fmt(agent.estimator.text(s.summary))}) + {n} сообщ.; {dropped} заменены summary"
    return f"system + вся история ({n} сообщ.)"


def full_projection(agent: StrategyAgent, turn: TurnStats) -> int:
    """≈ вход последнего запроса, если бы ушла вся история (оценка в масштабе точного prompt_tokens)."""
    last = turn.calls[-1] if turn.calls else None
    if not (last and turn.dropped and agent.full_estimate and last.estimate):
        return 0
    return round(agent.full_estimate * last.prompt / last.estimate)


def turn_helpers(agent: StrategyAgent, n: int) -> list[HelperCall]:
    return [c for s in agent.strategies.values() for c in s.calls if c.before_turn == n]


def print_turn(agent: StrategyAgent, turn: TurnStats) -> None:
    print(f"📊 Ход {turn.n} · {TITLES[agent.strategy_name]}"
          + (f", хвост {agent.keep_last}" if agent.strategy_name != "full" else ""))
    if not turn.calls:
        print(f"   ответа нет: {turn.error}")
        return
    last = turn.calls[-1]
    cached = f", из кэша {fmt(last.cached)}" if last.cached else ""
    print(f"   в запрос ушло:   {describe(agent, last.sent_messages, turn.dropped)}")
    print(f"   вход:            {fmt(last.prompt)} токенов{cached}; за ход {fmt(turn.prompt)}, запросов {len(turn.calls)}")
    full = full_projection(agent, turn)
    if full:
        print(f"   вся история:     было бы ≈{fmt(full)} — запрос {delta(last.prompt, full)} к ней")
    print(f"   ответ:           {fmt(turn.completion)} токенов")
    helpers = turn_helpers(agent, turn.n)
    if helpers:
        print(f"   служебные:       {len(helpers)} запрос(а): вход {fmt(sum(c.prompt for c in helpers))}, "
              f"выход {fmt(sum(c.completion for c in helpers))}, {money(sum(c.cost or 0 for c in helpers))}")
    t, h = agent.totals(), agent.helper_totals()
    total = None if t["cost"] is None or h["cost"] is None else t["cost"] + h["cost"]
    print(f"   стоимость:       ход {money(turn.cost)} · диалог {money(t['cost'])} + служебные {money(h['cost'])} "
          f"({h['count']}×) = {money(total)}")


def print_turn_line(agent: StrategyAgent, turn: TurnStats) -> None:
    if not turn.calls:
        print(f"  ❌ {turn.error}")
        return
    last = turn.calls[-1]
    cached = f" (кэш {fmt(last.cached)})" if last.cached else ""
    full = full_projection(agent, turn)
    saved = f" | вся история ≈{fmt(full)}, {delta(last.prompt, full)}" if full else ""
    print(f"  📊 вход {fmt(last.prompt)}{cached}{saved} | в запросе {last.sent_messages - 1} сообщ."
          f" | выход {fmt(turn.completion)} | {money(turn.cost)}")


def print_context(agent: StrategyAgent) -> None:
    """Из чего состоит запрос сейчас — и сколько он весил бы на других стратегиях с их текущим состоянием."""
    est, end = agent.estimator, agent.turn_end()
    print(f"  Стратегия: {TITLES[agent.strategy_name]}; хвост {agent.keep_last} сообщ., summary каждые {agent.chunk}")
    for name, s in agent.strategies.items():
        messages, dropped = s.build(agent, end)
        mark = "▶" if name == agent.strategy_name else " "
        stale = ""
        if name == "facts" and s.upto < end:
            stale = f" (не учтено сообщ. пользователя: {sum(m.get('role') == 'user' for m in agent.history[s.upto:end])})"
        if name == "summary" and s.pending(agent):
            stale = f" (ждут сжатия: {s.pending(agent)} сообщ.)"
        print(f"  {mark} {TITLES[name]:<17} ≈{fmt(est.apply(est.raw_messages(messages))):>7} токенов: "
              f"{len(messages) - 1} сообщ. как есть, {dropped} не отправлено{stale}")


def print_facts(agent: StrategyAgent) -> None:
    s: StickyFacts = agent.strategies["facts"]
    if not s.facts:
        print("  (фактов пока нет)" + ("" if agent.strategy_name == "facts" else " — включите /strategy facts"))
        return
    h = agent.helper_totals("facts")
    print(f"  {len(s.facts)} фактов, ≈{fmt(agent.estimator.text(s.text()))} токенов; обновлений {h['count']}, "
          f"{money(h['cost'])}")
    for k, v in s.facts.items():
        print(f"   · {k}: {v}")


def print_summary(agent: StrategyAgent) -> None:
    s: SummaryStrategy = agent.strategies["summary"]
    if not s.summary:
        print("  (summary ещё нет)")
        return
    for c in s.calls:
        print(f"  перед ходом {c.before_turn}: {c.note}, {money(c.cost)}")
    print(f"\n{s.summary}\n")


def print_branches(ws: Workspace, agent: StrategyAgent) -> None:
    def walk(name: str, depth: int) -> None:
        b = ws.meta["branches"][name]
        turns = len(agent.ledger) if name == ws.current else ws.turns(name)
        origin = f" ← {b['parent']}" + (f" @ {b['checkpoint']}" if b.get("checkpoint") else f", ход {b['turn']}") \
            if b.get("parent") else ""
        print(f"  {'  ' * depth}{'▶' if name == ws.current else '·'} {name}: {turns} ход.{origin}")
        for child, cb in ws.meta["branches"].items():
            if cb.get("parent") == name:
                walk(child, depth + 1)
    for name, b in ws.meta["branches"].items():
        if not b.get("parent") or b["parent"] not in ws.meta["branches"]:
            walk(name, 0)
    for name, cp in ws.meta["checkpoints"].items():
        print(f"  ⚑ checkpoint {name}: ветка {cp['branch']}, после хода {cp['turns']} ({cp['at'][:16].replace('T', ' ')})")


# ---------------------------------------------------------------- контрольные вопросы


def ask(agent: StrategyAgent, question: str) -> dict:
    """Вопрос отдельным запросом с текущим контекстом стратегии; в историю он не попадает."""
    schemas = [t.schema() for t in agent.tools.values()]
    priced = price_for(agent.client.model)
    try:
        result = agent.client.chat(agent.context() + [{"role": "user", "content": question + PROBE_SUFFIX}],
                                   tools=schemas)
    except LLMError as e:
        return {"answer": "", "error": str(e), "prompt": 0, "cached": 0, "completion": 0, "cost": 0.0}
    usage = Usage.parse(result.usage)
    answer = result.text.strip() or ("(модель попросила инструмент вместо ответа)" if result.tool_calls else "")
    return {"answer": answer, "error": "", "prompt": usage.prompt, "cached": usage.cached,
            "completion": usage.completion, "cost": usage.cost(priced[0] if priced else None)}


def run_probes(agent: StrategyAgent, details: list[Detail] = DETAILS, verbose: bool = True) -> list[dict]:
    out = []
    for d in details:
        r = {"id": d.id, "question": d.question, **ask(agent, d.question)}
        r["ok"] = not r["error"] and matches(d, r["answer"])
        r["denial"] = bool(DENIAL.search(normalize(r["answer"])))  # слова детали есть, но ответ её отрицает?
        out.append(r)
        if verbose:
            icon = "⚠️" if r["error"] else ("❔" if r["ok"] and r["denial"] else "✅" if r["ok"] else "❌")
            print(f"  {icon} [{d.kind}] {d.question} → {clip(r['answer'] or r['error'], 150)}")
    return out


# ---------------------------------------------------------------- чат


HELP = ("/strategy [sliding|facts|summary|full], /play [N] — реплики сценария Дня 9, /probe — 12 контрольных вопросов, "
        "/facts, /summary, /context, /compress, /checkpoint <имя>, /branch <имя> [checkpoint], /switch <ветка>, "
        "/branches, /tokens, /history, /reset, /exit")


def chat(agent: StrategyAgent, ws: Workspace) -> int:
    print(f"Агент на {agent.client.model}. Стратегия: {TITLES[agent.strategy_name]}, ветка {ws.current}.\n"
          f"Команды: {HELP}\n")
    while True:
        try:
            text = input(f"[{ws.current} · {agent.strategy_name}] Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return 0
        command, _, arg = text.partition(" ")
        args = arg.split()
        try:
            if command == "/strategy":
                if not args:
                    print("  " + " · ".join(("▶ " if n == agent.strategy_name else "") + f"{n} — {TITLES[n]}"
                                            for n in STRATEGIES) + "\n")
                else:
                    agent.switch(args[0])
                    print()
                continue
            if command == "/reset":
                agent.reset()
                continue
            if command == "/history":
                print_history(agent.history)
                print()
                continue
            if command == "/tokens":
                print_ledger(agent)
                for name in ("facts", "summary"):
                    h = agent.helper_totals(name)
                    if h["count"]:
                        print(f"  + {TITLES[name]}: служебных запросов {h['count']}, вход {fmt(h['prompt'])}, "
                              f"выход {fmt(h['completion'])}, {money(h['cost'])}")
                print()
                continue
            if command == "/facts":
                print_facts(agent)
                print()
                continue
            if command == "/summary":
                print_summary(agent)
                continue
            if command == "/context":
                print_context(agent)
                print()
                continue
            if command == "/compress":
                s = agent.strategies["summary"]
                before = len(s.calls)
                s.prepare(agent, None, force=True)
                agent.save_state()
                print("" if len(s.calls) > before else
                      f"  Сжимать нечего: за последними {agent.keep_last} сообщ. нет целых ходов.\n")
                continue
            if command == "/probe":
                agent.prepare()
                results = run_probes(agent)
                ok = sum(r["ok"] for r in results)
                doubt = sum(r["ok"] and r["denial"] for r in results)
                print(f"  Помнит {ok}/{len(results)} деталей" + (f" (❔ {doubt} — проверить вручную)" if doubt else "")
                      + f". Вход на вопрос ≈{fmt(results[0]['prompt'] if results else 0)} токенов.\n")
                continue
            if command == "/checkpoint":
                name = args[0] if args else f"cp{len(ws.meta['checkpoints']) + 1}"
                cp = ws.checkpoint(agent, name)
                print(f"  ⚑ checkpoint «{name}»: ветка {cp['branch']}, после хода {cp['turns']}. "
                      f"Ветка от него: /branch <имя> {name}\n")
                continue
            if command == "/branch":
                if not args:
                    print("  /branch <имя> [checkpoint] — без checkpoint ветка отходит от текущего места.\n")
                    continue
                info = ws.branch(agent, args[0], args[1] if len(args) > 1 else None)
                print(f"  🌿 ветка «{args[0]}» создана, ходов {info.turns}; текущая ветка — {args[0]}.\n")
                continue
            if command == "/switch":
                if not args:
                    print_branches(ws, agent)
                    print()
                    continue
                info = ws.switch(agent, args[0])
                print(f"  🔀 ветка {args[0]}: {info.turns} ход., {info.messages} сообщ.")
                for note in agent.notes:
                    print(f"  ⚠️  {note}")
                if info.turns:
                    print_history(agent.history, last_turns=1)
                print()
                continue
            if command == "/branches":
                print_branches(ws, agent)
                print()
                continue
            if command == "/play":
                done = len(agent.ledger)
                count = int(args[0]) if args and args[0].isdigit() else len(DIALOGUE) - done
                lines = DIALOGUE[done:done + count]
                if not lines:
                    print(f"  Сценарий пройден ({len(DIALOGUE)} реплик). Задайте /probe или /reset.\n")
                    continue
                for k, line in enumerate(lines, done + 1):
                    print(f"\n--- реплика {k}/{len(DIALOGUE)}\nВы: {line}")
                    reply = agent.run(line)
                    print(f"\nАгент: {reply.text.strip() or '(пустой ответ)'}\n")
                    print_turn(agent, agent.ledger[-1])
                print()
                continue
            if command.startswith("/"):
                print(f"  Неизвестная команда. Есть: {HELP}\n")
                continue
            reply = agent.run(text)
        except LLMError as e:
            print(f"❌ Ошибка: {e}\n", file=sys.stderr)
            continue
        except (StoreError, ValueError) as e:
            print(f"⚠️  {e}\n", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("\n(прервано — ход в историю не попал)\n")
            continue
        print(f"\nАгент: {reply.text.strip() or '(пустой ответ)'}\n")
        print_turn(agent, agent.ledger[-1])
        print()


# ---------------------------------------------------------------- сложность стратегий


def code_lines(*objects) -> int:
    """Строк кода без пустых строк, комментариев и docstring."""
    total = 0
    for obj in objects:
        src = textwrap.dedent(inspect.getsource(obj))
        doc: set[int] = set()
        for node in ast.walk(ast.parse(src)):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                    and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                doc.update(range(body[0].lineno, body[0].end_lineno + 1))
        total += sum(1 for i, line in enumerate(src.splitlines(), 1)
                     if line.strip() and not line.strip().startswith("#") and i not in doc)
    return total


def complexity() -> dict:
    """Сколько кода и служебных промптов нужно каждой стратегии сверх общего агента."""
    est = TokenEstimator()
    return {
        "full": {"lines": code_lines(FullHistory), "prompt": 0, "parts": "—"},
        "sliding": {"lines": code_lines(SlidingWindow, tail_start), "prompt": 0, "parts": "выбор хвоста"},
        "facts": {"lines": code_lines(StickyFacts, tail_start), "prompt": est.text(FACTS_PROMPT + FACTS_HEADER),
                  "parts": "выбор хвоста, промпт извлечения, разбор JSON, слияние, откат при ошибке, догон пропущенного"},
        "summary": {"lines": code_lines(SummaryStrategy, render), "prompt": est.text(SUMMARIZER_PROMPTS["facts"]
                                                                                        + SUMMARY_HEADER),
                    "parts": "граница сжатия, порог, промпт сжимателя, рендер истории, сверка с историей"},
    }


# ---------------------------------------------------------------- сравнение стратегий


def make_agent(client: LLMClient, helper: LLMClient, store: HistoryStore, strategy: str, keep: int, chunk: int,
               max_steps: int, event=on_event) -> StrategyAgent:
    watch_retries(client, helper, event=event)
    return StrategyAgent(client, TOOLS, store, strategy=strategy, keep_last=keep, chunk=chunk, helper_client=helper,
                         max_steps=max_steps, on_event=event)


def fresh(client: LLMClient) -> LLMClient:
    """Свой клиент (своя сессия OpenCode и свой счётчик запросов) на каждый прогон."""
    return LLMClient(client.base_url, client.api_key, client.model, client.timeout, retries=client.retries)


def where_kept(agent: StrategyAgent, detail: Detail) -> str:
    """Где деталь была в контексте на момент вопросов: в хвосте как есть, в фактах / summary или нигде."""
    s = agent.strategy
    sent, dropped = s.build(agent, len(agent.history))
    users = [i for i, m in enumerate(agent.history) if m.get("role") == "user"]
    if detail.turn > len(users):
        return "—"
    if users[detail.turn - 1] > dropped:
        return "хвост"
    if s.name == "facts":
        return "факты ✓" if matches(detail, s.text()) else "факты ✗"
    if s.name == "summary":
        return "summary ✓" if matches(detail, s.summary) else "summary ✗"
    return "отброшено"


def collect(agent: StrategyAgent, probes: list[dict], main: dict, helper: dict) -> dict:
    t, h = agent.totals(), agent.helper_totals()
    failed = [x.n for x in agent.ledger if x.status in ("error", "overflow")]
    contexts = [x.calls[-1].prompt for x in agent.ledger if x.calls]
    probe_prompt = [p["prompt"] for p in probes if p["prompt"]]
    total_cost = None if t["cost"] is None or h["cost"] is None else t["cost"] + h["cost"]
    s = agent.strategy
    return {
        "strategy": s.name, "title": TITLES[s.name], "keep_last": agent.keep_last, "chunk": agent.chunk,
        "remembered": sum(p["ok"] for p in probes),
        "doubtful": sum(p["ok"] and p["denial"] for p in probes),
        "failed_turns": failed, "probe_errors": sum(bool(p["error"]) for p in probes),
        "requests": {"main": main, "helper": helper},
        "where": {d.id: where_kept(agent, d) for d in DETAILS},
        "dialogue": {"prompt": t["prompt"], "cached": t["cached"], "completion": t["completion"], "cost": t["cost"]},
        "helper": h, "total_prompt": t["prompt"] + h["prompt"], "total_completion": t["completion"] + h["completion"],
        "total_cost": total_cost,
        "context_first": contexts[0] if contexts else 0, "context_last": contexts[-1] if contexts else 0,
        "context_max": max(contexts, default=0),
        "probe_prompt": round(sum(probe_prompt) / len(probe_prompt)) if probe_prompt else 0,
        "probe_cost": sum(p["cost"] or 0 for p in probes), "probes": probes,
        "facts": dict(agent.strategies["facts"].facts) if s.name == "facts" else {},
        "summary": agent.strategies["summary"].summary if s.name == "summary" else "",
        "helper_calls": [asdict(c) for c in s.calls],
        "turns": [{"n": x.n, "status": x.status, "prompt": x.calls[-1].prompt if x.calls else 0,
                   "cached": x.cached, "completion": x.completion, "cost": x.cost,
                   "sent_messages": x.calls[-1].sent_messages if x.calls else 0, "dropped": x.dropped,
                   "answer": x.answer, "error": x.error} for x in agent.ledger],
        "history_file": str(agent.store.path.relative_to(ROOT)), "state_file": str(agent.state_path.relative_to(ROOT)),
    }


def compare(client: LLMClient, helper_model: str | None, modes: list[str], keep: int, chunk: int, max_steps: int,
            root: Path) -> list[dict]:
    print(f"Сценарий Дня 9: {len(DIALOGUE)} реплик пользователя ({2 * len(DIALOGUE)} сообщений с ответами), "
          f"контрольных деталей {len(DETAILS)}:")
    for d in DETAILS:
        print(f"  · ход {d.turn:>2} [{d.kind}] {d.fact}")
    runs = []
    for mode in modes:
        main = fresh(client)
        helper = fresh(main) if not helper_model else LLMClient(client.base_url, client.api_key, helper_model,
                                                                 client.timeout, retries=client.retries)
        agent = make_agent(main, helper, HistoryStore(root / f"{mode}.json"), mode, keep, chunk, max_steps, quiet_event)
        how = {"full": "вся история в каждом запросе", "sliding": f"последние {keep} сообщ., остальное отброшено",
               "facts": f"факты + последние {keep} сообщ.; факты обновляются после каждого сообщения пользователя",
               "summary": f"summary + несжатая часть; сжатие, когда за {keep} сообщ. накопится {chunk}"}[mode]
        print(f"\n{'=' * 78}\n=== {TITLES[mode]}: {how}\n{'=' * 78}")
        for k, line in enumerate(DIALOGUE, 1):
            print(f"\n--- ход {k}/{len(DIALOGUE)}: {clip(line, 110)}")
            try:
                reply = agent.run(line)
                print(f"Агент: {clip(reply.text, 220) if reply.text.strip() else '(пустой ответ)'}")
            except LLMError as e:
                print(f"❌ {e}")
            print_turn_line(agent, agent.ledger[-1])
        print("\n--- перед вопросами (как перед следующим ходом):")
        try:
            agent.prepare()
        except LLMError as e:
            print(f"  ⚠️  не удалось: {e}")
        print(f"\n--- контрольные вопросы ({len(DETAILS)}), каждый — отдельный запрос с текущим контекстом:")
        probes = run_probes(agent)
        runs.append(collect(agent, probes, dict(main.stats), dict(helper.stats)))
        r = runs[-1]
        print(f"\nИтог «{TITLES[mode]}»: помнит {r['remembered']}/{len(DETAILS)}"
              + (f" (❔ {r['doubtful']} — ответ с отрицанием)" if r["doubtful"] else "")
              + f", вход за диалог {fmt(r['dialogue']['prompt'])} + служебные {fmt(r['helper']['prompt'])}, "
                f"стоимость {money(r['total_cost'])}")
    return runs


# ---------------------------------------------------------------- ветвление


BRANCH_SETUP = 4  # общих ходов сценария до checkpoint: имя, проект, пекарни, срок, бюджет
BRANCH_CHECKPOINT = "after-budget"
BRANCHES = {
    "growth": DIALOGUE[8],  # бюджет вырос до 2,7 млн, демо до 20 февраля
    "cut": ("Плохая новость: совет директоров урезал бюджет до 1,9 млн рублей. Зато публичный релиз разрешили "
            "сдвинуть на 4 апреля 2027 года. Что выкинуть из MVP?"),
}
BRANCH_QUESTION = "Напомни коротко: какой у нас сейчас бюджет на разработку и какой срок публичного релиза?"
# must — должно быть в ответе этой ветки; leak — значения, которые звучали только в другой ветке
BRANCH_CHECKS = {
    "growth": {"must": (r"2[,.]7|2700", r"\b14\s*марта"), "leak": (r"1[,.]9|1900", r"\b4\s*апрел")},
    "cut": {"must": (r"1[,.]9|1900", r"\b4\s*апрел"), "leak": (r"2[,.]7|2700", r"20\s*феврал")},
}
BRANCH_PROBES = [  # (вопрос, ветка → что должно быть; None — детали в ветке не было, она не должна её знать)
    ("Как меня зовут?", {"growth": ("марин",), "cut": ("марин",)}),
    ("К какой дате нужно показать демо инвесторам?", {"growth": (r"\b20\s*феврал",), "cut": None}),
]


def check_branch(branch: str, answer: str) -> dict:
    t = normalize(answer)
    c = BRANCH_CHECKS[branch]
    must = all(re.search(p, t) for p in c["must"])
    leak = [p for p in c["leak"] if re.search(p, t)]
    return {"ok": must and not leak, "must": must, "leak": leak}


def check_probe(expect: tuple | None, answer: str) -> bool:
    t = normalize(answer)
    if expect is None:  # детали в ветке не было: правильный ответ — «не знаю», без даты демо
        return not re.search(r"20\s*феврал", t)
    return all(re.search(p, t) for p in expect)


def branch_demo(client: LLMClient, helper_model: str | None, keep: int, chunk: int, max_steps: int, root: Path,
                strategy: str = "facts") -> dict:
    print(f"\n{'=' * 78}\n=== Ветвление диалога ({TITLES[strategy]}, хвост {keep})\n{'=' * 78}")
    main = fresh(client)
    helper = fresh(main) if not helper_model else LLMClient(client.base_url, client.api_key, helper_model,
                                                             client.timeout, retries=client.retries)
    ws = Workspace(root)
    agent = make_agent(main, helper, ws.store(ws.current), strategy, keep, chunk, max_steps, quiet_event)
    steps: list[dict] = []

    def turn(text: str) -> dict:
        print(f"\n[{ws.current}] Вы: {clip(text, 110)}")
        try:
            reply = agent.run(text)
            answer = reply.text.strip()
            print(f"Агент: {clip(answer, 260) or '(пустой ответ)'}")
        except LLMError as e:
            answer = ""
            print(f"❌ {e}")
        print_turn_line(agent, agent.ledger[-1])
        return {"branch": ws.current, "turn": len(agent.ledger), "user": text, "answer": answer,
                "prompt": agent.ledger[-1].calls[-1].prompt if agent.ledger[-1].calls else 0}

    print(f"\n--- общая часть: {BRANCH_SETUP} хода сценария в ветке main")
    for line in DIALOGUE[:BRANCH_SETUP]:
        steps.append({"action": "turn", **turn(line)})
    cp = ws.checkpoint(agent, BRANCH_CHECKPOINT)
    print(f"\n⚑ checkpoint «{BRANCH_CHECKPOINT}»: ветка {cp['branch']}, после хода {cp['turns']}")
    steps.append({"action": "checkpoint", "name": BRANCH_CHECKPOINT, **cp})
    for name, text in BRANCHES.items():
        ws.branch(agent, name, BRANCH_CHECKPOINT)
        print(f"\n🌿 ветка «{name}» от «{BRANCH_CHECKPOINT}» — продолжаем по-своему")
        steps.append({"action": "branch", "name": name, "from": BRANCH_CHECKPOINT})
        steps.append({"action": "turn", **turn(text)})
    checks = {}
    for name in BRANCHES:
        ws.switch(agent, name)
        print(f"\n🔀 переключились на «{name}»: {len(agent.ledger)} ход., фактов {len(agent.strategies['facts'].facts)}")
        steps.append({"action": "switch", "name": name})
        step = turn(BRANCH_QUESTION)
        step["check"] = check_branch(name, step["answer"])
        c = step["check"]
        print(f"  {'✅' if c['ok'] else '❌'} помнит своё: {'да' if c['must'] else 'нет'}; "
              f"утечка из другой ветки: {', '.join(c['leak']) or 'нет'}")
        steps.append({"action": "turn", **step})
        checks[name] = c
    probes = {}
    for name in BRANCHES:
        ws.switch(agent, name)
        agent.prepare()
        steps.append({"action": "switch", "name": name})
        probes[name] = []
        for question, expect in BRANCH_PROBES:
            r = ask(agent, question)
            r.update(question=question, ok=not r["error"] and check_probe(expect[name], r["answer"]),
                     expect="не знает" if expect[name] is None else "знает")
            probes[name].append(r)
            print(f"  [{name}] {'✅' if r['ok'] else '❌'} {question} → {clip(r['answer'] or r['error'], 140)}")
    branches = {}
    for name in ["main", *BRANCHES]:
        ws.switch(agent, name)
        facts = agent.strategies["facts"]
        branches[name] = {"turns": len(agent.ledger), "messages": len(agent.history), "facts": dict(facts.facts),
                          "session_id": agent.client.session_id, "file": str(agent.store.path.relative_to(ROOT)),
                          **ws.meta["branches"][name]}
    print("\nДерево веток:")
    print_branches(ws, agent)
    ok = all(c["ok"] for c in checks.values()) and all(p["ok"] for ps in probes.values() for p in ps)
    print(f"\nИтог ветвления: {'✅ ветки независимы' if ok else '❌ есть ошибки — см. отчёт'}; "
          f"запросов {main.stats['requests'] + helper.stats['requests']}")
    return {"strategy": strategy, "keep_last": keep, "checkpoint": BRANCH_CHECKPOINT, "setup_turns": BRANCH_SETUP,
            "steps": steps, "checks": checks, "probes": probes, "branches": branches, "ok": ok,
            "root": str(root.relative_to(ROOT)), "requests": {"main": dict(main.stats), "helper": dict(helper.stats)}}


# ---------------------------------------------------------------- отчёт


def delta(value: float | None, base: float | None) -> str:
    if value is None or not base:
        return ""
    pct = (value / base - 1) * 100
    return "±0%" if abs(pct) < 0.5 else f"{pct:+.0f}%"


def print_comparison(runs: list[dict], comp: dict) -> None:
    print(f"\n{'=' * 78}\nСравнение\n{'=' * 78}")
    print(f"  {'деталь':<34}" + "".join(f"{r['title']:>18}" for r in runs))
    for i, d in enumerate(DETAILS):
        cells = []
        for r in runs:
            p = r["probes"][i]
            cells.append("⚠️" if p["error"] else "❔" if p["ok"] and p["denial"] else "✅" if p["ok"] else "❌")
        print(f"  {clip(f'х{d.turn} {d.fact}', 33):<34}" + "".join(f"{c:>17}" for c in cells))
    rows = [("помнит", lambda r: f"{r['remembered']}/{len(DETAILS)}"),
            ("вход за диалог", lambda r: fmt(r["dialogue"]["prompt"])),
            ("служебные запросы / вход", lambda r: f"{r['helper']['count']} / {fmt(r['helper']['prompt'])}"),
            ("вход всего", lambda r: fmt(r["total_prompt"])),
            ("контекст последнего хода", lambda r: fmt(r["context_last"])),
            ("вход на контрольный вопрос", lambda r: fmt(r["probe_prompt"])),
            ("стоимость диалога", lambda r: money(r["total_cost"])),
            ("строк кода стратегии", lambda r: str(comp[r["strategy"]]["lines"]))]
    for label, value in rows:
        print(f"  {label:<34}" + "".join(f"{value(r):>18}" for r in runs))


def save_report(runs: list[dict], branches: dict | None, comp: dict, model: str, helper_model: str, keep: int,
                chunk: int, stamp: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path, md_path = RESULTS_DIR / f"day10_{stamp}.json", RESULTS_DIR / f"day10_{stamp}.md"
    json_path.write_text(json.dumps({"model": model, "helper_model": helper_model, "keep_last": keep, "chunk": chunk,
                                     "dialogue": DIALOGUE, "details": [asdict(d) for d in DETAILS],
                                     "complexity": comp, "runs": runs, "branching": branches},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    base = next((r for r in runs if r["strategy"] == "full"), None)
    titles = " | ".join(r["title"] for r in runs)
    lines = [f"# День 10 · Стратегии управления контекстом — прогон {stamp}", "",
             f"Модель: `{model}`, служебные запросы (факты, сжатие): `{helper_model}`. Сценарий и контрольные детали — "
             f"из Дня 9: {len(DIALOGUE)} реплик пользователя ({2 * len(DIALOGUE)} сообщений с ответами), "
             f"{len(DETAILS)} деталей. У всех стратегий хвост — последние {keep} сообщений (от начала хода); "
             f"summary сжимает, когда за хвостом накопится {chunk} сообщений. После диалога — подготовка контекста, "
             f"как перед следующим ходом, и {len(DETAILS)} контрольных вопросов: каждый — отдельный запрос "
             f"с контекстом стратегии, в историю он не попадает.", ""]

    if runs:
        lines += compare_section(runs, comp, base, titles)
    if branches:
        lines += branches_section(branches)
    if runs:
        lines += answers_section(runs)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {md_path.relative_to(ROOT)}, {json_path.name}")
    return md_path


def compare_section(runs: list[dict], comp: dict, base: dict | None, titles: str) -> list[str]:
    lines: list[str] = []
    lines += ["## Стратегия → деталей сохранено / токены / сложность", "",
              "| Стратегия | Деталей | Вход за диалог | Служебные запросы: число / вход / выход | Вход всего | "
              "Контекст 13-го хода | Вход на вопрос | Стоимость | Строк кода | Вызовов модели на ход |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        h, c = r["helper"], comp[r["strategy"]]
        turns = max(len(r["turns"]), 1)
        calls = f"1 + {h['count'] / turns:.2f}" if h["count"] else "1"
        doubt = f" (❔{r['doubtful']})" if r["doubtful"] else ""
        lines.append(
            f"| {r['title']} | **{r['remembered']}/{len(DETAILS)}**{doubt} | {fmt(r['dialogue']['prompt'])} | "
            f"{h['count']} / {fmt(h['prompt'])} / {fmt(h['completion'])} | **{fmt(r['total_prompt'])}**"
            + (f" ({delta(r['total_prompt'], base['total_prompt'])})" if base and r is not base else "") + " | "
            f"{fmt(r['context_last'])} | {fmt(r['probe_prompt'])}"
            + (f" ({delta(r['probe_prompt'], base['probe_prompt'])})" if base and r is not base else "") + " | "
            f"{money(r['total_cost'])}" + (f" ({delta(r['total_cost'], base['total_cost'])})"
                                           if base and r is not base else "") + " | "
            f"{c['lines']} | {calls} |")
    lines += ["", "«Деталей» — автоматическая проверка по ключевым словам; ❔ — ответ содержит слова детали, но звучит "
              "как отрицание («не упоминалось»), такие ответы надо проверить вручную. «Вызовов модели на ход» — один "
              "основной запрос (больше, если модель вызывает инструменты) плюс служебные в среднем на ход. "
              "«Строк кода» — код класса стратегии и его помощников без пустых строк, комментариев и docstring; "
              "общий агент (переключатель, хранение, учёт токенов) — сверх этого.", "",
              "| Стратегия | Что нужно для поддержки | Промпт служебного запроса |", "|---|---|---|"]
    for r in runs:
        c = comp[r["strategy"]]
        lines.append(f"| {r['title']} | {c['parts']} | {'≈' + fmt(c['prompt']) + ' токенов' if c['prompt'] else '—'} |")

    # --- детали
    lines += ["", "## Какие детали помнит агент", "",
              "В скобках — где деталь была на момент вопросов: «хвост» — в последних сообщениях как есть, "
              "«факты ✓/✗», «summary ✓/✗» — нашлась ли она в блоке фактов или в summary (той же проверкой, что и ответ), "
              "«отброшено» — Sliding Window её выкинул.", "",
              f"| Ход | Тип | Деталь | {titles} |", "|---|---|---|" + "---|" * len(runs)]
    for i, d in enumerate(DETAILS):
        cells = []
        for r in runs:
            p = r["probes"][i]
            mark = "⚠️ вопрос не прошёл" if p["error"] else ("❔" if p["ok"] and p["denial"] else "✅" if p["ok"] else "❌")
            if d.turn in r["failed_turns"]:
                mark += f" ⚠️ ход {d.turn} не прошёл"
            cells.append(f"{mark} ({r['where'][d.id]})")
        lines.append(f"| {d.turn} | {d.kind} | {d.fact} | " + " | ".join(cells) + " |")
    lines.append("| | | **помнит** | " + " | ".join(f"**{r['remembered']}/{len(DETAILS)}**" for r in runs) + " |")

    # --- токены
    lines += ["", "## Расход токенов", "", f"| | {titles} |", "|---|" + "---|" * len(runs)]
    rows = [("Запросов: основных / служебных (повторов, отказов)",
             lambda r: f"{r['requests']['main']['requests']} / {r['requests']['helper']['requests']} "
                       f"({r['requests']['main']['retries'] + r['requests']['helper']['retries']}, "
                       f"{r['requests']['main']['failed'] + r['requests']['helper']['failed']})"),
            ("Ходы, которые не прошли", lambda r: ", ".join(map(str, r["failed_turns"])) or "—"),
            ("Вход за диалог (из кэша)", lambda r: f"{fmt(r['dialogue']['prompt'])} ({fmt(r['dialogue']['cached'])})"),
            ("Выход за диалог", lambda r: fmt(r["dialogue"]["completion"])),
            ("Служебные: вход (из кэша) / выход",
             lambda r: f"{fmt(r['helper']['prompt'])} ({fmt(r['helper']['cached'])}) / {fmt(r['helper']['completion'])}"),
            ("**Вход всего**", lambda r: f"**{fmt(r['total_prompt'])}**"),
            ("Контекст: 1-й → последний ход (макс.)",
             lambda r: f"{fmt(r['context_first'])} → {fmt(r['context_last'])} ({fmt(r['context_max'])})"),
            ("Вход на контрольный вопрос", lambda r: fmt(r["probe_prompt"])),
            ("Стоимость диалога: ходы + служебные",
             lambda r: f"{money(r['dialogue']['cost'])} + {money(r['helper']['cost'])} = {money(r['total_cost'])}"),
            (f"Стоимость {len(DETAILS)} вопросов", lambda r: money(r["probe_cost"]))]
    for label, value in rows:
        lines.append(f"| {label} | " + " | ".join(value(r) for r in runs) + " |")
    lines += ["", "## Вход по ходам", "",
              "`prompt_tokens` последнего запроса хода; в скобках — сколько сообщений истории не ушло как есть; "
              "📌/🗜 — перед ходом был служебный запрос (факты / сжатие).", "",
              f"| Ход | {titles} |", "|---|" + "---|" * len(runs)]
    for k in range(len(DIALOGUE)):
        cells = []
        for r in runs:
            x = r["turns"][k] if k < len(r["turns"]) else None
            if not x:
                cells.append("—")
                continue
            icon = "".join(sorted({"📌 " if c["strategy"] == "facts" else "🗜 " for c in r["helper_calls"]
                                   if c["before_turn"] == x["n"]}))
            cells.append(f"{icon}{fmt(x['prompt'])}" + (f" (−{x['dropped']})" if x["dropped"] else "")
                         + ("" if x["status"] == "ok" else f" ❌ {x['status']}"))
        lines.append(f"| {k + 1} | " + " | ".join(cells) + " |")

    for r in runs:
        if r["facts"]:
            lines += ["", f"## Итоговые факты — {r['title']}", "", "| Ключ | Значение |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in r["facts"].items()]
            lines += ["", "Обновления фактов: " + "; ".join(f"х{c['before_turn']}: {c['note']}" for c in r["helper_calls"])]
        if r["summary"]:
            lines += ["", f"## Итоговое summary — {r['title']}", "",
                      *[f"> {s}" if s else ">" for s in r["summary"].splitlines()]]

    return lines


def branches_section(branches: dict) -> list[str]:
    lines: list[str] = []
    if branches:
        lines += ["", "## Ветвление диалога", "",
                  f"Стратегия {TITLES[branches['strategy']]}, хвост {branches['keep_last']}. В ветке `main` — первые "
                  f"{branches['setup_turns']} хода сценария, затем checkpoint `{branches['checkpoint']}` и две ветки "
                  f"от него. Каждая продолжается своей репликой, потом в каждой задан один и тот же вопрос — "
                  f"с переключением между ветками.", "",
                  "| Шаг | Ветка | Что | Ответ / результат |", "|---|---|---|---|"]
        for i, s in enumerate(branches["steps"], 1):
            if s["action"] == "turn":
                check = s.get("check")
                res = (" — " + ("✅" if check["ok"] else "❌") + (f" утечка: {', '.join(check['leak'])}"
                                                                 if check["leak"] else "")) if check else ""
                lines.append(f"| {i} | {s['branch']} | ход {s['turn']}: {clip(s['user'], 90)} | "
                             f"{clip(s['answer'], 160).replace('|', '/')}{res} |")
            elif s["action"] == "checkpoint":
                lines.append(f"| {i} | {s['branch']} | ⚑ checkpoint `{s['name']}` после хода {s['turns']} | |")
            elif s["action"] == "branch":
                lines.append(f"| {i} | {s['name']} | 🌿 ветка от `{s['from']}` | |")
            else:
                lines.append(f"| {i} | {s['name']} | 🔀 переключение | |")
        lines += ["", "Контрольные вопросы после переключения (отдельные запросы, в историю не попадают):", "",
                  "| Ветка | Вопрос | Ожидается | Ответ |", "|---|---|---|---|"]
        for name, ps in branches["probes"].items():
            for p in ps:
                lines.append(f"| {name} | {p['question']} | {p['expect']} | {'✅' if p['ok'] else '❌'} "
                             f"{clip(p['answer'] or p['error'], 160).replace('|', '/')} |")
        lines += ["", "Факты каждой ветки:", "", "| Ключ | " + " | ".join(branches["branches"]) + " |",
                  "|---|" + "---|" * len(branches["branches"])]
        keys = list(dict.fromkeys(k for b in branches["branches"].values() for k in b["facts"]))
        for k in keys:
            lines.append(f"| {k} | " + " | ".join(b["facts"].get(k, "—").replace("|", "/")
                                                 for b in branches["branches"].values()) + " |")
        lines += ["", "Итог: " + ("✅ ветки независимы — каждая помнит своё и не знает о другой."
                                  if branches["ok"] else "❌ есть ошибки — см. таблицы выше.")
                  + f" Файлы: `{branches['root']}/` (branches.json, <ветка>.json, <ветка>.state.json, checkpoints/)."]

    return lines


def answers_section(runs: list[dict]) -> list[str]:
    lines = ["", "## Ответы на контрольные вопросы"]
    for i, d in enumerate(DETAILS):
        lines += ["", f"**{d.question}** — ожидается: {d.fact}", ""]
        for r in runs:
            p = r["probes"][i]
            mark = "❔" if p["ok"] and p["denial"] else "✅" if p["ok"] else "❌"
            lines.append(f"- {mark} {r['title']}: {' '.join((p['answer'] or p['error'] or '—').split())}")
    lines += ["", "Файлы: " + ", ".join(f"`{r['history_file']}` + `.state.json`" for r in runs) + "."]
    return lines


def demo(client: LLMClient, helper_model: str | None, modes: list[str], keep: int, chunk: int, max_steps: int,
         branches: bool) -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = DAY_DIR / f"demo_{stamp}"
    comp = complexity()
    runs = compare(client, helper_model, modes, keep, chunk, max_steps, root / "compare") if modes else []
    tree = branch_demo(client, helper_model, keep, chunk, max_steps, root / "branches") if branches else None
    if runs:
        print_comparison(runs, comp)
    save_report(runs, tree, comp, client.model, helper_model or client.model, keep, chunk, stamp)
    return 0


# ---------------------------------------------------------------- запуск


def main() -> int:
    parser = argparse.ArgumentParser(description="День 10: агент с тремя стратегиями управления контекстом и ветками.")
    parser.add_argument("prompt", nargs="?", help="один ход и выход; без него — чат в консоли")
    parser.add_argument("--session", default="strategies", help="имя сессии: memory/day10/<имя>/")
    parser.add_argument("--new", action="store_true", help="начать заново (папка сессии уходит в архив)")
    parser.add_argument("--strategy", choices=STRATEGIES, help="стратегия (по умолчанию — сохранённая или facts)")
    parser.add_argument("--keep", type=int, default=6, help="сколько последних сообщений отправлять как есть")
    parser.add_argument("--chunk", type=int, default=10, help="summary: сжимать, когда за хвостом накопится столько")
    parser.add_argument("--helper-model", help="модель для служебных запросов — факты и сжатие (по умолчанию та же)")
    parser.add_argument("--demo", action="store_true", help="сценарий Дня 9 на всех стратегиях + ветвление, отчёт")
    parser.add_argument("--modes", default="sliding,facts,summary,full",
                        help="для --demo: стратегии через запятую (full — эталон, вся история); none — только ветки")
    parser.add_argument("--no-branches", action="store_true", help="для --demo: без проверки ветвления")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--max-steps", type=int, default=8, help="предел запросов к модели за один ход")
    args = parser.parse_args()

    if not NAME_RE.fullmatch(args.session):
        print("Ошибка: имя сессии — буквы, цифры, дефис, подчёркивание.", file=sys.stderr)
        return 1
    if args.keep < 0 or args.chunk < 1:
        print("Ошибка: --keep ≥ 0, --chunk ≥ 1.", file=sys.stderr)
        return 1
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    model = args.model or settings.model

    if args.demo:
        modes = [m.strip() for m in args.modes.split(",") if m.strip() and m.strip() != "none"]
        bad = [m for m in modes if m not in STRATEGIES]
        if bad:
            print(f"Ошибка: неизвестные стратегии {', '.join(bad)}. Есть: {', '.join(STRATEGIES)}.", file=sys.stderr)
            return 1
        client = LLMClient(settings.base_url, settings.api_key, model, settings.timeout, retries=settings.retries)
        return demo(client, args.helper_model, modes, args.keep, args.chunk, args.max_steps, not args.no_branches)

    try:
        ws = Workspace(DAY_DIR / args.session)
        if args.new:
            archived = ws.archive()
            if archived:
                print(f"Прошлая сессия сохранена в {archived.relative_to(ROOT)}.")
            ws = Workspace(DAY_DIR / args.session)
        store = ws.store(ws.current)
        saved = store.load()
    except StoreError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    client = LLMClient(settings.base_url, settings.api_key, model, settings.timeout,
                       session_id=(saved.session_id or None) if saved else None, retries=settings.retries)
    helper = (LLMClient(settings.base_url, settings.api_key, args.helper_model, settings.timeout,
                        retries=settings.retries) if args.helper_model else None)
    agent = make_agent(client, helper or client, store, args.strategy or "facts", args.keep, args.chunk,
                       args.max_steps)
    try:
        info = agent.restore(saved) if saved else RestoreInfo(1, 0, 0, 0, "")
        if args.strategy and args.strategy != agent.strategy_name:
            agent.switch(args.strategy)
        ws.save()
    except StoreError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print_restore(info, store, agent)
    for note in agent.notes:
        print(f"⚠️  {note}")
    if len(ws.meta["branches"]) > 1 or ws.meta["checkpoints"]:
        print_branches(ws, agent)
    print()

    if args.prompt:
        try:
            reply = agent.run(args.prompt)
        except (LLMError, StoreError) as e:
            print(f"❌ Ошибка: {e}", file=sys.stderr)
            return 1
        print(f"\nАгент: {reply.text.strip()}\n")
        print_turn(agent, agent.ledger[-1])
        return 0
    return chat(agent, ws)


if __name__ == "__main__":
    from run_timer import run_timed
    sys.exit(run_timed(main))
