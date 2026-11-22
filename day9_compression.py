"""День 9. Управление контекстом: сжатие истории.

Агент из Дня 8 (`TokenAgent`: инструменты, история в JSON, счётчик токенов и стоимости), который отправляет
в модель не всю историю, а:
    • системный промпт;
    • summary — сжатое содержание старой части диалога. Хранится отдельно от истории,
      в memory/<сессия>.summary.json, и подставляется в системный промпт;
    • последние N сообщений как есть (`--keep`, по умолчанию 6), плюс текущий ход.
Как только за пределами этого «хвоста» накапливается `--chunk` несжатых сообщений (по умолчанию 10),
они вместе с прежним summary уходят отдельным запросом к модели-«сжимателю», и summary обновляется.
Полная история остаётся в memory/<сессия>.json: сжимается только то, что уходит в модель.
Граница сжатия всегда проходит по началу хода — вызов инструмента и его результат не разрываются.

Стиль summary (`--style`):
    facts — структурированное резюме: имена, числа, даты, бюджет, ограничения и решения — дословно;
    brief — пересказ в 2–3 предложениях («просто попросить модель сжать») — чтобы увидеть, что теряется;
    off   — без сжатия, вся история, как в Дне 8.

    python day9_compression.py                       # чат со сжатием (сессия compress)
    python day9_compression.py --keep 4 --chunk 6    # сжимать чаще
    python day9_compression.py --style off           # тот же чат без сжатия
    python day9_compression.py --demo                # сценарий на 26 сообщений в режимах off / facts / brief,
                                                     # 12 контрольных вопросов по деталям + отчёт в results/
    python day9_compression.py --demo --modes off,facts

Команды чата: /play [N] — отправить следующие N реплик сценария, /probe — контрольные вопросы по деталям,
/summary, /context, /compress, /tokens, /history, /reset, /exit.
"""
import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from config import ConfigError, load_settings
from day6_agent import SYSTEM_PROMPT, TOOLS
from day7_memory import MEMORY_DIR, ROOT, HistoryStore, RestoreInfo, StoreError, now_iso, print_history, print_restore
from day8_tokens import TokenAgent, TurnStats, Usage, clip, fmt, money, price_for, print_ledger
from day8_tokens import on_event as day8_event
from llm_client import LLMClient, LLMError

RESULTS_DIR = ROOT / "results"
STYLES = ("facts", "brief", "off")
STYLE_TITLES = {"off": "без сжатия", "facts": "сжатие: факты", "brief": "сжатие: пересказ"}

AGENT_PROMPT = SYSTEM_PROMPT + (
    "\nНа вопросы о прошлой части разговора отвечай только по тому, что есть в контексте. "
    "Если такой детали в контексте нет — прямо скажи, что не знаешь, и не придумывай.")

SUMMARY_HEADER = ("\n\n## Краткое содержание начала разговора\n"
                  "Ранняя часть диалога заменена этим резюме, исходных сообщений у тебя нет. "
                  "Всё, что в нём написано, пользователь сообщил тебе раньше.\n\n")

SUMMARIZER_PROMPTS = {
    "facts": (
        "Ты сжимаешь историю диалога пользователя с ассистентом. Резюме заменит исходные сообщения: ассистент "
        "продолжит разговор, видя только резюме и несколько последних реплик.\n"
        "Объедини прежнее резюме и новые сообщения в одно обновлённое резюме. Правила:\n"
        "- сохрани все конкретные детали точно, как в оригинале: имена, фамилии и роли людей, названия, "
        "числа с единицами, суммы с валютой, даты и сроки, ограничения и требования, принятые решения "
        "и то, от чего отказались;\n"
        "- если значение изменилось — оставь новое и отметь старое, например «бюджет: 5 млн (было 4 млн)»;\n"
        "- советы и рассуждения ассистента опусти или сократи до нескольких слов, если в них нет решений;\n"
        "- ничего не придумывай и не добавляй от себя;\n"
        "- формат: markdown-список по разделам «Участники», «Проект», «Сроки», «Бюджет», «Ограничения», "
        "«Решения», «Прочее»; пустые разделы пропускай;\n"
        "- не длиннее 200 слов.\n"
        "Верни только текст резюме."),
    "brief": (
        "Ты сжимаешь историю диалога пользователя с ассистентом. Объедини прежнее резюме и новые сообщения "
        "и кратко перескажи, о чём шёл разговор, в 2–3 предложениях. Верни только пересказ."),
}


# ---------------------------------------------------------------- хранилище summary


@dataclass
class Compression:
    """Одно сжатие: какие сообщения истории ушли в summary и во что это обошлось."""
    n: int
    at: str
    before_turn: int  # перед каким ходом пользователя сжали
    start: int  # сжатые сообщения — history[start:end]
    end: int
    messages: int
    replaced_tokens: int  # ≈ токенов в сжатых сообщениях
    summary_tokens: int  # ≈ токенов в новом summary (он заменяет и старое summary, и эти сообщения)
    prompt: int  # точные числа запроса к сжимателю
    completion: int
    cached: int
    reasoning: int
    cost: float | None
    latency: float
    summary: str


class SummaryStore:
    """summary хранится отдельно от истории: memory/<сессия>.summary.json, запись атомарная."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("summary", ""), str):
                raise ValueError("ожидался объект с полем summary")
            return data
        except (OSError, ValueError) as e:
            raise StoreError(f"не удалось прочитать summary {self.path}: {e}. Начните заново с --new.")

    def save(self, data: dict) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            raise StoreError(f"не удалось сохранить summary в {self.path}: {e}")

    def archive(self) -> Path | None:
        if not self.path.exists():
            return None
        target = self.path.with_name(f"{self.path.name.removesuffix('.summary.json')}_"
                                     f"{datetime.now():%Y%m%d_%H%M%S}.summary.json")
        try:
            os.replace(self.path, target)
        except OSError as e:
            raise StoreError(f"не удалось переместить {self.path} в архив: {e}")
        return target


def summary_path(history_path: Path) -> Path:
    return history_path.with_name(history_path.stem + ".summary.json")


# ---------------------------------------------------------------- агент со сжатием


def render(messages: list[dict]) -> str:
    """Сообщения истории текстом для сжимателя: реплики, вызовы инструментов и их результаты."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            out.append(f"Пользователь: {m.get('content') or ''}")
        elif role == "assistant":
            for call in m.get("tool_calls") or []:
                f = call.get("function") or {}
                out.append(f"Ассистент вызвал {f.get('name')}({f.get('arguments')})")
            if m.get("content"):
                out.append(f"Ассистент: {m['content']}")
        elif role == "tool":
            out.append(f"Результат инструмента: {m.get('content')}")
    return "\n\n".join(out)


class CompressingAgent(TokenAgent):
    """TokenAgent из Дня 8, который заменяет старую часть истории на summary в каждом запросе к модели."""

    def __init__(self, client: LLMClient, tools, store: HistoryStore, summary_store: SummaryStore, *,
                 style: str = "facts", keep_last: int = 6, chunk: int = 10,
                 summary_client: LLMClient | None = None, **kwargs):
        if style not in STYLES:
            raise ValueError(f"style: {', '.join(STYLES)}")
        self.summary_store = summary_store
        self.style = style
        self.keep_last = max(keep_last, 0)
        self.chunk = max(chunk, 1)
        self.summary_client = summary_client or client
        self.summary = ""
        self.summarized_upto = 1  # history[1:summarized_upto] заменены summary; history[0] — системный промпт
        self.compressions: list[Compression] = []
        self.restore_note = ""
        self.full_estimate = 0  # ≈ токенов, если бы последний запрос ушёл с полной историей
        kwargs.setdefault("system_prompt", AGENT_PROMPT)
        super().__init__(client, tools, store, **kwargs)

    @property
    def enabled(self) -> bool:
        return self.style != "off"

    # --- что уходит в модель

    def context(self) -> list[dict]:
        """Сообщения для запроса: системный промпт + summary, затем несжатая часть истории как есть."""
        if not self.enabled or not self.summary:
            return self.history
        system = dict(self.history[0])
        system["content"] = (system.get("content") or "") + SUMMARY_HEADER + self.summary
        return [system] + self.history[self.summarized_upto:]

    def _select(self, schemas: list[dict]) -> tuple[list[dict], int]:
        self.full_estimate = self.estimator.apply(self.estimator.raw_messages(self.history, schemas))
        messages = self.context()
        return messages, (self.summarized_upto - 1 if messages is not self.history else 0)

    # --- сжатие

    def _cut(self) -> int | None:
        """Самое позднее начало хода, после которого в хвосте остаётся не меньше keep_last сообщений."""
        n = len(self.history)
        cuts = [i for i, m in enumerate(self.history)
                if m.get("role") == "user" and i > self.summarized_upto and n - i >= self.keep_last]
        return cuts[-1] if cuts else None

    def pending(self) -> int:
        """Сколько сообщений за пределами хвоста ещё не сжато."""
        cut = self._cut()
        return cut - self.summarized_upto if cut else 0

    def compress(self, force: bool = False) -> Compression | None:
        """Сжимает накопившееся за пределами хвоста, если его не меньше chunk сообщений (force — сколько есть)."""
        if not self.enabled:
            return None
        cut = self._cut()
        if cut is None or (not force and cut - self.summarized_upto < self.chunk):
            return None
        start, part = self.summarized_upto, self.history[self.summarized_upto:cut]
        user_turns = [i for i, m in enumerate(self.history) if m.get("role") == "user"]
        first, last = user_turns.index(start) + 1, user_turns.index(cut)  # номера сжимаемых ходов
        self.on_event("compress_start", {"messages": len(part), "turns": (first, last)})
        request = [{"role": "system", "content": SUMMARIZER_PROMPTS[self.style]},
                   {"role": "user", "content": f"Прежнее резюме:\n{self.summary or '(пока нет — это начало диалога)'}"
                                               f"\n\nНовые сообщения диалога:\n\n{render(part)}"}]
        result = self.summary_client.chat(request, temperature=0)
        text = result.text.strip()
        if not text:
            raise LLMError(f"сжиматель вернул пустой ответ (finish_reason={result.finish_reason})")
        usage = Usage.parse(result.usage)
        priced = price_for(self.summary_client.model)
        est = self.estimator
        c = Compression(n=len(self.compressions) + 1, at=now_iso(), before_turn=len(self.ledger) + 1, start=start,
                        end=cut, messages=len(part), replaced_tokens=est.apply(sum(est.raw_message(m, False) for m in part)),
                        summary_tokens=est.text(text), prompt=usage.prompt, completion=usage.completion,
                        cached=usage.cached, reasoning=usage.reasoning,
                        cost=usage.cost(priced[0] if priced else None), latency=result.latency, summary=text)
        self.summary, self.summarized_upto = text, cut
        self.compressions.append(c)
        self.save_summary()
        self.on_event("compress", {"c": c, "turns": (first, last)})
        return c

    def run(self, user_text: str):
        try:
            self.compress()
        except LLMError as e:  # без нового summary агент всё равно ответит — просто с более длинным контекстом
            self.on_event("compress_error", {"error": str(e)})
        return super().run(user_text)

    def compression_totals(self) -> dict:
        costs = [c.cost for c in self.compressions]
        return {"count": len(self.compressions), "prompt": sum(c.prompt for c in self.compressions),
                "completion": sum(c.completion for c in self.compressions),
                "cost": None if any(c is None for c in costs) else sum(costs)}

    # --- хранение

    def save_summary(self) -> None:
        self.summary_store.save({"format": 1, "style": self.style, "keep_last": self.keep_last, "chunk": self.chunk,
                                 "updated_at": now_iso(), "summarized_upto": self.summarized_upto,
                                 "summary": self.summary, "compressions": [asdict(c) for c in self.compressions]})

    def restore(self, saved) -> RestoreInfo:
        info = super().restore(saved)
        data = self.summary_store.load()
        if not data:
            return info
        upto = int(data.get("summarized_upto") or 1)
        n = len(self.history)
        if not (1 <= upto <= n and (upto == n or self.history[upto].get("role") == "user")):
            self.restore_note = (f"summary не совпадает с историей (сжато до сообщения {upto}, в истории {n}) — "
                                 f"не используется, история будет сжата заново.")
            return info
        known = Compression.__dataclass_fields__
        self.summary, self.summarized_upto = data.get("summary") or "", upto
        self.compressions = [Compression(**{k: v for k, v in c.items() if k in known})
                             for c in data.get("compressions") or []]
        if data.get("style") and data["style"] != self.style:
            self.restore_note = f"summary сделано в стиле {data['style']}, дальше сжимаю в стиле {self.style}."
        return info

    def reset(self) -> None:
        self.summary, self.summarized_upto, self.compressions = "", 1, []
        if getattr(self, "_ready", False):
            self.summary_store.archive()
        super().reset()


# ---------------------------------------------------------------- вывод


def on_event(kind: str, data: dict) -> None:
    if kind == "compress_start":
        first, last = data["turns"]
        print(f"  🗜  сжимаю {data['messages']} сообщений (ходы {first}–{last}) в summary…")
    elif kind == "compress":
        c: Compression = data["c"]
        print(f"  🗜  сжатие №{c.n}: {c.messages} сообщений ≈{fmt(c.replaced_tokens)} токенов → summary "
              f"≈{fmt(c.summary_tokens)} | сжиматель: вход {fmt(c.prompt)}, выход {fmt(c.completion)}, "
              f"{c.latency:.1f} с, {money(c.cost)}")
    elif kind == "compress_error":
        print(f"  ⚠️  сжать не удалось: {data['error']} — запрос уйдёт с несжатой историей")
    elif kind == "retry":
        print(f"  ⏳ {data['error']} Повтор {data['attempt']}/{data['retries']} через {data['delay']:.1f} с…")
    else:
        day8_event(kind, data)


def watch_retries(*clients: LLMClient | None, event=on_event) -> None:
    """Повторы запросов (таймаут, 429, 5xx) печатаются, чтобы паузы в выводе были понятны."""
    for c in clients:
        if c is not None:
            c.on_retry = lambda info: event("retry", info)


def quiet_event(kind: str, data: dict) -> None:
    """Для --demo: только сжатие и инструменты; строку хода печатает print_turn_line."""
    if kind in ("tokens", "llm", "truncate"):
        return
    on_event(kind, data)


def dialogue_cost(agent: CompressingAgent) -> tuple[float | None, float | None]:
    turns = agent.totals()["cost"]
    comp = agent.compression_totals()["cost"]
    return turns, (None if turns is None or comp is None else turns + comp)


def full_projection(agent: CompressingAgent, call) -> int:
    """≈ вход последнего запроса, если бы ушла вся история: оценка полной истории в масштабе точного prompt_tokens
    (оценка и факт расходятся на проценты — так сравниваются числа одной шкалы). 0 — если summary ещё нет."""
    if not (agent.enabled and agent.summary and agent.full_estimate and call.estimate):
        return 0
    return round(agent.full_estimate * call.prompt / call.estimate)


def print_turn(agent: CompressingAgent, turn: TurnStats) -> None:
    mode = STYLE_TITLES[agent.style] + (f", хвост {agent.keep_last}, порог {agent.chunk}" if agent.enabled else "")
    print(f"📊 Ход {turn.n} · {mode}")
    if not turn.calls:
        print(f"   ответа нет: {turn.error}")
        return
    last = turn.calls[-1]
    tail = len(agent.history) - agent.summarized_upto if agent.summary else len(agent.history) - 1
    if agent.enabled and agent.summary:
        sent = (f"system + summary (≈{fmt(agent.estimator.text(agent.summary))}) + {tail} сообщ. как есть; "
                f"{agent.summarized_upto - 1} старых заменены summary")
    else:
        sent = f"system + вся история ({tail} сообщ.)"
    cached = f", из кэша {fmt(last.cached)}" if last.cached else ""
    print(f"   в запрос ушло:   {sent}")
    print(f"   вход:            {fmt(last.prompt)} токенов{cached}; за ход {fmt(turn.prompt)}, запросов {len(turn.calls)}")
    full = full_projection(agent, last)
    if full:
        print(f"   без сжатия:      было бы ≈{fmt(full)} — экономия ≈{(1 - last.prompt / full) * 100:.0f}%")
    print(f"   ответ:           {fmt(turn.completion)} токенов")
    if agent.enabled:
        left = max(agent.chunk - agent.pending(), 0)
        when = f"до следующего ещё {left} сообщ." if left else "следующее — перед следующим ходом"
        print(f"   история в файле: {len(agent.history)} сообщ., ≈{fmt(agent.history_tokens())} токенов; "
              f"сжатий {len(agent.compressions)}, {when}")
    turns_cost, total = dialogue_cost(agent)
    comp = agent.compression_totals()
    extra = f" + сжатие {money(comp['cost'])} ({comp['count']}×) = {money(total)}" if comp["count"] else ""
    print(f"   стоимость:       ход {money(turn.cost)} · диалог {money(turns_cost)}{extra}")


def print_turn_line(agent: CompressingAgent, turn: TurnStats) -> None:
    if not turn.calls:
        print(f"  ❌ {turn.error}")
        return
    last = turn.calls[-1]
    cached = f" (кэш {fmt(last.cached)})" if last.cached else ""
    full = full_projection(agent, last)
    saved = f" | без сжатия было бы ≈{fmt(full)}, −{(1 - last.prompt / full) * 100:.0f}%" if full else ""
    print(f"  📊 вход {fmt(last.prompt)}{cached}{saved} | выход {fmt(turn.completion)} | {money(turn.cost)}")


def print_context(agent: CompressingAgent) -> None:
    est = agent.estimator
    system = est.apply(est.raw_message(agent.history[0]))
    print(f"  Стиль: {STYLE_TITLES[agent.style]}" + (f", хвост {agent.keep_last} сообщ., сжимать по {agent.chunk}"
                                                     if agent.enabled else ""))
    print(f"  system:           ≈{fmt(system)}")
    if agent.enabled and agent.summary:
        print(f"  summary:          ≈{fmt(est.text(agent.summary))} — заменяет сообщения 1–{agent.summarized_upto - 1}")
        tail = agent.history[agent.summarized_upto:]
    else:
        tail = agent.history[1:]
    print(f"  как есть:         {len(tail)} сообщ., ≈{fmt(est.apply(sum(est.raw_message(m, False) for m in tail)))}")
    sent = est.apply(est.raw_messages(agent.context()))
    full = est.apply(est.raw_messages(agent.history))
    print(f"  итого в запрос:   ≈{fmt(sent)} (без схем инструментов); вся история ≈{fmt(full)}")
    if agent.enabled:
        print(f"  ждут сжатия:      {agent.pending()} сообщ. из {agent.chunk}")


def print_summary(agent: CompressingAgent) -> None:
    if not agent.summary:
        print("  (summary ещё нет)")
        return
    for c in agent.compressions:
        print(f"  №{c.n} перед ходом {c.before_turn}: сообщения {c.start}–{c.end - 1} (≈{fmt(c.replaced_tokens)}) → "
              f"≈{fmt(c.summary_tokens)}, {money(c.cost)}")
    print(f"\n{agent.summary}\n")


# ---------------------------------------------------------------- сценарий и контрольные детали


DIALOGUE = [
    "Привет! Меня зовут Марина Ковалёва, я продакт-менеджер сети пекарен «Колосок». Хочу обсудить с тобой план "
    "мобильного приложения для наших покупателей. Внутри компании проект называется «Опара». Ответь коротко.",
    "У нас 37 пекарен в Казани и Набережных Челнах. Приложение нужно для предзаказа выпечки и программы "
    "лояльности. С чего обычно начинают такой проект?",
    "Жёсткий срок: публичный релиз в сторах — 14 марта 2027 года, в этот день открывается наша 38-я пекарня. "
    "Сдвигать нельзя. Как бы ты разбил работу на этапы?",
    "Бюджет на разработку — 2,4 млн рублей, из них 300 тысяч уже заложены на дизайн. Это реалистично для MVP?",
    "Важное ограничение от юристов: персональные данные покупателей храним только на серверах в России, "
    "из-за 152-ФЗ. Какие облака можно рассмотреть?",
    "Какие метрики успеха ты бы предложил отслеживать после запуска?",
    "Мы посовещались и решили: делаем на Flutter, одной кодовой базой под iOS и Android, а не двумя нативными "
    "командами. Какие риски у такого решения?",
    "Тимлидом разработки будет Арсений Гладков, он пришёл к нам из Яндекс Еды. Что ему подготовить к первой "
    "встрече с подрядчиками?",
    "Новость: совет директоров увеличил бюджет до 2,7 млн рублей, но с условием — до 20 февраля 2027 года "
    "показать инвесторам демо. Как это меняет план?",
    "Доставку в первой версии делать не будем — переносим её во вторую фазу, на осень 2027. В MVP только "
    "предзаказ с самовывозом и бонусы. Как объяснить это маркетингу?",
    "Для оплаты выбрали ЮKassa: у нашей бухгалтерии с ней уже есть договор. Что учесть при интеграции?",
    "Какие ошибки чаще всего допускают при запуске программ лояльности?",
    "Спасибо! Последний вопрос на сегодня: как часто проводить статус-встречи с подрядчиком?",
]

PROBE_SUFFIX = (" Ответь одним-двумя предложениями, только по нашему разговору. Если этого в разговоре не было "
                "или ты не знаешь — так и скажи. Инструменты не вызывай.")


@dataclass(frozen=True)
class Detail:
    id: str
    kind: str  # имя, название, число, срок, бюджет, ограничение, решение
    turn: int  # в каком ходе сценария названа
    fact: str  # что агент должен помнить
    question: str
    patterns: tuple[str, ...]  # все регулярные выражения должны найтись в ответе (после normalize)


DETAILS = [
    Detail("name", "имя", 1, "пользователь — Марина Ковалёва", "Как меня зовут?", ("марин", "ковалев")),
    Detail("codename", "название", 1, "проект называется «Опара»", "Как внутри компании называется наш проект?",
           ("опар",)),
    Detail("bakeries", "число", 2, "37 пекарен в Казани и Набережных Челнах",
           "Сколько у нас пекарен и в каких городах?", (r"\b37\b", "казан", "челн")),
    Detail("deadline", "срок", 3, "релиз 14 марта 2027", "Какой срок публичного релиза?",
           (r"\b14\s*(марта|\.0?3\b|/0?3\b)",)),
    Detail("budget", "бюджет", 9, "бюджет 2,7 млн ₽ (был 2,4 млн)", "Какой у нас сейчас бюджет на разработку?",
           (r"2[,.]7|2700",)),
    Detail("design", "бюджет", 4, "300 тыс. ₽ на дизайн", "Сколько из бюджета заложено на дизайн?", (r"\b300\b",)),
    Detail("storage", "ограничение", 5, "персональные данные — только на серверах в России (152-ФЗ)",
           "Какое ограничение поставили юристы по хранению данных?", (r"росси|\bрф\b", r"152|сервер|хран")),
    Detail("flutter", "решение", 7, "делаем на Flutter, одной кодовой базой", "На какой технологии решили делать приложение?",
           (r"flutter|флаттер",)),
    Detail("teamlead", "имя", 8, "тимлид — Арсений Гладков", "Как зовут тимлида разработки?", ("арсени", "гладков")),
    Detail("demo", "срок", 9, "демо инвесторам до 20 февраля 2027", "К какой дате нужно показать демо инвесторам?",
           (r"\b20\s*(феврал|\.0?2\b|/0?2\b)",)),
    Detail("delivery", "решение", 10, "доставка перенесена во вторую фазу (осень 2027)",
           "Что решили насчёт доставки?", (r"доставк", r"втор|2[\s-]*(?:-?й|ю)?\s*фаз|осен|позже|отлож|перен|не будем|не делаем")),
    Detail("payment", "решение", 11, "оплата через ЮKassa", "Какой платёжный сервис выбрали?",
           (r"ю\s?kassa|ю\s?касс|yoo\s?kassa",)),
]


def normalize(text: str) -> str:
    text = (text or "").lower().replace("ё", "е").replace(" ", " ").replace(" ", " ")
    return re.sub(r"(?<=\d) (?=\d{3}\b)", "", text)  # «2 700 000» → «2700000»


def matches(detail: Detail, text: str) -> bool:
    t = normalize(text)
    return all(re.search(p, t) for p in detail.patterns)


def run_probes(agent: CompressingAgent, details: list[Detail] = DETAILS, verbose: bool = True) -> list[dict]:
    """Каждый вопрос — отдельный запрос с текущим контекстом агента; в историю вопросы не попадают,
    поэтому ответ на один вопрос не подсказывает следующий."""
    context = agent.context()
    schemas = [t.schema() for t in agent.tools.values()]
    priced = price_for(agent.client.model)
    out = []
    for d in details:
        try:
            result = agent.client.chat(context + [{"role": "user", "content": d.question + PROBE_SUFFIX}], tools=schemas)
        except LLMError as e:
            out.append({"id": d.id, "question": d.question, "answer": "", "ok": False, "error": str(e),
                        "prompt": 0, "cached": 0, "completion": 0, "cost": 0.0})
            if verbose:
                print(f"  ❌ {d.question} — ошибка: {e}")
            continue
        usage = Usage.parse(result.usage)
        answer = result.text.strip() or ("(модель попросила инструмент вместо ответа)" if result.tool_calls else "")
        ok = matches(d, answer)
        out.append({"id": d.id, "question": d.question, "answer": answer, "ok": ok, "error": "",
                    "prompt": usage.prompt, "cached": usage.cached, "completion": usage.completion,
                    "cost": usage.cost(priced[0] if priced else None)})
        if verbose:
            print(f"  {'✅' if ok else '❌'} [{d.kind}] {d.question} → {clip(answer, 150)}")
    return out


# ---------------------------------------------------------------- чат


def chat(agent: CompressingAgent) -> int:
    print(f"Агент на {agent.client.model}. {STYLE_TITLES[agent.style]}"
          + (f": последние {agent.keep_last} сообщ. как есть, остальное — summary, сжатие каждые {agent.chunk} сообщ."
             if agent.enabled else "") +
          "\nКоманды: /play [N] — реплики сценария, /probe — контрольные вопросы, /summary, /context, /compress, "
          "/tokens, /history, /reset, /exit\n")
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
                print()
                continue
            if command == "/tokens":
                print_ledger(agent)
                comp = agent.compression_totals()
                if comp["count"]:
                    print(f"  + сжатий {comp['count']}: вход {fmt(comp['prompt'])}, выход {fmt(comp['completion'])}, "
                          f"{money(comp['cost'])}; всего {money(dialogue_cost(agent)[1])}")
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
                if not agent.enabled:
                    print("  Сжатие выключено (--style off).\n")
                elif not agent.compress(force=True):
                    print(f"  Сжимать нечего: за пределами хвоста из {agent.keep_last} сообщ. нет целых ходов.\n")
                else:
                    print()
                continue
            if command == "/probe":
                if agent.enabled:
                    agent.compress()  # как перед следующим ходом
                results = run_probes(agent)
                print(f"  Помнит {sum(r['ok'] for r in results)}/{len(results)} деталей. "
                      f"Вход на вопрос ≈{fmt(results[0]['prompt'] if results else 0)} токенов.\n")
                continue
            if command == "/play":
                done = len(agent.ledger)
                count = int(arg) if arg.strip().isdigit() else len(DIALOGUE) - done
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
                print("  Неизвестная команда. Есть: /play, /probe, /summary, /context, /compress, /tokens, /history, "
                      "/reset, /exit\n")
                continue
            reply = agent.run(text)
        except LLMError as e:
            print(f"❌ Ошибка: {e}\n", file=sys.stderr)
            continue
        except StoreError as e:
            print(f"⚠️  {e}\n", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("\n(прервано — ход в историю не попал)\n")
            continue
        print(f"\nАгент: {reply.text.strip() or '(пустой ответ)'}\n")
        print_turn(agent, agent.ledger[-1])
        print()


# ---------------------------------------------------------------- сравнение режимов


def make_agent(client: LLMClient, session: str, style: str, keep: int, chunk: int, max_steps: int,
               summary_client: LLMClient | None = None, event=on_event) -> CompressingAgent:
    store = HistoryStore(MEMORY_DIR / f"{session}.json")
    watch_retries(client, summary_client, event=event)
    return CompressingAgent(client, TOOLS, store, SummaryStore(summary_path(store.path)), style=style, keep_last=keep,
                            chunk=chunk, summary_client=summary_client, max_steps=max_steps, on_event=event)


def demo(client: LLMClient, modes: list[str], keep: int, chunk: int, max_steps: int,
         summary_client: LLMClient | None) -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Сценарий: {len(DIALOGUE)} реплик пользователя ({2 * len(DIALOGUE)} сообщений с ответами), "
          f"контрольных деталей {len(DETAILS)}:")
    for d in DETAILS:
        print(f"  · ход {d.turn:>2} [{d.kind}] {d.fact}")
    runs = []
    for style in modes:
        client = LLMClient(client.base_url, client.api_key, client.model, client.timeout,  # своя сессия OpenCode
                           retries=client.retries)
        before = dict(summary_client.stats) if summary_client else None
        agent = make_agent(client, f"day9_{style}_{stamp}", style, keep, chunk, max_steps, summary_client, quiet_event)
        print(f"\n{'=' * 78}\n=== {STYLE_TITLES[style]}"
              + (f": хвост {keep} сообщ., сжатие каждые {chunk}" if agent.enabled else ": вся история в каждом запросе")
              + f"\n{'=' * 78}")
        for k, line in enumerate(DIALOGUE, 1):
            print(f"\n--- ход {k}/{len(DIALOGUE)}: {clip(line, 110)}")
            try:
                reply = agent.run(line)
                print(f"Агент: {clip(reply.text, 220) if reply.text.strip() else '(пустой ответ)'}")
            except LLMError as e:
                print(f"❌ {e}")
            print_turn_line(agent, agent.ledger[-1])
        if agent.enabled:
            print("\n--- перед вопросами (как перед следующим ходом):")
            try:
                if not agent.compress():
                    print("  сжимать нечего")
            except LLMError as e:
                print(f"  ⚠️  сжать не удалось: {e}")
        print(f"\n--- контрольные вопросы ({len(DETAILS)}), каждый — отдельный запрос с текущим контекстом:")
        probes = run_probes(agent)
        stats = dict(client.stats)
        if summary_client:  # общий для всех режимов — берём прирост за этот режим
            stats = {k: v + summary_client.stats[k] - before[k] for k, v in stats.items()}
        runs.append(collect(agent, probes, stats))
        r = runs[-1]
        if r["failed_turns"]:
            print(f"\n⚠️  Не прошли ходы {', '.join(map(str, r['failed_turns']))} — их детали агенту не сообщены, "
                  f"в сравнении они помечены ⚠️")
        print(f"\nИтог «{STYLE_TITLES[style]}»: помнит {r['remembered']}/{len(DETAILS)} деталей, вход за диалог "
              f"{fmt(r['dialogue']['prompt'])} + сжатие {fmt(r['compression']['prompt'])}, "
              f"стоимость {money(r['total_cost'])}")
    print_comparison(runs)
    save_report(runs, client.model, keep, chunk, stamp, summary_client.model if summary_client else client.model)
    return 0


def compressed_turns(agent: CompressingAgent) -> int:
    """Сколько первых ходов сценария целиком ушли в summary."""
    return sum(1 for i, m in enumerate(agent.history) if m.get("role") == "user" and i < agent.summarized_upto) \
        if agent.summary else 0


def lost_turn(detail: Detail, failed: list[int]) -> bool:
    """Деталь не дошла до агента: ход, в котором её называют, не прошёл (агент откатил его из истории)."""
    return detail.turn in failed


def collect(agent: CompressingAgent, probes: list[dict], requests: dict | None = None) -> dict:
    t, comp = agent.totals(), agent.compression_totals()
    failed = [x.n for x in agent.ledger if x.status in ("error", "overflow")]
    fair = [(p, d) for p, d in zip(probes, DETAILS) if not lost_turn(d, failed) and not p["error"]]
    contexts = [x.calls[-1].prompt for x in agent.ledger if x.calls]
    probe_prompt = [p["prompt"] for p in probes if p["prompt"]]
    probe_cost = sum(p["cost"] or 0 for p in probes)
    turns_cost, total = dialogue_cost(agent)
    cut = compressed_turns(agent)
    return {
        "style": agent.style, "title": STYLE_TITLES[agent.style], "keep_last": agent.keep_last, "chunk": agent.chunk,
        "remembered": sum(p["ok"] for p in probes),
        "failed_turns": failed, "requests": requests or dict(agent.client.stats),
        "probe_errors": sum(bool(p["error"]) for p in probes),
        "remembered_fair": sum(p["ok"] for p, _ in fair), "fair_total": len(fair),
        "remembered_compressed": sum(p["ok"] for p, d in zip(probes, DETAILS) if d.turn <= cut),
        "compressed_details": sum(1 for d in DETAILS if d.turn <= cut), "compressed_turns": cut,
        "in_summary": {d.id: matches(d, agent.summary) for d in DETAILS} if agent.summary else {},
        "dialogue": {"prompt": t["prompt"], "cached": t["cached"], "completion": t["completion"], "cost": turns_cost},
        "compression": comp, "total_prompt": t["prompt"] + comp["prompt"],
        "total_completion": t["completion"] + comp["completion"], "total_cost": total,
        "context_first": contexts[0] if contexts else 0, "context_last": contexts[-1] if contexts else 0,
        "context_max": max(contexts, default=0),
        "probe_prompt": round(sum(probe_prompt) / len(probe_prompt)) if probe_prompt else 0,
        "probe_cost": probe_cost, "probes": probes, "summary": agent.summary,
        "compressions": [asdict(c) for c in agent.compressions],
        "turns": [{"n": x.n, "status": x.status, "prompt": x.calls[-1].prompt if x.calls else 0,
                   "prompt_turn": x.prompt, "cached": x.cached, "completion": x.completion, "cost": x.cost,
                   "sent_messages": x.calls[-1].sent_messages if x.calls else 0, "replaced": x.dropped,
                   "answer": x.answer, "error": x.error} for x in agent.ledger],
        "history_file": agent.store.path.name, "summary_file": agent.summary_store.path.name,
    }


def mark(r: dict, i: int, d: Detail) -> str:
    """✅/❌ ответа; ⚠️ — проверить нельзя: не прошёл ход с этой деталью или сам контрольный вопрос."""
    p = r["probes"][i]
    if p["error"]:
        return "⚠️ вопрос"
    if lost_turn(d, r["failed_turns"]):
        return ("✅" if p["ok"] else "❌") + f" ⚠️ х{d.turn}"
    return "✅" if p["ok"] else "❌"


def delta(value: float, base: float) -> str:
    return f"{(value / base - 1) * 100:+.0f}%" if base else ""


def print_comparison(runs: list[dict]) -> None:
    base = next((r for r in runs if r["style"] == "off"), None)
    print(f"\n{'=' * 78}\nСравнение\n{'=' * 78}")
    head = f"  {'деталь':<34}" + "".join(f"{r['title']:>18}" for r in runs)
    print(head)
    for i, d in enumerate(DETAILS):
        print(f"  {clip(f'х{d.turn} {d.fact}', 33):<34}" + "".join(f"{mark(r, i, d):>17}" for r in runs))
    print(f"  {'помнит':<34}" + "".join(f"{str(r['remembered']) + '/' + str(len(DETAILS)):>18}" for r in runs))
    if any(r["fair_total"] < len(DETAILS) for r in runs):
        print(f"  {'  без ⚠️ (упавших запросов)':<34}" + "".join(
            f"{str(r['remembered_fair']) + '/' + str(r['fair_total']):>18}" for r in runs))
    rows = [("повторов / отказов запросов", lambda r: f"{r['requests']['retries']} / {r['requests']['failed']}"),
            ("вход за диалог", lambda r: fmt(r["dialogue"]["prompt"])),
            ("  из них из кэша", lambda r: fmt(r["dialogue"]["cached"])),
            ("вход сжимателя", lambda r: fmt(r["compression"]["prompt"])),
            ("вход всего", lambda r: fmt(r["total_prompt"]) + (f" {delta(r['total_prompt'], base['total_prompt'])}"
                                                               if base and r is not base else "")),
            ("контекст последнего хода", lambda r: fmt(r["context_last"])),
            ("вход на контрольный вопрос", lambda r: fmt(r["probe_prompt"]) + (
                f" {delta(r['probe_prompt'], base['probe_prompt'])}" if base and r is not base else "")),
            ("стоимость диалога", lambda r: money(r["total_cost"]))]
    for label, value in rows:
        print(f"  {label:<34}" + "".join(f"{value(r):>18}" for r in runs))


def save_report(runs: list[dict], model: str, keep: int, chunk: int, stamp: str, summary_model: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path, md_path = RESULTS_DIR / f"day9_{stamp}.json", RESULTS_DIR / f"day9_{stamp}.md"
    json_path.write_text(json.dumps({"model": model, "summary_model": summary_model, "keep_last": keep, "chunk": chunk,
                                     "dialogue": DIALOGUE, "details": [asdict(d) for d in DETAILS], "runs": runs},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    base = next((r for r in runs if r["style"] == "off"), None)
    ok = lambda b: "✅" if b else "❌"  # noqa: E731
    lines = [f"# День 9 · Сжатие истории — прогон {stamp}", "",
             f"Модель: `{model}`, сжиматель: `{summary_model}`. Сценарий: {len(DIALOGUE)} реплик пользователя "
             f"({2 * len(DIALOGUE)} сообщений с ответами). Сжатие: последние {keep} сообщений как есть, "
             f"остальное — в summary каждые {chunk} сообщений; перед контрольными вопросами — ещё одно сжатие, "
             f"как перед следующим ходом. Каждый контрольный вопрос — отдельный запрос с контекстом агента "
             f"после диалога, в историю он не попадает.", "",
             "## Сколько деталей помнит агент", "",
             "| Ход | Тип | Деталь | " + " | ".join(r["title"] for r in runs) + " |",
             "|---|---|---|" + "---|" * len(runs)]
    for i, d in enumerate(DETAILS):
        cells = []
        for r in runs:
            p = r["probes"][i]
            if p["error"]:
                cells.append("⚠️ вопрос не прошёл")
                continue
            cell = ok(p["ok"])
            if lost_turn(d, r["failed_turns"]):
                cell += f" ⚠️ ход {d.turn} не прошёл"
            elif r["in_summary"]:
                cell += " (в summary ✓)" if r["in_summary"][d.id] else (
                    " (в summary ✗)" if d.turn <= r["compressed_turns"] else " (в хвосте)")
            cells.append(cell)
        lines.append(f"| {d.turn} | {d.kind} | {d.fact} | " + " | ".join(cells) + " |")
    lines.append("| | | **помнит** | " + " | ".join(
        f"**{r['remembered']}/{len(DETAILS)}**" + (f" (из сжатых {r['remembered_compressed']}/{r['compressed_details']})"
                                                  if r["compressed_turns"] else "") for r in runs) + " |")
    if any(r["fair_total"] < len(DETAILS) for r in runs):
        lines.append("| | | без ⚠️ | " + " | ".join(f"{r['remembered_fair']}/{r['fair_total']}" for r in runs) + " |")
    lines += ["", "«в summary ✓/✗» — нашлась ли деталь в тексте summary (той же проверкой, что и ответ): так видно, "
              "потерялась она при сжатии или при ответе. ⚠️ — проверить нельзя: ход с этой деталью или сам вопрос "
              "не прошёл даже после повторов (агент откатывает упавший ход, и деталь до него не доходит).", "",
              "## Расход токенов", "",
              "| | " + " | ".join(r["title"] for r in runs) + " |", "|---|" + "---|" * len(runs)]
    rows = [("Запросов / повторов / отказов", lambda r: f"{r['requests']['requests']} / {r['requests']['retries']} / "
                                                        f"{r['requests']['failed']}"),
            ("Ходы, которые не прошли", lambda r: ", ".join(map(str, r["failed_turns"])) or "—"),
            ("Вход за диалог (из кэша)", lambda r: f"{fmt(r['dialogue']['prompt'])} ({fmt(r['dialogue']['cached'])})"),
            ("Выход за диалог", lambda r: fmt(r["dialogue"]["completion"])),
            ("Сжатий / вход сжимателя / выход", lambda r: f"{r['compression']['count']} / {fmt(r['compression']['prompt'])}"
                                                          f" / {fmt(r['compression']['completion'])}"),
            ("**Вход всего**", lambda r: f"**{fmt(r['total_prompt'])}**" + (
                f" ({delta(r['total_prompt'], base['total_prompt'])})" if base and r is not base else "")),
            ("Контекст: 1-й → последний ход (макс.)", lambda r: f"{fmt(r['context_first'])} → {fmt(r['context_last'])} "
                                                                f"({fmt(r['context_max'])})"),
            ("Вход на контрольный вопрос", lambda r: fmt(r["probe_prompt"]) + (
                f" ({delta(r['probe_prompt'], base['probe_prompt'])})" if base and r is not base else "")),
            ("Стоимость диалога (со сжатием)", lambda r: money(r["total_cost"]) + (
                f" ({delta(r['total_cost'], base['total_cost'])})" if base and r is not base and r["total_cost"]
                and base["total_cost"] else "")),
            (f"Стоимость {len(DETAILS)} вопросов", lambda r: money(r["probe_cost"]))]
    for label, value in rows:
        lines.append(f"| {label} | " + " | ".join(value(r) for r in runs) + " |")
    lines += ["", "## Вход по ходам", "", "Число — `prompt_tokens` последнего запроса хода (весь контекст), "
              "🗜 — перед этим ходом было сжатие, в скобках — сколько сообщений истории заменено summary.", "",
              "| Ход | " + " | ".join(r["title"] for r in runs) + " |", "|---|" + "---|" * len(runs)]
    for k in range(len(DIALOGUE)):
        cells = []
        for r in runs:
            x = r["turns"][k] if k < len(r["turns"]) else None
            if not x:
                cells.append("—")
                continue
            mark = "🗜 " if any(c["before_turn"] == x["n"] for c in r["compressions"]) else ""
            cells.append(f"{mark}{fmt(x['prompt'])}" + (f" (−{x['replaced']})" if x["replaced"] else "")
                         + ("" if x["status"] == "ok" else f" ❌ {x['status']}"))
        lines.append(f"| {k + 1} | " + " | ".join(cells) + " |")
    for r in runs:
        if r["compressions"]:
            lines += ["", f"## Сжатия — {r['title']}", "",
                      "| № | Перед ходом | Сообщения | ≈ было → summary | Вход / выход сжимателя | $ |",
                      "|---|---|---|---|---|---|"]
            lines += [f"| {c['n']} | {c['before_turn'] if c['before_turn'] <= len(DIALOGUE) else 'вопросами'} | "
                      f"{c['start']}–{c['end'] - 1} ({c['messages']}) | "
                      f"{fmt(c['replaced_tokens'])} → {fmt(c['summary_tokens'])} | {fmt(c['prompt'])} / "
                      f"{fmt(c['completion'])} | {money(c['cost'])} |" for c in r["compressions"]]
            lines += ["", "Итоговое summary:", "", *[f"> {s}" if s else ">" for s in r["summary"].splitlines()]]
    lines += ["", "## Ответы на контрольные вопросы"]
    for i, d in enumerate(DETAILS):
        lines += ["", f"**{d.question}** — ожидается: {d.fact}", ""]
        for r in runs:
            p = r["probes"][i]
            answer = " ".join((p["answer"] or p["error"] or "—").split())
            lines.append(f"- {ok(p['ok'])} {r['title']}: {answer}")
    lines += ["", "Файлы истории: " + ", ".join(f"`memory/{r['history_file']}`" for r in runs) + "; summary: "
              + ", ".join(f"`memory/{r['summary_file']}`" for r in runs if r["compressions"]) + "."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {md_path.relative_to(ROOT)}, {json_path.name}")


# ---------------------------------------------------------------- запуск


def main() -> int:
    parser = argparse.ArgumentParser(description="День 9: агент со сжатием истории в summary.")
    parser.add_argument("prompt", nargs="?", help="один ход и выход; без него — чат в консоли")
    parser.add_argument("--session", default="compress", help="имя сессии: memory/<имя>.json и <имя>.summary.json")
    parser.add_argument("--new", action="store_true", help="начать новый диалог (старые файлы уходят в архив)")
    parser.add_argument("--style", choices=STYLES, default="facts", help="как сжимать: facts, brief или off")
    parser.add_argument("--keep", type=int, default=6, help="сколько последних сообщений отправлять как есть")
    parser.add_argument("--chunk", type=int, default=10, help="сжимать, когда за хвостом накопится столько сообщений")
    parser.add_argument("--summary-model", help="модель для сжатия (по умолчанию та же)")
    parser.add_argument("--demo", action="store_true", help="сценарий в нескольких режимах + отчёт в results/")
    parser.add_argument("--modes", default="off,facts,brief", help="для --demo: через запятую из off, facts, brief")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--max-steps", type=int, default=8, help="предел запросов к модели за один ход")
    args = parser.parse_args()

    if not re.fullmatch(r"[\w.-]+", args.session):
        print("Ошибка: имя сессии — буквы, цифры, точка, дефис, подчёркивание.", file=sys.stderr)
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
    summary_client = (LLMClient(settings.base_url, settings.api_key, args.summary_model, settings.timeout,
                                retries=settings.retries) if args.summary_model else None)

    if args.demo:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        bad = [m for m in modes if m not in STYLES]
        if bad or not modes:
            print(f"Ошибка: неизвестные режимы {', '.join(bad)}. Есть: {', '.join(STYLES)}.", file=sys.stderr)
            return 1
        client = LLMClient(settings.base_url, settings.api_key, model, settings.timeout, retries=settings.retries)
        return demo(client, modes, args.keep, args.chunk, args.max_steps, summary_client)

    store = HistoryStore(MEMORY_DIR / f"{args.session}.json")
    sstore = SummaryStore(summary_path(store.path))
    try:
        if args.new:
            archived = store.archive()
            sstore.archive()
            if archived:
                print(f"Прошлая история сохранена в {archived.name}.")
        saved = store.load()
    except StoreError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    client = LLMClient(settings.base_url, settings.api_key, model, settings.timeout,
                       session_id=(saved.session_id or None) if saved else None, retries=settings.retries)
    watch_retries(client, summary_client)
    agent = CompressingAgent(client, TOOLS, store, sstore, style=args.style, keep_last=args.keep, chunk=args.chunk,
                             summary_client=summary_client, max_steps=args.max_steps, on_event=on_event)
    try:
        info = agent.restore(saved) if saved else RestoreInfo(1, 0, 0, 0, "")
    except StoreError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print_restore(info, store, agent)
    if agent.restore_note:
        print(f"⚠️  {agent.restore_note}")
    if agent.summary:
        print(f"🗜  summary: {sstore.path.name} — {len(agent.compressions)} сжатий, заменяет сообщения "
              f"1–{agent.summarized_upto - 1}, ≈{fmt(agent.estimator.text(agent.summary))} токенов.")
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
    return chat(agent)


if __name__ == "__main__":
    from run_timer import run_timed
    sys.exit(run_timed(main))
