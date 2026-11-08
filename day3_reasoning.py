"""День 3. Разные способы рассуждения.

10 задач с известным ответом решаются четырьмя способами при двух уровнях рассуждения:

    direct  — прямой вопрос, без дополнительных инструкций;
    steps   — «Решай пошагово»;
    meta    — модель сначала пишет промпт для решения задачи, затем решает по нему (2 запроса);
    experts — в промпте группа экспертов (аналитик, инженер, критик), у каждого своё решение.

    python day3_reasoning.py --probe          # только узнать, как у модели включается/выключается рассуждение
    python day3_reasoning.py                  # полный прогон: 10 трудных задач × 4 способа × 2 уровня
    python day3_reasoning.py --set easy       # первый, лёгкий набор задач
    python day3_reasoning.py --tasks dice,josephus --methods direct,steps --levels low
    python day3_reasoning.py --low reasoning_effort=low --high reasoning_effort=high
    python day3_reasoning.py --low thinking.type=disabled --high thinking.type=enabled

Во всех способах к задаче добавлена одна и та же строка с форматом ответа («ОТВЕТ: ...»),
чтобы ответ можно было проверить кодом. Сырые ответы и отчёт пишутся в results/.
"""
import argparse
import json
import re
import sys
import threading
import time
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from config import ConfigError, load_settings
from llm_client import LLMClient, LLMError

RESULTS_DIR = Path(__file__).with_name("results")
PROBE_FILE = RESULTS_DIR / "day3_probe.json"
DEFAULT_TIMEOUT = 300.0  # с высоким рассуждением ответ легко идёт дольше минуты

# ---------------------------------------------------------------- задачи

@dataclass(frozen=True)
class Task:
    id: str
    kind: str      # логика / алгоритм / анализ
    text: str
    answer: str    # эталон, как показывать в отчёте
    check: str     # number | fraction | word | knights
    hint: str      # что писать после «ОТВЕТ:»


# Эталоны проверены кодом (перебором / прямым расчётом), а не взяты на веру.
# easy — первый набор: deepseek-v4.1-flash решил 9–10 из 10 при любой настройке, способы почти не различались.
EASY_TASKS = [
    Task("sisters", "логика",
         "У Алисы 4 брата и 1 сестра. Сколько сестёр у брата Алисы?",
         "2", "number", "число"),
    Task("clock", "анализ",
         "Часы показывают 3:15. Какой угол в градусах между часовой и минутной стрелками?",
         "7.5", "number", "число градусов"),
    Task("letters", "анализ",
         "Сколько раз буква «о» встречается в слове «обороноспособность»?",
         "7", "number", "число"),
    Task("knights", "логика",
         "На острове живут рыцари, которые всегда говорят правду, и лжецы, которые всегда лгут. "
         "Встретились трое: A, B и C.\n"
         "A сказал: «Ровно один из нас троих — рыцарь».\n"
         "B сказал: «A — лжец».\n"
         "C сказал: «A и B одного типа».\n"
         "Кто из них рыцарь, а кто лжец?",
         "РЛЛ", "knights", "три буквы Р или Л для A, B, C по порядку, например РЛР"),
    Task("weekday", "анализ",
         "1 марта 2026 года — воскресенье. Какой день недели будет 1 марта 2028 года?",
         "среда", "word", "день недели"),
    Task("stairs", "алгоритм",
         "Лестница из 10 ступенек. За один шаг можно подняться на 1, 2 или 3 ступеньки. "
         "Сколькими способами можно подняться на вершину лестницы?",
         "274", "number", "число"),
    Task("divisible", "алгоритм",
         "Сколько натуральных чисел от 1 до 1000 включительно делятся на 3 или на 5, но не делятся на 15?",
         "401", "number", "число"),
    Task("anagrams", "алгоритм",
         "Сколько различных слов (любых последовательностей букв) можно составить, "
         "переставляя буквы слова «МАТЕМАТИКА»?",
         "151200", "number", "число"),
    Task("zeros", "алгоритм",
         "Сколькими нулями заканчивается десятичная запись числа 2026! (факториал 2026)?",
         "505", "number", "число"),
    Task("coins", "логика",
         "Среди 27 одинаковых на вид монет одна фальшивая — она легче остальных. "
         "Какое наименьшее число взвешиваний на чашечных весах без гирь гарантированно позволяет её найти?",
         "3", "number", "число"),
]

# hard — задачи, где ответ нельзя вспомнить: надо считать, перебирать или ловить ловушку в условии.
HARD_TASKS = [
    Task("right_angles", "анализ",
         "Сколько раз за сутки (24 часа, с 00:00 до 24:00) часовая и минутная стрелки образуют прямой угол?",
         "44", "number", "число"),
    Task("letters_phrase", "анализ",
         "Сколько раз буква «а» встречается во фразе «Карл у Клары украл кораллы, а Клара у Карла украла кларнет»?",
         "12", "number", "число"),
    Task("knights4", "логика",
         "На острове живут рыцари, которые всегда говорят правду, и лжецы, которые всегда лгут. "
         "Встретились четверо: A, B, C и D.\n"
         "A сказал: «B и C одного типа».\n"
         "B сказал: «Среди нас четверых ровно двое лжецов».\n"
         "C сказал: «D — лжец».\n"
         "D сказал: «Хотя бы один из A и B — лжец».\n"
         "Кто из них рыцарь, а кто лжец?",
         "ЛРЛР", "knights", "четыре буквы Р или Л для A, B, C, D по порядку, например РЛРЛ"),
    Task("leap_day", "анализ",
         "1 марта 2026 года — воскресенье. Какой день недели будет 29 февраля 2032 года?",
         "воскресенье", "word", "день недели"),
    Task("friday13", "анализ",
         "1 марта 2026 года — воскресенье. Сколько пятниц, приходящихся на 13-е число, будет в 2027 году?",
         "1", "number", "число"),
    Task("dice", "анализ",
         "Бросают две игральные кости. Известно, что хотя бы на одной выпала шестёрка. "
         "Какова вероятность, что шестёрка выпала на обеих?",
         "1/11", "fraction", "несократимая дробь вида a/b"),
    Task("last_digits", "алгоритм",
         "Какие две последние цифры у числа 7^2026?",
         "49", "number", "двузначное число"),
    Task("digit_sum", "алгоритм",
         "Сколько существует пятизначных чисел, сумма цифр которых равна 9?",
         "495", "number", "число"),
    Task("divisors", "алгоритм",
         "Сколько натуральных делителей у числа 2026²?",
         "9", "number", "число"),
    Task("josephus", "алгоритм",
         "30 человек стоят по кругу и пронумерованы по часовой стрелке от 1 до 30. Начиная с первого, "
         "считают по кругу «раз, два, три», и каждый третий выбывает (первым выбывает № 3, затем № 6 и так далее; "
         "круг смыкается). Какой номер у последнего оставшегося?",
         "29", "number", "число"),
]

TASK_SETS = {"hard": HARD_TASKS, "easy": EASY_TASKS}
ALL_TASKS = {t.id: t for tasks in TASK_SETS.values() for t in tasks}
TASKS = HARD_TASKS  # набор по умолчанию; переключается через --set

# ---------------------------------------------------------------- способы

def answer_format(task: Task) -> str:
    return f"В последней строке напиши только итоговый ответ в виде «ОТВЕТ: <{task.hint}>»."


STEPS_INSTRUCTION = "Решай пошагово."

META_PROMPT = """Ты — эксперт по промпт-инжинирингу. Составь промпт, с которым другая языковая модель решит задачу ниже максимально надёжно.
Промпт должен содержать полное условие задачи без изменений и инструкции, как её решать: роль, план решения, проверки, возможные ловушки.
Сам задачу не решай и ответ в промпт не вписывай. Верни только текст промпта, без пояснений и без ```.

Задача:
{task}"""

EXPERTS = [("аналитик", "АНАЛИТИКА"), ("инженер", "ИНЖЕНЕРА"), ("критик", "КРИТИКА")]

EXPERTS_PROMPT = """Реши задачу силами группы из трёх экспертов. Каждый работает в своей роли:
- Аналитик — разбирает условие, формализует его, отмечает ловушки, решает и даёт свой ответ.
- Инженер — решает независимо от аналитика: прямым расчётом, перебором или алгоритмом — и даёт свой ответ.
- Критик — проверяет оба решения, ищет ошибки и даёт свой ответ.
Затем сведите решения в один итоговый ответ.

Пиши строго в таком формате:
АНАЛИТИК: <решение>
ОТВЕТ АНАЛИТИКА: <ответ>
ИНЖЕНЕР: <решение>
ОТВЕТ ИНЖЕНЕРА: <ответ>
КРИТИК: <разбор>
ОТВЕТ КРИТИКА: <ответ>
ОТВЕТ: <итоговый ответ>

Задача:
{task}

Ответ ({hint}) — после «ОТВЕТ:»."""

METHODS = {
    "direct": "Прямой ответ",
    "steps": "«Решай пошагово»",
    "meta": "Сначала промпт, потом решение",
    "experts": "Группа экспертов",
}

LEVELS = {"low": "минимальное", "high": "максимальное"}

# ---------------------------------------------------------------- проверка ответа

ANSWER_RE = re.compile(r"ОТВЕТ\s*[:：]\s*(.+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?")
NUMBER_WORDS = {"ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
                "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10}
WEEKDAYS = ["понедельник", "вторник", "сред", "четверг", "пятниц", "суббот", "воскресень"]


def clean(text: str) -> str:
    """Снимает markdown и LaTeX-обвязку, которая мешает регуляркам."""
    text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)
    return re.sub(r"[*`_$\\]", "", text)


def extract_answer(text: str, label: str = "") -> str | None:
    """Последняя строка «ОТВЕТ: ...» (или «ОТВЕТ АНАЛИТИКА: ...», если задан label)."""
    pattern = re.compile(rf"ОТВЕТ\s+{label}\s*[:：]\s*(.+)", re.IGNORECASE) if label else ANSWER_RE
    found = pattern.findall(clean(text))
    return found[-1].strip() if found else None


def to_number(s: str) -> float | None:
    m = NUMBER_RE.search(s)
    if m:
        return float(re.sub(r"[ \u00a0\u202f]", "", m.group()).replace(",", "."))
    for word in re.findall(r"[а-яё]+", s.lower()):
        if word in NUMBER_WORDS:
            return float(NUMBER_WORDS[word])
    return None


def to_fraction(s: str) -> Fraction | None:
    """«1/11», «1 / 11», «0,0909» → Fraction; десятичную дробь приводим к ближайшей простой."""
    m = re.search(r"(\d+)\s*/\s*(\d+)", s)
    if m and int(m.group(2)):
        return Fraction(int(m.group(1)), int(m.group(2)))
    n = to_number(s)
    return Fraction(n).limit_denominator(100) if n is not None else None


def knights_code(s: str) -> str:
    low = s.lower().replace("ё", "е")
    words = re.findall(r"рыцар|лжец|лжиц", low)
    if len(words) >= 2:
        return "".join("Р" if w == "рыцар" else "Л" for w in words)
    return re.sub(r"[^РЛ]", "", s.upper())


def is_correct(task: Task, answer: str | None) -> bool:
    if not answer:
        return False
    if task.check == "number":
        n = to_number(answer)
        return n is not None and abs(n - float(task.answer)) < 1e-6
    if task.check == "word":
        low = answer.lower().replace("ё", "е")
        named = [d for d in WEEKDAYS if d in low]
        return named == [task.answer[:4]] or (len(named) == 1 and task.answer.startswith(named[0]))
    if task.check == "fraction":
        return to_fraction(answer) == to_fraction(task.answer)
    if task.check == "knights":
        return knights_code(answer) == task.answer
    raise ValueError(task.check)

# ---------------------------------------------------------------- вызовы модели

@dataclass
class Call:
    step: str                 # solve | make_prompt
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency: float = 0.0
    finish_reason: str | None = None
    text: str = ""
    reasoning: str = ""
    error: str | None = None
    attempts: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


RETRYABLE = ("HTTP 429", "HTTP 5", "таймаут", "нет связи", "неожиданный ответ")


def call_llm(client: LLMClient, prompt: str, params: dict, step: str, retries: int = 2) -> Call:
    call = Call(step)
    for attempt in range(1, retries + 2):
        call.attempts = attempt
        try:
            reply = client.chat([{"role": "user", "content": prompt}], **params)
        except LLMError as e:
            call.error = str(e)
            if attempt <= retries and any(marker in call.error for marker in RETRYABLE):
                time.sleep(5 * attempt)
                continue
            return call
        usage = reply.usage
        details = usage.get("completion_tokens_details") or {}
        call.error = None
        call.prompt_tokens = usage.get("prompt_tokens") or 0
        call.completion_tokens = usage.get("completion_tokens") or 0
        call.reasoning_tokens = (details.get("reasoning_tokens") if isinstance(details, dict) else 0) or 0
        call.latency = reply.latency
        call.finish_reason = reply.finish_reason
        call.text = reply.text
        call.reasoning = reply.reasoning
        return call
    return call


@dataclass
class Solution:
    task: str
    method: str
    level: str
    run: int
    answer: str | None = None
    correct: bool = False
    calls: list[Call] = field(default_factory=list)
    generated_prompt: str | None = None            # для meta
    experts: dict = field(default_factory=dict)    # для experts: роль -> {"answer", "correct"}
    problem: str | None = None

    @property
    def tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def reasoning_tokens(self) -> int:
        return sum(c.reasoning_tokens for c in self.calls)

    @property
    def latency(self) -> float:
        return sum(c.latency for c in self.calls)

    @property
    def error(self) -> str | None:
        return next((c.error for c in self.calls if c.error), None)


def strip_fences(text: str) -> str:
    m = re.match(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def solve(client: LLMClient, task: Task, method: str, level: str, params: dict, run: int) -> Solution:
    sol = Solution(task.id, method, level, run)
    fmt = answer_format(task)

    if method == "direct":
        sol.calls.append(call_llm(client, f"{task.text}\n\n{fmt}", params, "solve"))
    elif method == "steps":
        sol.calls.append(call_llm(client, f"{task.text}\n\n{STEPS_INSTRUCTION}\n{fmt}", params, "solve"))
    elif method == "meta":
        first = call_llm(client, META_PROMPT.format(task=task.text), params, "make_prompt")
        sol.calls.append(first)
        if first.error or not first.text.strip():
            sol.problem = first.error or "модель не вернула промпт"
            return sol
        sol.generated_prompt = strip_fences(first.text)
        sol.calls.append(call_llm(client, f"{sol.generated_prompt}\n\n{fmt}", params, "solve"))
    elif method == "experts":
        sol.calls.append(call_llm(client, EXPERTS_PROMPT.format(task=task.text, hint=task.hint), params, "solve"))
    else:
        raise ValueError(method)

    last = sol.calls[-1]
    if last.error:
        sol.problem = last.error
        return sol
    sol.answer = extract_answer(last.text)
    sol.correct = is_correct(task, sol.answer)
    if method == "experts":
        for role, label in EXPERTS:
            a = extract_answer(last.text, label)
            sol.experts[role] = {"answer": a, "correct": is_correct(task, a)}
    if sol.answer is None:
        sol.problem = ("пустой ответ, finish_reason=length" if last.finish_reason == "length" and not last.text.strip()
                       else "нет строки «ОТВЕТ:»")
    elif not sol.correct:
        sol.problem = f"неверно: {sol.answer[:60]!r}, эталон {task.answer}"
    return sol

# ---------------------------------------------------------------- как у модели управлять рассуждением

PROBE_CANDIDATES = [
    ("по умолчанию", {}),
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    ("reasoning_effort=minimal", {"reasoning_effort": "minimal"}),
    ("reasoning_effort=low", {"reasoning_effort": "low"}),
    ("reasoning_effort=high", {"reasoning_effort": "high"}),
    ("reasoning_effort=max", {"reasoning_effort": "max"}),
    ("reasoning.effort=low", {"reasoning": {"effort": "low"}}),
    ("reasoning.effort=high", {"reasoning": {"effort": "high"}}),
    ("thinking=disabled", {"thinking": {"type": "disabled"}}),
    ("thinking=enabled", {"thinking": {"type": "enabled"}}),
]
PROBE_TASK = "stairs"


def reasoning_size(call: Call) -> int:
    """Токены рассуждения; если провайдер их не считает — грубая оценка по длине текста рассуждения."""
    return call.reasoning_tokens or len(call.reasoning) // 3


def probe(client: LLMClient, workers: int, runs: int) -> list[dict]:
    task = ALL_TASKS[PROBE_TASK]  # задача из easy-набора: так проба сравнима с прошлыми
    prompt = f"{task.text}\n\n{answer_format(task)}"
    jobs = [(name, params) for name, params in PROBE_CANDIDATES for _ in range(runs)]
    calls: dict[str, list[Call]] = {name: [] for name, _ in PROBE_CANDIDATES}
    print(f"Проба: задача «{task.id}», {len(PROBE_CANDIDATES)} вариантов параметров × {runs}")
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(call_llm, client, prompt, params, "probe", 1): name for name, params in jobs}
        for f in as_completed(futures):
            calls[futures[f]].append(f.result())

    rows = []
    for name, params in PROBE_CANDIDATES:
        ok = [c for c in calls[name] if not c.error]
        row = {"name": name, "params": params, "accepted": bool(ok),
               "error": None if ok else calls[name][0].error,
               "reasoning": round(sum(map(reasoning_size, ok)) / len(ok)) if ok else None,
               "completion": round(sum(c.completion_tokens for c in ok) / len(ok)) if ok else None,
               "latency": round(sum(c.latency for c in ok) / len(ok), 1) if ok else None,
               "correct": sum(is_correct(task, extract_answer(c.text)) for c in ok)}
        rows.append(row)
    print(f"\n{'параметры':<26} {'рассужд.':>9} {'ответ':>7} {'время':>7}  верно")
    for r in rows:
        if r["accepted"]:
            print(f"{r['name']:<26} {r['reasoning']:>9} {r['completion']:>7} {r['latency']:>6}с  {r['correct']}/{runs}")
        else:
            print(f"{r['name']:<26} отклонено: {(r['error'] or '')[:90]}")
    return rows


def pick_levels(rows: list[dict]) -> tuple[dict, dict, str]:
    """low — вариант с самым коротким рассуждением, high — с самым длинным (среди принятых API)."""
    accepted = [r for r in rows if r["accepted"]]
    if not accepted:
        raise RuntimeError("ни один вариант параметров не принят API — проверьте ключ и модель.")
    low = min(accepted, key=lambda r: (r["reasoning"], r["name"] == "по умолчанию"))
    high = max(accepted, key=lambda r: (r["reasoning"], r["name"] != "по умолчанию"))
    note = f"low = {low['name']} (~{low['reasoning']} ток. рассуждения), high = {high['name']} (~{high['reasoning']})"
    if high["reasoning"] <= low["reasoning"] * 1.5 + 50:
        note += " — ВНИМАНИЕ: разница небольшая, похоже, провайдер игнорирует параметры рассуждения"
    return low["params"], high["params"], note

# ---------------------------------------------------------------- сводка и отчёт

def summarize(solutions: list[Solution], tasks: list[Task], methods: list[str], levels: list[str]) -> list[dict]:
    rows = []
    for level in levels:
        for method in methods:
            group = [s for s in solutions if s.method == method and s.level == level]
            if not group:
                continue
            n = len(group)
            row = {
                "method": method, "level": level, "title": METHODS[method], "level_title": LEVELS[level],
                "n": n, "correct": sum(s.correct for s in group),
                "errors": sum(1 for s in group if s.error),
                "tokens": sum(s.tokens for s in group),
                "reasoning_tokens": sum(s.reasoning_tokens for s in group),
                "completion_tokens": sum(c.completion_tokens for s in group for c in s.calls),
                "time": round(sum(s.latency for s in group), 1),
                "calls": sum(len(s.calls) for s in group),
            }
            row["tokens_per_task"] = round(row["tokens"] / n)
            row["time_per_task"] = round(row["time"] / n, 1)
            if method == "experts":
                row["experts"] = {role: sum(s.experts.get(role, {}).get("correct", False) for s in group)
                                  for role, _ in EXPERTS}
            rows.append(row)
    return rows


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 96)
    print(f"{'Способ':<32} {'рассужд.':<13} {'верно':>7} {'токенов':>9} {'из них рассужд.':>16} "
          f"{'на задачу':>10} {'время, с':>9} {'с/задачу':>9}")
    for r in rows:
        print(f"{r['title']:<32} {r['level_title']:<13} {r['correct']:>4}/{r['n']:<2} {r['tokens']:>9} "
              f"{r['reasoning_tokens']:>16} {r['tokens_per_task']:>10} {r['time']:>9} {r['time_per_task']:>9}")
        if r["errors"]:
            print(f"    ошибок API: {r['errors']}")


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def observations(rows: list[dict], levels: list[str]) -> list[str]:
    """Факты для выводов: прирост каждого приёма относительно прямого ответа и цена этого прироста."""
    out = []
    by = {(r["method"], r["level"]): r for r in rows}
    for level in levels:
        base = by.get(("direct", level))
        if not base:
            continue
        out.append(f"**Рассуждение {LEVELS[level]}** (база — прямой ответ: {base['correct']}/{base['n']}, "
                   f"{fmt_int(base['tokens'])} ток., {base['time']} с):")
        for method in METHODS:
            r = by.get((method, level))
            if not r or method == "direct":
                continue
            d = r["correct"] - base["correct"]
            k_tok = r["tokens"] / base["tokens"] if base["tokens"] else 0
            k_time = r["time"] / base["time"] if base["time"] else 0
            out.append(f"- {r['title']}: {d:+d} верных, токенов ×{k_tok:.1f}, времени ×{k_time:.1f}")
        out.append("")
    if set(levels) == {"low", "high"}:
        out.append("**Что даёт включённое рассуждение** (high против low, тот же способ):")
        for method in METHODS:
            lo, hi = by.get((method, "low")), by.get((method, "high"))
            if lo and hi:
                k = hi["tokens"] / lo["tokens"] if lo["tokens"] else 0
                out.append(f"- {lo['title']}: {lo['correct']} → {hi['correct']} верных, токенов ×{k:.1f}")
        out.append("")
    return out


def write_report(path: Path, model: str, level_params: dict, probe_note: str, rows: list[dict],
                 solutions: list[Solution], tasks: list[Task], methods: list[str], levels: list[str],
                 runs: int, task_set: str) -> None:
    lines = [
        "# День 3 · Разные способы рассуждения — результаты прогона", "",
        f"Модель: `{model}` · дата: {datetime.now():%Y-%m-%d %H:%M} · набор: {task_set} · задач: {len(tasks)} · прогонов: {runs}", "",
        "Уровни рассуждения (параметры запроса):", "",
    ]
    for level in levels:
        lines.append(f"- {LEVELS[level]}: `{json.dumps(level_params[level], ensure_ascii=False)}`")
    if probe_note:
        lines += ["", f"Выбраны пробой: {probe_note}"]
    lines += [
        "", "## Способ × рассуждение → верных ответов / токены / время", "",
        "| Способ | Рассуждение | Верных | Токенов всего | из них рассуждение | Токенов на задачу | Время, с | с на задачу |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        err = f" (ошибок API: {r['errors']})" if r["errors"] else ""
        lines.append(f"| {r['title']} | {r['level_title']} | **{r['correct']}/{r['n']}**{err} | {fmt_int(r['tokens'])} | "
                     f"{fmt_int(r['reasoning_tokens'])} | {fmt_int(r['tokens_per_task'])} | {r['time']} | {r['time_per_task']} |")
    lines += ["", "Токены — prompt + completion по всем запросам способа (у «сначала промпт» их два на задачу). "
              "Время — сумма времени ответа модели, без учёта параллельности.", ""]

    lines += ["## Наблюдения (посчитано автоматически)", ""] + observations(rows, levels)

    experts_rows = [r for r in rows if r.get("experts")]
    if experts_rows:
        lines += ["## Группа экспертов: кто прав", "",
                  "| Рассуждение | " + " | ".join(role.capitalize() for role, _ in EXPERTS) + " | Итог |",
                  "|---|" + "---|" * (len(EXPERTS) + 1)]
        for r in experts_rows:
            lines.append(f"| {r['level_title']} | " + " | ".join(f"{r['experts'][role]}/{r['n']}" for role, _ in EXPERTS)
                         + f" | {r['correct']}/{r['n']} |")
        lines.append("")

    cols = [(m, l) for l in levels for m in methods]
    lines += ["## По задачам", "",
              "| Задача | Эталон | " + " | ".join(f"{m} · {l}" for m, l in cols) + " |",
              "|---|---|" + "---|" * len(cols)]
    for t in tasks:
        cells = []
        for m, l in cols:
            group = [s for s in solutions if s.task == t.id and s.method == m and s.level == l]
            ok = sum(s.correct for s in group)
            cells.append(("✅" if ok == len(group) else "❌" if ok == 0 else f"{ok}/{len(group)}") if group else "—")
        lines.append(f"| {t.id} ({t.kind}) | {t.answer} | " + " | ".join(cells) + " |")

    wrong = [s for s in solutions if s.problem]
    lines += ["", "## Ошибки", ""]
    lines += [f"- `{s.task}` · {METHODS[s.method]} · {LEVELS[s.level]} · прогон {s.run}: {s.problem}" for s in wrong] \
        or ["Все задачи решены верно."]

    metas = [s for s in solutions if s.generated_prompt]
    if metas:
        s = metas[0]
        lines += ["", f"## Пример промпта, который модель составила сама (`{s.task}`, {LEVELS[s.level]})", "",
                  "~~~text", s.generated_prompt[:2000], "~~~"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- запуск

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="День 3: 4 способа рассуждения × 2 уровня reasoning.")
    p.add_argument("--probe", action="store_true", help="только проба параметров рассуждения, без прогона")
    p.add_argument("--reprobe", action="store_true", help="повторить пробу, даже если results/day3_probe.json уже есть")
    p.add_argument("--probe-runs", type=int, default=2, help="запросов на каждый вариант параметров в пробе")
    p.add_argument("--low", help="параметры минимального рассуждения: reasoning_effort=low или JSON")
    p.add_argument("--high", help="параметры максимального рассуждения: reasoning_effort=high или JSON")
    p.add_argument("--set", default="hard", choices=list(TASK_SETS),
                   help="набор задач: hard (по умолчанию) или easy — первый, слишком лёгкий набор")
    p.add_argument("--tasks", help="id задач через запятую (из выбранного набора)")
    p.add_argument("--methods", help="через запятую: " + ",".join(METHODS))
    p.add_argument("--levels", default="low,high", help="low,high (по умолчанию оба)")
    p.add_argument("--runs", type=int, default=1, help="прогонов каждой пары задача × способ (по умолчанию 1)")
    p.add_argument("--workers", type=int, default=4, help="параллельных запросов (по умолчанию 4)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"таймаут запроса, с (по умолчанию {DEFAULT_TIMEOUT:g})")
    p.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    return p.parse_args()


def pick(raw: str | None, allowed: list[str], what: str) -> list[str]:
    if not raw:
        return allowed
    wanted = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = set(wanted) - set(allowed)
    if unknown:
        raise ValueError(f"неизвестные {what}: {sorted(unknown)}; допустимо: {', '.join(allowed)}")
    return [x for x in allowed if x in wanted]


def parse_params(raw: str) -> dict:
    """JSON ('{"reasoning_effort": "low"}') или короткая форма: reasoning_effort=low,thinking.type=disabled.
    Короткая форма нужна для PowerShell 5, который съедает кавычки внутри JSON-аргумента."""
    raw = raw.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    params: dict = {}
    for pair in filter(None, (x.strip() for x in raw.split(","))):
        if "=" not in pair:
            raise ValueError(f"ожидалось ключ=значение, получено {pair!r}")
        key, value = (x.strip() for x in pair.split("=", 1))
        *path, last = key.split(".")
        node = params
        for part in path:
            node = node.setdefault(part, {})
        node[last] = int(value) if value.isdigit() else value
    return params


def resolve_levels(args, client: LLMClient, model: str) -> tuple[dict, str]:
    if args.low and args.high and not args.probe:
        return {"low": parse_params(args.low), "high": parse_params(args.high)}, "заданы вручную (--low/--high)"
    cached = None
    if PROBE_FILE.exists() and not args.reprobe and not args.probe:
        cached = json.loads(PROBE_FILE.read_text(encoding="utf-8"))
        if cached.get("model") != model:
            cached = None
    if cached:
        rows = cached["rows"]
        print(f"Параметры рассуждения — из прошлой пробы ({PROBE_FILE.name}; --reprobe, чтобы повторить)")
    else:
        rows = probe(client, args.workers, args.probe_runs)
        RESULTS_DIR.mkdir(exist_ok=True)
        PROBE_FILE.write_text(json.dumps({"model": model, "date": datetime.now().isoformat(timespec="seconds"),
                                          "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    low, high, note = pick_levels(rows)
    if args.low:
        low = parse_params(args.low)
    if args.high:
        high = parse_params(args.high)
    return {"low": low, "high": high}, note


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings()
        pool = TASK_SETS[args.set]
        tasks = [t for t in pool if t.id in pick(args.tasks, [t.id for t in pool], "задачи")]
        methods = pick(args.methods, list(METHODS), "способы")
        levels = pick(args.levels, list(LEVELS), "уровни")
    except (ConfigError, ValueError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    model = args.model or settings.model
    client = LLMClient(settings.base_url, settings.api_key, model, args.timeout)
    print(f"Модель: {model} · набор задач: {args.set}")

    try:
        level_params, probe_note = resolve_levels(args, client, model)
    except (RuntimeError, ValueError) as e:  # JSONDecodeError — подкласс ValueError
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print(f"Уровни: {probe_note}")
    for level in levels:
        print(f"  {level}: {json.dumps(level_params[level], ensure_ascii=False)}")
    if args.probe:
        return 0

    jobs = [(t, m, l, run) for l in levels for m in methods for t in tasks for run in range(1, args.runs + 1)]
    print(f"\nЗапуск: {len(tasks)} задач × {len(methods)} способа × {len(levels)} уровня × {args.runs} = "
          f"{len(jobs)} решений, параллельно {args.workers}\n")
    solutions: list[Solution] = []
    lock = threading.Lock()
    started = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(solve, client, t, m, l, level_params[l], run) for t, m, l, run in jobs]
        for f in as_completed(futures):
            s = f.result()
            with lock:
                solutions.append(s)
                mark = "✅" if s.correct else "❌"
                info = s.error or f"{s.tokens} ток. (рассужд. {s.reasoning_tokens}), {s.latency:.1f} с"
                print(f"[{len(solutions):>3}/{len(jobs)}] {mark} {s.level:<4} {s.method:<8} {s.task:<10} "
                      f"ответ={str(s.answer)[:24]!r:<26} | {info}")
    print(f"\nВсего на прогон: {time.perf_counter() - started:.0f} с")

    order = {(t.id, m, l, r): i for i, (t, m, l, r) in enumerate(jobs)}
    solutions.sort(key=lambda s: order[(s.task, s.method, s.level, s.run)])
    rows = summarize(solutions, tasks, methods, levels)
    print_summary(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = RESULTS_DIR / f"day3_{args.set}_{stamp}.json"
    report_path = RESULTS_DIR / f"day3_{args.set}_{stamp}.md"
    raw_path.write_text(json.dumps({
        "model": model, "task_set": args.set, "levels": level_params, "probe_note": probe_note, "summary": rows,
        "tasks": [asdict(t) for t in tasks],
        "solutions": [{**asdict(s), "tokens": s.tokens, "latency": s.latency} for s in solutions],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, model, level_params, probe_note, rows, solutions, tasks, methods, levels, args.runs, args.set)
    print(f"\nСырые ответы: {raw_path}\nОтчёт: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
