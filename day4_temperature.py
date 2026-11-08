"""День 4. Температура.

Один и тот же запрос отправляется с temperature = 0 / 0.7 / 1.2, каждая комбинация — 5 раз.
Два типа запросов:

    exact — задачи с одним правильным ответом (подсчёт букв во фразе, число дней между датами);
    gen   — задачи на генерацию (5 названий для кофейни, слоган).

Для каждого ответа код проверяет формат и (для exact) верность, а для каждой комбинации
считает, насколько ответы отличаются между прогонами.

    python day4_temperature.py                          # 4 задачи × 3 температуры × 5 прогонов = 60 запросов
    python day4_temperature.py --runs 10                # больше прогонов
    python day4_temperature.py --temps 0,0.7,1.2,1.8    # другие температуры
    python day4_temperature.py --tasks letters,names    # только часть задач
    python day4_temperature.py --reasoning model        # не выключать рассуждение (как у модели по умолчанию)
    python day4_temperature.py --recheck results/day4_<время>.json   # перепроверить сохранённые ответы

По умолчанию рассуждение выключено (reasoning_effort=none): у reasoning-моделей температура
может не применяться к скрытому рассуждению, и сравнение было бы нечистым.
Сырые ответы (.json) и отчёт с примерами (.md) пишутся в results/.
"""
import argparse
import json
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

from config import ConfigError, load_settings
from llm_client import LLMClient, LLMError

RESULTS_DIR = Path(__file__).with_name("results")
DEFAULT_TEMPS = "0,0.7,1.2"
DEFAULT_TIMEOUT = 120.0
REASONING_OFF = {"reasoning_effort": "none"}

# ---------------------------------------------------------------- задачи

@dataclass(frozen=True)
class Task:
    id: str
    kind: str       # exact | gen
    title: str
    prompt: str
    answer: int | None = None   # эталон для exact (проверен кодом)


EXACT_FORMAT = ('\n\nОтветь строго JSON-объектом вида {"answer": <целое число>} — '
                "без пояснений, без markdown, без текста до и после.")

TASKS = [
    Task("letters", "exact", "Буквы во фразе",
         "Сколько раз буква «а» (строчная или заглавная) встречается во фразе "
         "«Карл у Клары украл кораллы, а Клара у Карла украла кларнет»?" + EXACT_FORMAT,
         answer="Карл у Клары украл кораллы, а Клара у Карла украла кларнет".lower().count("а")),  # 12
    Task("days", "exact", "Дни между датами",
         "Сколько дней проходит от 1 января 2026 года до 1 марта 2028 года "
         "(первый день не считаем, последний считаем)?" + EXACT_FORMAT,
         answer=(datetime(2028, 3, 1) - datetime(2026, 1, 1)).days),  # 790
    Task("names", "gen", "5 названий кофейни",
         "Придумай 5 вариантов названия для маленькой кофейни у моря в старом портовом городе. "
         "Ответ — только JSON-массив из 5 строк на русском языке, каждое название не длиннее 4 слов, "
         "без пояснений и без markdown."),
    Task("slogan", "gen", "Слоган кофейни",
         "Напиши рекламный слоган для маленькой кофейни у моря в старом портовом городе. "
         "Ответ — одна строка на русском языке, не длиннее 8 слов, без кавычек и без пояснений."),
]
TASKS_BY_ID = {t.id: t for t in TASKS}

# ---------------------------------------------------------------- проверки

def strip_fences(text: str) -> str:
    m = re.match(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def is_cyrillic(c: str) -> bool:
    return "а" <= c.lower() <= "я" or c.lower() == "ё"


def non_russian(text: str) -> bool:
    """True, если в тексте есть буквы не кириллицей (латиница, иероглифы…) или букв нет вовсе.
    Проверяется каждый элемент ответа отдельно: доля кириллицы по всему ответу пропускала
    «Порт and Кофе» среди четырёх русских названий."""
    letters = [c for c in text if c.isalpha()]
    return not letters or not all(is_cyrillic(c) for c in letters)


def words(text: str) -> list[str]:
    return re.findall(r"[\w\-]+", text)


def check_exact(task: Task, text: str) -> tuple[bool, list[str], int | None]:
    """→ (формат ок, проблемы формата, извлечённый ответ). Ответ извлекается мягко,
    даже если формат сломан, — чтобы отдельно видеть «сломан формат» и «неверный ответ»."""
    problems = []
    raw = text.strip()
    try:
        data = json.loads(raw)
        if not (isinstance(data, dict) and set(data) == {"answer"} and type(data["answer"]) is int):
            problems.append("JSON не вида {\"answer\": <целое>}")
    except ValueError:
        problems.append("не чистый JSON" + (" (обёрнут в ```)" if raw.startswith("```") else ""))

    value = None
    m = re.search(r"\{.*?\}", strip_fences(raw), re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0)).get("answer")
            value = int(v) if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().isdigit()) else None
        except (ValueError, AttributeError, TypeError):
            pass
    if value is None:
        nums = re.findall(r"-?\d+", raw)
        value = int(nums[-1]) if nums else None
    return not problems, problems, value


def check_names(text: str) -> tuple[bool, list[str], list[str]]:
    problems = []
    raw = text.strip()
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
        problems.append("не чистый JSON" + (" (обёрнут в ```)" if raw.startswith("```") else ""))
        try:
            data = json.loads(strip_fences(raw))
        except ValueError:
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            try:
                data = json.loads(m.group(0)) if m else None
            except ValueError:
                pass
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        problems.append("не массив строк")
        return False, problems, []
    names = [x.strip() for x in data if x.strip()]
    if len(data) != 5:
        problems.append(f"названий {len(data)}, а не 5")
    if len({n.lower() for n in names}) != len(names):
        problems.append("повторы внутри ответа")
    long = [n for n in names if len(words(n)) > 4]
    if long:
        problems.append(f"длиннее 4 слов: {len(long)}")
    foreign = [n for n in names if non_russian(n)]
    if foreign:
        problems.append("не на русском: " + ", ".join(f"«{n}»" for n in foreign))
    return not problems, problems, names


def check_slogan(text: str) -> tuple[bool, list[str]]:
    problems = []
    raw = text.strip()
    if not raw:
        return False, ["пустой ответ"]
    if "\n" in raw:
        problems.append("больше одной строки")
    n = len(words(raw))
    if n > 8:
        problems.append(f"{n} слов (> 8)")
    if raw[0] in "«\"'„“" or raw[-1] in "»\"'“”":
        problems.append("в кавычках")
    if non_russian(raw):
        problems.append("не на русском")
    return not problems, problems


# ---------------------------------------------------------------- запросы

@dataclass
class Answer:
    task: str
    temperature: float
    run: int
    text: str = ""
    format_ok: bool = False
    problems: list[str] = field(default_factory=list)
    value: int | None = None          # exact: извлечённое число
    correct: bool | None = None       # exact: совпало с эталоном; gen: None
    names: list[str] = field(default_factory=list)  # names: разобранные названия
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency: float = 0.0
    finish_reason: str | None = None
    error: str | None = None
    attempts: int = 1


RETRYABLE = ("HTTP 429", "HTTP 5", "таймаут", "нет связи", "неожиданный ответ")


def ask(client: LLMClient, task: Task, temperature: float, run: int, extra: dict, retries: int = 2) -> Answer:
    a = Answer(task.id, temperature, run)
    for attempt in range(1, retries + 2):
        a.attempts = attempt
        try:
            reply = client.chat([{"role": "user", "content": task.prompt}], temperature=temperature, **extra)
        except LLMError as e:
            a.error = str(e)
            if attempt <= retries and any(m in a.error for m in RETRYABLE):
                time.sleep(5 * attempt)
                continue
            return a
        a.error = None
        usage = reply.usage
        details = usage.get("completion_tokens_details") or {}
        a.prompt_tokens = usage.get("prompt_tokens") or 0
        a.completion_tokens = usage.get("completion_tokens") or 0
        a.reasoning_tokens = (details.get("reasoning_tokens") if isinstance(details, dict) else 0) or 0
        a.latency = reply.latency
        a.finish_reason = reply.finish_reason
        a.text = reply.text
        break
    else:
        return a
    return evaluate(task, a)


def evaluate(task: Task, a: Answer) -> Answer:
    """Проверяет формат и верность ответа; используется и в прогоне, и в --recheck."""
    a.problems, a.names, a.value, a.correct = [], [], None, None
    if task.kind == "exact":
        a.format_ok, a.problems, a.value = check_exact(task, a.text)
        a.correct = a.value == task.answer
    elif task.id == "names":
        a.format_ok, a.problems, a.names = check_names(a.text)
    else:
        a.format_ok, a.problems = check_slogan(a.text)
    if a.finish_reason == "length":
        a.format_ok = False
        a.problems.append("обрезан по лимиту")
    return a


# ---------------------------------------------------------------- метрики

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def similarity(texts: list[str]) -> float:
    """Средняя попарная похожесть текстов (difflib, 0..1). 1 — все ответы одинаковые."""
    pairs = list(combinations(texts, 2))
    if not pairs:
        return 1.0
    return sum(SequenceMatcher(None, a, b).ratio() for a, b in pairs) / len(pairs)


def summarize(answers: list[Answer], tasks: list[Task], temps: list[float]) -> list[dict]:
    rows = []
    for t in tasks:
        for temp in temps:
            group = [a for a in answers if a.task == t.id and a.temperature == temp]
            ok = [a for a in group if not a.error]
            texts = [norm(a.text) for a in ok]
            row = {
                "task": t.id, "kind": t.kind, "title": t.title, "temperature": temp,
                "n": len(group), "errors": len(group) - len(ok),
                "distinct": len(set(texts)),
                "all_identical": len(set(texts)) == 1 and len(ok) == len(group) and len(ok) > 0,
                "similarity": round(similarity(texts), 3),
                "format_broken": sum(not a.format_ok for a in ok),
                "wrong": sum(a.correct is False for a in ok) if t.kind == "exact" else None,
                "values": [a.value for a in ok] if t.kind == "exact" else None,
                "completion_tokens": round(statistics.mean([a.completion_tokens for a in ok]), 1) if ok else 0,
                "latency_median": round(statistics.median([a.latency for a in ok]), 2) if ok else 0,
            }
            if t.id == "names":
                all_names = [n.lower() for a in ok for n in a.names]
                per_run = [{n.lower() for n in a.names} for a in ok if a.names]
                row["names_total"] = len(all_names)
                row["names_unique"] = len(set(all_names))
                row["names_in_every_run"] = len(set.intersection(*per_run)) if per_run else 0
            rows.append(row)
    return rows


# ---------------------------------------------------------------- вывод

def temp_str(t: float) -> str:
    return f"{t:g}"


def short(text: str, n: int = 70) -> str:
    s = re.sub(r"\s+", " ", text.strip())
    return s if len(s) <= n else s[: n - 1] + "…"


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{'Задача':<22} {'temp':>5} {'разных':>7} {'похожесть':>10} {'формат сломан':>14} "
          f"{'неверно':>8} {'уник. названий':>15} {'ошибок API':>11}")
    for r in rows:
        wrong = "—" if r["wrong"] is None else f"{r['wrong']}/{r['n']}"
        uniq = f"{r['names_unique']}/{r['names_total']}" if "names_unique" in r else "—"
        print(f"{r['title']:<22} {temp_str(r['temperature']):>5} {r['distinct']:>4}/{r['n']:<2} "
              f"{r['similarity'] * 100:>9.0f}% {r['format_broken']:>11}/{r['n']:<2} {wrong:>8} {uniq:>15} "
              f"{r['errors']:>11}")


def observations(rows: list[dict], temps: list[float]) -> list[str]:
    notes = []
    zero = [r for r in rows if r["temperature"] == 0]
    if zero:
        same = [r["title"] for r in zero if r["all_identical"]]
        diff = [f"{r['title']} ({r['distinct']} разных из {r['n']})" for r in zero if not r["all_identical"]]
        if same and not diff:
            notes.append("temperature = 0: во всех задачах все прогоны дали посимвольно одинаковый ответ.")
        else:
            notes.append("temperature = 0: одинаковый ответ — " + (", ".join(same) or "ни в одной задаче")
                         + "; разные — " + (", ".join(diff) or "нет") + ".")
    for kind, label in (("exact", "Задачи с одним ответом"), ("gen", "Генерация")):
        parts = []
        for temp in temps:
            rs = [r for r in rows if r["kind"] == kind and r["temperature"] == temp]
            if not rs:
                continue
            n = sum(r["n"] for r in rs)
            broken = sum(r["format_broken"] for r in rs)
            p = f"t={temp_str(temp)}: формат сломан {broken}/{n}"
            if kind == "exact":
                p += f", неверно {sum(r['wrong'] for r in rs)}/{n}"
            p += f", похожесть {statistics.mean(r['similarity'] for r in rs) * 100:.0f}%"
            parts.append(p)
        if parts:
            notes.append(f"{label}: " + "; ".join(parts) + ".")
    names = [r for r in rows if "names_unique" in r]
    if names:
        notes.append("Названия: уникальных за все прогоны — " + ", ".join(
            f"t={temp_str(r['temperature'])}: {r['names_unique']} из {r['names_total']} "
            f"(во всех прогонах повторились {r['names_in_every_run']})" for r in names) + ".")
    return notes


def md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ⏎ ")


def write_report(path: Path, model: str, extra: dict, rows: list[dict], answers: list[Answer],
                 tasks: list[Task], temps: list[float], runs: int, when: datetime | None = None) -> None:
    L = [f"# День 4 · Температура — прогон {when or datetime.now():%Y-%m-%d %H:%M}", "",
         f"Модель: `{model}` · параметры кроме temperature: `{json.dumps(extra, ensure_ascii=False) or '{}'}` · "
         f"{len(tasks)} задачи × {len(temps)} температуры × {runs} прогонов = {len(answers)} запросов", "",
         "## Задачи", ""]
    for t in tasks:
        ref = f" Эталон: **{t.answer}**." if t.answer is not None else ""
        L.append(f"- `{t.id}` ({'один ответ' if t.kind == 'exact' else 'генерация'}) — {t.title}.{ref}")
        L.append(f"  > {md_cell(t.prompt)}")
    L += ["", "## Сводка", "",
          "«Разных» — сколько различных ответов (после нормализации пробелов и регистра) из всех прогонов; "
          "«похожесть» — средняя попарная похожесть текстов (difflib), 100% — все ответы одинаковые.", "",
          "| Задача | temperature | Разных | Похожесть | Формат сломан | Неверно | Ответы / уникальных названий | "
          "Ток. ответа (ср.) | Медиана времени, с |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["kind"] == "exact":
            extra_col = ", ".join("—" if v is None else str(v) for v in r["values"])
        else:
            extra_col = f"{r['names_unique']}/{r['names_total']}" if "names_unique" in r else "—"
        wrong = "—" if r["wrong"] is None else f"{r['wrong']}/{r['n']}"
        err = f" (ошибок API: {r['errors']})" if r["errors"] else ""
        L.append(f"| {r['title']} | {temp_str(r['temperature'])} | {r['distinct']}/{r['n']}{err} | "
                 f"{r['similarity'] * 100:.0f}% | {r['format_broken']}/{r['n']} | {wrong} | {extra_col} | "
                 f"{r['completion_tokens']:g} | {r['latency_median']:g} |")
    L += ["", "## Наблюдения (автоматически)", ""] + [f"- {n}" for n in observations(rows, temps)]
    L += ["", "## Примеры ответов", ""]
    for t in tasks:
        L += [f"### {t.title} (`{t.id}`)", ""]
        for temp in temps:
            L += [f"**temperature = {temp_str(temp)}**", "", "| # | Ответ | Проверка |", "|---|---|---|"]
            for a in sorted((a for a in answers if a.task == t.id and a.temperature == temp), key=lambda a: a.run):
                if a.error:
                    check = f"ошибка API: {md_cell(a.error[:120])}"
                else:
                    marks = []
                    if t.kind == "exact":
                        marks.append("✅ верно" if a.correct else f"❌ неверно ({a.value})")
                    marks.append("формат ок" if a.format_ok else "⚠️ " + "; ".join(a.problems))
                    check = " · ".join(marks)
                L.append(f"| {a.run} | {md_cell(a.text.strip()) or '(пусто)'} | {check} |")
            L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="День 4: один запрос при разной температуре.")
    p.add_argument("--temps", default=DEFAULT_TEMPS, help=f"температуры через запятую (по умолчанию {DEFAULT_TEMPS})")
    p.add_argument("--runs", type=int, default=5, help="прогонов каждой комбинации (по умолчанию 5)")
    p.add_argument("--tasks", help="id задач через запятую: " + ",".join(TASKS_BY_ID))
    p.add_argument("--reasoning", choices=["none", "model"], default="none",
                   help="none — выключить рассуждение (reasoning_effort=none, по умолчанию); "
                        "model — не передавать параметр, как у модели по умолчанию")
    p.add_argument("--workers", type=int, default=4, help="параллельных запросов (по умолчанию 4)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"таймаут запроса, с (по умолчанию {DEFAULT_TIMEOUT:g})")
    p.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    p.add_argument("--recheck", metavar="JSON", help="без запросов к модели: заново проверить ответы из "
                   "сохранённого results/day4_<время>.json и перезаписать его и .md-отчёт")
    return p.parse_args()


def save_run(raw_path: Path, model: str, extra: dict, temps: list[float], runs: int, tasks: list[Task],
             rows: list[dict], answers: list[Answer], when: datetime | None = None) -> Path:
    raw_path.write_text(json.dumps({
        "model": model, "extra_params": extra, "temperatures": temps, "runs": runs,
        "tasks": [asdict(t) for t in tasks], "summary": rows, "answers": [asdict(a) for a in answers],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = raw_path.with_suffix(".md")
    write_report(report_path, model, extra, rows, answers, tasks, temps, runs, when)
    return report_path


def recheck(path: Path) -> int:
    """Перепроверяет сохранённые ответы текущими проверками: тексты ответов не меняются, только оценки."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks = [TASKS_BY_ID[t["id"]] for t in data["tasks"]]
        answers = [Answer(**a) for a in data["answers"]]
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"Ошибка: не удалось прочитать {path}: {e}", file=sys.stderr)
        return 1
    changed = 0
    for a in answers:
        if a.error:
            continue
        before = (a.format_ok, a.correct, list(a.problems))
        evaluate(TASKS_BY_ID[a.task], a)
        if (a.format_ok, a.correct, a.problems) != before:
            changed += 1
            print(f"t={temp_str(a.temperature):<4} {a.task:<8} #{a.run}: "
                  f"{'ок' if before[0] else '; '.join(before[2])} → {'ок' if a.format_ok else '; '.join(a.problems)}")
    temps = data["temperatures"]
    rows = summarize(answers, tasks, temps)
    print_summary(rows)
    print()
    for n in observations(rows, temps):
        print("• " + n)
    m = re.search(r"(\d{8}_\d{6})", path.name)
    when = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else None
    report = save_run(path, data["model"], data["extra_params"], temps, data["runs"], tasks, rows, answers, when)
    print(f"\nИзменилось оценок: {changed}. Перезаписаны: {path}, {report}")
    return 0


def main() -> int:
    args = parse_args()
    if args.recheck:
        return recheck(Path(args.recheck))
    try:
        settings = load_settings()
        temps = [float(x) for x in args.temps.split(",") if x.strip()]
        wanted = [x.strip() for x in (args.tasks or ",".join(TASKS_BY_ID)).split(",") if x.strip()]
        unknown = set(wanted) - set(TASKS_BY_ID)
        if unknown:
            raise ValueError(f"неизвестные задачи: {sorted(unknown)}; допустимо: {', '.join(TASKS_BY_ID)}")
        if args.runs < 1 or not temps:
            raise ValueError("нужен хотя бы 1 прогон и 1 температура")
    except (ConfigError, ValueError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    tasks = [t for t in TASKS if t.id in wanted]
    extra = REASONING_OFF if args.reasoning == "none" else {}

    model = args.model or settings.model
    client = LLMClient(settings.base_url, settings.api_key, model, args.timeout)
    jobs = [(t, temp, run) for t in tasks for temp in temps for run in range(1, args.runs + 1)]
    print(f"Модель: {model} · доп. параметры: {json.dumps(extra, ensure_ascii=False)}")
    print(f"Запуск: {len(tasks)} задачи × {len(temps)} температуры × {args.runs} прогонов = "
          f"{len(jobs)} запросов, параллельно {args.workers}\n")

    answers: list[Answer] = []
    lock = threading.Lock()
    started = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(ask, client, t, temp, run, extra) for t, temp, run in jobs]
        for f in as_completed(futures):
            a = f.result()
            with lock:
                answers.append(a)
                if a.error:
                    mark, info = "💥", a.error[:80]
                else:
                    mark = ("✅" if a.correct else "❌") if a.correct is not None else "  "
                    mark += "  " if a.format_ok else "⚠️"
                    info = short(a.text) + ("" if a.format_ok else f"  [{'; '.join(a.problems)}]")
                print(f"[{len(answers):>3}/{len(jobs)}] t={temp_str(a.temperature):<4} {a.task:<8} #{a.run} {mark} {info}")
    print(f"\nВсего на прогон: {time.perf_counter() - started:.0f} с")

    order = {(t.id, temp, run): i for i, (t, temp, run) in enumerate(jobs)}
    answers.sort(key=lambda a: order[(a.task, a.temperature, a.run)])
    rows = summarize(answers, tasks, temps)
    print_summary(rows)
    print()
    for n in observations(rows, temps):
        print("• " + n)

    RESULTS_DIR.mkdir(exist_ok=True)
    raw_path = RESULTS_DIR / f"day4_{datetime.now():%Y%m%d_%H%M%S}.json"
    report_path = save_run(raw_path, model, extra, temps, args.runs, tasks, rows, answers)
    print(f"\nСырые ответы: {raw_path}\nОтчёт: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
