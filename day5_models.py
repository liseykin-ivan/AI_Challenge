"""День 5. Версии моделей.

Одни и те же 10 задач (трудный набор из Дня 3) решают три модели одного семейства (Qwen в OpenCode Go):

    small  — qwen3.8-flash  — маленькая дешёвая;
    medium — qwen3.7-plus   — средняя;
    large  — qwen3.8-max    — флагман.

Для каждого ответа замеряется время, токены (вход / выход / рассуждение) и стоимость по прайсу OpenCode Go,
ответ проверяется кодом по эталону. Итог — таблица «модель → верных / время / токены / стоимость».

    python day5_models.py --probe                   # проверить, что модели доступны и как у них выключается рассуждение
    python day5_models.py                           # 10 задач × 3 модели × 3 прогона = 90 запросов
    python day5_models.py --reasoning model         # рассуждение как у модели по умолчанию
    python day5_models.py --method steps            # к условию добавить «Решай пошагово.»
    python day5_models.py --models qwen3.8-flash,glm-5.3-flash,qwen3.8-max   # другие модели из PRICES
    python day5_models.py --recheck results\\day5_<время>.json   # перепроверить сохранённые ответы без запросов

Задачи, проверка ответа и формат «ОТВЕТ: ...» взяты из day3_reasoning.py, чтобы результаты были сравнимы с Днём 3.
"""
import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from config import ConfigError, load_settings
from day3_reasoning import HARD_TASKS, STEPS_INSTRUCTION, Task, answer_format, extract_answer, is_correct
from llm_client import LLMClient, LLMError

RESULTS_DIR = Path(__file__).with_name("results")
DEFAULT_TIMEOUT = 300.0
TASKS = HARD_TASKS
TASKS_BY_ID = {t.id: t for t in TASKS}

# ---------------------------------------------------------------- модели и цены

@dataclass(frozen=True)
class Price:
    """USD за 1 млн токенов, прайс OpenCode Go (https://opencode.ai/docs/go/), на 2026-11-08."""
    name: str
    input: float
    output: float
    cached: float


PRICES = {
    # Qwen — три размера одного провайдера; это основной набор
    "qwen3.8-flash": Price("Qwen3.8 Flash", 0.15, 0.47, 0.016),
    "qwen3.7-plus": Price("Qwen3.7 Plus", 0.40, 1.60, 0.04),      # ≤256K контекста
    "qwen3.8-max": Price("Qwen3.8 Max", 2.00, 6.00, 0.25),
    # запасные варианты для --models
    "deepseek-v4.1-flash": Price("DeepSeek V4.1 Flash (off-peak)", 0.15, 0.60, 0.003),
    "deepseek-v4-pro": Price("DeepSeek V4 Pro (off-peak)", 0.66, 1.98, 0.022),
    "glm-5.3-flash": Price("GLM-5.3-Flash", 0.15, 0.50, 0.03),
    "glm-5.3": Price("GLM-5.3", 1.40, 4.40, 0.26),
    "mimo-v2.6-flash": Price("MiMo-V2.6-Flash", 0.14, 0.28, 0.0028),
    "mimo-v2.6-pro": Price("MiMo-V2.6-Pro", 0.435, 0.87, 0.003625),
    "kimi-k3.1-pro": Price("Kimi K3.1 Pro", 2.00, 10.00, 0.20),
}
DEFAULT_MODELS = ["qwen3.8-flash", "qwen3.7-plus", "qwen3.8-max"]
TIERS = ["маленькая", "средняя", "флагман"]

REASONING = {"off": {"reasoning_effort": "none"}, "model": {}}
METHODS = {"direct": "только условие", "steps": "условие + «Решай пошагово.»"}

# ---------------------------------------------------------------- запрос

RETRYABLE = ("HTTP 429", "HTTP 5", "таймаут", "нет связи", "неожиданный ответ")


@dataclass
class Answer:
    model: str
    task: str
    run: int
    text: str = ""
    reasoning: str = ""
    answer: str | None = None
    correct: bool = False
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency: float = 0.0
    cost: float = 0.0          # по прайсу, считаем сами
    list_cost: float = 0.0     # стоимость, которую вернул OpenCode (usage.opencode.list_cost)
    allowance: float = 0.0     # сколько списано из лимита подписки (usage.opencode.allowance)
    finish_reason: str | None = None
    served_model: str = ""
    usage: dict = field(default_factory=dict)
    error: str | None = None
    attempts: int = 1

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def build_prompt(task: Task, method: str) -> str:
    extra = f"\n\n{STEPS_INSTRUCTION}\n" if method == "steps" else "\n\n"
    return f"{task.text}{extra}{answer_format(task)}"


def cost_of(a: Answer) -> float:
    p = PRICES.get(a.model)
    if not p:
        return 0.0
    fresh = max(a.prompt_tokens - a.cached_tokens, 0)
    return (fresh * p.input + a.cached_tokens * p.cached + a.completion_tokens * p.output) / 1_000_000


def evaluate(task: Task, a: Answer) -> Answer:
    a.answer = extract_answer(a.text) if not a.error else None
    a.correct = is_correct(task, a.answer)
    a.cost = cost_of(a)
    oc = a.usage.get("opencode") if isinstance(a.usage, dict) else None
    if isinstance(oc, dict):
        a.list_cost = oc.get("list_cost") or 0.0
        a.allowance = oc.get("allowance") or 0.0
    return a


def ask(client: LLMClient, task: Task, run: int, method: str, params: dict, retries: int = 2) -> Answer:
    a = Answer(client.model, task.id, run)
    prompt = build_prompt(task, method)
    for attempt in range(1, retries + 2):
        a.attempts = attempt
        try:
            reply = client.chat([{"role": "user", "content": prompt}], **params)
        except LLMError as e:
            a.error = str(e)
            if attempt <= retries and any(m in a.error for m in RETRYABLE):
                time.sleep(5 * attempt)
                continue
            return evaluate(task, a)
        u = reply.usage
        cd = u.get("completion_tokens_details") or {}
        pd = u.get("prompt_tokens_details") or {}
        a.error = None
        a.usage = u
        a.prompt_tokens = u.get("prompt_tokens") or 0
        a.completion_tokens = u.get("completion_tokens") or 0
        a.reasoning_tokens = (cd.get("reasoning_tokens") if isinstance(cd, dict) else 0) or 0
        a.cached_tokens = (pd.get("cached_tokens") if isinstance(pd, dict) else 0) or 0
        a.latency = reply.latency
        a.finish_reason = reply.finish_reason
        a.served_model = reply.model
        a.text = reply.text
        a.reasoning = reply.reasoning
        return evaluate(task, a)
    return evaluate(task, a)

# ---------------------------------------------------------------- сводка

def fmt_int(n: float) -> str:
    return f"{round(n):,}".replace(",", " ")


def fmt_usd(x: float) -> str:
    return f"${x:.4f}" if x < 1 else f"${x:.2f}"


def summarize(answers: list[Answer], models: list[str]) -> list[dict]:
    rows = []
    for i, m in enumerate(models):
        group = [a for a in answers if a.model == m]
        ok = [a for a in group if not a.error]
        if not group:
            continue
        lat = [a.latency for a in ok]
        out_speed = [a.completion_tokens / a.latency for a in ok if a.latency > 0]
        row = {
            "model": m, "name": PRICES[m].name if m in PRICES else m,
            "tier": TIERS[i] if len(models) == 3 else "",
            "served": sorted({a.served_model for a in ok if a.served_model}),
            "n": len(group), "correct": sum(a.correct for a in group),
            "errors": len(group) - len(ok),
            "no_answer": sum(1 for a in ok if a.answer is None),
            "prompt_tokens": sum(a.prompt_tokens for a in group),
            "completion_tokens": sum(a.completion_tokens for a in group),
            "reasoning_tokens": sum(a.reasoning_tokens for a in group),
            "tokens": sum(a.tokens for a in group),
            "cost": sum(a.cost for a in group),
            "list_cost": sum(a.list_cost for a in group),
            "allowance": sum(a.allowance for a in group),
            "time_median": round(statistics.median(lat), 1) if lat else 0,
            "time_p90": round(sorted(lat)[max(0, int(len(lat) * 0.9) - 1)], 1) if lat else 0,
            "time_total": round(sum(lat), 1),
            "speed": round(statistics.median(out_speed)) if out_speed else 0,
        }
        row["accuracy"] = row["correct"] / row["n"]
        row["completion_per_task"] = row["completion_tokens"] / row["n"]
        row["cost_per_1000"] = row["cost"] / row["n"] * 1000
        row["cost_per_correct"] = row["cost"] / row["correct"] if row["correct"] else 0
        rows.append(row)
    return rows


def per_task(answers: list[Answer], tasks: list[Task], models: list[str]) -> dict:
    """task -> model -> (верных, всего)."""
    return {t.id: {m: (sum(a.correct for a in answers if a.task == t.id and a.model == m),
                       sum(1 for a in answers if a.task == t.id and a.model == m)) for m in models} for t in tasks}


def comparison(rows: list[dict]) -> list[str]:
    """Флагман против маленькой: во сколько раз дороже / медленнее и на сколько точнее."""
    if len(rows) < 2:
        return []
    small, large = rows[0], rows[-1]
    out = []

    def ratio(a, b):
        return f"×{a / b:.1f}" if b else "—"

    out.append(f"- **Стоимость:** {large['name']} дороже {small['name']} в "
               f"**{ratio(large['cost'], small['cost'])}** на этом наборе "
               f"({fmt_usd(large['cost'])} против {fmt_usd(small['cost'])}); по прайсу — "
               f"вход {ratio(PRICES[large['model']].input, PRICES[small['model']].input)}, "
               f"выход {ratio(PRICES[large['model']].output, PRICES[small['model']].output)}.")
    d = (large["accuracy"] - small["accuracy"]) * 100
    out.append(f"- **Точность:** {large['correct']}/{large['n']} против {small['correct']}/{small['n']} — "
               f"**{d:+.0f} п. п.**")
    out.append(f"- **Время (медиана):** {large['time_median']} с против {small['time_median']} с "
               f"({ratio(large['time_median'], small['time_median'])}); скорость генерации "
               f"{large['speed']} против {small['speed']} ток/с.")
    out.append(f"- **Токены ответа на задачу:** {fmt_int(large['completion_per_task'])} против "
               f"{fmt_int(small['completion_per_task'])} ({ratio(large['completion_tokens'], small['completion_tokens'])}).")
    if small["allowance"] and large["allowance"]:
        out.append(f"- **Списание из лимита подписки** (поле `allowance` от OpenCode): "
                   f"{fmt_usd(large['allowance'])} против {fmt_usd(small['allowance'])} — "
                   f"**{ratio(large['allowance'], small['allowance'])}**.")
    extra_correct = large["correct"] - small["correct"]
    if extra_correct > 0:
        per = (large["cost"] - small["cost"]) / extra_correct
        out.append(f"- **Цена одного дополнительного верного ответа** флагмана: {fmt_usd(per)}.")
    return out


def sufficient(answers: list[Answer], tasks: list[Task], models: list[str]) -> tuple[list[str], list[str]]:
    """Задачи, где маленькая модель решила все прогоны, и где нет."""
    pt = per_task(answers, tasks, models)
    small = models[0]
    enough = [t.id for t in tasks if pt[t.id][small][0] == pt[t.id][small][1] > 0]
    return enough, [t.id for t in tasks if t.id not in enough]


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 110)
    print(f"{'Модель':<22} {'верно':>7} {'медиана, с':>11} {'p90, с':>7} {'ток/с':>6} {'вход':>8} {'выход':>8} "
          f"{'рассужд.':>9} {'стоимость':>11} {'за 1000 задач':>14} {'из лимита':>10}")
    for r in rows:
        print(f"{r['model']:<22} {r['correct']:>4}/{r['n']:<2} {r['time_median']:>11} {r['time_p90']:>7} {r['speed']:>6} "
              f"{fmt_int(r['prompt_tokens']):>8} {fmt_int(r['completion_tokens']):>8} {fmt_int(r['reasoning_tokens']):>9} "
              f"{fmt_usd(r['cost']):>11} {fmt_usd(r['cost_per_1000']):>14} {fmt_usd(r['allowance']):>10}")
        if r["errors"]:
            print(f"    ошибок API: {r['errors']}")


def write_report(path: Path, meta: dict, rows: list[dict], answers: list[Answer], tasks: list[Task],
                 models: list[str]) -> None:
    lines = [
        "# День 5 · Версии моделей — результаты прогона", "",
        f"Дата: {meta['date']} · задач: {len(tasks)} (набор hard из Дня 3) · прогонов: {meta['runs']} · "
        f"промпт: {METHODS[meta['method']]} · рассуждение: `{json.dumps(meta['params'], ensure_ascii=False)}`", "",
        "## Модели", "",
        "| Уровень | Модель | id | Вход, $/1M | Выход, $/1M | Кэш, $/1M |", "|---|---|---|---|---|---|",
    ]
    for r in rows:
        p = PRICES.get(r["model"])
        lines.append(f"| {r['tier']} | {r['name']} | `{r['model']}` | {p.input if p else '?'} | "
                     f"{p.output if p else '?'} | {p.cached if p else '?'} |")
    lines += ["", "Цены — прайс OpenCode Go (https://opencode.ai/docs/go/). По подписке Go Plus деньги за запросы не "
              "списываются, но расход идёт из месячного лимита модели по этим же ценам.", "",
              "## Модель → верных ответов / время / токены / стоимость", "",
              "| Модель | Верных | Медиана времени, с | p90, с | Скорость, ток/с | Токенов вход | Токенов выход | "
              "из них рассуждение | Стоимость прогона | За 1000 задач | За верный ответ | Списано из лимита |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        err = f" (ошибок API: {r['errors']})" if r["errors"] else ""
        lines.append(f"| {r['tier']} · `{r['model']}` | **{r['correct']}/{r['n']}**{err} | {r['time_median']} | "
                     f"{r['time_p90']} | {r['speed']} | {fmt_int(r['prompt_tokens'])} | "
                     f"{fmt_int(r['completion_tokens'])} | {fmt_int(r['reasoning_tokens'])} | "
                     f"{fmt_usd(r['cost'])} | {fmt_usd(r['cost_per_1000'])} | {fmt_usd(r['cost_per_correct'])} | "
                     f"{fmt_usd(r['allowance']) if r['allowance'] else '—'} |")
    lines += ["", "Время — по каждому запросу отдельно (медиана и 90-й перцентиль), скорость — выходных токенов "
              "в секунду (медиана). Стоимость = (вход − кэш) × цена входа + кэш × цена кэша + выход × цена выхода; "
              "сверена с `list_cost`, который возвращает OpenCode" + (" — совпадает" if all(
                  abs(r["cost"] - r["list_cost"]) < 1e-6 for r in rows) and any(r["list_cost"] for r in rows) else "")
              + ". «Списано из лимита» — поле `allowance`: столько уходит из лимита подписки Go Plus.", "",
              "## Флагман против маленькой модели (посчитано автоматически)", ""] + comparison(rows)

    pt = per_task(answers, tasks, models)
    lines += ["", "## По задачам", "",
              "| Задача | Эталон | " + " | ".join(f"`{m}`" for m in models) + " |",
              "|---|---|" + "---|" * len(models)]
    for t in tasks:
        cells = []
        for m in models:
            ok, n = pt[t.id][m]
            cells.append("—" if not n else "✅" if ok == n else "❌" if ok == 0 else f"{ok}/{n}")
        lines.append(f"| {t.id} ({t.kind}) | {t.answer} | " + " | ".join(cells) + " |")

    enough, not_enough = sufficient(answers, tasks, models)
    lines += ["", f"Маленькой модели (`{models[0]}`) хватило во всех прогонах: "
              + (", ".join(f"`{x}`" for x in enough) or "ни в одной задаче") + ".",
              "Не хватило: " + (", ".join(f"`{x}`" for x in not_enough) or "—") + "."]

    wrong = [a for a in answers if not a.correct]
    lines += ["", "## Неверные ответы", ""]
    for a in wrong:
        why = a.error or ("нет строки «ОТВЕТ:»" + (" (обрезан по лимиту)" if a.finish_reason == "length" else ""))
        lines.append(f"- `{a.model}` · `{a.task}` · прогон {a.run}: "
                     + (f"{a.answer[:60]!r}, эталон {TASKS_BY_ID[a.task].answer}" if a.answer else why))
    if not wrong:
        lines.append("Все ответы верные.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(raw_path: Path, report_path: Path, meta: dict, answers: list[Answer], tasks: list[Task], models: list[str]):
    rows = summarize(answers, models)
    raw_path.write_text(json.dumps({**meta, "models": models, "summary": rows, "tasks": [asdict(t) for t in tasks],
                                    "prices": {m: asdict(PRICES[m]) for m in models if m in PRICES},
                                    "answers": [asdict(a) for a in answers]},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, meta, rows, answers, tasks, models)
    return rows

# ---------------------------------------------------------------- проба

def probe(settings, models: list[str], timeout: float) -> int:
    """Отвечает ли модель и выключается ли у неё рассуждение параметром reasoning_effort=none."""
    task = TASKS_BY_ID["divisors"]
    prompt = build_prompt(task, "direct")
    print(f"Проба на задаче «{task.id}»:\n")
    print(f"{'модель':<22} {'параметры':<26} {'рассужд.':>9} {'выход':>7} {'время':>7}  ответ")
    for m in models:
        client = LLMClient(settings.base_url, settings.api_key, m, timeout)
        for label, params in [("по умолчанию", {}), ("reasoning_effort=none", REASONING["off"])]:
            try:
                r = client.chat([{"role": "user", "content": prompt}], **params)
            except LLMError as e:
                print(f"{m:<22} {label:<26} ошибка: {str(e)[:80]}")
                continue
            cd = r.usage.get("completion_tokens_details") or {}
            rt = (cd.get("reasoning_tokens") if isinstance(cd, dict) else 0) or len(r.reasoning) // 3
            ans = extract_answer(r.text)
            mark = "✅" if is_correct(task, ans) else "❌"
            print(f"{m:<22} {label:<26} {rt:>9} {r.usage.get('completion_tokens', '?'):>7} {r.latency:>6.1f}с  "
                  f"{mark} {ans!r} (served: {r.model})")
    return 0

# ---------------------------------------------------------------- запуск

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="День 5: одни и те же задачи на маленькой, средней и флагманской модели.")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help="id моделей через запятую, от маленькой к флагману (по умолчанию " + ",".join(DEFAULT_MODELS) + ")")
    p.add_argument("--tasks", help="id задач через запятую: " + ",".join(TASKS_BY_ID))
    p.add_argument("--runs", type=int, default=3, help="прогонов каждой пары модель × задача (по умолчанию 3)")
    p.add_argument("--method", choices=list(METHODS), default="direct", help="промпт: direct (по умолчанию) или steps")
    p.add_argument("--reasoning", choices=list(REASONING), default="off",
                   help="off — reasoning_effort=none (по умолчанию), model — как у модели по умолчанию")
    p.add_argument("--workers", type=int, default=6, help="параллельных запросов (по умолчанию 6)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"таймаут запроса, с (по умолчанию {DEFAULT_TIMEOUT:g})")
    p.add_argument("--probe", action="store_true", help="только проба: доступность моделей и выключение рассуждения")
    p.add_argument("--recheck", metavar="JSON", help="без запросов: заново проверить ответы из сохранённого .json")
    return p.parse_args()


def recheck(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data["models"]
    tasks = [TASKS_BY_ID[t["id"]] for t in data["tasks"]]
    answers = [evaluate(TASKS_BY_ID[a["task"]], Answer(**a)) for a in data["answers"]]
    meta = {k: data[k] for k in ("date", "runs", "method", "reasoning", "params")}
    rows = save(path, path.with_suffix(".md"), meta, answers, tasks, models)
    print_summary(rows)
    print(f"\nПерепроверено: {path}\nОтчёт: {path.with_suffix('.md')}")
    return 0


def main() -> int:
    args = parse_args()
    if args.recheck:
        return recheck(Path(args.recheck))
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in PRICES]
    if unknown:
        print(f"Внимание: нет цены для {unknown} — стоимость будет 0. Добавьте их в PRICES.", file=sys.stderr)
    if args.probe:
        return probe(settings, models, args.timeout)

    wanted = [x.strip() for x in (args.tasks or "").split(",") if x.strip()]
    if set(wanted) - set(TASKS_BY_ID):
        print(f"Ошибка: неизвестные задачи {sorted(set(wanted) - set(TASKS_BY_ID))}", file=sys.stderr)
        return 1
    tasks = [t for t in TASKS if not wanted or t.id in wanted]
    params = REASONING[args.reasoning]
    clients = {m: LLMClient(settings.base_url, settings.api_key, m, args.timeout) for m in models}

    # Чередуем модели в очереди, чтобы нагрузка у провайдера в момент замера была для всех одинаковой.
    jobs = [(m, t, run) for run in range(1, args.runs + 1) for t in tasks for m in models]
    print(f"Модели: {', '.join(models)}\nЗадач: {len(tasks)} × моделей: {len(models)} × прогонов: {args.runs} = "
          f"{len(jobs)} запросов, параллельно {args.workers}; промпт: {args.method}; "
          f"рассуждение: {json.dumps(params)}\n")
    answers: list[Answer] = []
    lock = threading.Lock()
    started = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(ask, clients[m], t, run, args.method, params) for m, t, run in jobs]
        for f in as_completed(futures):
            a = f.result()
            with lock:
                answers.append(a)
                mark = "✅" if a.correct else "❌"
                info = a.error or (f"{a.prompt_tokens}+{a.completion_tokens} ток. (рассужд. {a.reasoning_tokens}), "
                                   f"{a.latency:.1f} с, {fmt_usd(a.cost)}")
                print(f"[{len(answers):>3}/{len(jobs)}] {mark} {a.model:<16} {a.task:<15} "
                      f"ответ={str(a.answer)[:20]!r:<22} | {info}")
    print(f"\nВсего на прогон: {time.perf_counter() - started:.0f} с")

    order = {(m, t.id, r): i for i, (m, t, r) in enumerate(jobs)}
    answers.sort(key=lambda a: order[(a.model, a.task, a.run)])
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path, report_path = RESULTS_DIR / f"day5_{stamp}.json", RESULTS_DIR / f"day5_{stamp}.md"
    meta = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "runs": args.runs, "method": args.method,
            "reasoning": args.reasoning, "params": params}
    rows = save(raw_path, report_path, meta, answers, tasks, models)
    print_summary(rows)
    print("\n" + "\n".join(comparison(rows)).replace("**", ""))
    print(f"\nСырые ответы: {raw_path}\nОтчёт: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
