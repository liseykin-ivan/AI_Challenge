"""День 2. Формат ответа: один и тот же запрос с разным уровнем контроля.

    python day2_format.py                  # все варианты, по 3 прогона
    python day2_format.py --runs 5
    python day2_format.py --variants plain,schema --runs 2
    python day2_format.py --max-tokens 40 --model <id>

Для каждого ответа проверяется:
  json_strict — json.loads(ответ) без всякой подготовки (так парсит «честный» код);
  json_loose  — JSON удалось вытащить, если срезать ```-обёртку и текст вокруг {...};
  schema      — распарсенный объект соответствует ожидаемой структуре.
Сырые ответы и отчёт пишутся в results/.
"""
import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from config import ConfigError, load_settings
from llm_client import LLMClient, LLMError

RESULTS_DIR = Path(__file__).with_name("results")

# ---------------------------------------------------------------- запрос и формат

QUESTION = "Посоветуй 3 языка программирования для начинающего бэкенд-разработчика и объясни выбор."

STOP_MARKER = "<END>"

FORMAT_INSTRUCTIONS = """Ответь строго в формате JSON, без markdown, без ```-блоков и без текста до или после JSON.
Структура:
{
  "languages": [
    {"name": "<название языка>", "reason": "<почему, одно предложение>"}
  ],
  "summary": "<общий вывод, одно предложение>"
}
В "languages" ровно 3 элемента. Других полей не добавляй."""

FORMAT_WITH_STOP = FORMAT_INSTRUCTIONS + f"\nСразу после JSON напиши на новой строке {STOP_MARKER}."

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "reason"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["languages", "summary"],
    "additionalProperties": False,
}

JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "language_advice", "strict": True, "schema": RESPONSE_SCHEMA},
}


@dataclass
class Variant:
    name: str
    title: str
    system: str | None  # None — без системного промта, «как есть»
    params: dict = field(default_factory=dict)


def build_variants(max_tokens: int) -> list[Variant]:
    return [
        Variant("plain", "Без ограничений", None),
        Variant("format", "Формат описан в промте", FORMAT_INSTRUCTIONS),
        Variant("max_tokens", f"Формат + max_tokens={max_tokens}", FORMAT_INSTRUCTIONS,
                {"max_tokens": max_tokens}),
        Variant("stop", f"Формат + stop={STOP_MARKER!r}", FORMAT_WITH_STOP,
                {"stop": [STOP_MARKER]}),
        Variant("json_mode", "JSON mode (json_object)", FORMAT_INSTRUCTIONS,
                {"response_format": {"type": "json_object"}}),
        Variant("schema", "Строгая JSON Schema", FORMAT_INSTRUCTIONS,
                {"response_format": JSON_SCHEMA_FORMAT}),
        Variant("schema_max_tokens", f"Строгая схема + max_tokens={max_tokens}", FORMAT_INSTRUCTIONS,
                {"response_format": JSON_SCHEMA_FORMAT, "max_tokens": max_tokens}),
    ]


# ---------------------------------------------------------------- проверка ответа

FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def parse_strict(text: str):
    """Ровно json.loads. Возвращает (объект, ошибка)."""
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e.msg} (строка {e.lineno}, столбец {e.colno})"


def parse_loose(text: str):
    """Типичные костыли: срезать ```json ... ``` и текст вокруг первого {...последнего}."""
    candidate = text.strip()
    m = FENCE_RE.match(candidate)
    if m:
        candidate = m.group(1)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        candidate = candidate[start:end + 1]
    return parse_strict(candidate)


def schema_errors(obj) -> list[str]:
    """Проверка структуры по RESPONSE_SCHEMA + «ровно 3 языка» из промта."""
    if not isinstance(obj, dict):
        return [f"ожидался объект, пришёл {type(obj).__name__}"]
    errors = []
    extra = set(obj) - {"languages", "summary"}
    if extra:
        errors.append(f"лишние поля: {sorted(extra)}")
    if not isinstance(obj.get("summary"), str):
        errors.append("нет строки summary")
    langs = obj.get("languages")
    if not isinstance(langs, list):
        errors.append("нет массива languages")
        return errors
    if len(langs) != 3:
        errors.append(f"языков {len(langs)}, а не 3")
    for i, item in enumerate(langs):
        if not isinstance(item, dict):
            errors.append(f"languages[{i}] не объект")
            continue
        for key in ("name", "reason"):
            if not isinstance(item.get(key), str):
                errors.append(f"languages[{i}].{key} не строка")
        if set(item) - {"name", "reason"}:
            errors.append(f"languages[{i}]: лишние поля {sorted(set(item) - {'name', 'reason'})}")
    return errors


@dataclass
class RunResult:
    variant: str
    run: int
    text: str = ""
    finish_reason: str | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None  # сколько из completion_tokens ушло на скрытое рассуждение
    reasoning: str = ""
    latency: float | None = None
    api_error: str | None = None
    json_strict: bool = False
    json_loose: bool = False
    schema_ok: bool = False
    problem: str = ""  # почему сломалось — коротко


def check(result: RunResult) -> RunResult:
    if result.api_error:
        result.problem = f"API: {result.api_error}"
        return result
    if not result.text.strip():
        thinking = bool(result.reasoning.strip()) or bool(result.reasoning_tokens)
        if result.finish_reason == "length":
            result.problem = ("пустой ответ: лимит max_tokens съело рассуждение, до JSON модель не дошла"
                              if thinking else "пустой ответ (finish_reason=length)")
        elif thinking:
            result.problem = "пустой ответ: stop sequence сработал внутри рассуждения, до ответа модель не дошла"
        else:
            result.problem = "пустой ответ"
        return result

    obj, err = parse_strict(result.text)
    result.json_strict = err is None
    if obj is None:
        obj, loose_err = parse_loose(result.text)
        result.json_loose = loose_err is None
        if obj is None:
            cut = " — обрезан по лимиту (finish_reason=length)" if result.finish_reason == "length" else ""
            result.problem = f"не JSON{cut}: {loose_err}"
            return result
        result.problem = "JSON только после чистки (```-обёртка или текст вокруг)"
    else:
        result.json_loose = True

    errors = schema_errors(obj)
    result.schema_ok = not errors
    if errors:
        result.problem = "; ".join(filter(None, [result.problem, "схема: " + "; ".join(errors)]))
    return result


# ---------------------------------------------------------------- прогон

def run_variant(client: LLMClient, variant: Variant, run: int, extra: dict) -> RunResult:
    messages = []
    if variant.system:
        messages.append({"role": "system", "content": variant.system})
    messages.append({"role": "user", "content": QUESTION})

    result = RunResult(variant=variant.name, run=run)
    try:
        reply = client.chat(messages, **{**extra, **variant.params})
    except LLMError as e:
        result.api_error = str(e)
        return check(result)
    result.text = reply.text
    result.finish_reason = reply.finish_reason
    result.completion_tokens = reply.usage.get("completion_tokens")
    details = reply.usage.get("completion_tokens_details") or {}
    result.reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
    result.reasoning = reply.reasoning
    result.latency = reply.latency
    return check(result)


def mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def summarize(variants: list[Variant], results: list[RunResult]) -> list[dict]:
    rows = []
    for v in variants:
        rs = [r for r in results if r.variant == v.name]
        n = len(rs)
        tokens = [r.completion_tokens for r in rs if r.completion_tokens is not None]
        problems = {}
        for r in rs:
            if r.problem:
                key = r.problem[:160]
                problems[key] = problems.get(key, 0) + 1
        rows.append({
            "variant": v.name,
            "title": v.title,
            "runs": n,
            "strict": sum(r.json_strict for r in rs),
            "loose": sum(r.json_loose for r in rs),
            "schema": sum(r.schema_ok for r in rs),
            "avg_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
            "finish": sorted({str(r.finish_reason) for r in rs if not r.api_error}),
            "problems": problems,
        })
    return rows


def print_summary(rows: list[dict]) -> None:
    print("\nИтог (сколько ответов из N разобрал код):")
    head = f"{'вариант':<38} {'json.loads':>10} {'с чисткой':>10} {'схема':>7} {'токены':>7}  finish_reason"
    print(head)
    print("-" * len(head))
    for r in rows:
        n = r["runs"]
        print(f"{r['title']:<38} {r['strict']:>6}/{n:<3} {r['loose']:>6}/{n:<3} {r['schema']:>3}/{n:<3}"
              f" {str(r['avg_tokens'] or '-'):>7}  {', '.join(r['finish']) or '-'}")
        for problem, count in r["problems"].items():
            print(f"    {count}× {problem}")


def write_report(path: Path, model: str, rows: list[dict], results: list[RunResult]) -> None:
    lines = [
        "# День 2 · Формат ответа — результаты прогона",
        "",
        f"Модель: `{model}` · дата: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Запрос: «{QUESTION}»",
        "",
        "| Вариант | `json.loads` | JSON после чистки | Схема | Ср. токенов | finish_reason |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        n = r["runs"]
        lines.append(f"| {r['title']} | {r['strict']}/{n} | {r['loose']}/{n} | {r['schema']}/{n} | "
                     f"{r['avg_tokens'] or '—'} | {', '.join(r['finish']) or '—'} |")
    lines += ["", "## Что сломалось", ""]
    any_problem = False
    for r in rows:
        for problem, count in r["problems"].items():
            any_problem = True
            lines.append(f"- **{r['title']}** — {count}×: {problem}")
    if not any_problem:
        lines.append("Все ответы разобраны и прошли схему.")
    lines += ["", "## Примеры ответов (первый прогон каждого варианта)", ""]
    for r in rows:
        first = next((x for x in results if x.variant == r["variant"]), None)
        if not first:
            continue
        body = first.api_error or first.text or "(пусто)"
        if not first.text and first.reasoning:
            body = f"(content пустой; хвост рассуждения)\n…{first.reasoning[-600:]}"
        lines += [f"### {r['title']} — {first.problem or 'OK'}", "", "~~~text", body[:1200], "~~~", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="День 2: один запрос с разным уровнем контроля формата.")
    parser.add_argument("--runs", type=int, default=3, help="прогонов на вариант (по умолчанию 3)")
    parser.add_argument("--variants", help="через запятую: plain,format,max_tokens,stop,json_mode,schema,schema_max_tokens")
    parser.add_argument("--max-tokens", type=int, default=60, help="лимит для вариантов с max_tokens (по умолчанию 60)")
    parser.add_argument("--temperature", type=float, help="температура для всех вариантов (по умолчанию — как у модели)")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--timeout", type=float, help="таймаут запроса, с (по умолчанию LLM_TIMEOUT из .env)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    variants = build_variants(args.max_tokens)
    if args.variants:
        wanted = [x.strip() for x in args.variants.split(",") if x.strip()]
        unknown = set(wanted) - {v.name for v in variants}
        if unknown:
            print(f"Ошибка: неизвестные варианты {sorted(unknown)}", file=sys.stderr)
            return 1
        variants = [v for v in variants if v.name in wanted]

    model = args.model or settings.model
    client = LLMClient(settings.base_url, settings.api_key, model, args.timeout or settings.timeout)
    extra = {"temperature": args.temperature}

    print(f"Модель: {model} · вариантов: {len(variants)} · прогонов: {args.runs}")
    print(f"Запрос: {QUESTION}\n")
    results = []
    for v in variants:
        print(f"== {v.title}")
        for i in range(1, args.runs + 1):
            r = run_variant(client, v, i, extra)
            results.append(r)
            thinking = f" (рассуждение {r.reasoning_tokens})" if r.reasoning_tokens else ""
            info = r.api_error or (f"{r.completion_tokens} ток.{thinking}, {r.latency:.1f} с, finish={r.finish_reason}")
            print(f"  #{i} json.loads {mark(r.json_strict)}  чистка {mark(r.json_loose)}  "
                  f"схема {mark(r.schema_ok)}  | {info}")
            if r.problem:
                print(f"      ↳ {r.problem}")

    rows = summarize(variants, results)
    print_summary(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = RESULTS_DIR / f"day2_{stamp}.json"
    report_path = RESULTS_DIR / f"day2_{stamp}.md"
    raw_path.write_text(json.dumps({"model": model, "question": QUESTION, "summary": rows,
                                    "runs": [asdict(r) for r in results]},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, model, rows, results)
    print(f"\nСырые ответы: {raw_path}\nОтчёт: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
