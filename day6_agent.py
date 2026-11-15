"""День 6. Первый агент.

Агент — отдельный класс `Agent`: хранит историю диалога и набор инструментов и сам крутит цикл
«запрос к модели → вызов инструментов → результат обратно в модель → …», пока модель не перестанет
запрашивать инструменты и не даст финальный ответ. Инструменты — обычные функции в несколько строк,
модели они описываются JSON-схемой в поле `tools` запроса (OpenAI-совместимый function calling).

Инструменты:
    get_current_time  — текущие дата и время в часовом поясе IANA;
    calculator        — арифметическое выражение (безопасный разбор через ast, без eval);
    convert_currency  — конвертер валют по зашитому в код курсу.

    python day6_agent.py                         # чат в консоли (/tools, /reset, /exit)
    python day6_agent.py "Который час в Токио?"  # один запрос и выход
    python day6_agent.py --demo                  # три проверочных сценария + отчёт в results/
    python day6_agent.py --model <id> --demo     # другая модель (должна поддерживать tools)
"""
import argparse
import ast
import json
import operator
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from config import ConfigError, load_settings
from llm_client import LLMClient, LLMError

RESULTS_DIR = Path(__file__).with_name("results")

# ---------------------------------------------------------------- инструменты: обычные функции


def get_current_time(timezone: str = "Europe/Moscow") -> dict:
    try:
        now = datetime.now(ZoneInfo(timezone))
    except (ZoneInfoNotFoundError, ValueError):  # на Windows базу поясов даёт пакет tzdata из requirements.txt
        return {"error": f"неизвестный часовой пояс «{timezone}». Нужно имя IANA, например Asia/Tokyo."}
    return {"timezone": timezone, "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": now.strftime("%A"), "utc_offset": now.strftime("%z")}


_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def calculator(expression: str) -> dict:
    def ev(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("слишком большая степень")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](ev(node.operand))
        raise ValueError("допустимы только числа, скобки и + - * / // % **")

    try:
        value = ev(ast.parse(expression.replace(",", ".").replace("^", "**"), mode="eval").body)
    except ZeroDivisionError:
        return {"error": "деление на ноль"}
    except (SyntaxError, ValueError, TypeError) as e:
        return {"error": f"не удалось вычислить «{expression}»: {e}"}
    return {"expression": expression, "result": round(value, 10) if isinstance(value, float) else value}


# Курс зашит в код: сколько рублей стоит 1 единица валюты (условные значения для учебного задания).
RATES_RUB = {"RUB": 1.0, "USD": 81.5, "EUR": 94.2, "CNY": 11.3, "GBP": 107.6, "JPY": 0.53}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    src, dst = from_currency.upper(), to_currency.upper()
    missing = [c for c in (src, dst) if c not in RATES_RUB]
    if missing:
        return {"error": f"нет курса для {', '.join(missing)}. Доступны: {', '.join(RATES_RUB)}."}
    result = amount * RATES_RUB[src] / RATES_RUB[dst]
    return {"amount": amount, "from": src, "to": dst, "rate": round(RATES_RUB[src] / RATES_RUB[dst], 6),
            "result": round(result, 2)}


# ---------------------------------------------------------------- описание инструментов для модели


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema аргументов
    func: Callable[..., dict]

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


TOOLS = [
    Tool("get_current_time",
         "Текущие дата, время и день недели в заданном часовом поясе. Используй для любых вопросов о текущем времени или дате.",
         {"type": "object",
          "properties": {"timezone": {"type": "string",
                                      "description": "Часовой пояс IANA, например Europe/Moscow, Asia/Tokyo, America/New_York"}},
          "required": ["timezone"]},
         get_current_time),
    Tool("calculator",
         "Вычисляет арифметическое выражение: числа, скобки, + - * / // % ** (степень).",
         {"type": "object",
          "properties": {"expression": {"type": "string", "description": "Выражение, например (150 * 81.5) * 1.035"}},
          "required": ["expression"]},
         calculator),
    Tool("convert_currency",
         f"Переводит сумму из одной валюты в другую по фиксированному курсу. Валюты: {', '.join(RATES_RUB)}.",
         {"type": "object",
          "properties": {"amount": {"type": "number", "description": "Сумма в исходной валюте"},
                         "from_currency": {"type": "string", "description": "Код исходной валюты, например USD"},
                         "to_currency": {"type": "string", "description": "Код целевой валюты, например RUB"}},
          "required": ["amount", "from_currency", "to_currency"]},
         convert_currency),
]

SYSTEM_PROMPT = (
    "Ты — полезный ассистент-агент. Отвечай кратко, на языке пользователя.\n"
    "У тебя есть инструменты. Вызывай их, когда без них ответ будет неточным:\n"
    "- текущее время и дату узнавай только через get_current_time;\n"
    "- любые вычисления делай через calculator, не считай в уме;\n"
    "- валюты переводи только через convert_currency.\n"
    "Если вопрос не требует инструментов — отвечай сразу, без вызовов."
)


# ---------------------------------------------------------------- агент


@dataclass
class ToolCall:
    step: int  # номер запроса к модели внутри одного run(), начиная с 1
    name: str
    arguments: dict
    result: dict


@dataclass
class AgentReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0


class Agent:
    """LLM-агент с инструментами. Не знает об интерфейсе: события отдаёт в колбэк on_event."""

    def __init__(self, client: LLMClient, tools: list[Tool], system_prompt: str = SYSTEM_PROMPT,
                 max_steps: int = 8, on_event: Callable[[str, dict], None] | None = None):
        self.client = client
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.on_event = on_event or (lambda kind, data: None)
        self.history: list[dict] = []
        self.reset()

    def reset(self) -> None:
        self.history = [{"role": "system", "content": self.system_prompt}]

    def run(self, user_text: str) -> AgentReply:
        """Один ход пользователя: крутит цикл, пока модель не ответит без вызова инструментов.
        Если запрос к модели упал посреди цикла, история откатывается к состоянию до хода."""
        start = len(self.history)
        try:
            return self._loop(user_text)
        except LLMError:
            del self.history[start:]
            raise

    def _loop(self, user_text: str) -> AgentReply:
        self.history.append({"role": "user", "content": user_text})
        reply = AgentReply(text="")
        schemas = [t.schema() for t in self.tools.values()]

        for step in range(1, self.max_steps + 1):
            result = self.client.chat(self.history, tools=schemas)
            reply.llm_calls += 1
            reply.prompt_tokens += result.usage.get("prompt_tokens") or 0
            reply.completion_tokens += result.usage.get("completion_tokens") or 0
            reply.latency += result.latency
            self.history.append(self._assistant_message(result))
            self.on_event("llm", {"step": step, "latency": result.latency, "usage": result.usage,
                                  "tool_calls": len(result.tool_calls), "text": result.text})

            if not result.tool_calls:  # модель больше не просит инструменты — это финальный ответ
                reply.text = result.text
                return reply

            for call in result.tool_calls:
                name, args, output = self._execute(call)
                reply.tool_calls.append(ToolCall(step, name, args, output))
                self.on_event("tool", {"step": step, "name": name, "arguments": args, "result": output})
                self.history.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                     "content": json.dumps(output, ensure_ascii=False)})

        reply.text = f"(агент остановлен: модель запрашивала инструменты {self.max_steps} раз подряд без финального ответа)"
        return reply

    @staticmethod
    def _assistant_message(result) -> dict:
        """Сообщение ассистента для истории: с tool_calls, чтобы модель видела, что именно она вызывала."""
        message = {"role": "assistant", "content": result.text or None}
        if result.tool_calls:
            message["tool_calls"] = result.tool_calls
        # reasoning-модели (DeepSeek и др.) в цикле с инструментами ждут своё рассуждение обратно
        if result.message.get("reasoning_content"):
            message["reasoning_content"] = result.message["reasoning_content"]
        return message

    def _execute(self, call: dict) -> tuple[str, dict, dict]:
        """Выполняет один вызов. Ошибки не роняют агента, а уходят модели как результат инструмента."""
        function = call.get("function") or {}
        name = function.get("name", "")
        raw = function.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            return name, {"_raw": raw}, {"error": f"аргументы не являются JSON: {raw!r}"}
        tool = self.tools.get(name)
        if tool is None:
            return name, args, {"error": f"нет инструмента «{name}». Доступны: {', '.join(self.tools)}."}
        try:
            return name, args, tool.func(**args)
        except TypeError as e:
            return name, args, {"error": f"неверные аргументы: {e}"}


# ---------------------------------------------------------------- проверочные сценарии


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    prompt: str
    expected_tools: tuple[str, ...]  # какие инструменты должны быть вызваны (в любом порядке)
    min_tool_steps: int  # сколько запросов к модели подряд должны закончиться вызовом инструмента


SCENARIOS = [
    Scenario("no_tools", "без инструментов",
             "Объясни в двух предложениях, чем LLM-агент отличается от одного вызова модели.", (), 0),
    Scenario("one_tool", "один инструмент",
             "Который час сейчас в Токио?", ("get_current_time",), 1),
    Scenario("two_tools", "два инструмента подряд",
             "Сколько рублей я заплачу за покупку на 150 долларов, если банк сверху берёт комиссию 3,5%?",
             ("convert_currency", "calculator"), 2),
]


def check(s: Scenario, reply: AgentReply) -> tuple[bool, str]:
    used = [c.name for c in reply.tool_calls]
    tool_steps = len({c.step for c in reply.tool_calls})
    if not s.expected_tools:
        ok = not used
        return ok, "инструменты не вызывались" if ok else f"лишние вызовы: {', '.join(used)}"
    missing = [t for t in s.expected_tools if t not in used]
    if missing:
        return False, f"не вызваны: {', '.join(missing)}"
    if tool_steps < s.min_tool_steps:
        return False, f"вызовы в {tool_steps} шаге(ах), ожидалось подряд в {s.min_tool_steps}"
    return True, f"вызваны {' → '.join(used)} за {tool_steps} шаг(а)"


# ---------------------------------------------------------------- интерфейс (CLI)


def short(value, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def print_event(kind: str, data: dict) -> None:
    if kind == "llm":
        usage = data["usage"]
        what = f"просит инструменты: {data['tool_calls']}" if data["tool_calls"] else "финальный ответ"
        print(f"  · шаг {data['step']}: модель → {what} | токены {usage.get('prompt_tokens', '?')}/"
              f"{usage.get('completion_tokens', '?')} | {data['latency']:.2f} с")
        if data["tool_calls"] and data["text"].strip():
            print(f"    💭 {data['text'].strip()}")
    elif kind == "tool":
        print(f"    🔧 {data['name']}({short(data['arguments'])}) → {short(data['result'])}")


def print_reply(reply: AgentReply, model: str) -> None:
    print(f"\nАгент: {reply.text.strip()}\n")
    print(f"{model} | запросов к модели: {reply.llm_calls} | вызовов инструментов: {len(reply.tool_calls)} | "
          f"токены: {reply.prompt_tokens}/{reply.completion_tokens} | {reply.latency:.2f} с")


def chat(agent: Agent, model: str) -> int:
    print(f"Агент на {model}. Инструменты: {', '.join(agent.tools)}. Команды: /tools, /reset, /exit\n")
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
        if text == "/reset":
            agent.reset()
            print("История очищена.\n")
            continue
        if text == "/tools":
            for t in agent.tools.values():
                print(f"  {t.name}: {t.description}")
            print()
            continue
        try:
            reply = agent.run(text)
        except LLMError as e:
            print(f"Ошибка: {e}\n", file=sys.stderr)
            continue
        print_reply(reply, model)
        print()


def demo(client: LLMClient) -> int:
    report = []
    for s in SCENARIOS:
        print(f"=== {s.title}: {s.prompt}")
        agent = Agent(client, TOOLS, on_event=print_event)  # у каждого сценария своя чистая история
        try:
            reply = agent.run(s.prompt)
        except LLMError as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 1
        ok, note = check(s, reply)
        print_reply(reply, client.model)
        print(f"Проверка: {'✅' if ok else '❌'} {note}\n")
        report.append({"scenario": asdict(s), "ok": ok, "note": note, "reply": asdict(reply),
                       "history": agent.history})

    passed = sum(r["ok"] for r in report)
    print(f"Итог: {passed}/{len(report)} сценариев прошли проверку.")
    save_report(report, client.model)
    return 0 if passed == len(report) else 2


def save_report(report: list[dict], model: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, md_path = RESULTS_DIR / f"day6_{stamp}.json", RESULTS_DIR / f"day6_{stamp}.md"
    json_path.write_text(json.dumps({"model": model, "scenarios": report}, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    lines = [f"# День 6 · Первый агент — прогон {stamp}", "", f"Модель: `{model}`", "",
             "| Сценарий | Вызовы инструментов | Запросов к модели | Токены вход/выход | Время, с | Проверка |",
             "|---|---|---|---|---|---|"]
    for r in report:
        rep = r["reply"]
        calls = " → ".join(c["name"] for c in rep["tool_calls"]) or "—"
        lines.append(f"| {r['scenario']['title']} | {calls} | {rep['llm_calls']} | "
                     f"{rep['prompt_tokens']}/{rep['completion_tokens']} | {rep['latency']:.2f} | "
                     f"{'✅' if r['ok'] else '❌'} {r['note']} |")
    for r in report:
        rep = r["reply"]
        lines += ["", f"## {r['scenario']['title']}", "", f"**Запрос:** {r['scenario']['prompt']}", ""]
        for c in rep["tool_calls"]:
            lines.append(f"- шаг {c['step']}: `{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})` → "
                         f"`{json.dumps(c['result'], ensure_ascii=False)}`")
        if rep["tool_calls"]:
            lines.append("")
        lines += [f"**Ответ:** {rep['text'].strip()}"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Отчёт: {md_path.relative_to(Path(__file__).parent)}, {json_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="День 6: LLM-агент с инструментами.")
    parser.add_argument("prompt", nargs="?", help="один запрос; без него — чат в консоли")
    parser.add_argument("--demo", action="store_true", help="прогнать три проверочных сценария и сохранить отчёт")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--max-steps", type=int, default=8, help="предел запросов к модели за один ход")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    client = LLMClient(settings.base_url, settings.api_key, args.model or settings.model, settings.timeout)

    if args.demo:
        return demo(client)
    agent = Agent(client, TOOLS, max_steps=args.max_steps, on_event=print_event)
    if args.prompt:
        try:
            print_reply(agent.run(args.prompt), client.model)
        except LLMError as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 1
        return 0
    return chat(agent, client.model)


if __name__ == "__main__":
    sys.exit(main())
