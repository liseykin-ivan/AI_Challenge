"""День 7. Сохранение контекста.

Агент из Дня 6 (`Agent`, те же инструменты), который хранит историю диалога в JSON-файле и при запуске
загружает её обратно. В файл пишется весь список `messages` в том виде, в каком он уходит в API:
system, user, assistant с `tool_calls`, `role: "tool"` с результатами инструментов. Поэтому после
перезапуска модель получает ровно тот же контекст и продолжает диалог, как будто агент не выключался.

Файл истории: memory/<сессия>.json (сессия по умолчанию — default). Пишется после каждого завершённого
хода атомарно (временный файл + os.replace): если процесс убит посреди записи, старая версия остаётся целой.

    python day7_memory.py                          # чат; история сессии default подхватывается сама
    python day7_memory.py --session work           # другая сессия — другой файл
    python day7_memory.py "А сколько это в евро?"  # один ход с памятью и выход
    python day7_memory.py --new                    # начать заново (старая история уходит в архив)
    python day7_memory.py --demo                   # проверка: 3 запуска в отдельных процессах + отчёт в results/

Команды чата: /history, /tools, /reset, /exit.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import ConfigError, load_settings
from day6_agent import TOOLS, Agent, AgentReply, print_event, print_reply, short
from llm_client import LLMClient, LLMError

ROOT = Path(__file__).parent
MEMORY_DIR = ROOT / "memory"
RESULTS_DIR = ROOT / "results"
FORMAT_VERSION = 1


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- хранилище истории


class StoreError(Exception):
    """Файл истории не читается или не пишется; текст сразу понятен пользователю."""


@dataclass
class Saved:
    """Содержимое файла истории."""
    session_id: str  # x-opencode-session: тот же id после перезапуска — для провайдера это тот же диалог
    model: str
    created_at: str
    updated_at: str
    messages: list[dict]  # история целиком, в формате запроса к API
    turns: list[dict]  # служебная статистика по ходам (в API не отправляется)


class HistoryStore:
    """История диалога в одном JSON-файле."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Saved | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            messages = data["messages"]
            if not isinstance(messages, list) or not all(isinstance(m, dict) and "role" in m for m in messages):
                raise ValueError("messages — не список сообщений с полем role")
            return Saved(session_id=str(data.get("session_id") or ""), model=str(data.get("model") or ""),
                         created_at=data.get("created_at") or "", updated_at=data.get("updated_at") or "",
                         messages=messages, turns=list(data.get("turns") or []))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            raise StoreError(f"не удалось прочитать историю {self.path}: {e}. "
                             f"Исправьте файл или начните заново с --new (старый файл уйдёт в архив).")

    def save(self, saved: Saved) -> None:
        payload = {"format": FORMAT_VERSION, "session_id": saved.session_id, "model": saved.model,
                   "created_at": saved.created_at, "updated_at": saved.updated_at,
                   "messages": saved.messages, "turns": saved.turns}
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)  # атомарно: в файле всегда либо старая, либо новая версия
        except OSError as e:
            raise StoreError(f"не удалось сохранить историю в {self.path}: {e}")

    def archive(self) -> Path | None:
        """Переименовывает текущий файл в <имя>_<время>.json, чтобы /reset и --new ничего не теряли."""
        if not self.path.exists():
            return None
        target = self.path.with_name(f"{self.path.stem}_{datetime.now():%Y%m%d_%H%M%S}.json")
        try:
            os.replace(self.path, target)
        except OSError as e:
            raise StoreError(f"не удалось переместить {self.path} в архив: {e}")
        return target


def complete_prefix(messages: list[dict]) -> int:
    """Длина самого длинного начала истории, которое заканчивается завершённым ходом.

    Ход завершён, когда на каждый tool_call пришёл ответ `role: "tool"` и после этого модель дала
    ответ без tool_calls. Хвост без такого конца (файл правили руками или обрезали) API может не принять:
    сообщение assistant с tool_calls без ответов инструментов — ошибка 400 у большинства провайдеров.
    """
    good = 1 if messages and messages[0].get("role") == "system" else 0
    pending: set[str] = set()
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            ids = {c.get("id", "") for c in m.get("tool_calls") or []}
            if ids:
                pending |= ids
            elif not pending:
                good = i + 1
        elif role == "tool":
            pending.discard(m.get("tool_call_id", ""))
    return good


# ---------------------------------------------------------------- агент с памятью


@dataclass
class RestoreInfo:
    messages: int
    turns: int
    tool_calls: int
    dropped: int  # сколько сообщений незавершённого хода отброшено
    updated_at: str


class PersistentAgent(Agent):
    """Agent из Дня 6, который сохраняет историю после каждого хода и умеет её восстановить."""

    def __init__(self, client: LLMClient, tools, store: HistoryStore, **kwargs):
        self.store = store
        self.created_at = now_iso()
        self.turns: list[dict] = []
        self._ready = False  # Agent.__init__ вызывает reset(): в этот момент архивировать и сохранять нечего
        super().__init__(client, tools, **kwargs)
        self._ready = True

    # --- загрузка

    def restore(self, saved: Saved) -> RestoreInfo:
        keep = complete_prefix(saved.messages)
        messages = saved.messages[:keep]
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        self.history = messages
        self.created_at = saved.created_at or self.created_at
        user_count = sum(m.get("role") == "user" for m in messages)
        self.turns = saved.turns[:user_count]
        return RestoreInfo(messages=len(messages), turns=user_count,
                           tool_calls=sum(m.get("role") == "tool" for m in messages),
                           dropped=len(saved.messages) - keep, updated_at=saved.updated_at)

    # --- сохранение

    def save(self) -> None:
        self.store.save(Saved(session_id=self.client.session_id, model=self.client.model,
                              created_at=self.created_at, updated_at=now_iso(),
                              messages=self.history, turns=self.turns))

    def run(self, user_text: str) -> AgentReply:
        start = len(self.history)
        try:
            reply = super().run(user_text)  # при LLMError Agent сам откатывает историю к началу хода
        except KeyboardInterrupt:  # Ctrl+C посреди цикла: полуготовый ход в историю не попадает
            del self.history[start:]
            raise
        self.turns.append({"at": now_iso(), "user": user_text, "llm_calls": reply.llm_calls,
                           "tool_calls": len(reply.tool_calls), "prompt_tokens": reply.prompt_tokens,
                           "completion_tokens": reply.completion_tokens, "latency": round(reply.latency, 2),
                           "history_before": start})
        self.save()
        return reply

    def reset(self) -> None:
        super().reset()
        if not self._ready:
            return
        self.turns = []
        self.created_at = now_iso()
        archived = self.store.archive()
        self.save()
        self.on_event("reset", {"archived": archived})


# ---------------------------------------------------------------- вывод истории


def tool_names(history: list[dict]) -> dict[str, tuple[str, dict]]:
    """tool_call_id → (имя инструмента, аргументы) — чтобы печатать результат рядом с вызовом."""
    out = {}
    for m in history:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {"_raw": fn.get("arguments")}
            out[c.get("id", "")] = (fn.get("name", "?"), args)
    return out


def print_history(history: list[dict], last_turns: int | None = None) -> None:
    """Печатает диалог: реплики, вызовы инструментов с результатами, ответы. last_turns — только последние ходы."""
    starts = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if not starts:
        print("  (история пуста)")
        return
    if last_turns is not None:
        starts = starts[-last_turns:]
    calls = tool_names(history)
    for m in history[starts[0]:]:
        role = m.get("role")
        if role == "user":
            print(f"  Вы: {m.get('content', '')}")
        elif role == "tool":
            name, args = calls.get(m.get("tool_call_id", ""), ("?", {}))
            try:
                result = json.loads(m.get("content") or "null")
            except ValueError:
                result = m.get("content")
            print(f"    🔧 {name}({short(args)}) → {short(result)}")
        elif role == "assistant" and not m.get("tool_calls"):
            print(f"  Агент: {(m.get('content') or '').strip()}")


def print_restore(info: RestoreInfo, store: HistoryStore, agent: PersistentAgent) -> None:
    where = store.path.relative_to(ROOT) if store.path.is_relative_to(ROOT) else store.path
    if info.dropped:
        print(f"⚠️  В конце {where} незавершённый ход: отброшено сообщений — {info.dropped}.")
    if not info.turns:
        print(f"💾 История: {where} — новый диалог.")
        return
    when = info.updated_at.replace("T", " ")[:16]
    print(f"💾 История: {where} — восстановлено {info.messages} сообщений: {info.turns} ход(ов), "
          f"{info.tool_calls} результат(ов) инструментов. Последнее сохранение: {when}.")
    print("Последний обмен:")
    print_history(agent.history, last_turns=1)


def on_event(kind: str, data: dict) -> None:
    if kind == "reset":
        archived = data.get("archived")
        note = f" Прошлая история сохранена в {archived.name}." if archived else ""
        print(f"История очищена.{note}\n")
    else:
        print_event(kind, data)


# ---------------------------------------------------------------- чат


def chat(agent: PersistentAgent, model: str) -> int:
    print(f"Агент на {model}. Инструменты: {', '.join(agent.tools)}. "
          f"Команды: /history, /tools, /reset, /exit\n")
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
        try:
            if text == "/reset":
                agent.reset()
                continue
            if text == "/history":
                print_history(agent.history)
                print()
                continue
            if text == "/tools":
                for t in agent.tools.values():
                    print(f"  {t.name}: {t.description}")
                print()
                continue
            reply = agent.run(text)
        except LLMError as e:
            print(f"Ошибка: {e}\n", file=sys.stderr)
            continue
        except StoreError as e:  # ответ уже получен и есть в памяти процесса, но на диск не попал
            print(f"⚠️  {e}\n", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("\n(ход прерван — в историю он не попал)\n")
            continue
        print_reply(reply, model)
        print()


# ---------------------------------------------------------------- проверка с перезапусками


@dataclass(frozen=True)
class Step:
    title: str
    prompt: str


DEMO_STEPS = [
    Step("запуск 1 — вопрос с инструментами",
         "Который час сейчас в Токио? И сколько рублей стоят 150 долларов?"),
    Step("запуск 2 — после перезапуска, по памяти",
         "Какое точное время, с секундами, вернул инструмент, когда я спрашивал про Токио? "
         "Инструменты не вызывай — ответь по истории диалога."),
    Step("запуск 3 — после перезапуска, с результатом из прошлого",
         "Прибавь 10% к рублёвой сумме из моего первого вопроса."),
]


def turn_slice(history: list[dict], start: int) -> tuple[list[tuple[str, dict, object]], str]:
    """Вызовы инструментов и финальный ответ среди сообщений, добавленных начиная с start."""
    calls = tool_names(history)
    used, answer = [], ""
    for m in history[start:]:
        if m.get("role") == "tool":
            name, args = calls.get(m.get("tool_call_id", ""), ("?", {}))
            try:
                result = json.loads(m.get("content") or "null")
            except ValueError:
                result = m.get("content")
            used.append((name, args, result))
        elif m.get("role") == "assistant" and not m.get("tool_calls"):
            answer = m.get("content") or ""
    return used, answer


def digits(text: str) -> str:
    """«12 225,00» и «12225.0» → «12225» — для сравнения чисел в тексте ответа."""
    return re.sub(r"[\s  ]", "", text).replace(",", ".")


def check_step(n: int, used, answer: str, first_turn) -> tuple[bool, str]:
    names = [u[0] for u in used]
    if n == 0:
        missing = [t for t in ("get_current_time", "convert_currency") if t not in names]
        return (not missing, f"вызваны {', '.join(names)}" if not missing else f"не вызваны: {', '.join(missing)}")
    tokyo = next((r for nm, _, r in first_turn if nm == "get_current_time" and isinstance(r, dict)), {})
    rub = next((r for nm, _, r in first_turn if nm == "convert_currency" and isinstance(r, dict)), {})
    if n == 1:
        clock = (tokyo.get("datetime") or "").split(" ")[-1]  # HH:MM:SS из результата первого запуска
        if names:
            return False, f"модель снова вызвала инструменты ({', '.join(names)}) вместо ответа по памяти"
        ok = bool(clock) and clock in answer
        return ok, (f"без вызовов, в ответе время {clock} из результата запуска 1" if ok
                    else f"в ответе нет времени {clock or '?'} из запуска 1")
    value = rub.get("result")
    if value is None:
        return False, "в запуске 1 нет результата convert_currency"
    num = f"{float(value):.2f}".rstrip("0").rstrip(".")
    in_calc = any(nm == "calculator" and num in digits(str(a.get("expression", ""))) for nm, a, _ in used)
    expected = f"{float(value) * 1.1:.2f}".rstrip("0").rstrip(".")
    in_answer = expected in digits(answer)
    ok = in_calc or in_answer
    return ok, (f"взята сумма {num} ₽ из запуска 1" + (", calculator" if in_calc else "") + f" → {expected}"
                if ok else f"сумма {num} ₽ из запуска 1 не использована")


def demo(model: str | None, max_steps: int) -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = f"demo_{stamp}"
    store = HistoryStore(MEMORY_DIR / f"{session}.json")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    base = [sys.executable, str(Path(__file__).resolve()), "--session", session, "--max-steps", str(max_steps)]
    if model:
        base += ["--model", model]

    report, previous, first_turn = [], [], []
    for n, step in enumerate(DEMO_STEPS):
        print(f"\n{'=' * 70}\n=== {step.title}: новый процесс python\n{'=' * 70}")
        code = subprocess.run(base + [step.prompt], env=env).returncode
        if code != 0:
            print(f"Процесс завершился с кодом {code}.", file=sys.stderr)
            return 1
        saved = store.load()
        history = saved.messages if saved else []
        kept = history[:len(previous)] == previous  # прошлые сообщения на месте, а не начаты заново
        used, answer = turn_slice(history, len(previous))
        if n == 0:
            first_turn = used
        ok, note = check_step(n, used, answer, first_turn)
        ok = ok and kept and len(history) > len(previous)
        if not kept:
            note = "история не продолжена: прошлые сообщения изменились или пропали; " + note
        turn = saved.turns[-1] if saved and saved.turns else {}
        print(f"\nПроверка: {'✅' if ok else '❌'} {note}")
        report.append({"step": step.title, "prompt": step.prompt, "ok": ok, "note": note,
                       "messages_before": len(previous), "messages_after": len(history),
                       "tools": [{"name": nm, "arguments": a, "result": r} for nm, a, r in used],
                       "answer": answer, "turn": turn})
        previous = history

    passed = sum(r["ok"] for r in report)
    print(f"\nИтог: {passed}/{len(report)} проверок прошли. Файл истории: {store.path.relative_to(ROOT)}")
    save_report(report, store, stamp)
    return 0 if passed == len(report) else 2


def save_report(report: list[dict], store: HistoryStore, stamp: str) -> None:
    saved = store.load()
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path, md_path = RESULTS_DIR / f"day7_{stamp}.json", RESULTS_DIR / f"day7_{stamp}.md"
    json_path.write_text(json.dumps({"model": saved.model, "history_file": store.path.name, "steps": report,
                                     "saved_history": saved.__dict__}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    lines = [f"# День 7 · Сохранение контекста — прогон {stamp}", "",
             f"Модель: `{saved.model}`. Файл истории: `memory/{store.path.name}`, session_id `{saved.session_id}`.",
             "Каждый запуск — отдельный процесс `python day7_memory.py --session ...`: агент загружает историю "
             "из файла, делает ход и сохраняет её обратно.", "",
             "| Запуск | Сообщений в истории до → после | Вызовы инструментов | Токены вход/выход | Проверка |",
             "|---|---|---|---|---|"]
    for r in report:
        t = r["turn"]
        calls = " → ".join(c["name"] for c in r["tools"]) or "—"
        lines.append(f"| {r['step']} | {r['messages_before']} → {r['messages_after']} | {calls} | "
                     f"{t.get('prompt_tokens', '?')}/{t.get('completion_tokens', '?')} | "
                     f"{'✅' if r['ok'] else '❌'} {r['note']} |")
    for r in report:
        lines += ["", f"## {r['step']}", "", f"**Запрос:** {r['prompt']}", ""]
        for c in r["tools"]:
            lines.append(f"- `{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})` → "
                         f"`{json.dumps(c['result'], ensure_ascii=False)}`")
        if r["tools"]:
            lines.append("")
        lines.append(f"**Ответ:** {r['answer'].strip()}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Отчёт: {md_path.relative_to(ROOT)}, {json_path.name}")


# ---------------------------------------------------------------- запуск


def main() -> int:
    parser = argparse.ArgumentParser(description="День 7: агент, который сохраняет контекст между запусками.")
    parser.add_argument("prompt", nargs="?", help="один ход с памятью; без него — чат в консоли")
    parser.add_argument("--session", default="default", help="имя сессии: файл memory/<имя>.json")
    parser.add_argument("--new", action="store_true", help="начать новый диалог (старая история уходит в архив)")
    parser.add_argument("--demo", action="store_true", help="проверка с перезапусками в отдельных процессах")
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
    if args.demo:
        return demo(args.model, args.max_steps)

    store = HistoryStore(MEMORY_DIR / f"{args.session}.json")
    try:
        if args.new and (archived := store.archive()):
            print(f"Прошлая история сохранена в {archived.name}.")
        saved = store.load()
    except StoreError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    # тот же session_id, что до перезапуска: для провайдера это продолжение того же диалога
    client = LLMClient(settings.base_url, settings.api_key, args.model or settings.model, settings.timeout,
                       session_id=(saved.session_id or None) if saved else None)
    agent = PersistentAgent(client, TOOLS, store, max_steps=args.max_steps, on_event=on_event)
    info = agent.restore(saved) if saved else RestoreInfo(1, 0, 0, 0, "")
    print_restore(info, store, agent)
    print()

    if args.prompt:
        try:
            reply = agent.run(args.prompt)
        except (LLMError, StoreError) as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\n(ход прерван — в историю он не попал)")
            return 130
        print_reply(reply, client.model)
        return 0
    return chat(agent, client.model)


if __name__ == "__main__":
    from run_timer import run_timed
    sys.exit(run_timed(main))
