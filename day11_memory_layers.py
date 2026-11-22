"""День 11. Модель памяти ассистента: три слоя, которые хранятся отдельно.

Агент из Дня 8 (`TokenAgent`: инструменты, история в JSON, учёт токенов и стоимости) с явной моделью памяти.
Каждый слой — свой файл в memory/day11/<профиль>/ и своё правило, что из него уходит в промпт:

    краткосрочная  short_term.json        текущий диалог как есть (формат Дня 7).
                                          В промпт: последние N сообщений (--keep, 6) + текущий ход.
    рабочая        working.json,          данные активной задачи: цель, сроки, бюджет, этап, договорённости.
                   tasks/<id>.json        В промпт: все поля только активной задачи; задачи на паузе — нет.
    долговременная long_term.json         профиль, решения, знания — живут между задачами и диалогами.
                                          В промпт: профиль целиком; решения и знания — от новых к старым,
                                          пока укладываются в --long-budget токенов (1500).

Что и куда сохраняется, выбирается явно:
    - роутер памяти — Sticky Facts из Дня 10, разведённые по слоям. После каждого сообщения пользователя служебный
      запрос возвращает изменения по разделам {"profile" | "decisions" | "knowledge" | "task": {"set", "delete"}}.
      Таблица SECTION_LAYER в коде решает, в какой слой идёт раздел; слияние делает код; ответы ассистента
      в рабочую и долговременную память не попадают;
    - команды пользователя поверх роутера: /remember, /forget, /move;
    - каждое изменение пишется в журнал log.jsonl: слой, раздел, ключ, значение, откуда.
Факты Дня 10 импортируются в слои: --import-day10 [файл] (по умолчанию — последний results/day10_*.json).

    python day11_memory_layers.py                      # чат (профиль default)
    python day11_memory_layers.py --profile marina --keep 4
    python day11_memory_layers.py --import-day10       # разложить факты Дня 10 по слоям текущего профиля
    python day11_memory_layers.py --demo               # импорт, диалог Дня 9, очистка слоёв, смена задачи, отчёт

Команды чата: /memory, /context, /rules, /remember <раздел> <ключ> = <значение>, /forget <раздел> <ключ>,
/move <ключ> <раздел>, /clear short|working|long, /newchat, /task [new <название> | switch <id> | done | list],
/play [N], /probe, /experiment, /log, /tokens, /history, /exit.
"""
import argparse
import copy
import json
import os
import re
import shutil
import sys
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from config import ConfigError, load_settings
from day6_agent import TOOLS
from day7_memory import MEMORY_DIR, ROOT, HistoryStore, RestoreInfo, StoreError, now_iso, print_history, \
    print_restore
from day8_tokens import TokenAgent, TurnStats, Usage, clip, fmt, money, price_for, print_ledger
from day9_compression import AGENT_PROMPT, DETAILS, DIALOGUE, PROBE_SUFFIX, Detail, matches, normalize, watch_retries
from day10_strategies import DENIAL, HelperCall, StickyFacts, read_json, tail_start, write_json
from day10_strategies import on_event as day10_event
from llm_client import LLMClient, LLMError

RESULTS_DIR = ROOT / "results"
DAY_DIR = MEMORY_DIR / "day11"
NAME_RE = re.compile(r"[\w-]+")

LAYERS = ("short", "working", "long")
LAYER_TITLES = {"short": "краткосрочная", "working": "рабочая", "long": "долговременная"}
LAYER_ALIASES = {"short": "short", "s": "short", "краткосрочная": "short", "working": "working", "w": "working",
                 "рабочая": "working", "long": "long", "l": "long", "долговременная": "long"}
# Разделы, в которые пишет роутер, и в какой слой уходит каждый. Это и есть явный выбор «что куда»:
# поменять слой раздела — поменять одну строку здесь.
SECTIONS = ("profile", "decisions", "knowledge", "task")
SECTION_LAYER = {"profile": "long", "decisions": "long", "knowledge": "long", "task": "working"}
SECTION_TITLES = {"profile": "профиль", "decisions": "решения", "knowledge": "знания", "task": "задача"}
SECTION_ALIASES = {"profile": "profile", "профиль": "profile", "decisions": "decisions", "decision": "decisions",
                   "решения": "decisions", "решение": "decisions", "knowledge": "knowledge", "знания": "knowledge",
                   "task": "task", "задача": "task", "working": "task", "рабочая": "task"}
LONG_SECTIONS = ("profile", "decisions", "knowledge")

RULES = {
    "short": {
        "stores": "текущий диалог: все сообщения как есть, с вызовами инструментов",
        "file": "short_term.json",
        "writes": "автоматически: каждое сообщение диалога",
        "prompt": "последние {keep} сообщений (граница — начало хода) + текущий ход; старше — не отправляются",
        "cleared": "/newchat или /clear short — диалог уходит в архив (сначала роутер разбирает то, что пропустил), "
                   "рабочая и долговременная память остаются",
    },
    "working": {
        "stores": "данные активной задачи: название и цель, объём, сроки и вехи, бюджет, этап, договорённости",
        "file": "working.json (активная задача), tasks/<id>.json (на паузе и закрытые)",
        "writes": "роутер — раздел task; /remember task …",
        "prompt": "все поля только активной задачи; задачи на паузе и закрытые — не отправляются",
        "cleared": "/task done или /clear working; /task new ставит текущую задачу на паузу",
    },
    "long": {
        "stores": "профиль пользователя, принятые решения, устойчивые знания о компании и окружении",
        "file": "long_term.json",
        "writes": "роутер — разделы profile, decisions, knowledge; /remember …; импорт фактов Дня 10",
        "prompt": "профиль — всегда целиком; решения и знания — от новых к старым, пока влезают в {budget} токенов",
        "cleared": "только явно: /clear long или /forget по ключу",
    },
}
NOT_SAVED = ("вопросы и просьбы пользователя, советы и ответы ассистента в рабочую и долговременную память не идут — "
             "они живут только в краткосрочной, пока не выпадут из хвоста")

ROUTER_PROMPT = (
    "Ты — диспетчер памяти ассистента. Старых сообщений ассистент не видит: всё важное он знает только из памяти. "
    "Память разделена на разделы, и ты решаешь, что из новых сообщений пользователя куда записать:\n"
    "- profile — о самом пользователе: имя, должность и роль, компания, как ему удобно получать ответы. "
    "Хранится всегда, во всех задачах;\n"
    "- decisions — решения, которые пользователь объявил принятыми: что выбрали и вместо чего, от чего отказались, "
    "что отложили. Советы ассистента — не решения. Хранятся всегда;\n"
    "- knowledge — устойчивые знания о компании и её окружении, которые останутся верными и после текущей задачи: "
    "сколько точек и где, люди и их роли, юридические и технические ограничения, действующие договоры. "
    "Хранятся всегда;\n"
    "- task — данные текущей задачи: её название и цель, объём, сроки и вехи, бюджет и его части, этап, "
    "договорённости и открытые вопросы именно этой задачи. Устаревают, когда задача закрыта.\n"
    "Тебе дают текущую память и новые сообщения пользователя. Верни JSON вида "
    '{"profile": {"set": {}, "delete": []}, "decisions": {"set": {}, "delete": []}, '
    '"knowledge": {"set": {}, "delete": []}, "task": {"set": {}, "delete": []}}:\n'
    "- set — новые записи и записи, значение которых изменилось; delete — записи, которые пользователь отменил;\n"
    "- значения пиши точно, как у пользователя, и коротко (до 20 слов). Если значение изменилось — новое, "
    "а в скобках старое: «2,7 млн ₽ (было 2,4 млн ₽)»;\n"
    "- меняй число, только если пользователь прямо сказал, что прежнее значение изменилось. Другое число в другом "
    "смысле (порядковый номер, дата, номер следующей точки) — не изменение;\n"
    "- ключи короткие, по-русски, в нижнем регистре через подчёркивание. Если запись уже есть — обнови тот же ключ "
    "в том же разделе, не заводи новый. Один ключ — в одном разделе: переносить запись можно, "
    "только если она явно лежит не там;\n"
    "- вопросы пользователя и просьбы что-то объяснить — не записи. Ничего не придумывай;\n"
    "- если записывать нечего — верни пустые set и delete.\n"
    "Верни только JSON.")

IMPORT_NOTE = ("Это не новое сообщение, а готовые факты прошлого разговора (ключ: значение). Разложи каждый факт "
               "по разделам, ключи и значения сохрани как есть.")

MEMORY_INTRO = ("\n\n# Память ассистента\n"
                "Старые сообщения диалога и прошлые разговоры тебе не отправлены — всё важное из них записано здесь. "
                "Это пользователь сообщил тебе раньше; опирайся на это, как на его слова.\n")
LONG_HEADER = "\n## Долговременная память — из всех разговоров и задач\n"
WORKING_HEADER = "\n## Рабочая память — текущая задача «{title}»\n"
NO_TASK = "Текущая задача"


# ---------------------------------------------------------------- слои памяти


def stamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def archive_file(path: Path) -> Path | None:
    """<имя>.json → <имя>_<время>.json: очистка слоя ничего не теряет безвозвратно."""
    if not path.exists():
        return None
    target = path.with_name(f"{path.stem}_{stamp_now()}{path.suffix}")
    n = 1
    while target.exists():
        n += 1
        target = path.with_name(f"{path.stem}_{stamp_now()}_{n}{path.suffix}")
    try:
        os.replace(path, target)
    except OSError as e:
        raise StoreError(f"не удалось переместить {path} в архив: {e}")
    return target


def entries(raw: dict | None) -> dict[str, dict]:
    """Записи слоя из файла: ключ → {value, at, source, task}; мусор отбрасывается."""
    out = {}
    for k, v in (raw or {}).items():
        if isinstance(v, dict) and str(v.get("value") or "").strip():
            out[str(k)] = {"value": str(v["value"]), "at": str(v.get("at") or ""), "source": str(v.get("source") or ""),
                           "task": str(v.get("task") or "")}
        elif isinstance(v, str) and v.strip():  # плоский словарь «ключ: значение» — например, руками
            out[str(k)] = {"value": v, "at": "", "source": "файл", "task": ""}
    return out


class LongTermMemory:
    """Долговременная память: профиль, решения, знания. Один файл, живёт между диалогами и задачами."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict[str, dict]] = {s: {} for s in LONG_SECTIONS}

    def load(self) -> None:
        self.data = {s: {} for s in LONG_SECTIONS}
        if self.path.exists():
            raw = read_json(self.path)
            for s in LONG_SECTIONS:
                self.data[s] = entries(raw.get(s))

    def save(self) -> None:
        write_json(self.path, {"format": 1, "updated_at": now_iso(), **self.data})

    def count(self) -> int:
        return sum(len(v) for v in self.data.values())

    def clear(self) -> Path | None:
        archived = archive_file(self.path)
        self.data = {s: {} for s in LONG_SECTIONS}
        return archived


class WorkingMemory:
    """Рабочая память: активная задача в working.json; задачи на паузе и закрытые — в tasks/<id>.json."""

    def __init__(self, root: Path):
        self.path = root / "working.json"
        self.tasks_dir = root / "tasks"
        self.task: dict = {}
        self.task = self.blank(NO_TASK)

    def blank(self, title: str) -> dict:
        return {"id": self.next_id(), "title": title, "status": "active", "created_at": now_iso(), "data": {}}

    def next_id(self) -> str:
        used = [int(p.stem) for p in self.tasks_dir.glob("*.json") if p.stem.isdigit()] if self.tasks_dir.exists() else []
        current = self.task
        if current and str(current.get("id", "")).isdigit():
            used.append(int(current["id"]))
        return str(max(used, default=0) + 1)

    @property
    def data(self) -> dict[str, dict]:
        return self.task["data"]

    def load(self) -> None:
        if not self.path.exists():
            self.task = {}
            self.task = self.blank(NO_TASK)
            return
        raw = read_json(self.path)
        self.task = {"id": str(raw.get("id") or self.next_id()), "title": str(raw.get("title") or NO_TASK),
                     "status": "active", "created_at": str(raw.get("created_at") or now_iso()),
                     "data": entries(raw.get("data"))}

    def save(self) -> None:
        write_json(self.path, {"format": 1, "updated_at": now_iso(), **self.task})

    def park(self, status: str) -> Path:
        """Убрать активную задачу из рабочей памяти в tasks/<id>.json (paused | done)."""
        path = self.tasks_dir / f"{self.task['id']}.json"
        write_json(path, {"format": 1, "updated_at": now_iso(), **self.task, "status": status})
        return path

    def new(self, title: str) -> dict:
        if self.task["data"] or self.task["title"] != NO_TASK:
            self.park("paused")
            self.task = self.blank(title)
        else:  # пустая задача по умолчанию — просто даём ей название
            self.task = dict(self.blank(title), id=self.task["id"])
        self.save()
        return self.task

    def switch(self, task_id: str) -> dict:
        path = self.tasks_dir / f"{task_id}.json"
        if task_id == self.task["id"]:
            return self.task
        if not path.exists():
            raise ValueError(f"нет задачи {task_id}. Есть: " + ", ".join(t["id"] for t in self.list()))
        raw = read_json(path)
        self.park("paused")
        self.task = {"id": task_id, "title": str(raw.get("title") or NO_TASK), "status": "active",
                     "created_at": str(raw.get("created_at") or now_iso()), "data": entries(raw.get("data"))}
        self.save()
        return self.task

    def done(self) -> dict:
        closed = dict(self.task)
        self.park("done")
        self.task = self.blank(NO_TASK)
        self.save()
        return closed

    def clear(self) -> Path | None:
        """Забыть данные активной задачи; копия остаётся в tasks/<id>_cleared_<время>.json."""
        archived = None
        if self.task["data"]:
            archived = self.tasks_dir / f"{self.task['id']}_cleared_{stamp_now()}.json"
            write_json(archived, {"format": 1, "updated_at": now_iso(), **self.task, "status": "cleared"})
        self.task = dict(self.task, data={})
        self.save()
        return archived

    def list(self) -> list[dict]:
        out = [{"id": self.task["id"], "title": self.task["title"], "status": "active", "fields": len(self.data)}]
        if self.tasks_dir.exists():
            for p in sorted(self.tasks_dir.glob("*.json"), key=lambda p: (len(p.stem), p.stem)):
                if not p.stem.isdigit() or p.stem == self.task["id"]:
                    continue
                try:
                    raw = read_json(p)
                except StoreError:
                    continue
                out.append({"id": p.stem, "title": raw.get("title") or NO_TASK, "status": raw.get("status") or "?",
                            "fields": len(raw.get("data") or {})})
        return out


class MemoryLog:
    """Журнал записей в память (log.jsonl). Записи хода копятся в pending и пишутся после успешного хода —
    если ход откатился, его записи в журнал не попадают."""

    def __init__(self, path: Path):
        self.path = path
        self.pending: list[dict] = []

    def add(self, op: str, section: str, key: str = "", value: str = "", source: str = "", **extra) -> dict:
        layer = SECTION_LAYER.get(section, section)
        item = {"at": now_iso(), "op": op, "layer": layer, "section": section, "key": key, "value": value,
                "source": source, **{k: v for k, v in extra.items() if v not in (None, "")}}
        self.pending.append(item)
        return item

    def flush(self) -> None:
        if not self.pending:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                for item in self.pending:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except OSError as e:
            raise StoreError(f"не удалось дописать журнал {self.path}: {e}")
        self.pending = []

    def drop(self) -> None:
        self.pending = []

    def tail(self, n: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(x) for x in lines if x.strip()]


def parse_route(text: str) -> dict[str, tuple[dict[str, str], list[str]]]:
    """Ответ роутера → {раздел: (set, delete)}. Каждый раздел разбирает парсер Sticky Facts из Дня 10."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("нет JSON")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("ожидался объект")
    out = {}
    for s in SECTIONS:
        part = data.get(s) or {}
        if not isinstance(part, dict):
            raise ValueError(f"{s} — не объект")
        out[s] = StickyFacts.parse(json.dumps(part, ensure_ascii=False)) if part else ({}, [])
    return out


def op_label(item: dict) -> str:
    if item["op"] == "clear":
        return f"🧹 очищен слой: {LAYER_TITLES.get(item['layer'], item['layer'])}"
    if item["op"] == "task":
        return f"📋 задача {item['key']} «{item['value']}» — {item.get('status', '')}"
    where = f"{LAYER_TITLES.get(item['layer'], item['layer'])} · {SECTION_TITLES.get(item['section'], item['section'])}"
    op = item["op"]
    if op == "move":
        src = item.get("from", "")
        where = f"{SECTION_TITLES.get(src, src)} → {where}"
    sign = {"add": "+", "update": "~", "move": "↪", "delete": "−", "clear": "🧹"}.get(op, op)
    value = f" = {clip(item['value'], 80)}" if item.get("value") and op != "delete" else ""
    old = (f" (было: {clip(item['old'], 40)})" if item.get("old") and op == "update"
           and normalize(item["old"]) not in normalize(item.get("value") or "") else "")
    return f"{sign} [{where}] {item['key']}{value}{old}"


# ---------------------------------------------------------------- агент


class LayeredAgent(TokenAgent):
    """TokenAgent из Дня 8 с тремя слоями памяти. Краткосрочная — его же история (short_term.json),
    рабочая и долговременная — отдельные файлы; запрос к модели собирает build() по правилам слоёв."""

    def __init__(self, client: LLMClient, tools, root: Path, *, keep_last: int = 6, long_budget: int = 1500,
                 helper_client: LLMClient | None = None, **kwargs):
        self.root = root
        self.keep_last = max(keep_last, 0)
        self.long_budget = max(long_budget, 0)
        self.long = LongTermMemory(root / "long_term.json")
        self.working = WorkingMemory(root)
        self.log = MemoryLog(root / "log.jsonl")
        self.router_path = root / "router.json"
        self.router_upto = 1  # сообщения пользователя из history[1:router_upto] уже разобраны роутером
        self.router_calls: list[HelperCall] = []
        self.dialogue_start = 0  # router_calls[dialogue_start:] — запросы роутера в текущем диалоге
        self.helper_client = helper_client or client
        self.layer_stats: list[dict] = []  # токены по слоям для каждого хода текущего диалога
        self.notes: list[str] = []
        self._turn_start: int | None = None
        self._parts: dict | None = None
        kwargs.setdefault("system_prompt", AGENT_PROMPT)
        super().__init__(client, tools, HistoryStore(root / "short_term.json"), **kwargs)

    # --- загрузка и сохранение: каждый слой — свой файл

    def load(self) -> RestoreInfo:
        self.long.load()
        self.working.load()
        saved = self.store.load()
        info = self.restore(saved) if saved else RestoreInfo(1, 0, 0, 0, "")
        if saved and saved.session_id:
            self.client.session_id = saved.session_id
        router = read_json(self.router_path) if self.router_path.exists() else {}
        upto = int(router.get("upto") or 1)
        if not 1 <= upto <= len(self.history):
            self.notes.append(f"роутер: учтено до сообщения {upto}, а в диалоге {len(self.history)} — "
                              f"считаю диалог разобранным")
            upto = len(self.history)
        self.router_upto = upto
        known = HelperCall.__dataclass_fields__
        self.router_calls = [HelperCall(**{k: v for k, v in c.items() if k in known}) for c in router.get("calls") or []]
        self.dialogue_start = min(int(router.get("dialogue_start") or 0), len(self.router_calls))
        self.layer_stats = [t["layers"] for t in self.turns if isinstance(t.get("layers"), dict)]
        return info

    def save_memory(self) -> None:
        self.long.save()
        self.working.save()
        write_json(self.router_path, {"format": 1, "updated_at": now_iso(), "upto": self.router_upto,
                                      "dialogue_start": self.dialogue_start,
                                      "calls": [asdict(c) for c in self.router_calls]})
        self.log.flush()

    def save(self) -> None:
        super().save()  # краткосрочная — история диалога
        self.save_memory()

    # --- запись в память

    def section(self, name: str) -> dict[str, dict]:
        return self.working.data if name == "task" else self.long.data[name]

    def find(self, key: str) -> str | None:
        return next((s for s in SECTIONS if key in self.section(s)), None)

    def write(self, section: str, key: str, value: str, source: str) -> list[dict]:
        """Записать ключ в раздел. Один ключ — в одном разделе: если он лежал в другом, он переносится."""
        ops, moved = [], None
        for other in SECTIONS:
            if other != section and key in self.section(other):
                moved = other
                self.section(other).pop(key)
        target = self.section(section)
        old = target.get(key)
        if old and old["value"] == value and not moved:
            return []
        target[key] = {"value": value, "at": now_iso(), "source": source, "task": self.working.task["title"]}
        op = "move" if moved else ("update" if old else "add")
        ops.append(self.log.add(op, section, key, value, source, old=old["value"] if old else None, **{"from": moved}))
        return ops

    def forget(self, section: str, key: str, source: str) -> list[dict]:
        old = self.section(section).pop(key, None)
        return [self.log.add("delete", section, key, old["value"], source)] if old else []

    def apply(self, changes: dict, source: str) -> list[dict]:
        ops = []
        for s in SECTIONS:
            update, delete = changes.get(s) or ({}, [])
            for key, value in update.items():
                ops += self.write(s, key, value, source)
            for key in delete:
                ops += self.forget(s, key, source)
        return ops

    def memory_json(self) -> dict:
        return {s: {k: e["value"] for k, e in self.section(s).items()} for s in SECTIONS}

    def route(self, texts: list[str], source: str, imported: bool = False) -> list[dict]:
        """Служебный запрос роутера: новые сообщения пользователя (и пропущенные, если роутер падал) → изменения
        по разделам. Возвращает журнал применённых изменений."""
        end = self.turn_end()
        pending = [] if imported else [m.get("content") or "" for m in self.history[self.router_upto:end]
                                       if m.get("role") == "user"]
        pending += [t for t in texts if t]
        if not pending:
            return []
        numbered = "\n\n".join(pending) if imported else "\n\n".join(f"{i}. {t}" for i, t in enumerate(pending, 1))
        user = (f"Текущая задача: «{self.working.task['title']}»\n\nТекущая память:\n"
                f"{json.dumps(self.memory_json(), ensure_ascii=False, indent=1)}\n\n"
                + (f"{IMPORT_NOTE}\n\n{numbered}" if imported else f"Новые сообщения пользователя:\n\n{numbered}"))
        result = self.helper_client.chat([{"role": "system", "content": ROUTER_PROMPT},
                                          {"role": "user", "content": user}], temperature=0)
        usage = Usage.parse(result.usage)
        priced = price_for(self.helper_client.model)
        call = HelperCall(strategy="router", before_turn=len(self.ledger) + (0 if imported else 1),
                          messages=len(pending), prompt=usage.prompt, completion=usage.completion, cached=usage.cached,
                          cost=usage.cost(priced[0] if priced else None), latency=result.latency, at=now_iso())
        self.router_calls.append(call)
        try:
            changes = parse_route(result.text)
        except ValueError as e:
            call.note = f"ответ не разобран: {e}"
            raise LLMError(f"ответ роутера не разобран ({e}): {clip(result.text, 120)!r}")
        ops = self.apply(changes, source)
        if not imported:
            self.router_upto = end + (1 if self._turn_start is not None else 0)
        counts = {layer: sum(o["layer"] == layer for o in ops) for layer in ("working", "long")}
        call.note = (f"изменений {len(ops)}: рабочая {counts['working']}, долговременная {counts['long']}"
                     if ops else "изменений нет")
        self.on_event("route", {"call": call, "ops": ops})
        return ops

    def pending(self) -> int:
        """Сколько сообщений пользователя роутер ещё не разобрал (его запрос падал)."""
        return sum(m.get("role") == "user" for m in self.history[self.router_upto:self.turn_end()])

    def catch_up(self) -> list[dict]:
        """Разобрать сообщения пользователя, которые роутер пропустил (служебный запрос падал)."""
        return self.route([], "догон пропущенного")

    def flush_pending(self, before: str) -> None:
        """Перед очисткой диалога и сменой задачи: пропущенное роутером — сначала в память, иначе оно потеряется
        вместе с диалогом или ляжет не в ту задачу. Если роутер снова падает — действие не выполняется."""
        if not self.pending():
            return
        try:
            self.catch_up()
        except LLMError as e:
            self.log.drop()
            raise LLMError(f"{before}: отменено — роутер не разобрал {self.pending()} сообщ. пользователя ({e}). "
                           f"Повторите команду: иначе эти сообщения не попадут в память.")
        self.save_memory()

    # --- что уходит в модель: правило каждого слоя

    def turn_end(self) -> int:
        return len(self.history) if self._turn_start is None else self._turn_start

    def long_block(self) -> tuple[str, dict]:
        """Профиль — целиком; решения и знания — от новых к старым, пока влезают в long_budget токенов."""
        est = self.estimator
        active = self.working.task["title"]
        line = lambda k, e: f"- {k}: {e['value']}" + (  # noqa: E731  из другой задачи — с пометкой, из какой
            f" (задача «{e['task']}»)" if e.get("task") and e["task"] not in (active, NO_TASK) else "")
        profile = [f"- {k}: {e['value']}" for k, e in self.long.data["profile"].items()]
        used = est.text("\n".join(profile))
        rest = sorted(((s, k, e) for s in ("decisions", "knowledge") for k, e in self.long.data[s].items()),
                      key=lambda x: x[2].get("at") or "", reverse=True)
        chosen, skipped = set(), []
        for s, k, e in rest:
            cost = est.text(line(k, e))
            if used + cost > self.long_budget:
                skipped.append(f"{s}.{k}")
                continue
            used += cost
            chosen.add((s, k))
        parts = []
        if profile:
            parts.append("### Профиль пользователя\n" + "\n".join(profile))
        for s, title in (("decisions", "Принятые решения"), ("knowledge", "Знания о компании и окружении")):
            items = [line(k, e) for k, e in self.long.data[s].items() if (s, k) in chosen]
            if items:
                parts.append(f"### {title}\n" + "\n".join(items))
        info = {"profile": len(profile), "decisions": sum(s == "decisions" for s, _ in chosen),
                "knowledge": sum(s == "knowledge" for s, _ in chosen), "skipped": skipped}
        return (LONG_HEADER + "\n\n".join(parts) + "\n") if parts else "", info

    def working_block(self) -> str:
        """Только активная задача, все её поля."""
        t = self.working.task
        if not t["data"]:
            return ""
        return WORKING_HEADER.format(title=t["title"]) + "\n".join(f"- {k}: {e['value']}" for k, e in t["data"].items()) + "\n"

    def build(self, end: int | None = None, layers=LAYERS, tools: list[dict] | None = None) -> tuple[list[dict], int, dict]:
        """(сообщения запроса, сколько сообщений диалога не отправлено, ≈токены по частям запроса)."""
        end = self.turn_end() if end is None else end
        h, est = self.history, self.estimator
        long_text, long_info = self.long_block() if "long" in layers else ("", {})
        work_text = self.working_block() if "working" in layers else ""
        intro = MEMORY_INTRO if long_text or work_text else ""
        base = h[0].get("content") or ""
        system = dict(h[0], content=base + intro + long_text + work_text)
        start = tail_start(h, end, self.keep_last) if "short" in layers else end
        tail, current = h[start:end], h[end:]
        raw = {"system": est.PER_REQUEST + est.PER_MESSAGE + est.raw_text(base + intro)
                         + (est.raw_text(json.dumps(tools, ensure_ascii=False)) if tools else 0),
               "long": est.raw_text(long_text), "working": est.raw_text(work_text),
               "short": sum(est.raw_message(m, False) for m in tail), "current": sum(est.raw_message(m) for m in current)}
        parts = {"raw": raw, "estimate": {k: est.apply(v) for k, v in raw.items()}, "tail_messages": len(tail),
                 "dropped": max(end - 1 - len(tail), 0), "long_info": long_info}
        return [system] + tail + current, parts["dropped"], parts

    def context(self, layers=LAYERS) -> list[dict]:
        return self.build(layers=layers)[0]

    def _select(self, schemas):
        messages, dropped, parts = self.build(tools=schemas)
        if self._turn is not None and not self._turn.calls:  # первый запрос хода — его и раскладываем по слоям
            self._parts = parts
        return messages, dropped

    def layer_record(self, turn: TurnStats) -> dict:
        """Токены по слоям в первом запросе хода: оценка по частям, отмасштабированная к точному prompt_tokens."""
        parts = self._parts or {}
        raw = parts.get("raw") or {}
        actual = turn.calls[0].prompt if turn.calls else 0
        total = sum(raw.values())
        scaled = {k: round(v * actual / total) for k, v in raw.items()} if total and actual else {}
        return {"turn": turn.n, "prompt": actual, "cached": turn.calls[0].cached if turn.calls else 0,
                "tokens": scaled, "tail_messages": parts.get("tail_messages", 0), "dropped": parts.get("dropped", 0),
                "long_info": parts.get("long_info") or {}, "working_fields": len(self.working.data),
                "long_entries": self.long.count(), "history_messages": len(self.history) - 1,
                "router": [asdict(c) for c in self.dialogue_calls() if c.before_turn == turn.n]}

    def dialogue_calls(self) -> list[HelperCall]:
        return self.router_calls[self.dialogue_start:]

    # --- ход

    def snapshot(self) -> dict:
        return copy.deepcopy({"long": self.long.data, "task": self.working.task, "upto": self.router_upto})

    def rollback(self, snap: dict) -> None:
        self.long.data, self.working.task, self.router_upto = snap["long"], snap["task"], snap["upto"]
        self.log.drop()

    def run(self, user_text: str):
        self._turn_start, self._parts = len(self.history), None
        snap = self.snapshot()
        try:
            try:
                self.route([user_text], f"диалог, ход {len(self.ledger) + 1}")
            except LLMError as e:  # без свежей памяти агент всё равно ответит, а роутер догонит в следующий ход
                self.log.drop()
                self.on_event("route_error", {"error": str(e)})
            reply = super().run(user_text)
        except BaseException:  # ход не прошёл (или Ctrl+C): то, что роутер успел записать из него, откатывается
            self.rollback(snap)
            self.save_memory()
            raise
        finally:
            self._turn_start = None
        stats = self.layer_record(self.ledger[-1])
        self.layer_stats.append(stats)
        self.turns[-1]["layers"] = stats
        self.save()
        return reply

    # --- очистка слоёв и задачи

    def reset(self) -> None:
        """Новый диалог: краткосрочная память уходит в архив; рабочая и долговременная остаются."""
        self.router_upto = 1
        self.layer_stats = []
        self.dialogue_start = len(self.router_calls)
        if getattr(self, "_ready", False):
            self.client.session_id = uuid.uuid4().hex  # новый диалог — новая сессия у провайдера
        super().reset()

    def clear(self, layer: str) -> str:
        if layer == "short":
            self.flush_pending("очищение диалога")
            messages = len(self.history) - 1
            self.reset()
            what = f"диалог ({messages} сообщ.) ушёл в архив, начат новый"
        elif layer == "working":
            archived = self.working.clear()
            what = f"данные задачи «{self.working.task['title']}» очищены" + (
                f", копия — {archived.relative_to(self.root)}" if archived else "")
        elif layer == "long":
            archived = self.long.clear()
            what = "профиль, решения и знания очищены" + (f", копия — {archived.name}" if archived else "")
        else:
            raise ValueError(f"нет слоя «{layer}». Есть: short, working, long")
        self.log.add("clear", layer, source="команда пользователя")
        self.save_memory()
        return what

    def new_task(self, title: str) -> dict:
        self.flush_pending("смена задачи")
        task = self.working.new(title)
        self.log.add("task", "task", task["id"], title, "команда пользователя", status="new")
        self.save_memory()
        return task

    def switch_task(self, task_id: str) -> dict:
        self.flush_pending("смена задачи")
        task = self.working.switch(task_id)
        self.log.add("task", "task", task["id"], task["title"], "команда пользователя", status="switch")
        self.save_memory()
        return task

    def router_totals(self) -> dict:
        costs = [c.cost for c in self.router_calls]
        return {"count": len(self.router_calls), "prompt": sum(c.prompt for c in self.router_calls),
                "completion": sum(c.completion for c in self.router_calls),
                "cost": None if any(c is None for c in costs) else sum(costs)}


# ---------------------------------------------------------------- вывод


def on_event(kind: str, data: dict) -> None:
    if kind == "route":
        c: HelperCall = data["call"]
        ops = data["ops"]
        print(f"  🧠 роутер памяти: {c.note} | вход {fmt(c.prompt)}, выход {fmt(c.completion)}, {c.latency:.1f} с, "
              f"{money(c.cost)}")
        for item in ops:
            print(f"     {op_label(item)}")
    elif kind == "route_error":
        print(f"  ⚠️  роутер памяти не сработал: {data['error']} — отвечу с прежней памятью, догоню в следующий ход")
    else:
        day10_event(kind, data)


def quiet_event(kind: str, data: dict) -> None:
    if kind in ("tokens", "llm", "truncate", "reset"):
        return
    on_event(kind, data)


def layer_line(tokens: dict) -> str:
    names = (("system", "system"), ("long", "долговр."), ("working", "рабочая"), ("short", "краткоср."),
             ("current", "ход"))
    return " · ".join(f"{label} {fmt(tokens.get(k, 0))}" for k, label in names)


def print_turn(agent: LayeredAgent, turn: TurnStats) -> None:
    if not turn.calls:
        print(f"  ❌ ответа нет: {turn.error}")
        return
    s = agent.layer_stats[-1] if agent.layer_stats and agent.layer_stats[-1]["turn"] == turn.n else None
    cached = f", из кэша {fmt(turn.calls[0].cached)}" if turn.calls[0].cached else ""
    print(f"  📊 ход {turn.n}: вход {fmt(turn.calls[0].prompt)}{cached} = "
          + (layer_line(s["tokens"]) if s else "?")
          + (f" | хвост {s['tail_messages']} сообщ., не отправлено {s['dropped']}" if s else ""))
    router = [c for c in agent.dialogue_calls() if c.before_turn == turn.n]
    print(f"     выход {fmt(turn.completion)}, запросов {len(turn.calls)} | {money(turn.cost)}"
          + (f" + роутер {money(sum(c.cost or 0 for c in router))}" if router else ""))


def print_memory(agent: LayeredAgent) -> None:
    est = agent.estimator
    h = agent.history
    tail = tail_start(h, len(h), agent.keep_last)
    print(f"  ① краткосрочная — {agent.store.path.name}: {len(h) - 1} сообщ. диалога, в промпт — последние "
          f"{len(h) - tail} (≈{fmt(est.apply(sum(est.raw_message(m, False) for m in h[tail:])))} токенов)")
    t = agent.working.task
    print(f"  ② рабочая — working.json: задача {t['id']} «{t['title']}», полей {len(t['data'])}"
          f" (≈{fmt(est.text(agent.working_block()))} токенов)")
    for k, e in t["data"].items():
        print(f"     · {k}: {e['value']}  ⟵ {e['source']}")
    others = [x for x in agent.working.list() if x["status"] != "active"]
    if others:
        print("     задачи вне промпта: " + "; ".join(f"{x['id']} «{x['title']}» ({x['status']}, {x['fields']} пол.)"
                                                    for x in others))
    text, info = agent.long_block()
    print(f"  ③ долговременная — long_term.json: записей {agent.long.count()} (≈{fmt(est.text(text))} токенов)"
          + (f"; не влезли в бюджет {agent.long_budget}: {', '.join(info['skipped'])}" if info.get("skipped") else ""))
    for s in LONG_SECTIONS:
        for k, e in agent.long.data[s].items():
            print(f"     · [{SECTION_TITLES[s]}] {k}: {e['value']}  ⟵ {e['source']}")


def print_context(agent: LayeredAgent) -> None:
    schemas = [t.schema() for t in agent.tools.values()]
    messages, dropped, parts = agent.build(end=len(agent.history), tools=schemas)
    e = parts["estimate"]
    total = sum(e.values())
    print(f"  Следующий запрос ≈{fmt(total)} токенов (без нового сообщения):")
    rows = [("system", "системный промпт и схемы инструментов", "всегда"),
            ("long", "долговременная", RULES["long"]["prompt"].format(budget=agent.long_budget)),
            ("working", "рабочая", RULES["working"]["prompt"]),
            ("short", "краткосрочная", RULES["short"]["prompt"].format(keep=agent.keep_last))]
    for k, title, rule in rows:
        print(f"   {title:<38} ≈{fmt(e[k]):>6}  — {rule}")
    print(f"   в хвосте {parts['tail_messages']} сообщ., не отправлено {dropped} старых сообщений диалога")


def print_rules(agent: LayeredAgent) -> None:
    for layer in LAYERS:
        r = RULES[layer]
        print(f"  {LAYER_TITLES[layer].upper()} ({r['file']})")
        print(f"     хранит:   {r['stores']}")
        print(f"     пишет:    {r['writes']}")
        print(f"     в промпт: {r['prompt'].format(keep=agent.keep_last, budget=agent.long_budget)}")
        print(f"     очистка:  {r['cleared']}")
    print("  РОУТЕР: " + ", ".join(f"{s} → {LAYER_TITLES[SECTION_LAYER[s]]}" for s in SECTIONS))
    print(f"  НЕ СОХРАНЯЕТСЯ: {NOT_SAVED}")


# ---------------------------------------------------------------- контрольные вопросы


ROLE = Detail("role", "профиль", 1, "продакт-менеджер сети пекарен «Колосок»", "Кем я работаю и в какой компании?",
              (r"продакт|product|менеджер продукт", "колос"))
LAST_QUESTION = Detail("last_question", "диалог", 13, "последний вопрос — как часто проводить статус-встречи",
                       "Какой вопрос я задал тебе последним, до этого сообщения?", (r"статус|встреч",))
ADVICE = Detail("advice", "диалог", 13, "совет ассистента о частоте статус-встреч (только в ответе ассистента)",
                "Как часто ты посоветовал проводить статус-встречи с подрядчиком?",
                (r"недел|ежедн|раз в|дв[аеу]\s*раза|спринт|кажд",))
PROBES = [DETAILS[0], ROLE] + DETAILS[1:] + [LAST_QUESTION, ADVICE]

CONFIGS = [  # (id, что очищено, какие слои остаются)
    ("all", "ничего", ("short", "working", "long")),
    ("no_short", "краткосрочная", ("working", "long")),
    ("no_working", "рабочая", ("short", "long")),
    ("no_long", "долговременная", ("short", "working")),
    ("long_only", "краткосрочная + рабочая", ("long",)),
    ("none", "все три слоя", ()),
]


def ask(agent: LayeredAgent, question: str) -> dict:
    """Вопрос отдельным запросом с текущей памятью агента; в диалог он не попадает."""
    schemas = [t.schema() for t in agent.tools.values()]
    messages, _, parts = agent.build(end=len(agent.history), tools=schemas)
    priced = price_for(agent.client.model)
    try:
        result = agent.client.chat(messages + [{"role": "user", "content": question + PROBE_SUFFIX}], tools=schemas)
    except LLMError as e:
        return {"answer": "", "error": str(e), "prompt": 0, "cached": 0, "completion": 0, "cost": 0.0,
                "estimate": parts["estimate"]}
    usage = Usage.parse(result.usage)
    answer = result.text.strip() or ("(модель попросила инструмент вместо ответа)" if result.tool_calls else "")
    return {"answer": answer, "error": "", "prompt": usage.prompt, "cached": usage.cached,
            "completion": usage.completion, "cost": usage.cost(priced[0] if priced else None),
            "estimate": parts["estimate"]}


def icon(r: dict) -> str:
    return "⚠️" if r["error"] else ("❔" if r["ok"] and r["denial"] else "✅" if r["ok"] else "❌")


def run_probes(agent: LayeredAgent, probes: list[Detail] = PROBES, verbose: bool = True) -> list[dict]:
    out = []
    for d in probes:
        r = {"id": d.id, "question": d.question, **ask(agent, d.question)}
        r["ok"] = not r["error"] and matches(d, r["answer"])
        r["denial"] = bool(DENIAL.search(normalize(r["answer"])))
        r["answered"] = r["ok"] and not r["denial"]
        out.append(r)
        if verbose:
            print(f"  {icon(r)} [{d.kind}] {d.question} → {clip(r['answer'] or r['error'], 150)}")
    return out


def where(agent: LayeredAgent, d: Detail) -> dict:
    """Где лежит деталь: в каком слое (в файле) и попадает ли она оттуда в промпт."""
    h = agent.history
    tail = h[tail_start(h, len(h), agent.keep_last):]
    text = lambda ms: "\n".join(m.get("content") or "" for m in ms if isinstance(m.get("content"), str))  # noqa: E731
    long_all = "\n".join(e["value"] + " " + k for s in LONG_SECTIONS for k, e in agent.long.data[s].items())
    return {"short_file": matches(d, text(h[1:])), "short_prompt": matches(d, text(tail)),
            "working": matches(d, agent.working_block()), "long_file": matches(d, long_all),
            "long_prompt": matches(d, agent.long_block()[0])}


def where_label(w: dict) -> str:
    out = []
    if w["short_prompt"]:
        out.append("К")
    elif w["short_file"]:
        out.append("к (вне хвоста)")
    if w["working"]:
        out.append("Р")
    if w["long_prompt"]:
        out.append("Д")
    elif w["long_file"]:
        out.append("д (вне бюджета)")
    return " + ".join(out) or "нигде"


def clone(agent: LayeredAgent, root: Path, client: LLMClient | None = None, event=quiet_event) -> LayeredAgent:
    """Копия профиля в другой папке — чтобы чистить слои, не трогая оригинал."""
    agent.save()
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(agent.root, root)
    # своя копия клиента: своя сессия у провайдера (очистка краткосрочной её меняет), общий счётчик запросов
    copy_ = LayeredAgent(client or copy.copy(agent.client), list(agent.tools.values()), root, keep_last=agent.keep_last,
                         long_budget=agent.long_budget, helper_client=agent.helper_client,
                         max_steps=agent.max_steps, on_event=event)
    copy_.load()
    copy_.estimator.factor, copy_.estimator.samples = agent.estimator.factor, max(agent.estimator.samples, 1)
    return copy_


def experiment(agent: LayeredAgent, root: Path, configs=CONFIGS, probes: list[Detail] = PROBES) -> dict:
    """Очистить слои по очереди (каждая конфигурация — на своей копии профиля, очистка настоящая: /clear)
    и задать одни и те же контрольные вопросы."""
    try:
        agent.catch_up()
    except LLMError as e:
        print(f"  ⚠️  роутер не догнал пропущенные сообщения: {e}")
    placement = {d.id: where(agent, d) for d in probes}
    runs = []
    for cid, cleared, keep in configs:
        print(f"\n--- очищено: {cleared} (в промпте: {', '.join(LAYER_TITLES[x] for x in keep) or 'только system'})")
        a = clone(agent, root / cid)
        for layer in LAYERS:
            if layer not in keep:
                a.clear(layer)
        probe = run_probes(a, probes)
        schemas = [t.schema() for t in a.tools.values()]
        est = a.build(end=len(a.history), tools=schemas)[2]["estimate"]
        runs.append({"id": cid, "cleared": cleared, "layers": list(keep), "estimate": est, "probes": probe,
                     "answered": sum(p["answered"] for p in probe), "doubtful": sum(p["ok"] and p["denial"] for p in probe),
                     "prompt_avg": round(sum(p["prompt"] for p in probe) / len(probe)) if probe else 0,
                     "cost": sum(p["cost"] or 0 for p in probe), "dir": str(a.root.relative_to(ROOT))})
        r = runs[-1]
        print(f"  → отвечает на {r['answered']}/{len(probes)}" + (f" (❔ {r['doubtful']})" if r["doubtful"] else "")
              + f"; вход на вопрос ≈{fmt(r['prompt_avg'])} токенов")
    return {"placement": placement, "runs": runs, "layer_tokens": layer_tokens(runs, probes)}


def layer_tokens(runs: list[dict], probes: list[Detail]) -> dict:
    """Сколько токенов добавляет слой: разница точных prompt_tokens одного и того же вопроса с ним и без него."""
    by = {r["id"]: r for r in runs}
    out = {}
    for layer, without in (("short", "no_short"), ("working", "no_working"), ("long", "no_long")):
        if "all" in by and without in by:
            diffs = [a["prompt"] - b["prompt"] for a, b in zip(by["all"]["probes"], by[without]["probes"])
                     if a["prompt"] and b["prompt"]]
            out[layer] = {"exact": round(sum(diffs) / len(diffs)) if diffs else None,
                          "estimate": by["all"]["estimate"].get(layer, 0)}
    if "none" in by:
        prompts = [p["prompt"] for p in by["none"]["probes"] if p["prompt"]]
        out["base"] = {"exact": round(sum(prompts) / len(prompts)) if prompts else None,
                       "estimate": by["none"]["estimate"].get("system", 0) + by["none"]["estimate"].get("current", 0)}
    if "all" in by:
        prompts = [p["prompt"] for p in by["all"]["probes"] if p["prompt"]]
        out["all"] = {"exact": round(sum(prompts) / len(prompts)) if prompts else None,
                      "estimate": sum(by["all"]["estimate"].values())}
    return out


def print_experiment(exp: dict, probes: list[Detail] = PROBES) -> None:
    runs = exp["runs"]
    print(f"\n{'=' * 78}\nОчистка слоёв: на какие вопросы агент перестаёт отвечать\n{'=' * 78}")
    print("  К — краткосрочная (в хвосте), Р — рабочая, Д — долговременная")
    print(f"  {'деталь':<36}{'где лежит':<14}" + "".join(f"{r['id']:>11}" for r in runs))
    for i, d in enumerate(probes):
        cells = "".join(f"{icon(r['probes'][i]):>10}" for r in runs)
        print(f"  {clip(d.fact, 35):<36}{where_label(exp['placement'][d.id]):<14}{cells}")
    print(f"  {'отвечает':<50}" + "".join(f"{str(r['answered']) + '/' + str(len(probes)):>11}" for r in runs))
    print(f"  {'вход на вопрос':<50}" + "".join(f"{fmt(r['prompt_avg']):>11}" for r in runs))
    lt = exp["layer_tokens"]
    print("\n  Сколько токенов добавляет слой (точно — разница prompt_tokens одного вопроса со слоем и без):")
    for k, title in (("base", "без памяти (system + вопрос)"), ("long", "долговременная"), ("working", "рабочая"),
                     ("short", "краткосрочная"), ("all", "всё вместе")):
        if k in lt:
            print(f"   {title:<32} точно {fmt(lt[k]['exact']):>6}, оценка {fmt(lt[k]['estimate']):>6}")


# ---------------------------------------------------------------- импорт фактов Дня 10


def day10_facts(path: Path | None = None) -> tuple[dict[str, str], Path]:
    """Факты Sticky Facts из Дня 10: из отчёта results/day10_*.json или из <ветка>.state.json."""
    if path is None:
        found = sorted(RESULTS_DIR.glob("day10_*.json"))
        if not found:
            raise StoreError("не найден results/day10_*.json — укажите файл: --import-day10 <путь>")
        path = found[-1]
    data = read_json(path)
    facts = {}
    for run in data.get("runs") or []:
        if run.get("strategy") == "facts" and run.get("facts"):
            facts = run["facts"]
    if not facts:
        facts = ((data.get("strategies") or {}).get("facts") or {}).get("facts") or {}
    if not facts:
        raise StoreError(f"в {path} нет фактов Sticky Facts (ни runs[facts].facts, ни strategies.facts.facts)")
    return {str(k): str(v) for k, v in facts.items()}, path


def import_day10(agent: LayeredAgent, path: Path | None = None) -> dict:
    facts, path = day10_facts(path)
    text = "\n".join(f"- {k}: {v}" for k, v in facts.items())
    where_ = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"  📥 факты Дня 10 из {where_}: {len(facts)} шт. → роутер раскладывает по слоям")
    ops = agent.route([text], f"импорт фактов Дня 10 ({path.name})", imported=True)
    agent.save_memory()
    placed = {}
    for k in facts:
        s = agent.find(k)
        placed[k] = s
    renamed = [o["key"] for o in ops if o["key"] not in facts]
    return {"file": str(where_), "facts": facts, "placed": placed, "ops": ops, "renamed": renamed,
            "call": asdict(agent.router_calls[-1]) if agent.router_calls else None}


def print_import(result: dict) -> None:
    by = {}
    for k, s in result["placed"].items():
        by.setdefault(s, []).append(k)
    for s in SECTIONS + (None,):
        if by.get(s):
            title = f"{LAYER_TITLES[SECTION_LAYER[s]]} · {SECTION_TITLES[s]}" if s else "не сохранено под своим ключом"
            print(f"     {title}: {', '.join(by[s])}")
    if result["renamed"]:
        print(f"     записано под новыми ключами: {', '.join(result['renamed'])}")


# ---------------------------------------------------------------- чат


HELP = ("/memory, /context, /rules, /remember <раздел> <ключ> = <значение>, /forget <раздел> <ключ>, "
        "/move <ключ> <раздел>, /clear short|working|long, /newchat, /task [new <название>|switch <id>|done|list], "
        "/import-day10 [файл], /play [N], /probe, /experiment, /log, /tokens, /history, /exit")


def section_arg(name: str) -> str:
    s = SECTION_ALIASES.get(name.lower())
    if not s:
        raise ValueError(f"нет раздела «{name}». Есть: profile, decisions, knowledge, task")
    return s


def print_ops(ops: list[dict]) -> None:
    for item in ops:
        print(f"  {op_label(item)}")
    if not ops:
        print("  (ничего не изменилось)")


def chat(agent: LayeredAgent) -> int:
    print(f"Агент на {agent.client.model}, профиль {agent.root.name}. Задача: «{agent.working.task['title']}».\n"
          f"Команды: {HELP}\n")
    while True:
        try:
            text = input(f"[{agent.root.name} · задача {agent.working.task['id']}] Вы: ").strip()
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
            if command == "/memory":
                print_memory(agent)
            elif command == "/context":
                print_context(agent)
            elif command == "/rules":
                print_rules(agent)
            elif command == "/remember":
                head, sep, value = arg.partition("=")
                parts = head.split()
                if not sep or len(parts) != 2 or not value.strip():
                    print("  /remember <profile|decisions|knowledge|task> <ключ> = <значение>")
                else:
                    print_ops(agent.write(section_arg(parts[0]), parts[1], value.strip(), "команда пользователя"))
                    agent.save_memory()
            elif command == "/forget":
                if len(args) != 2:
                    print("  /forget <раздел> <ключ>")
                else:
                    print_ops(agent.forget(section_arg(args[0]), args[1], "команда пользователя"))
                    agent.save_memory()
            elif command == "/move":
                src = agent.find(args[0]) if len(args) == 2 else None
                if not src:
                    print("  /move <ключ> <раздел> — ключ должен быть в памяти (см. /memory)")
                else:
                    value = agent.section(src)[args[0]]["value"]
                    print_ops(agent.write(section_arg(args[1]), args[0], value, "команда пользователя"))
                    agent.save_memory()
            elif command == "/clear":
                layer = LAYER_ALIASES.get(args[0].lower()) if args else None
                if not layer:
                    print("  /clear short|working|long")
                else:
                    print(f"  🧹 {LAYER_TITLES[layer]}: {agent.clear(layer)}")
            elif command == "/newchat":
                print(f"  🧹 {agent.clear('short')}; рабочая и долговременная память остаются")
            elif command == "/task":
                sub = args[0] if args else "list"
                if sub == "new":
                    title = arg.partition(" ")[2].strip() or NO_TASK
                    t = agent.new_task(title)
                    print(f"  📋 новая задача {t['id']} «{t['title']}»; прежняя — на паузе (вне промпта)")
                elif sub == "switch" and len(args) > 1:
                    t = agent.switch_task(args[1])
                    print(f"  📋 активная задача {t['id']} «{t['title']}», полей {len(t['data'])}")
                elif sub == "done":
                    agent.flush_pending("закрытие задачи")
                    t = agent.working.done()
                    agent.log.add("task", "task", t["id"], t["title"], "команда пользователя", status="done")
                    agent.save_memory()
                    print(f"  ✅ задача {t['id']} «{t['title']}» закрыта и убрана из рабочей памяти")
                else:
                    for t in agent.working.list():
                        print(f"  {'▶' if t['status'] == 'active' else '·'} {t['id']} «{t['title']}» — {t['status']}, "
                              f"полей {t['fields']}")
            elif command == "/import-day10":
                print_import(import_day10(agent, Path(arg.strip()) if arg.strip() else None))
            elif command == "/log":
                for item in agent.log.tail(int(args[0]) if args and args[0].isdigit() else 20):
                    print(f"  {item['at'][11:19]} {op_label(item)}  ⟵ {item.get('source', '')}")
            elif command == "/tokens":
                print_ledger(agent)
                h = agent.router_totals()
                print(f"  + роутер памяти: запросов {h['count']}, вход {fmt(h['prompt'])}, выход {fmt(h['completion'])}, "
                      f"{money(h['cost'])}")
                for s in agent.layer_stats:
                    print(f"  ход {s['turn']}: вход {fmt(s['prompt'])} = {layer_line(s['tokens'])}")
            elif command == "/history":
                print_history(agent.history)
            elif command == "/probe":
                agent.catch_up()
                results = run_probes(agent)
                print(f"  Отвечает на {sum(r['answered'] for r in results)}/{len(results)}.")
            elif command == "/experiment":
                exp = experiment(agent, agent.root.parent / f"{agent.root.name}_experiment_{stamp_now()}")
                print_experiment(exp)
            elif command == "/play":
                done = len(agent.ledger)
                count = int(args[0]) if args and args[0].isdigit() else len(DIALOGUE) - done
                lines = DIALOGUE[done:done + count]
                if not lines:
                    print(f"  Сценарий пройден ({len(DIALOGUE)} реплик). /probe, /experiment или /newchat.")
                for k, line in enumerate(lines, done + 1):
                    print(f"\n--- реплика {k}/{len(DIALOGUE)}\nВы: {line}")
                    reply = agent.run(line)
                    print(f"\nАгент: {reply.text.strip() or '(пустой ответ)'}\n")
                    print_turn(agent, agent.ledger[-1])
            elif command.startswith("/"):
                print(f"  Неизвестная команда. Есть: {HELP}")
            else:
                reply = agent.run(text)
                print(f"\nАгент: {reply.text.strip() or '(пустой ответ)'}\n")
                print_turn(agent, agent.ledger[-1])
            print()
        except LLMError as e:
            print(f"❌ Ошибка: {e}\n", file=sys.stderr)
        except (StoreError, ValueError) as e:
            print(f"⚠️  {e}\n", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n(прервано — ход в историю не попал)\n")


# ---------------------------------------------------------------- демо


TASK_A = "План MVP мобильного приложения"
TASK_B = "Правила бонусной программы"
TASK_B_TURN = ("Начинаем новую задачу — правила начисления бонусов в программе лояльности. Бюджет на неё отдельный — "
               "400 тысяч рублей, правила нужно согласовать с маркетингом до 1 февраля 2027 года. С чего начать?")
SWITCH_PROBES = [  # (id, вопрос, задача → (что должно быть, что пришло из другой задачи))
    ("name", "Как меня зовут?", {"B": (("марин",), ()), "A": (("марин",), ())}),
    ("flutter", "На какой технологии решили делать приложение?",
     {"B": ((r"flutter|флаттер",), ()), "A": ((r"flutter|флаттер",), ())}),
    ("budget", "Какой бюджет у текущей задачи?",
     {"B": ((r"\b400\b|0[,.]4\s*млн",), (r"2[,.]7|2700",)), "A": ((r"2[,.]7|2700",), (r"\b400\b",))}),
    ("deadline", "До какой даты нужно сделать текущую задачу?",
     {"B": ((r"(?<!\d)1\s*феврал|01\.02",), (r"14\s*марта",)),
      "A": ((r"14\s*марта|20\s*феврал",), (r"(?<!\d)1\s*феврал",))}),
]


def check_switch(task: str, expect: dict, answer: str) -> dict:
    t = normalize(answer)
    must, leak = expect[task]
    found_leak = [p for p in leak if re.search(p, t)]
    has = all(re.search(p, t) for p in must)
    return {"ok": has and not found_leak, "must": has, "leak": found_leak}


def switch_demo(agent: LayeredAgent) -> dict:
    """Новый диалог + новая задача: долговременная память переходит, рабочая — своя у каждой задачи."""
    print(f"\n{'=' * 78}\n=== Новый диалог и новая задача: что переходит, а что остаётся в своей задаче\n{'=' * 78}")
    task_a = agent.working.task["id"]
    steps = []

    def probe(task: str) -> list[dict]:
        out = []
        for pid, question, expect in SWITCH_PROBES:
            r = ask(agent, question)
            r.update({"id": pid, "question": question, **check_switch(task, expect, r["answer"] or r["error"])})
            out.append(r)
            mark = "✅" if r["ok"] else ("⚠️ утечка" if r["leak"] else "❌")
            print(f"  {mark} {question} → {clip(r['answer'] or r['error'], 140)}")
        return out

    print(f"  🧹 /newchat: {agent.clear('short')}")
    t = agent.new_task(TASK_B)
    print(f"  📋 /task new: задача {t['id']} «{TASK_B}», задача {task_a} «{TASK_A}» — на паузе")
    print(f"\nВы: {TASK_B_TURN}")
    try:
        reply = agent.run(TASK_B_TURN)
        print(f"Агент: {clip(reply.text, 220)}")
        print_turn(agent, agent.ledger[-1])
        steps.append({"step": "turn", "task": t["id"], "text": TASK_B_TURN, "answer": reply.text,
                      "layers": agent.layer_stats[-1] if agent.layer_stats else None})
    except LLMError as e:
        print(f"  ❌ {e}")
        steps.append({"step": "turn", "task": t["id"], "text": TASK_B_TURN, "error": str(e)})
    task_b = t["id"]
    print(f"\n--- вопросы в задаче «{TASK_B}» (новый диалог, рабочая память — задача {task_b}):")
    steps.append({"step": "probe", "task": task_b, "title": TASK_B, "probes": probe("B"),
                  "working": dict(agent.memory_json()["task"])})
    print(f"\n  🧹 /newchat: {agent.clear('short')}")
    t = agent.switch_task(task_a)
    print(f"  📋 /task switch {task_a}: «{t['title']}», полей {len(t['data'])}; задача {task_b} — на паузе")
    print(f"\n--- вопросы в задаче «{TASK_A}» (снова новый диалог):")
    steps.append({"step": "probe", "task": task_a, "title": TASK_A, "probes": probe("A"),
                  "working": dict(agent.memory_json()["task"])})
    checks = [p for s in steps if s["step"] == "probe" for p in s["probes"]]
    print(f"\n  Итог: {sum(p['ok'] for p in checks)}/{len(checks)} проверок, утечек между задачами: "
          f"{sum(bool(p['leak']) for p in checks)}")
    return {"task_a": task_a, "task_b": task_b, "steps": steps, "passed": sum(p["ok"] for p in checks),
            "total": len(checks), "leaks": sum(bool(p["leak"]) for p in checks),
            "tasks": agent.working.list(), "long_term": agent.memory_json()}


def make_agent(client: LLMClient, helper: LLMClient, root: Path, keep: int, budget: int, max_steps: int,
               event=on_event) -> LayeredAgent:
    watch_retries(client, helper, event=event)
    return LayeredAgent(client, TOOLS, root, keep_last=keep, long_budget=budget, helper_client=helper,
                        max_steps=max_steps, on_event=event)


def layers_snapshot(agent: LayeredAgent) -> dict:
    return {"short": {"file": agent.store.path.name, "messages": len(agent.history) - 1,
                      "tail": len(agent.history) - tail_start(agent.history, len(agent.history), agent.keep_last)},
            "working": copy.deepcopy(agent.working.task), "long": copy.deepcopy(agent.long.data),
            "tasks": agent.working.list()}


def demo(client: LLMClient, helper_model: str | None, keep: int, budget: int, max_steps: int, do_import: bool,
         do_switch: bool) -> int:
    stamp = stamp_now()
    root = DAY_DIR / f"demo_{stamp}"
    helper = (LLMClient(client.base_url, client.api_key, helper_model, client.timeout, retries=client.retries)
              if helper_model else client)
    report: dict = {"model": client.model, "helper_model": helper.model, "keep_last": keep, "long_budget": budget,
                    "stamp": stamp, "dir": str(root.relative_to(ROOT)), "rules": RULES, "router_prompt": ROUTER_PROMPT,
                    "section_layer": SECTION_LAYER, "dialogue": DIALOGUE, "probes": [asdict(d) for d in PROBES]}

    if do_import:
        print(f"{'=' * 78}\n=== 1. Факты Дня 10 → слои памяти\n{'=' * 78}")
        imp = make_agent(client, helper, root / "day10_import", keep, budget, max_steps, quiet_event)
        imp.load()
        try:
            report["import"] = import_day10(imp)
            print_import(report["import"])
            report["import"]["layers"] = layers_snapshot(imp)
        except (LLMError, StoreError) as e:
            print(f"  ⚠️  импорт не удался: {e}")
            report["import"] = {"error": str(e)}

    print(f"\n{'=' * 78}\n=== 2. Диалог Дня 9 ({len(DIALOGUE)} реплик): роутер раскладывает сообщения по слоям\n"
          f"{'=' * 78}")
    agent = make_agent(client, helper, root / "main", keep, budget, max_steps, quiet_event)
    agent.load()
    agent.new_task(TASK_A)
    print(f"  📋 /task new «{TASK_A}» — рабочая память этой задачи")
    for k, line in enumerate(DIALOGUE, 1):
        print(f"\n--- ход {k}/{len(DIALOGUE)}: {clip(line, 110)}")
        try:
            reply = agent.run(line)
            print(f"  Агент: {clip(reply.text, 200) if reply.text.strip() else '(пустой ответ)'}")
        except LLMError as e:
            print(f"  ❌ {e}")
        print_turn(agent, agent.ledger[-1])
    try:
        agent.catch_up()
    except LLMError as e:
        print(f"  ⚠️  догон роутера не удался: {e}")
    print("\n--- память после диалога:")
    print_memory(agent)
    t = agent.totals()
    report["dialogue_run"] = {
        "turns": [{"n": x.n, "status": x.status, "prompt": x.calls[0].prompt if x.calls else 0,
                   "cached": x.cached, "completion": x.completion, "cost": x.cost, "answer": x.answer,
                   "error": x.error} for x in agent.ledger],
        "layer_stats": copy.deepcopy(agent.layer_stats), "totals": t, "router": agent.router_totals(),
        "router_calls": [asdict(c) for c in agent.router_calls], "log": agent.log.tail(500),
        "layers": layers_snapshot(agent)}

    print(f"\n{'=' * 78}\n=== 3. Очистка слоёв по очереди: {len(PROBES)} контрольных вопросов × {len(CONFIGS)} "
          f"конфигураций\n{'=' * 78}")
    exp = experiment(agent, root / "experiment")
    print_experiment(exp)
    report["experiment"] = exp

    if do_switch:
        report["switch"] = switch_demo(agent)

    report["requests"] = {"main": dict(client.stats), "helper": dict(helper.stats) if helper is not client else None}
    save_report(report, stamp)
    return 0


# ---------------------------------------------------------------- отчёт


def md_escape(text: str) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def save_report(report: dict, stamp: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path, md_path = RESULTS_DIR / f"day11_{stamp}.json", RESULTS_DIR / f"day11_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    keep, budget = report["keep_last"], report["long_budget"]
    L = [f"# День 11 · Модель памяти ассистента — прогон {stamp}", "",
         f"Модель: `{report['model']}`, роутер памяти: `{report['helper_model']}`. Хвост краткосрочной памяти — "
         f"{keep} сообщений, бюджет долговременной — {budget} токенов. Файлы прогона: `{report['dir']}/`.", "",
         "## Модель памяти", "",
         "| Слой | Что хранит | Файл | Кто пишет | Что попадает в промпт | Когда очищается |", "|---|---|---|---|---|---|"]
    for layer in LAYERS:
        r = RULES[layer]
        L.append(f"| {LAYER_TITLES[layer]} | {r['stores']} | `{r['file']}` | {r['writes']} | "
                 f"{r['prompt'].format(keep=keep, budget=budget)} | {r['cleared']} |")
    L += ["", "Роутер памяти: " + ", ".join(f"`{s}` → {LAYER_TITLES[SECTION_LAYER[s]]}" for s in SECTIONS)
          + f". Не сохраняется: {NOT_SAVED}.", ""]

    imp = report.get("import")
    if imp:
        L += ["## 1. Факты Дня 10 → слои", ""]
        if imp.get("error"):
            L += [f"Импорт не удался: {imp['error']}", ""]
        else:
            call = imp.get("call") or {}
            L += [f"Файл: `{imp['file']}`, фактов: {len(imp['facts'])}. Один запрос роутера: вход {fmt(call.get('prompt'))}, "
                  f"выход {fmt(call.get('completion'))}, {money(call.get('cost'))}.", "",
                  "| Факт Дня 10 | Значение | Слой · раздел |", "|---|---|---|"]
            for k, v in imp["facts"].items():
                s = imp["placed"].get(k)
                place = f"{LAYER_TITLES[SECTION_LAYER[s]]} · {SECTION_TITLES[s]}" if s else "— (не под своим ключом)"
                L.append(f"| `{k}` | {md_escape(clip(v, 90))} | {place} |")
            if imp.get("renamed"):
                L += ["", "Записано под новыми ключами: " + ", ".join(f"`{k}`" for k in imp["renamed"])]
            L.append("")

    run = report["dialogue_run"]
    L += [f"## 2. Диалог Дня 9: что роутер положил в каждый слой", "",
          "| Ход | Реплика | Записи в память | Вход | system | долговр. | рабочая | краткоср. | ход | Хвост / не отпр. |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    by_turn: dict[int, list[str]] = {}
    for item in run["log"]:
        m = re.search(r"ход (\d+)", item.get("source") or "")
        if m and item["op"] in ("add", "update", "move", "delete"):
            by_turn.setdefault(int(m.group(1)), []).append(md_escape(op_label(item)))
    stats = {s["turn"]: s for s in run["layer_stats"]}
    for k, line in enumerate(report["dialogue"], 1):
        s = stats.get(k) or {}
        tk = s.get("tokens") or {}
        L.append(f"| {k} | {md_escape(clip(line, 70))} | {'<br>'.join(by_turn.get(k, [])) or '—'} | "
                 f"{fmt(s.get('prompt'))} | {fmt(tk.get('system'))} | {fmt(tk.get('long'))} | {fmt(tk.get('working'))} | "
                 f"{fmt(tk.get('short'))} | {fmt(tk.get('current'))} | {s.get('tail_messages', '?')} / {s.get('dropped', '?')} |")
    t, h = run["totals"], run["router"]
    total_cost = None if t["cost"] is None or h["cost"] is None else t["cost"] + h["cost"]
    L += ["", f"Токены по слоям — оценка по частям запроса, отмасштабированная к точному `prompt_tokens` первого "
              f"запроса хода. Вход за диалог {fmt(t['prompt'])} (из кэша {fmt(t['cached'])}), выход {fmt(t['completion'])}; "
              f"роутер: {h['count']} запросов, вход {fmt(h['prompt'])}, выход {fmt(h['completion'])}. "
              f"Стоимость: ходы {money(t['cost'])} + роутер {money(h['cost'])} = {money(total_cost)}.", ""]

    lay = run["layers"]
    L += ["### Содержимое слоёв после диалога", "",
          f"**Краткосрочная** (`{lay['short']['file']}`): {lay['short']['messages']} сообщений диалога, "
          f"в промпт уходят последние {lay['short']['tail']}.", "",
          f"**Рабочая** (`working.json`): задача {lay['working']['id']} «{lay['working']['title']}»", ""]
    L += [f"- `{k}`: {md_escape(e['value'])} — _{e['source']}_" for k, e in lay["working"]["data"].items()] or ["- (пусто)"]
    L += ["", "**Долговременная** (`long_term.json`)", ""]
    for s in LONG_SECTIONS:
        items = lay["long"][s]
        L.append(f"- {SECTION_TITLES[s]}: " + ("; ".join(f"`{k}` = {md_escape(e['value'])}" for k, e in items.items())
                                              or "(пусто)"))
    L.append("")

    exp = report["experiment"]
    runs = exp["runs"]
    L += ["## 3. Очистка слоёв: на какие вопросы агент перестаёт отвечать", "",
          "Каждая конфигурация — копия профиля после диалога, в которой слои очищены настоящей командой `/clear` "
          "(файлы — в `experiment/<конфигурация>/`). Вопросы — отдельные запросы, в диалог не попадают. "
          "Где лежит: К — краткосрочная (в хвосте), к — в диалоге, но вне хвоста, Р — рабочая, Д — долговременная. "
          "✅ — ответ с деталью, ❔ — слова детали есть, но ответ звучит как отрицание, ❌ — не знает или неверно.", "",
          "| Деталь | Где лежит | " + " | ".join(f"очищено: {r['cleared']}" for r in runs) + " |",
          "|---|---|" + "---|" * len(runs)]
    for i, d in enumerate(PROBES):
        L.append(f"| {md_escape(d.fact)} | {where_label(exp['placement'][d.id])} | "
                 + " | ".join(icon(r["probes"][i]) for r in runs) + " |")
    L.append("| **отвечает** | | " + " | ".join(f"**{r['answered']}/{len(PROBES)}**" for r in runs) + " |")
    L.append("| вход на вопрос | | " + " | ".join(fmt(r["prompt_avg"]) for r in runs) + " |")
    L += ["", "Перестаёт отвечать (по сравнению со всеми слоями):", ""]
    base = next((r for r in runs if r["id"] == "all"), None)
    for r in runs:
        if base is None or r is base:
            continue
        lost = [PROBES[i].fact for i, p in enumerate(r["probes"]) if base["probes"][i]["answered"] and not p["answered"]]
        L.append(f"- очищено **{r['cleared']}**: " + ("; ".join(lost) if lost else "ничего"))
    lt = exp["layer_tokens"]
    L += ["", "### Сколько токенов добавляет каждый слой", "",
          "Точно — средняя разница `prompt_tokens` одного и того же вопроса со слоем и без него; оценка — по тексту слоя.",
          "", "| Часть запроса | Точно, токенов | Оценка |", "|---|---|---|"]
    for k, title in (("base", "без памяти: system, схемы инструментов, вопрос"), ("long", "долговременная"),
                     ("working", "рабочая"), ("short", "краткосрочная"), ("all", "всё вместе")):
        if k in lt:
            L.append(f"| {title} | {fmt(lt[k]['exact'])} | {fmt(lt[k]['estimate'])} |")
    L.append("")

    sw = report.get("switch")
    if sw:
        L += ["## 4. Новый диалог и новая задача", "",
              f"`/newchat` → `/task new «{TASK_B}»` → одна реплика → вопросы; затем `/newchat` → "
              f"`/task switch {sw['task_a']}` → те же вопросы. Итог: {sw['passed']}/{sw['total']} проверок, "
              f"утечек между задачами: {sw['leaks']}.", "",
              f"Реплика в новой задаче: «{TASK_B_TURN}»", ""]
        for s in sw["steps"]:
            if s["step"] != "probe":
                continue
            L += [f"**Задача {s['task']} «{s['title']}»** — рабочая память: "
                  + ("; ".join(f"`{k}` = {md_escape(v)}" for k, v in s["working"].items()) or "(пусто)"), "",
                  "| Вопрос | Ответ | Проверка |", "|---|---|---|"]
            for p in s["probes"]:
                mark = "✅" if p["ok"] else ("⚠️ утечка" if p["leak"] else "❌")
                L.append(f"| {p['question']} | {md_escape(clip(p['answer'] or p['error'], 200))} | {mark} |")
            L.append("")

    L += ["## Все ответы на контрольные вопросы", ""]
    for r in runs:
        L += [f"### Очищено: {r['cleared']}", ""]
        for i, p in enumerate(r["probes"]):
            L.append(f"- {icon(p)} **{PROBES[i].question}** {md_escape(p['answer'] or p['error'])}")
        L.append("")
    req = report["requests"]
    L += [f"Запросов к модели: {req['main']['requests']}, повторов {req['main']['retries']}, отказов {req['main']['failed']}"
          + (f"; отдельный клиент роутера: {req['helper']['requests']}" if req.get("helper") else "") + "."]
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {md_path.relative_to(ROOT)}, {json_path.name}")
    return md_path


# ---------------------------------------------------------------- запуск


def main() -> int:
    parser = argparse.ArgumentParser(description="День 11: агент с явной моделью памяти — краткосрочная, рабочая, "
                                                 "долговременная.")
    parser.add_argument("prompt", nargs="?", help="один ход и выход; без него — чат в консоли")
    parser.add_argument("--profile", default="default", help="профиль памяти: memory/day11/<имя>/")
    parser.add_argument("--new", action="store_true", help="начать заново: папка профиля уходит в архив")
    parser.add_argument("--keep", type=int, default=6, help="краткосрочная: сколько последних сообщений в промпт")
    parser.add_argument("--long-budget", type=int, default=1500, help="долговременная: бюджет токенов в промпте")
    parser.add_argument("--helper-model", help="модель роутера памяти (по умолчанию та же)")
    parser.add_argument("--import-day10", nargs="?", const="", metavar="ФАЙЛ",
                        help="разложить факты Sticky Facts Дня 10 по слоям (по умолчанию — последний results/day10_*.json)")
    parser.add_argument("--demo", action="store_true", help="импорт Дня 10, диалог Дня 9, очистка слоёв, смена задачи")
    parser.add_argument("--no-import", action="store_true", help="для --demo: без импорта фактов Дня 10")
    parser.add_argument("--no-switch", action="store_true", help="для --demo: без проверки смены задачи")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--max-steps", type=int, default=8, help="предел запросов к модели за один ход")
    args = parser.parse_args()

    if not NAME_RE.fullmatch(args.profile):
        print("Ошибка: имя профиля — буквы, цифры, дефис, подчёркивание.", file=sys.stderr)
        return 1
    if args.keep < 0 or args.long_budget < 0:
        print("Ошибка: --keep ≥ 0, --long-budget ≥ 0.", file=sys.stderr)
        return 1
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    model = args.model or settings.model
    client = LLMClient(settings.base_url, settings.api_key, model, settings.timeout, retries=settings.retries)
    if args.demo:
        return demo(client, args.helper_model, args.keep, args.long_budget, args.max_steps, not args.no_import,
                    not args.no_switch)

    root = DAY_DIR / args.profile
    if args.new and root.exists():
        target = root.with_name(f"{root.name}_{stamp_now()}")
        os.replace(root, target)
        print(f"Прошлый профиль сохранён в {target.relative_to(ROOT)}.")
    helper = (LLMClient(settings.base_url, settings.api_key, args.helper_model, settings.timeout,
                        retries=settings.retries) if args.helper_model else client)
    agent = make_agent(client, helper, root, args.keep, args.long_budget, args.max_steps)
    try:
        info = agent.load()
    except StoreError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print_restore(info, agent.store, agent)
    for note in agent.notes:
        print(f"⚠️  {note}")
    print(f"🧠 Память: рабочая — задача «{agent.working.task['title']}» ({len(agent.working.data)} пол.), "
          f"долговременная — {agent.long.count()} записей. /rules — правила слоёв.\n")

    if args.import_day10 is not None:
        try:
            print_import(import_day10(agent, Path(args.import_day10) if args.import_day10 else None))
        except (LLMError, StoreError) as e:
            print(f"❌ Ошибка: {e}", file=sys.stderr)
            return 1
        if not args.prompt:
            return 0
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
