"""День 11. Модель памяти: текстовая ролевая игра в жанре уся, у героя — три слоя памяти, которые хранятся отдельно.

Модель — Мастер игры: ведёт мир, озвучивает людей, бросает кости. Игрок управляет своим героем. Агент Дня 8
(`TokenAgent`: инструменты, история в JSON, учёт токенов и стоимости) получает явную модель памяти героя.
Каждый слой — свои файлы в memory/day11/<герой>/ и своё правило, что из него уходит в промпт:

    долговременная  long_term.json      постоянное: личность, школа, характер и клятвы, врождённые особенности, связи.
                                        В промпт: личность, школа, характер — целиком; особенности и связи — от новых
                                        к старым, пока укладываются в --long-budget токенов (1500).
    рабочая         working.json,       текущий круг развития: круг и ступень, год обучения, техника, ци, условия
                    circles/<id>.json   прорыва; снаряжение и запасы. В промпт: всё о текущем круге; прошлые круги — нет.
                                        Новый круг (/circle next) — прошлый уходит в архив, снаряжение переносится.
    краткосрочная   mission.json,       текущее задание (заказчик, цель, место, срок, награда, улики) и его сцена —
                    missions/<id>.json, диалог как есть. В промпт: из карточки — записи игры целиком, записи летописца —
                    short_term.json     от новых к старым в пределах --card-budget (600 токенов); из сцены — последние
                                        N сообщений (--keep, 6). Задание сдано или сменилось — карточка и сцена в архив.

Что и куда сохраняется, выбирается явно:
    - летописец (роутер памяти) — Sticky Facts из Дня 10, разведённые по слоям. Перед каждым ходом служебный запрос
      читает последний рассказ Мастера и новое сообщение игрока и возвращает изменения по разделам
      {"persona" | "school" | "nature" | "traits" | "bonds" | "circle" | "gear" | "mission": {"set", "delete"}}.
      Таблица SECTION_LAYER в коде решает, в какой слой идёт раздел; слияние делает код;
    - игра пишет сама: карточка задания (MISSIONS) — сразу в краткосрочную, новый круг — в рабочую;
    - команды игрока поверх летописца: /remember, /forget, /move;
    - каждое изменение пишется в журнал log.jsonl: слой, раздел, ключ, значение, откуда.
Факты в формате Sticky Facts Дня 10 (ключ → значение) раскладываются по слоям: --import-facts [файл]
(по умолчанию — легенда героини Фэн Ся; подходит и results/day10_*.json).

    python day11_memory_layers.py                      # игра (герой default)
    python day11_memory_layers.py --hero luchen --keep 4
    python day11_memory_layers.py --import-facts       # разложить легенду по слоям текущего героя
    python day11_memory_layers.py --demo               # импорт, сценарий, очистка слоёв, смена задания и круга, отчёт

Команды: /memory, /context, /rules, /remember <раздел> <ключ> = <значение>, /forget <раздел> <ключ>,
/move <ключ> <раздел>, /clear short|working|long, /newchat, /mission [new <id|название> | switch <id> | done | list],
/circle [next <название> | list], /import-facts [файл], /play [N], /probe, /experiment, /log, /tokens, /history, /exit.
"""
import argparse
import copy
import json
import os
import random
import re
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from config import ConfigError, load_settings
from day6_agent import Tool
from day7_memory import MEMORY_DIR, ROOT, HistoryStore, RestoreInfo, StoreError, now_iso, print_history, \
    print_restore
from day8_tokens import TokenAgent, TurnStats, Usage, clip, fmt, money, price_for, print_ledger
from day9_compression import Detail, matches, normalize, watch_retries
from day10_strategies import DENIAL, HelperCall, StickyFacts, extract_json_object, read_json, tail_start, write_json
from day10_strategies import on_event as day10_event
from llm_client import LLMClient, LLMError

RESULTS_DIR = ROOT / "results"
DAY_DIR = MEMORY_DIR / "day11"
NAME_RE = re.compile(r"[\w-]+")

LAYERS = ("short", "working", "long")
LAYER_TITLES = {"short": "краткосрочная", "working": "рабочая", "long": "долговременная"}
LAYER_ALIASES = {"short": "short", "s": "short", "краткосрочная": "short", "mission": "short", "задание": "short",
                 "working": "working", "w": "working", "рабочая": "working", "circle": "working", "круг": "working",
                 "long": "long", "l": "long", "долговременная": "long"}
# Разделы, в которые пишет летописец, и в какой слой уходит каждый. Это и есть явный выбор «что куда»:
# поменять слой раздела — поменять одну строку здесь.
SECTIONS = ("persona", "school", "nature", "traits", "bonds", "circle", "gear", "mission")
SECTION_LAYER = {"persona": "long", "school": "long", "nature": "long", "traits": "long", "bonds": "long",
                 "circle": "working", "gear": "working", "mission": "short"}
SECTION_TITLES = {"persona": "личность", "school": "школа", "nature": "характер", "traits": "особенности",
                  "bonds": "связи", "circle": "круг", "gear": "снаряжение", "mission": "задание"}
SECTION_ALIASES = {**{s: s for s in SECTIONS}, **{v: k for k, v in SECTION_TITLES.items()},
                   "личность": "persona", "школа": "school", "секта": "school", "характер": "nature",
                   "клятвы": "nature", "особенности": "traits", "связи": "bonds", "круг": "circle",
                   "развитие": "circle", "снаряжение": "gear", "вещи": "gear", "задание": "mission",
                   "миссия": "mission"}
LONG_SECTIONS = ("persona", "school", "nature", "traits", "bonds")
LONG_ALWAYS = ("persona", "school", "nature")  # в промпт всегда целиком
LONG_BUDGETED = ("traits", "bonds")  # от новых к старым, пока влезают в бюджет
WORKING_SECTIONS = ("circle", "gear")
SHORT_SECTIONS = ("mission",)
CARRY_OVER = ("gear",)  # при переходе на новый круг переносится в него целиком, остальное — в архив круга
CARD_BUDGET = 600  # токенов на записи летописца в карточке задания: без предела карточка растёт как журнал
NO_MISSION = "Без задания"
NO_CIRCLE = "Текущий круг"

RULES = {
    "short": {
        "stores": "текущее задание: карточка (заказчик, цель, место, срок, награда, улики) и сцена — диалог как есть, "
                  "с бросками костей",
        "file": "mission.json (активное задание), missions/<id>.json (отложенные и сданные), short_term.json (сцена)",
        "writes": "игра — карточка задания (/mission new); летописец — раздел mission; каждое сообщение — в сцену",
        "prompt": "из карточки задания — записи игры целиком, записи летописца — от новых к старым, пока влезают "
                  "в {card} токенов; из сцены — последние {keep} сообщений (граница — начало хода) + текущий ход; "
                  "старше — не отправляются",
        "cleared": "/mission done или смена задания — карточка и сцена уходят в архив; /newchat — только сцена; "
                   "/clear short — и карточка, и сцена (сначала летописец разбирает то, что пропустил)",
    },
    "working": {
        "stores": "текущий круг развития: круг и ступень, год обучения, изучаемая техника и формы, ци, условия "
                  "прорыва, наставления; снаряжение и запасы",
        "file": "working.json (текущий круг), circles/<id>.json (пройденные)",
        "writes": "летописец — разделы circle и gear; игра — новый круг (/circle next); /remember circle|gear …",
        "prompt": "все записи текущего круга и снаряжение; пройденные круги — не отправляются",
        "cleared": "/circle next — круг уходит в архив, снаряжение переносится в новый; /clear working",
    },
    "long": {
        "stores": "постоянное: личность, школа и наставник, характер и клятвы, врождённые особенности, связи и вражда",
        "file": "long_term.json",
        "writes": "летописец — разделы persona, school, nature, traits, bonds; /remember …; импорт фактов",
        "prompt": "личность, школа, характер — всегда целиком; особенности и связи — от новых к старым, пока влезают "
                  "в {budget} токенов",
        "cleared": "только явно: /clear long или /forget по ключу",
    },
}
NOT_SAVED = ("описания, угрозы и догадки Мастера, вопросы игрока вне игры и подробности сцены в рабочую "
             "и долговременную память не идут — они живут только в сцене, пока не выпадут из хвоста")

WORLD = (
    "\n\n## Мир игры\n"
    "Земли Девяти Рек. Праведные школы: Секта Лазурного Облака (мечи и техники ветра), Долина Нефритового Лотоса "
    "(целители и алхимики), Крепость Железной Горы (закалка тела, кулачный бой), Павильон Тысячи Свитков "
    "(учёные и стратеги). Демоническая секта Кровавой Луны грабит караваны и носит красное. Круги совершенствования "
    "по порядку: Закалка тела, Очищение ци, Основание, Золотое ядро; в каждом круге три ступени. Деньги — ляны серебра.")

MASTER_PROMPT = (
    "Ты — Мастер текстовой ролевой игры в жанре уся. Игрок управляет своим героем, ты — всем остальным: миром, "
    "людьми, погодой и случаем. Правила:\n"
    "- пиши по-русски, во втором лице («ты видишь…»), 80–180 слов, живо и в духе уся: цзянху, секты, техники, ци;\n"
    "- не решай и не говори за героя: заканчивай ход ситуацией, в которой игроку нужно выбрать;\n"
    "- исход рискованных действий решай броском костей через roll_dice, а не по своей воле;\n"
    "- опирайся на память героя: не противоречь его школе, характеру и клятвам, кругу развития, снаряжению и заданию;\n"
    "- на вопрос вне игры (о герое, задании, правилах) отвечай коротко, по памяти и сцене. Чего там нет — честно "
    "скажи, что не знаешь, и не придумывай." + WORLD)

ROUTER_PROMPT = (
    "Ты — летописец текстовой ролевой игры в жанре уся. Мастер игры не видит старых сообщений: всё важное о герое "
    "он знает только из памяти. Память героя разделена на разделы, и ты решаешь, что из новых сообщений куда "
    "записать.\n"
    "Долговременная память — постоянное, живёт всю игру:\n"
    "- persona — кто герой: имя, возраст, происхождение, родные места, внешность;\n"
    "- school — школа или секта героя, наставник, положение в школе, боевой стиль школы;\n"
    "- nature — характер, принципы, клятвы и обеты;\n"
    "- traits — врождённые и постоянные особенности: корень духа, телосложение, родовые дары, шрамы;\n"
    "- bonds — связи: кровные враги, побратимы, долги, кому герой обязан.\n"
    "Рабочая память — текущий круг развития, устаревает с прорывом:\n"
    "- circle — круг и ступень, год обучения, изучаемая техника и сколько форм освоено, накоплено ци, условия "
    "прорыва, наставления на этот круг;\n"
    "- gear — снаряжение и запасы: оружие, пилюли, деньги, вещи. Только то, что назвал игрок или что герой по ходу "
    "игры получил, потратил или потерял. Как Мастер описал одежду и вещи героя — не запись.\n"
    "Краткосрочная память — текущее задание, устаревает, когда задание сдано:\n"
    "- mission — заказчик, цель, место, срок, награда, найденные улики, договорённости и люди этого задания. "
    "Записывай итоги — что найдено, кто встречен, что выяснилось, — а не пересказ хода: где герой сейчас и что "
    "делает, видно в сцене.\n"
    "Тебе дают текущую память, последний рассказ Мастера и новые сообщения игрока. Верни JSON вида "
    '{"persona": {"set": {}, "delete": []}, "school": {"set": {}, "delete": []}, "nature": {"set": {}, "delete": []}, '
    '"traits": {"set": {}, "delete": []}, "bonds": {"set": {}, "delete": []}, "circle": {"set": {}, "delete": []}, '
    '"gear": {"set": {}, "delete": []}, "mission": {"set": {}, "delete": []}}:\n'
    "- set — новые записи и записи, значение которых изменилось; delete — записи, которые больше не верны;\n"
    "- из сообщений игрока бери то, что он сказал о герое, и то, что герой сделал или решил. Из рассказа Мастера — "
    "только то, что с героем случилось на самом деле: что он получил или потерял, что узнал, кого встретил. "
    "Описания, угрозы и догадки Мастера — не записи;\n"
    "- значения пиши точно и коротко (до 15 слов). Если значение изменилось — новое, а в скобках старое: "
    "«3 пилюли (было 5)»;\n"
    "- меняй число, только если прямо сказано, что оно изменилось. Другое число в другом смысле — не изменение;\n"
    "- ключи короткие, по-русски, в нижнем регистре через подчёркивание. Если запись уже есть — обнови тот же ключ "
    "в том же разделе, не заводи новый. Один ключ — в одном разделе;\n"
    "- вопросы игрока вне игры — не записи. Ничего не придумывай;\n"
    "- если записывать нечего — верни пустые set и delete.\n"
    "Верни только JSON.")

IMPORT_NOTE = ("Это не новое сообщение, а готовые факты о герое (ключ: значение) — легенда, записанная заранее. "
               "Разложи каждый факт по разделам, ключи и значения сохрани как есть.")

MEMORY_INTRO = ("\n\n# Память героя\n"
                "Старые сообщения и прошлые сцены тебе не отправлены — всё важное из них записано здесь. Это правда "
                "игрового мира: опирайся на неё, как на слова игрока.\n")
LONG_HEADER = "\n## Долговременная память — постоянное: личность, школа, характер, особенности, связи\n"
WORKING_HEADER = "\n## Рабочая память — текущий круг развития «{title}»\n"
SHORT_HEADER = "\n## Краткосрочная память — текущее задание «{title}»\n"
PROBE_SUFFIX = (" (Вопрос вне игры.) Ответь одним-двумя предложениями, только по тому, что известно о герое и игре. "
                "Если этого нет ни в памяти, ни в сцене — так и скажи. Кости не бросай.")


# ---------------------------------------------------------------- инструмент Мастера: кости


DICE = (4, 6, 8, 10, 12, 20, 100)


def roll_dice(count: int = 1, sides: int = 20, bonus: int = 0, reason: str = "") -> dict:
    try:
        count, sides, bonus = int(count), int(sides), int(bonus)
    except (TypeError, ValueError):
        return {"error": "count, sides и bonus — целые числа"}
    if sides not in DICE or not 1 <= count <= 10:
        return {"error": f"кости: от 1 до 10 штук, граней {', '.join(map(str, DICE))}"}
    rolls = [random.randint(1, sides) for _ in range(count)]
    return {"rolls": rolls, "bonus": bonus, "total": sum(rolls) + bonus, "reason": reason}


GAME_TOOLS = [
    Tool("roll_dice",
         "Бросок костей для исхода рискованного действия героя или случая в мире: count костей по sides граней "
         "плюс bonus. Обычная проверка — 1d20: 10 и выше — удача.",
         {"type": "object",
          "properties": {"count": {"type": "integer", "description": "сколько костей, 1–10"},
                         "sides": {"type": "integer", "description": "граней: 4, 6, 8, 10, 12, 20 или 100"},
                         "bonus": {"type": "integer", "description": "прибавка к сумме, например за навык"},
                         "reason": {"type": "string", "description": "что проверяется"}},
          "required": ["count", "sides", "reason"]},
         roll_dice),
]


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


class SceneStore(HistoryStore):
    """Сцена — история Дня 7. Архив — без перезаписи: две сцены, закрытые в одну секунду, не затирают друг друга."""

    def archive(self) -> Path | None:
        return archive_file(self.path)


def entries(raw: dict | None) -> dict[str, dict]:
    """Записи раздела из файла: ключ → {value, at, source, mission}; мусор отбрасывается."""
    out = {}
    for k, v in (raw or {}).items():
        if isinstance(v, dict) and str(v.get("value") or "").strip():
            out[str(k)] = {"value": str(v["value"]), "at": str(v.get("at") or ""), "source": str(v.get("source") or ""),
                           "mission": str(v.get("mission") or "")}
        elif isinstance(v, str) and v.strip():  # плоский словарь «ключ: значение» — например, руками
            out[str(k)] = {"value": v, "at": "", "source": "файл", "mission": ""}
    return out


class LongTermMemory:
    """Долговременная память: личность, школа, характер, особенности, связи. Один файл, живёт всю игру."""

    def __init__(self, path: Path, sections=LONG_SECTIONS):
        self.path = path
        self.sections = tuple(sections)
        self.data: dict[str, dict[str, dict]] = {s: {} for s in self.sections}

    def load(self) -> None:
        self.data = {s: {} for s in self.sections}
        if self.path.exists():
            raw = read_json(self.path)
            for s in self.sections:
                self.data[s] = entries(raw.get(s))

    def save(self) -> None:
        write_json(self.path, {"format": 1, "updated_at": now_iso(), **self.data})

    def count(self) -> int:
        return sum(len(v) for v in self.data.values())

    def clear(self) -> Path | None:
        archived = archive_file(self.path)
        self.data = {s: {} for s in self.sections}
        return archived


class SlotMemory:
    """Слот с одной активной записью и архивом прошлых: круг развития (рабочая память) или задание (краткосрочная).
    Активная — в <file>, прошлые — в <folder>/<id>.json. В промпт идёт только активная."""

    def __init__(self, root: Path, file: str, folder: str, sections, blank_title: str, noun: str):
        self.root = root
        self.path = root / file
        self.dir = root / folder
        self.sections = tuple(sections)
        self.blank_title = blank_title
        self.noun = noun
        self.slot: dict = {}
        self.slot = self.blank(blank_title)

    def blank(self, title: str) -> dict:
        return {"id": self.next_id(), "title": title, "status": "active", "created_at": now_iso(),
                "data": {s: {} for s in self.sections}}

    def next_id(self) -> str:
        used = [int(p.stem) for p in self.dir.glob("*.json") if p.stem.isdigit()] if self.dir.exists() else []
        current = self.slot
        if current and str(current.get("id", "")).isdigit():
            used.append(int(current["id"]))
        return str(max(used, default=0) + 1)

    @property
    def data(self) -> dict[str, dict[str, dict]]:
        return self.slot["data"]

    @property
    def title(self) -> str:
        return self.slot["title"]

    def count(self) -> int:
        return sum(len(v) for v in self.data.values())

    def _read(self, raw: dict, task_id: str) -> dict:
        data = raw.get("data") or {}
        return {"id": task_id, "title": str(raw.get("title") or self.blank_title), "status": "active",
                "created_at": str(raw.get("created_at") or now_iso()),
                "data": {s: entries(data.get(s)) for s in self.sections}}

    def load(self) -> None:
        if not self.path.exists():
            self.slot = {}
            self.slot = self.blank(self.blank_title)
            return
        raw = read_json(self.path)
        self.slot = self._read(raw, str(raw.get("id") or self.next_id()))

    def save(self) -> None:
        write_json(self.path, {"format": 1, "updated_at": now_iso(), **self.slot})

    def park(self, status: str) -> Path:
        """Убрать активную запись в архив <folder>/<id>.json со статусом (paused | done | passed)."""
        path = self.dir / f"{self.slot['id']}.json"
        write_json(path, {"format": 1, "updated_at": now_iso(), **self.slot, "status": status})
        return path

    def new(self, title: str, park_status: str = "paused", carry=()) -> dict:
        """Новая активная запись; прежняя уходит в архив. Разделы из carry переносятся в новую как есть."""
        carried = {s: copy.deepcopy(self.data.get(s) or {}) for s in carry}
        if self.count() or self.title != self.blank_title:
            self.park(park_status)
            self.slot = self.blank(title)
        else:  # пустая запись по умолчанию — просто даём ей название
            self.slot = dict(self.blank(title), id=self.slot["id"])
        self.slot["data"].update(carried)
        self.save()
        return self.slot

    def switch(self, slot_id: str) -> dict:
        path = self.dir / f"{slot_id}.json"
        if slot_id == self.slot["id"]:
            return self.slot
        if not path.exists():
            raise ValueError(f"нет записи «{self.noun}» {slot_id}. Есть: " + ", ".join(t["id"] for t in self.list()))
        raw = read_json(path)
        if raw.get("status") == "done":
            raise ValueError(f"{self.noun} {slot_id} «{raw.get('title')}» уже закрыто — вернуться к нему нельзя")
        self.park("paused")
        self.slot = self._read(raw, slot_id)
        self.save()
        return self.slot

    def done(self, status: str = "done") -> dict:
        closed = copy.deepcopy(self.slot)
        self.park(status)
        self.slot = self.blank(self.blank_title)
        self.save()
        return closed

    def clear(self) -> Path | None:
        """Забыть данные активной записи; копия остаётся в <folder>/<id>_cleared_<время>.json."""
        archived = None
        if self.count():
            archived = self.dir / f"{self.slot['id']}_cleared_{stamp_now()}.json"
            write_json(archived, {"format": 1, "updated_at": now_iso(), **self.slot, "status": "cleared"})
        self.slot = dict(self.slot, data={s: {} for s in self.sections})
        self.save()
        return archived

    def list(self) -> list[dict]:
        out = [{"id": self.slot["id"], "title": self.title, "status": "active", "fields": self.count()}]
        if self.dir.exists():
            for p in sorted(self.dir.glob("*.json"), key=lambda p: (len(p.stem), p.stem)):
                if not p.stem.isdigit() or p.stem == self.slot["id"]:
                    continue
                try:
                    raw = read_json(p)
                except StoreError:
                    continue
                fields = sum(len(v or {}) for v in (raw.get("data") or {}).values())
                out.append({"id": p.stem, "title": raw.get("title") or self.blank_title,
                            "status": raw.get("status") or "?", "fields": fields})
        return out


class MemoryLog:
    """Журнал записей в память (log.jsonl). Записи хода копятся в pending и пишутся после успешного хода —
    если ход откатился, его записи в журнал не попадают."""

    def __init__(self, path: Path, section_layer: dict | None = None):
        self.path = path
        self.section_layer = section_layer or SECTION_LAYER
        self.pending: list[dict] = []

    def add(self, op: str, section: str, key: str = "", value: str = "", source: str = "", **extra) -> dict:
        layer = self.section_layer.get(section, section)
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


def flatten_value(v) -> str:
    """Значение записи — строка. Вложенный объект или список (летописец иногда кладёт {"set": {…}} внутрь значения)
    разворачивается в строку из листьев через «; »."""
    if isinstance(v, dict):
        inner = v["set"] if isinstance(v.get("set"), dict) else v
        return "; ".join(flatten_value(x) for x in inner.values() if x not in (None, "", {}, []))
    if isinstance(v, list):
        return "; ".join(flatten_value(x) for x in v if x not in (None, "", {}, []))
    if isinstance(v, str) and v.lstrip().startswith(("{", "[")):  # JSON, положенный строкой
        try:
            parsed = json.loads(v)
        except ValueError:
            return v
        return flatten_value(parsed) if isinstance(parsed, (dict, list)) else v
    return str(v)


def clean_part(part: dict) -> dict:
    """{"set": {ключ: значение}, "delete": […]} с гарантированно строковыми значениями."""
    st = part.get("set")
    odd = lambda v: not isinstance(v, str) or v.lstrip().startswith(("{", "["))  # noqa: E731
    if isinstance(st, dict) and any(odd(v) for v in st.values()):
        part = dict(part, set={k: flatten_value(v) if odd(v) else v for k, v in st.items()})
    return part


def parse_route(text: str, sections=SECTIONS) -> dict[str, tuple[dict[str, str], list[str]]]:
    """Ответ летописца → {раздел: (set, delete)}. Каждый раздел разбирает парсер Sticky Facts из Дня 10."""
    data = extract_json_object(text)
    out = {}
    for s in sections:
        part = data.get(s) or {}
        if not isinstance(part, dict):
            raise ValueError(f"{s} — не объект")
        out[s] = StickyFacts.parse(json.dumps(clean_part(part), ensure_ascii=False)) if part else ({}, [])
    return out


SLOT_ICONS = {"mission": "📜", "circle": "☯"}


def op_label(item: dict, layer_titles: dict | None = None, section_titles: dict | None = None) -> str:
    lt, st = layer_titles or LAYER_TITLES, section_titles or SECTION_TITLES
    if item["op"] == "clear":
        return f"🧹 очищен слой: {lt.get(item['layer'], item['layer'])}"
    if item["op"] in ("mission", "circle"):
        noun = "задание" if item["op"] == "mission" else "круг"
        return f"{SLOT_ICONS[item['op']]} {noun} {item['key']} «{item['value']}» — {item.get('status', '')}"
    where = f"{lt.get(item['layer'], item['layer'])} · {st.get(item['section'], item['section'])}"
    op = item["op"]
    if op == "move":
        src = item.get("from", "")
        where = f"{st.get(src, src)} → {where}"
    sign = {"add": "+", "update": "~", "move": "↪", "delete": "−", "rejected": "⛔"}.get(op, op)
    value = f" = {clip(item['value'], 80)}" if item.get("value") and op != "delete" else ""
    old = (f" (было: {clip(item['old'], 40)})" if item.get("old") and op == "update"
           and normalize(item["old"]) not in normalize(item.get("value") or "") else "")
    why = f" — {item['why']}" if item.get("why") else ""
    return f"{sign} [{where}] {item['key']}{value}{old}{why}"


# ---------------------------------------------------------------- агент


def turn_source(scene: int | None, n: int | None) -> str:
    """Откуда запись летописца: номер хода сбрасывается с новой сценой, поэтому — вместе со сквозным номером сцены."""
    return f"сцена {scene}, ход {n}"


class LayeredAgent(TokenAgent):
    """TokenAgent из Дня 8 — Мастер игры с тремя слоями памяти героя. Сцена — его же история (short_term.json),
    карточка задания, круг развития и постоянное — отдельные файлы; запрос к модели собирает build() по правилам."""

    sections = SECTIONS
    section_layer = SECTION_LAYER
    router_prompt = ROUTER_PROMPT

    def __init__(self, client: LLMClient, tools, root: Path, *, keep_last: int = 6, long_budget: int = 1500,
                 card_budget: int = CARD_BUDGET, helper_client: LLMClient | None = None, **kwargs):
        self.root = root
        self.keep_last = max(keep_last, 0)
        self.long_budget = max(long_budget, 0)
        self.card_budget = max(card_budget, 0)
        self.long = LongTermMemory(root / "long_term.json")
        self.circle = SlotMemory(root, "working.json", "circles", WORKING_SECTIONS, NO_CIRCLE, "круг")
        self.mission = SlotMemory(root, "mission.json", "missions", SHORT_SECTIONS, NO_MISSION, "задание")
        self.log = MemoryLog(root / "log.jsonl", self.section_layer)
        self.router_path = root / "router.json"
        self.router_upto = 1  # сообщения history[1:router_upto] уже разобраны летописцем
        self.router_calls: list[HelperCall] = []
        self.dialogue_start = 0  # router_calls[dialogue_start:] — запросы летописца в текущей сцене
        self.scene = 1  # сквозной номер сцены: растёт с каждой новой сценой
        self.helper_client = helper_client or client
        self.layer_stats: list[dict] = []  # токены по слоям для каждого хода текущей сцены
        self.notes: list[str] = []
        self._turn_start: int | None = None
        self._parts: dict | None = None
        kwargs.setdefault("system_prompt", MASTER_PROMPT)
        super().__init__(client, tools, SceneStore(root / "short_term.json"), **kwargs)

    # --- загрузка и сохранение: каждый слой — свои файлы

    def load(self) -> RestoreInfo:
        self.long.load()
        self.circle.load()
        self.mission.load()
        saved = self.store.load()
        info = self.restore(saved) if saved else RestoreInfo(1, 0, 0, 0, "")
        if saved and saved.session_id:
            self.client.session_id = saved.session_id
        router = read_json(self.router_path) if self.router_path.exists() else {}
        upto = int(router.get("upto") or 1)
        if not 1 <= upto <= len(self.history):
            self.notes.append(f"летописец: учтено до сообщения {upto}, а в сцене {len(self.history)} — "
                              f"считаю сцену разобранной")
            upto = len(self.history)
        self.router_upto = upto
        known = HelperCall.__dataclass_fields__
        self.router_calls = [HelperCall(**{k: v for k, v in c.items() if k in known}) for c in router.get("calls") or []]
        self.dialogue_start = min(int(router.get("dialogue_start") or 0), len(self.router_calls))
        self.scene = int(router.get("scene") or 1)
        self.layer_stats = [t["layers"] for t in self.turns if isinstance(t.get("layers"), dict)]
        return info

    def save_memory(self) -> None:
        self.long.save()
        self.circle.save()
        self.mission.save()
        write_json(self.router_path, {"format": 1, "updated_at": now_iso(), "upto": self.router_upto,
                                      "dialogue_start": self.dialogue_start, "scene": self.scene,
                                      "calls": [asdict(c) for c in self.router_calls]})
        self.log.flush()

    def save(self) -> None:
        super().save()  # краткосрочная — сцена
        self.save_memory()

    # --- запись в память

    def section(self, name: str) -> dict[str, dict]:
        layer = SECTION_LAYER.get(name)
        if layer == "long":
            return self.long.data[name]
        if layer == "working":
            return self.circle.data[name]
        if layer == "short":
            return self.mission.data[name]
        raise ValueError(f"нет раздела «{name}»")

    def find(self, key: str) -> str | None:
        return next((s for s in self.sections if key in self.section(s)), None)

    def write(self, section: str, key: str, value: str, source: str) -> list[dict]:
        """Записать ключ в раздел. Один ключ — в одном разделе: если он лежал в другом, он переносится."""
        moved = None
        for other in self.sections:
            if other != section and key in self.section(other):
                moved = other
                self.section(other).pop(key)
        target = self.section(section)
        old = target.get(key)
        if old and old["value"] == value and not moved:
            return []
        target[key] = {"value": value, "at": now_iso(), "source": source, "mission": self.mission.title}
        op = "move" if moved else ("update" if old else "add")
        return [self.log.add(op, section, key, value, source, old=old["value"] if old else None, **{"from": moved})]

    def forget(self, section: str, key: str, source: str) -> list[dict]:
        old = self.section(section).pop(key, None)
        return [self.log.add("delete", section, key, old["value"], source)] if old else []

    def apply(self, changes: dict, source: str) -> list[dict]:
        ops = []
        for s in self.sections:
            update, delete = changes.get(s) or ({}, [])
            for key, value in update.items():
                ops += self.write(s, key, value, source)
            for key in delete:
                ops += self.forget(s, key, source)
        return ops

    def memory_json(self) -> dict:
        return {s: {k: e["value"] for k, e in self.section(s).items()} for s in self.sections}

    def pending_messages(self, end: int) -> list[tuple[str, str]]:
        """Что летописец ещё не разобрал: сообщения игрока и финальные рассказы Мастера (без вызовов костей)."""
        out = []
        for m in self.history[self.router_upto:end]:
            if m.get("role") == "user":
                out.append(("Игрок", m.get("content") or ""))
            elif m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls"):
                out.append(("Мастер", clip(m["content"], 1500)))
        return out

    def router_context(self) -> str:
        return f"Задание: «{self.mission.title}», круг развития: «{self.circle.title}»"

    def route(self, texts: list[str], source: str, imported: bool = False) -> list[dict]:
        """Служебный запрос летописца: последний рассказ Мастера и новые сообщения игрока (и пропущенные, если
        летописец падал) → изменения по разделам. Возвращает журнал применённых изменений."""
        end = self.turn_end()
        pending = [] if imported else self.pending_messages(end)
        pending += [("Игрок", t) for t in texts if t]
        if not pending:
            return []
        if imported:
            body = f"{IMPORT_NOTE}\n\n" + "\n\n".join(t for _, t in pending)
        else:
            body = "Новые сообщения:\n\n" + "\n\n".join(f"{i}. {who}: {t}" for i, (who, t) in enumerate(pending, 1))
        user = (f"{self.router_context()}\n\nТекущая память:\n"
                f"{json.dumps(self.memory_json(), ensure_ascii=False, indent=1)}\n\n{body}")
        result = self.helper_client.chat([{"role": "system", "content": self.router_prompt},
                                          {"role": "user", "content": user}], temperature=0)
        usage = Usage.parse(result.usage)
        priced = price_for(self.helper_client.model)
        call = HelperCall(strategy="router", before_turn=len(self.ledger) + (0 if imported else 1),
                          messages=len(pending), prompt=usage.prompt, completion=usage.completion, cached=usage.cached,
                          cost=usage.cost(priced[0] if priced else None), latency=result.latency, at=now_iso())
        self.router_calls.append(call)
        try:
            changes = self.parse_route(result.text)
        except ValueError as e:
            call.note = f"ответ не разобран: {e}"
            raise LLMError(f"ответ летописца не разобран ({e}): {clip(result.text, 120)!r}")
        ops = self.apply(changes, source)
        if not imported:
            self.router_upto = end + (1 if self._turn_start is not None else 0)
        call.note = self.route_note(ops)
        self.on_event("route", {"call": call, "ops": ops})
        return ops

    def parse_route(self, text: str) -> dict:
        return parse_route(text, self.sections)

    def route_note(self, ops: list[dict]) -> str:
        written = [o for o in ops if o["op"] != "rejected"]
        if not ops:
            return "изменений нет"
        counts = {layer: sum(o["layer"] == layer for o in written) for layer in LAYERS}
        rejected = len(ops) - len(written)
        return (f"изменений {len(written)}: долговременная {counts['long']}, рабочая {counts['working']}, "
                f"краткосрочная {counts['short']}" + (f"; отклонено {rejected}" if rejected else ""))

    def pending(self) -> int:
        """Сколько сообщений летописец ещё не разобрал."""
        return len(self.pending_messages(self.turn_end()))

    def catch_up(self) -> list[dict]:
        """Разобрать то, что летописец пропустил: последний рассказ Мастера и сообщения, на которых он падал."""
        return self.route([], "догон пропущенного")

    def flush_pending(self, before: str) -> None:
        """Перед концом сцены и сменой задания или круга: неразобранное — сначала в память, иначе оно уйдёт
        в архив вместе со сценой или ляжет не в то задание. Если летописец снова падает — действие не выполняется."""
        if not self.pending():
            return
        try:
            self.catch_up()
        except LLMError as e:
            self.log.drop()
            raise LLMError(f"{before}: отменено — летописец не разобрал {self.pending()} сообщ. ({e}). "
                           f"Повторите команду: иначе эти сообщения не попадут в память.")
        self.save_memory()

    # --- что уходит в модель: правило каждого слоя

    def turn_end(self) -> int:
        return len(self.history) if self._turn_start is None else self._turn_start

    def long_block(self) -> tuple[str, dict]:
        """Личность, школа, характер — целиком; особенности и связи — от новых к старым, пока влезают в бюджет."""
        est = self.estimator
        always = {s: [f"- {k}: {e['value']}" for k, e in self.long.data[s].items()] for s in LONG_ALWAYS}
        used = est.text("\n".join(line for lines in always.values() for line in lines))
        rest = sorted(((s, k, e) for s in LONG_BUDGETED for k, e in self.long.data[s].items()),
                      key=lambda x: x[2].get("at") or "", reverse=True)
        chosen, skipped = set(), []
        for s, k, e in rest:
            cost = est.text(f"- {k}: {e['value']}")
            if used + cost > self.long_budget:
                skipped.append(f"{s}.{k}")
                continue
            used += cost
            chosen.add((s, k))
        parts = []
        for s in LONG_SECTIONS:
            items = always.get(s) if s in always else [f"- {k}: {e['value']}" for k, e in self.long.data[s].items()
                                                         if (s, k) in chosen]
            if items:
                parts.append(f"### {SECTION_TITLES[s].capitalize()}\n" + "\n".join(items))
        info = {s: len(always[s]) for s in LONG_ALWAYS}
        info.update({s: sum(x == s for x, _ in chosen) for s in LONG_BUDGETED})
        info["skipped"] = skipped
        return (LONG_HEADER + "\n\n".join(parts) + "\n") if parts else "", info

    def working_block(self) -> str:
        """Текущий круг: все записи круга и снаряжение."""
        d = self.circle.data
        parts = [f"### {SECTION_TITLES[s].capitalize()}\n" + "\n".join(f"- {k}: {e['value']}" for k, e in d[s].items())
                 for s in WORKING_SECTIONS if d[s]]
        return (WORKING_HEADER.format(title=self.circle.title) + "\n\n".join(parts) + "\n") if parts else ""

    def mission_card(self) -> tuple[str, dict]:
        """Карточка задания в промпт: записи игры — целиком, записи летописца — от новых к старым в пределах
        card_budget токенов. В файле остаётся всё; то, что не влезло, просто не отправляется."""
        d = self.mission.data["mission"]
        if not d:
            return "", {"shown": 0, "skipped": []}
        est = self.estimator
        line = lambda k, e: f"- {k}: {e['value']}"  # noqa: E731
        game = {k for k, e in d.items() if e["source"].startswith("игра")}
        used, chosen, skipped = 0, set(), []
        for k, e in sorted(((k, e) for k, e in d.items() if k not in game), key=lambda x: x[1].get("at") or "",
                           reverse=True):
            cost = est.text(line(k, e))
            if used + cost > self.card_budget:
                skipped.append(k)
                continue
            used += cost
            chosen.add(k)
        lines = [line(k, e) for k, e in d.items() if k in game or k in chosen]
        return (SHORT_HEADER.format(title=self.mission.title) + "\n".join(lines) + "\n",
                {"shown": len(lines), "skipped": skipped})

    def mission_block(self) -> str:
        return self.mission_card()[0]

    def mission_full(self) -> str:
        """Вся карточка из файла — для поиска, где лежит деталь."""
        return "\n".join(f"- {k}: {e['value']}" for k, e in self.mission.data["mission"].items())

    def build(self, end: int | None = None, layers=LAYERS, tools: list[dict] | None = None) -> tuple[list[dict], int, dict]:
        """(сообщения запроса, сколько сообщений сцены не отправлено, ≈токены по частям запроса)."""
        end = self.turn_end() if end is None else end
        h, est = self.history, self.estimator
        long_text, long_info = self.long_block() if "long" in layers else ("", {})
        work_text = self.working_block() if "working" in layers else ""
        card_text = self.mission_block() if "short" in layers else ""
        intro = MEMORY_INTRO if long_text or work_text or card_text else ""
        base = h[0].get("content") or ""
        system = dict(h[0], content=base + intro + long_text + work_text + card_text)
        start = tail_start(h, end, self.keep_last) if "short" in layers else end
        tail, current = h[start:end], h[end:]
        raw = {"system": est.PER_REQUEST + est.PER_MESSAGE + est.raw_text(base + intro)
                         + (est.raw_text(json.dumps(tools, ensure_ascii=False)) if tools else 0),
               "long": est.raw_text(long_text), "working": est.raw_text(work_text),
               "short": est.raw_text(card_text) + sum(est.raw_message(m, False) for m in tail),
               "current": sum(est.raw_message(m) for m in current)}
        parts = {"raw": raw, "estimate": {k: est.apply(v) for k, v in raw.items()}, "tail_messages": len(tail),
                 "dropped": max(end - 1 - len(tail), 0), "long_info": long_info,
                 "card": est.apply(est.raw_text(card_text)) if card_text else 0}
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
        return {"turn": turn.n, "scene": self.scene, "mission_id": self.mission.slot["id"], "user": turn.user,
                "prompt": actual, "cached": turn.calls[0].cached if turn.calls else 0,
                "tokens": scaled, "tail_messages": parts.get("tail_messages", 0), "dropped": parts.get("dropped", 0),
                "long_info": parts.get("long_info") or {}, "card": parts.get("card", 0),
                "working_fields": self.circle.count(), "mission_fields": self.mission.count(),
                "long_entries": self.long.count(), "history_messages": len(self.history) - 1,
                "router": [asdict(c) for c in self.dialogue_calls() if c.before_turn == turn.n]}

    def dialogue_calls(self) -> list[HelperCall]:
        return self.router_calls[self.dialogue_start:]

    # --- ход

    def snapshot(self) -> dict:
        return copy.deepcopy({"long": self.long.data, "circle": self.circle.slot, "mission": self.mission.slot,
                              "upto": self.router_upto})

    def rollback(self, snap: dict) -> None:
        self.long.data, self.circle.slot, self.mission.slot = snap["long"], snap["circle"], snap["mission"]
        self.router_upto = snap["upto"]
        self.log.drop()

    def before_turn(self, user_text: str) -> None:
        """Перед ответом Мастера: летописец разбирает прошлый рассказ и новое сообщение игрока."""
        try:
            self.route([user_text], turn_source(self.scene, len(self.ledger) + 1))
        except LLMError as e:  # без свежей памяти Мастер всё равно ответит, а летописец догонит в следующий ход
            self.log.drop()
            self.on_event("route_error", {"error": str(e)})

    def run(self, user_text: str):
        self._turn_start, self._parts = len(self.history), None
        snap = self.snapshot()
        try:
            self.before_turn(user_text)
            reply = super().run(user_text)
        except BaseException:  # ход не прошёл (или Ctrl+C): то, что летописец успел записать из него, откатывается
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

    # --- сцена, задание, круг, очистка слоёв

    def reset(self) -> None:
        """Новая сцена: диалог уходит в архив; карточка задания, круг и постоянное остаются."""
        self.router_upto = 1
        self.layer_stats = []
        self.dialogue_start = len(self.router_calls)
        if getattr(self, "_ready", False):
            self.client.session_id = uuid.uuid4().hex  # новая сцена — новая сессия у провайдера
            self.scene += 1
        super().reset()

    def new_scene(self, before: str = "новая сцена") -> str:
        self.flush_pending(before)
        messages = len(self.history) - 1
        self.reset()
        return f"сцена ({messages} сообщ.) ушла в архив, начата новая"

    def clear(self, layer: str) -> str:
        if layer == "short":
            what = self.new_scene("очищение краткосрочной памяти")
            archived = self.mission.clear()
            what += f"; карточка задания «{self.mission.title}» очищена" + (
                f", копия — {archived.relative_to(self.root)}" if archived else "")
        elif layer == "working":
            archived = self.circle.clear()
            what = f"данные круга «{self.circle.title}» очищены" + (
                f", копия — {archived.relative_to(self.root)}" if archived else "")
        elif layer == "long":
            archived = self.long.clear()
            what = "личность, школа, характер, особенности и связи очищены" + (
                f", копия — {archived.name}" if archived else "")
        else:
            raise ValueError(f"нет слоя «{layer}». Есть: short, working, long")
        self.log.add("clear", layer, source="команда игрока")
        self.save_memory()
        return what

    def new_mission(self, title: str, card: dict[str, str] | None = None, source: str = "игра (карточка задания)") -> dict:
        """Новое задание: текущее откладывается, сцена начинается заново, карточка пишется прямо в краткосрочную."""
        self.new_scene("смена задания")
        m = self.mission.new(title)
        self.log.add("mission", "mission", m["id"], title, "команда игрока", status="new")
        for k, v in (card or {}).items():
            self.write("mission", k, v, source)
        self.save_memory()
        return m

    def switch_mission(self, mission_id: str) -> dict:
        self.new_scene("смена задания")
        m = self.mission.switch(mission_id)
        self.log.add("mission", "mission", m["id"], m["title"], "команда игрока", status="switch")
        self.save_memory()
        return m

    def done_mission(self, status: str = "done") -> dict:
        self.new_scene("закрытие задания")
        m = self.mission.done(status)
        self.log.add("mission", "mission", m["id"], m["title"], "команда игрока", status=status)
        self.save_memory()
        return m

    def next_circle(self, title: str, fields: dict[str, str] | None = None, source: str = "игра (новый круг)") -> dict:
        """Новый круг развития: прошлый — в архив, снаряжение переносится, поля нового круга пишет игра."""
        self.flush_pending("переход на новый круг")
        c = self.circle.new(title, park_status="passed", carry=CARRY_OVER)
        self.log.add("circle", "circle", c["id"], title, source, status="new")
        for k, v in (fields or {}).items():
            self.write("circle", k, v, source)
        self.save_memory()
        return c

    def router_totals(self) -> dict:
        costs = [c.cost for c in self.router_calls]
        return {"count": len(self.router_calls), "prompt": sum(c.prompt for c in self.router_calls),
                "completion": sum(c.completion for c in self.router_calls),
                "cost": None if any(c is None for c in costs) else sum(costs)}


# ---------------------------------------------------------------- вывод


def on_event(kind: str, data: dict) -> None:
    if kind == "route":
        c: HelperCall = data["call"]
        print(f"  📜 летописец: {c.note} | вход {fmt(c.prompt)}, выход {fmt(c.completion)}, {c.latency:.1f} с, "
              f"{money(c.cost)}")
        for item in data["ops"]:
            print(f"     {op_label(item)}")
    elif kind == "route_error":
        print(f"  ⚠️  летописец не сработал: {data['error']} — Мастер ответит с прежней памятью, догоню в следующий ход")
    elif kind == "tool" and data.get("name") == "roll_dice":
        r = data["result"]
        print(f"  🎲 {data['arguments'].get('reason', 'бросок')}: "
              + (f"{r['rolls']} + {r['bonus']} = {r['total']}" if "total" in r else r.get("error", "")))
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
          + (f" + летописец {money(sum(c.cost or 0 for c in router))}" if router else ""))


def print_memory(agent: LayeredAgent) -> None:
    est = agent.estimator
    h = agent.history
    tail = tail_start(h, len(h), agent.keep_last)
    m = agent.mission
    print(f"  ① краткосрочная — задание {m.slot['id']} «{m.title}», карточка: {m.count()} зап. "
          f"(≈{fmt(est.text(agent.mission_block()))} токенов); сцена {agent.store.path.name}: {len(h) - 1} сообщ., "
          f"в промпт — последние {len(h) - tail} (≈{fmt(est.apply(sum(est.raw_message(x, False) for x in h[tail:])))})")
    skipped = agent.mission_card()[1]["skipped"]
    for k, e in m.data["mission"].items():
        print(f"     · {k}: {e['value']}  ⟵ {e['source']}" + ("  (вне бюджета карточки)" if k in skipped else ""))
    others = [x for x in m.list() if x["status"] != "active"]
    if others:
        print("     задания вне промпта: " + "; ".join(f"{x['id']} «{x['title']}» ({x['status']})" for x in others))
    c = agent.circle
    print(f"  ② рабочая — круг {c.slot['id']} «{c.title}», записей {c.count()} "
          f"(≈{fmt(est.text(agent.working_block()))} токенов)")
    for s in WORKING_SECTIONS:
        for k, e in c.data[s].items():
            print(f"     · [{SECTION_TITLES[s]}] {k}: {e['value']}  ⟵ {e['source']}")
    passed = [x for x in c.list() if x["status"] != "active"]
    if passed:
        print("     прошлые круги вне промпта: " + "; ".join(f"{x['id']} «{x['title']}»" for x in passed))
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
    rows = [("system", "правила Мастера, мир, схемы инструментов", "всегда"),
            ("long", "долговременная", RULES["long"]["prompt"].format(budget=agent.long_budget)),
            ("working", "рабочая", RULES["working"]["prompt"]),
            ("short", "краткосрочная", RULES["short"]["prompt"].format(keep=agent.keep_last, card=agent.card_budget))]
    for k, title, rule in rows:
        print(f"   {title:<42} ≈{fmt(e[k]):>6}  — {rule}")
    print(f"   из них карточка задания ≈{fmt(parts['card'])}; в хвосте {parts['tail_messages']} сообщ., "
          f"не отправлено {dropped} старых сообщений сцены")


def print_rules(agent: LayeredAgent) -> None:
    for layer in ("long", "working", "short"):
        r = RULES[layer]
        print(f"  {LAYER_TITLES[layer].upper()} ({r['file']})")
        print(f"     хранит:   {r['stores']}")
        print(f"     пишет:    {r['writes']}")
        print(f"     в промпт: {r['prompt'].format(keep=agent.keep_last, budget=agent.long_budget, card=agent.card_budget)}")
        print(f"     очистка:  {r['cleared']}")
    print("  ЛЕТОПИСЕЦ: " + ", ".join(f"{s} → {LAYER_TITLES[SECTION_LAYER[s]]}" for s in SECTIONS))
    print(f"  ПРИ НОВОМ КРУГЕ ПЕРЕНОСИТСЯ: {', '.join(SECTION_TITLES[s] for s in CARRY_OVER)}")
    print(f"  НЕ СОХРАНЯЕТСЯ: {NOT_SAVED}")


# ---------------------------------------------------------------- сценарий и контрольные вопросы


@dataclass(frozen=True)
class MissionCard:
    id: str
    title: str
    card: dict


MISSIONS = {
    "caravan": MissionCard("caravan", "Пропавший караван", {
        "заказчик": "управляющий Ван из города Речная Переправа",
        "цель": "найти караван с лекарственными травами и вернуть груз",
        "место": "Ущелье Тысячи Ветров",
        "срок": "до полнолуния, через пять дней",
        "награда": "две пилюли собирания ци и 30 лянов серебра"}),
    "escort": MissionCard("escort", "Охрана лекарки", {
        "заказчик": "лекарка Фан Жоу из Долины Нефритового Лотоса",
        "цель": "довести её с сундуком лекарств до деревни Каменный Ручей",
        "место": "горная тропа к деревне Каменный Ручей",
        "срок": "три дня",
        "награда": "20 лянов серебра"}),
}

# Сценарий: строка — реплика игрока, ("mission", id) — игра выдаёт задание (карточка → краткосрочная, новая сцена).
SCENARIO = [
    "Начнём. Мой герой — Лу Чэнь, семнадцать лет, сирота из рыбацкой деревни Ивовая Заводь. Три года назад деревню "
    "сжёг Ци Ху из секты Кровавой Луны — это кровный враг Лу Чэня.",
    "Лу Чэнь — ученик Секты Лазурного Облака, его наставник — старейшина Мо Шуй. У него двойной корень духа: ветер "
    "и вода. Характер упрямый, лжи не терпит, а над могилой матери он поклялся никогда не поднимать меч "
    "на безоружного.",
    "Сейчас у него третий год обучения: круг Очищения ци, вторая ступень. Он изучает технику «Девять порывов ветра» — "
    "освоено пять форм из девяти. Ци накоплено 140 из 300.",
    "Наставник сказал: для прорыва на третью ступень нужно собрать 300 ци и освоить шестую форму. Из снаряжения — "
    "меч «Тихий тростник», три пилюли собирания ци и двенадцать лянов серебра. Иду в зал заданий секты.",
    ("mission", "caravan"),
    "Еду в Речную Переправу и нахожу управляющего Вана. Расспрашиваю, сколько было повозок и кто охранял караван.",
    "Бросаю кости на внимательность и осматриваю дорогу у входа в ущелье.",
    "Среди обломков нахожу колесо с меткой красного полумесяца. Забираю его как улику и иду глубже в ущелье.",
    "Ночую в пещере у водопада. Перед сном медитирую — пусть ветер ущелья очистит мой ци.",
    "Утром замечаю следы трёх всадников, они ведут к заброшенному храму Серого Журавля. Крадусь туда.",
    "У храма вижу часового в красном. Он безоружен — сидит у костра и греет руки. Не нападаю: помню клятву. "
    "Пытаюсь подслушать, о чём он бормочет.",
    "Отхожу назад и отправляю наставнику Мо Шую письмо с почтовым голубем: прошу совета.",
    "Пока жду ответа, решаю обойти храм с севера, по скалам.",
]
PLAYER_LINES = [s for s in SCENARIO if isinstance(s, str)]

PROBES = [
    Detail("name", "личность", 1, "герой — Лу Чэнь из деревни Ивовая Заводь", "Как зовут моего героя и откуда он родом?",
           (r"лу\s*чэн", r"ивов")),
    Detail("enemy", "связи", 1, "кровный враг — Ци Ху из Кровавой Луны", "Кто кровный враг моего героя?",
           (r"ци\s*ху",)),
    Detail("school", "школа", 2, "Секта Лазурного Облака, наставник — старейшина Мо Шуй",
           "К какой школе принадлежит мой герой и кто его наставник?", (r"лазурн", r"мо\s*шу")),
    Detail("root", "особенности", 2, "двойной корень духа: ветер и вода", "Какой корень духа у моего героя?",
           (r"ветер|ветр", r"\bвод[аыуе]\b")),
    Detail("oath", "характер", 2, "клятва: не поднимать меч на безоружного", "Какую клятву дал мой герой?",
           (r"безоруж",)),
    Detail("circle", "круг", 3, "Очищение ци, вторая ступень", "На каком круге и ступени сейчас мой герой?",
           (r"очищени", r"втор|\b2\b")),
    Detail("technique", "круг", 3, "«Девять порывов ветра», освоено 5 форм из 9",
           "Какую технику изучает мой герой и сколько форм уже освоено?", (r"порыв", r"пят|\b5\b")),
    Detail("breakthrough", "круг", 4, "прорыв: 300 ци и шестая форма", "Что нужно моему герою для прорыва?",
           (r"300", r"шест|\b6\b")),
    Detail("pills", "снаряжение", 4, "три пилюли собирания ци", "Сколько пилюль собирания ци у моего героя?",
           (r"(?:\bтри\b|\b3\b)\s+(?:\w+\s+){0,2}пилюл|пилюл\w*\D{0,40}(?:\bтри\b|\b3\b)",)),
    Detail("giver", "задание", 5, "заказчик — управляющий Ван из Речной Переправы", "Кто дал текущее задание?",
           (r"\bван",)),
    Detail("place", "задание", 5, "место — Ущелье Тысячи Ветров", "Где пропал караван?", (r"ущель",)),
    Detail("deadline", "задание", 5, "срок — до полнолуния", "До какого срока нужно выполнить задание?",
           (r"полнолун",)),
    Detail("clue", "задание", 8, "улика — колесо с меткой красного полумесяца", "Какую улику нашёл мой герой?",
           (r"полумесяц",)),
    Detail("last", "сцена", 13, "последнее решение — обойти храм с севера по скалам",
           "Что мой герой решил сделать последним, до этого вопроса?", (r"обойти|север|скал",)),
]

CONFIGS = [  # (id, что очищено, какие слои остаются)
    ("all", "ничего", ("short", "working", "long")),
    ("no_short", "краткосрочная", ("working", "long")),
    ("no_working", "рабочая", ("short", "long")),
    ("no_long", "долговременная", ("short", "working")),
    ("long_only", "краткосрочная + рабочая", ("long",)),
    ("none", "все три слоя", ()),
]


def ask(agent: LayeredAgent, question: str, suffix: str = PROBE_SUFFIX) -> dict:
    """Вопрос вне игры отдельным запросом с текущей памятью; в сцену он не попадает."""
    schemas = [t.schema() for t in agent.tools.values()]
    messages, _, parts = agent.build(end=len(agent.history), tools=schemas)
    priced = price_for(agent.client.model)
    try:
        result = agent.client.chat(messages + [{"role": "user", "content": question + suffix}], tools=schemas)
    except LLMError as e:
        return {"answer": "", "error": str(e), "prompt": 0, "cached": 0, "completion": 0, "cost": 0.0,
                "estimate": parts["estimate"]}
    usage = Usage.parse(result.usage)
    answer = result.text.strip() or ("(модель попросила бросок костей вместо ответа)" if result.tool_calls else "")
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
    card = agent.mission_block()
    return {"short_file": matches(d, text(h[1:]) + agent.mission_full()), "short_prompt": matches(d, text(tail) + card),
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
    """Копия героя в другой папке — чтобы чистить слои, не трогая оригинал."""
    agent.save()
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(agent.root, root)
    # своя копия клиента: своя сессия у провайдера (очистка сцены её меняет), общий счётчик запросов
    copy_ = type(agent)(client or copy.copy(agent.client), list(agent.tools.values()), root, keep_last=agent.keep_last,
                        long_budget=agent.long_budget, card_budget=agent.card_budget, helper_client=agent.helper_client,
                        max_steps=agent.max_steps, on_event=event)
    copy_.load()
    copy_.estimator.factor, copy_.estimator.samples = agent.estimator.factor, max(agent.estimator.samples, 1)
    return copy_


def experiment(agent: LayeredAgent, root: Path, configs=CONFIGS, probes: list[Detail] = PROBES) -> dict:
    """Очистить слои по очереди (каждая конфигурация — на своей копии героя, очистка настоящая: /clear)
    и задать одни и те же контрольные вопросы."""
    try:
        agent.catch_up()
    except LLMError as e:
        print(f"  ⚠️  летописец не догнал пропущенное: {e}")
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
    print(f"\n{'=' * 78}\nОчистка слоёв: на какие вопросы Мастер перестаёт отвечать\n{'=' * 78}")
    print("  К — краткосрочная (карточка задания или хвост сцены), Р — рабочая, Д — долговременная")
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


# ---------------------------------------------------------------- импорт фактов (формат Sticky Facts Дня 10)


# Легенда героини в формате Sticky Facts Дня 10: плоский словарь «ключ → значение», как его вёл блок фактов.
LEGEND = {
    "имя": "Фэн Ся", "возраст": "19 лет", "родина": "город Туманная Пристань",
    "школа": "Долина Нефритового Лотоса", "наставница": "мастер Цин И", "корень_духа": "дерево",
    "характер": "осторожная, насмешливая, держит слово", "обет": "лечить любого раненого, даже врага",
    "побратим": "Лу Чэнь из Секты Лазурного Облака", "круг": "Закалка тела, третья ступень",
    "техника": "«Ладонь весенней росы», освоено 2 формы из 5", "условие_прорыва": "100 ци и третья форма",
    "снаряжение": "игольница с серебряными иглами, пять пилюль заживления",
    "задание": "собрать лунную траву у Озера Тумана", "срок_задания": "до рассвета",
    "заказчик": "старшая сестра Лань из Долины",
}


def load_facts(path: Path | None = None) -> tuple[dict[str, str], str]:
    """Факты «ключ → значение»: легенда по умолчанию; файл — отчёт Дня 10 (results/day10_*.json),
    <ветка>.state.json Дня 10, {"facts": {...}} или плоский JSON-словарь."""
    if path is None:
        return dict(LEGEND), "легенда Фэн Ся (встроенная)"
    data = read_json(path)
    facts = {}
    for run in data.get("runs") or []:
        if run.get("strategy") == "facts" and run.get("facts"):
            facts = run["facts"]
    if not facts:
        facts = ((data.get("strategies") or {}).get("facts") or {}).get("facts") or data.get("facts") or {}
    if not facts and all(isinstance(v, str) for v in data.values()):
        facts = data
    if not facts:
        raise StoreError(f"в {path} нет фактов: ни runs[facts].facts, ни strategies.facts.facts, ни facts")
    where_ = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return {str(k): str(v) for k, v in facts.items()}, str(where_)


def import_facts(agent: LayeredAgent, path: Path | None = None) -> dict:
    facts, where_ = load_facts(path)
    text = "\n".join(f"- {k}: {v}" for k, v in facts.items())
    print(f"  📥 факты из {where_}: {len(facts)} шт. → летописец раскладывает по слоям")
    ops = agent.route([text], f"импорт фактов ({clip(where_, 60)})", imported=True)
    agent.save_memory()
    placed = {k: agent.find(k) for k in facts}
    renamed = [o["key"] for o in ops if o["key"] not in facts]
    return {"file": where_, "facts": facts, "placed": placed, "ops": ops, "renamed": renamed,
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


# ---------------------------------------------------------------- игра в консоли


HELP = ("/memory, /context, /rules, /remember <раздел> <ключ> = <значение>, /forget <раздел> <ключ>, "
        "/move <ключ> <раздел>, /clear short|working|long, /newchat, "
        "/mission [new <id|название>|switch <id>|done|list], /circle [next <название>|list], /import-facts [файл], "
        "/play [N], /probe, /experiment, /log, /tokens, /history, /exit")


def section_arg(name: str, aliases: dict = SECTION_ALIASES) -> str:
    s = aliases.get(name.lower())
    if not s:
        raise ValueError(f"нет раздела «{name}». Есть: {', '.join(sorted(set(aliases.values())))}")
    return s


def print_ops(ops: list[dict]) -> None:
    for item in ops:
        print(f"  {op_label(item)}")
    if not ops:
        print("  (ничего не изменилось)")


def start_mission(agent: LayeredAgent, arg: str) -> dict:
    card = MISSIONS.get(arg.strip())
    return agent.new_mission(card.title, card.card) if card else agent.new_mission(arg.strip() or NO_MISSION)


def mission_command(agent: LayeredAgent, arg: str) -> None:
    args = arg.split()
    sub = args[0] if args else "list"
    if sub == "new":
        m = start_mission(agent, arg.partition(" ")[2])
        print(f"  📜 задание {m['id']} «{m['title']}», карточка: {agent.mission.count()} зап.; новая сцена"
              + (f"; есть готовые: {', '.join(MISSIONS)}" if not agent.mission.count() else ""))
    elif sub == "switch" and len(args) > 1:
        m = agent.switch_mission(args[1])
        print(f"  📜 активное задание {m['id']} «{m['title']}», карточка: {agent.mission.count()} зап.; новая сцена")
    elif sub == "done":
        m = agent.done_mission()
        print(f"  ✅ задание {m['id']} «{m['title']}» сдано: карточка и сцена — в архиве")
    else:
        for t in agent.mission.list():
            print(f"  {'▶' if t['status'] == 'active' else '·'} {t['id']} «{t['title']}» — {t['status']}, "
                  f"записей {t['fields']}")


def circle_command(agent: LayeredAgent, arg: str) -> None:
    args = arg.split()
    sub = args[0] if args else "list"
    if sub == "next":
        title = arg.partition(" ")[2].strip() or NO_CIRCLE
        c = agent.next_circle(title)
        print(f"  ☯ новый круг {c['id']} «{c['title']}»; перенесено: "
              + ", ".join(f"{SECTION_TITLES[s]} ({len(c['data'][s])})" for s in CARRY_OVER))
    else:
        for t in agent.circle.list():
            print(f"  {'▶' if t['status'] == 'active' else '·'} {t['id']} «{t['title']}» — {t['status']}, "
                  f"записей {t['fields']}")


def scenario_progress(agent: LayeredAgent, value: int | None = None) -> int:
    """Сколько шагов сценария уже пройдено этим героем (scenario.json рядом со слоями)."""
    path = agent.root / "scenario.json"
    if value is not None:
        write_json(path, {"done": value, "updated_at": now_iso()})
        return value
    return int(read_json(path).get("done") or 0) if path.exists() else 0


def play_scenario(agent: LayeredAgent, scenario=SCENARIO, count: int | None = None, verbose: bool = True) -> list[dict]:
    """Пройти сценарий с места, где остановились: строка — ход игрока, ("mission", id) — игра выдаёт задание."""
    done = scenario_progress(agent)
    items = scenario[done:] if count is None else scenario[done:done + count]
    out = []
    for k, item in enumerate(items, done + 1):
        if isinstance(item, tuple):
            m = start_mission(agent, item[1])
            print(f"\n--- {k}/{len(scenario)}: 📜 игра выдаёт задание «{m['title']}» — карточка в краткосрочную, "
                  f"новая сцена")
            out.append({"n": k, "mission": item[1], "card": agent.memory_json()["mission"]})
        else:
            print(f"\n--- {k}/{len(scenario)}\nИгрок: {item}" if verbose
                  else f"\n--- {k}/{len(scenario)}: {clip(item, 110)}")
            try:
                reply = agent.run(item)
                print(f"\nМастер: {reply.text.strip() or '(пустой ответ)'}\n" if verbose
                      else f"  Мастер: {clip(reply.text, 200) if reply.text.strip() else '(пустой ответ)'}")
            except LLMError as e:
                print(f"  ❌ {e}")
            print_turn(agent, agent.ledger[-1])
            out.append({"n": k, "line": item})
        scenario_progress(agent, k)
    if not items:
        print(f"  Сценарий пройден ({len(scenario)} шагов). /probe, /experiment или /mission new.")
    return out


def chat(agent: LayeredAgent) -> int:
    print(f"Мастер на {agent.client.model}, герой {agent.root.name}. Задание: «{agent.mission.title}», "
          f"круг: «{agent.circle.title}».\nКоманды: {HELP}\n")
    while True:
        try:
            text = input(f"[{agent.root.name} · задание {agent.mission.slot['id']}] Игрок: ").strip()
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
                    print("  /remember <раздел> <ключ> = <значение>; разделы: " + ", ".join(SECTIONS))
                else:
                    print_ops(agent.write(section_arg(parts[0]), parts[1], value.strip(), "команда игрока"))
                    agent.save_memory()
            elif command == "/forget":
                if len(args) != 2:
                    print("  /forget <раздел> <ключ>")
                else:
                    print_ops(agent.forget(section_arg(args[0]), args[1], "команда игрока"))
                    agent.save_memory()
            elif command == "/move":
                src = agent.find(args[0]) if len(args) == 2 else None
                if not src:
                    print("  /move <ключ> <раздел> — ключ должен быть в памяти (см. /memory)")
                else:
                    value = agent.section(src)[args[0]]["value"]
                    print_ops(agent.write(section_arg(args[1]), args[0], value, "команда игрока"))
                    agent.save_memory()
            elif command == "/clear":
                layer = LAYER_ALIASES.get(args[0].lower()) if args else None
                if not layer:
                    print("  /clear short|working|long")
                else:
                    print(f"  🧹 {LAYER_TITLES[layer]}: {agent.clear(layer)}")
            elif command == "/newchat":
                print(f"  🧹 {agent.new_scene()}; карточка задания, круг и постоянное остаются")
            elif command == "/mission":
                mission_command(agent, arg)
            elif command == "/circle":
                circle_command(agent, arg)
            elif command == "/import-facts":
                print_import(import_facts(agent, Path(arg.strip()) if arg.strip() else None))
            elif command == "/log":
                for item in agent.log.tail(int(args[0]) if args and args[0].isdigit() else 20):
                    print(f"  {item['at'][11:19]} {op_label(item)}  ⟵ {item.get('source', '')}")
            elif command == "/tokens":
                print_ledger(agent)
                h = agent.router_totals()
                print(f"  + летописец: запросов {h['count']}, вход {fmt(h['prompt'])}, выход {fmt(h['completion'])}, "
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
                play_scenario(agent, count=int(args[0]) if args and args[0].isdigit() else None)
            elif command.startswith("/"):
                print(f"  Неизвестная команда. Есть: {HELP}")
            else:
                reply = agent.run(text)
                print(f"\nМастер: {reply.text.strip() or '(пустой ответ)'}\n")
                print_turn(agent, agent.ledger[-1])
            print()
        except LLMError as e:
            print(f"❌ Ошибка: {e}\n", file=sys.stderr)
        except (StoreError, ValueError) as e:
            print(f"⚠️  {e}\n", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n(прервано — ход в сцену не попал)\n")


# ---------------------------------------------------------------- демо: смена задания и круга


ESCORT_TURN = ("Встречаю лекарку Фан Жоу у ворот секты, помогаю привязать сундук с лекарствами к мулу. Говорю, "
               "что выходим на рассвете.")
NEXT_CIRCLE = ("Очищение ци, третья ступень", {
    "круг": "Очищение ци", "ступень": "третья", "год_обучения": "третий", "ци": "0 из 450",
    "техника": "«Девять порывов ветра», освоено шесть форм из девяти",
    "условие_прорыва": "450 ци и седьмая форма «Девяти порывов ветра»"})
# (id, вопрос, фаза → (что должно быть, что пришло из другого задания или круга)); фазы: B — второе задание,
# A — снова первое, C — первое задание на новом круге.
SWITCH_PROBES = [
    ("name", "Как зовут моего героя?", {"B": ((r"лу\s*чэн",), ()), "A": ((r"лу\s*чэн",), ()),
                                       "C": ((r"лу\s*чэн",), ())}),
    ("mentor", "Кто наставник моего героя?", {"B": ((r"мо\s*шу",), ()), "A": ((r"мо\s*шу",), ()),
                                             "C": ((r"мо\s*шу",), ())}),
    ("giver", "Кто заказчик текущего задания?", {"B": ((r"фан\s*жо",), (r"\bван",)), "A": ((r"\bван",), (r"фан\s*жо",)),
                                                "C": ((r"\bван",), (r"фан\s*жо",))}),
    ("deadline", "Какой срок у текущего задания?", {"B": ((r"тр[иеё]\s*дн|\b3\s*дн",), (r"полнолун",)),
                                                    "A": ((r"полнолун",), (r"тр[иеё]\s*дн|\b3\s*дн",))}),
    ("qi", "Сколько ци нужно накопить моему герою для следующего прорыва?",
     {"A": ((r"\b300\b",), (r"\b450\b",)), "C": ((r"\b450\b",), ())}),
    ("pills", "Сколько пилюль собирания ци у моего героя?", {"C": ((r"(?:\bтри\b|\b3\b)\s+(?:\w+\s+){0,2}пилюл|пилюл\w*\D{0,40}(?:\bтри\b|\b3\b)",), ())}),
]


def check_switch(phase: str, expect: dict, answer: str) -> dict:
    t = normalize(answer)
    must, leak = expect[phase]
    found_leak = [p for p in leak if re.search(p, t)]
    has = all(re.search(p, t) for p in must)
    return {"ok": has and not found_leak, "must": has, "leak": found_leak}


def switch_demo(agent: LayeredAgent) -> dict:
    """Смена задания и круга: что переходит (постоянное, снаряжение), а что остаётся в своём задании или круге."""
    print(f"\n{'=' * 78}\n=== 4. Смена задания и круга: что переходит, а что остаётся на месте\n{'=' * 78}")
    mission_a = agent.mission.slot["id"]
    title_a = agent.mission.title
    steps = []

    def probe(phase: str, title: str) -> None:
        out = []
        for pid, question, expect in SWITCH_PROBES:
            if phase not in expect:
                continue
            r = ask(agent, question)
            r.update({"id": pid, "question": question, **check_switch(phase, expect, r["answer"] or r["error"])})
            out.append(r)
            mark = "✅" if r["ok"] else ("⚠️ утечка" if r["leak"] else "❌")
            print(f"  {mark} {question} → {clip(r['answer'] or r['error'], 140)}")
        steps.append({"step": "probe", "phase": phase, "title": title, "mission": agent.mission.title,
                      "circle": agent.circle.title, "probes": out, "card": agent.memory_json()["mission"],
                      "working": {s: agent.memory_json()[s] for s in WORKING_SECTIONS}})

    card = MISSIONS["escort"]
    m = agent.new_mission(card.title, card.card)
    print(f"  📜 /mission new escort: задание {m['id']} «{card.title}», задание {mission_a} «{title_a}» — отложено; "
          f"новая сцена")
    print(f"\nИгрок: {ESCORT_TURN}")
    try:
        reply = agent.run(ESCORT_TURN)
        print(f"Мастер: {clip(reply.text, 220)}")
        print_turn(agent, agent.ledger[-1])
        steps.append({"step": "turn", "text": ESCORT_TURN, "answer": reply.text})
    except LLMError as e:
        print(f"  ❌ {e}")
        steps.append({"step": "turn", "text": ESCORT_TURN, "error": str(e)})
    print(f"\n--- B. Второе задание «{card.title}»:")
    probe("B", f"задание «{card.title}»")
    m = agent.switch_mission(mission_a)
    print(f"\n  📜 /mission switch {mission_a}: «{m['title']}», карточка {agent.mission.count()} зап.; новая сцена")
    print(f"\n--- A. Снова первое задание «{title_a}»:")
    probe("A", f"снова «{title_a}»")
    c = agent.next_circle(*NEXT_CIRCLE)
    print(f"\n  ☯ /circle next: круг {c['id']} «{c['title']}», прошлый — в архиве; снаряжение перенесено "
          f"({len(c['data']['gear'])} зап.)")
    print(f"\n--- C. Новый круг «{c['title']}», то же задание:")
    probe("C", f"новый круг «{c['title']}»")
    checks = [p for s in steps if s["step"] == "probe" for p in s["probes"]]
    print(f"\n  Итог: {sum(p['ok'] for p in checks)}/{len(checks)} проверок, утечек: {sum(bool(p['leak']) for p in checks)}")
    return {"mission_a": mission_a, "steps": steps, "passed": sum(p["ok"] for p in checks), "total": len(checks),
            "leaks": sum(bool(p["leak"]) for p in checks), "missions": agent.mission.list(),
            "circles": agent.circle.list(), "memory": agent.memory_json()}


# ---------------------------------------------------------------- демо


def make_agent(client: LLMClient, helper: LLMClient, root: Path, keep: int, budget: int, max_steps: int,
               event=on_event, card: int = CARD_BUDGET) -> LayeredAgent:
    watch_retries(client, helper, event=event)
    return LayeredAgent(client, GAME_TOOLS, root, keep_last=keep, long_budget=budget, card_budget=card,
                        helper_client=helper, max_steps=max_steps, on_event=event)


def layers_snapshot(agent: LayeredAgent) -> dict:
    return {"short": {"file": agent.store.path.name, "messages": len(agent.history) - 1,
                      "tail": len(agent.history) - tail_start(agent.history, len(agent.history), agent.keep_last),
                      "mission": copy.deepcopy(agent.mission.slot)},
            "working": copy.deepcopy(agent.circle.slot), "long": copy.deepcopy(agent.long.data),
            "missions": agent.mission.list(), "circles": agent.circle.list()}


def demo(client: LLMClient, helper_model: str | None, keep: int, budget: int, max_steps: int, do_import: bool,
         do_switch: bool, card: int = CARD_BUDGET) -> int:
    stamp = stamp_now()
    root = DAY_DIR / f"demo_{stamp}"
    helper = (LLMClient(client.base_url, client.api_key, helper_model, client.timeout, retries=client.retries)
              if helper_model else client)
    report: dict = {"model": client.model, "helper_model": helper.model, "keep_last": keep, "long_budget": budget,
                    "card_budget": card,
                    "stamp": stamp, "dir": str(root.relative_to(ROOT)), "rules": RULES, "router_prompt": ROUTER_PROMPT,
                    "master_prompt": MASTER_PROMPT, "section_layer": SECTION_LAYER,
                    "scenario": [x if isinstance(x, str) else {"mission": x[1]} for x in SCENARIO],
                    "missions": {k: asdict(v) for k, v in MISSIONS.items()}, "probes": [asdict(d) for d in PROBES]}

    if do_import:
        print(f"{'=' * 78}\n=== 1. Легенда в формате Sticky Facts Дня 10 → слои памяти\n{'=' * 78}")
        imp = make_agent(client, helper, root / "import", keep, budget, max_steps, quiet_event, card)
        imp.load()
        try:
            report["import"] = import_facts(imp)
            print_import(report["import"])
            report["import"]["layers"] = layers_snapshot(imp)
        except (LLMError, StoreError) as e:
            print(f"  ⚠️  импорт не удался: {e}")
            report["import"] = {"error": str(e)}

    print(f"\n{'=' * 78}\n=== 2. Сценарий: {len(PLAYER_LINES)} реплик игрока и одно задание; летописец раскладывает "
          f"по слоям\n{'=' * 78}")
    agent = make_agent(client, helper, root / "main", keep, budget, max_steps, quiet_event, card)
    agent.load()
    played = play_scenario(agent, verbose=False)
    try:
        agent.catch_up()
        agent.save_memory()
    except LLMError as e:
        print(f"  ⚠️  догон летописца не удался: {e}")
    print("\n--- память после сценария:")
    print_memory(agent)
    log = agent.log.tail(10_000)
    report["dialogue_run"] = {
        "played": played, "log": log,
        "scene_turns": [{"n": x.n, "status": x.status, "prompt": x.calls[0].prompt if x.calls else 0,
                         "cached": x.cached, "completion": x.completion, "cost": x.cost, "answer": x.answer,
                         "error": x.error, "user": x.user} for x in agent.ledger],
        "layer_stats": copy.deepcopy(agent.layer_stats), "totals": agent.totals(), "router": agent.router_totals(),
        "router_calls": [asdict(c) for c in agent.router_calls], "layers": layers_snapshot(agent),
        "prologue": prologue_stats(agent)}

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


def prologue_stats(agent: LayeredAgent) -> list[dict]:
    """Токены по слоям ходов пролога (до задания): их сцена ушла в архив, а статистика — в архивном файле."""
    out = []
    for p in sorted(agent.root.glob("short_term_*.json")):
        try:
            data = read_json(p)
        except StoreError:
            continue
        for t in data.get("turns") or []:
            if isinstance(t.get("layers"), dict):
                out.append({"user": t.get("user", ""), **t["layers"]})
    return out


# ---------------------------------------------------------------- отчёт


def md_escape(text: str) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def save_report(report: dict, stamp: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path, md_path = RESULTS_DIR / f"day11_{stamp}.json", RESULTS_DIR / f"day11_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    keep, budget, card = report["keep_last"], report["long_budget"], report.get("card_budget", CARD_BUDGET)
    L = [f"# День 11 · Модель памяти героя (уся-РПГ) — прогон {stamp}", "",
         f"Мастер игры: `{report['model']}`, летописец (роутер памяти): `{report['helper_model']}`. Хвост сцены — "
         f"{keep} сообщений, бюджет долговременной — {budget} токенов, записей летописца в карточке задания — {card}. "
         f"Файлы прогона: `{report['dir']}/`.", "",
         "## Модель памяти", "",
         "| Слой | Что хранит | Файлы | Кто пишет | Что попадает в промпт | Когда очищается |", "|---|---|---|---|---|---|"]
    for layer in ("long", "working", "short"):
        r = RULES[layer]
        L.append(f"| {LAYER_TITLES[layer]} | {r['stores']} | `{r['file']}` | {r['writes']} | "
                 f"{r['prompt'].format(keep=keep, budget=budget, card=card)} | {r['cleared']} |")
    L += ["", "Летописец: " + ", ".join(f"`{s}` → {LAYER_TITLES[SECTION_LAYER[s]]}" for s in SECTIONS)
          + f". При новом круге переносится: {', '.join(SECTION_TITLES[s] for s in CARRY_OVER)}. "
            f"Не сохраняется: {NOT_SAVED}.", ""]

    imp = report.get("import")
    if imp:
        L += ["## 1. Легенда в формате Sticky Facts Дня 10 → слои", ""]
        if imp.get("error"):
            L += [f"Импорт не удался: {imp['error']}", ""]
        else:
            call = imp.get("call") or {}
            L += [f"Источник: {imp['file']}, фактов: {len(imp['facts'])}. Один запрос летописца: вход "
                  f"{fmt(call.get('prompt'))}, выход {fmt(call.get('completion'))}, {money(call.get('cost'))}.", "",
                  "| Факт | Значение | Слой · раздел |", "|---|---|---|"]
            for k, v in imp["facts"].items():
                s = imp["placed"].get(k)
                place = f"{LAYER_TITLES[SECTION_LAYER[s]]} · {SECTION_TITLES[s]}" if s else "— (не под своим ключом)"
                L.append(f"| `{k}` | {md_escape(clip(v, 90))} | {place} |")
            if imp.get("renamed"):
                L += ["", "Записано под новыми ключами: " + ", ".join(f"`{k}`" for k in imp["renamed"])]
            L.append("")

    run = report["dialogue_run"]
    L += ["## 2. Сценарий: что летописец положил в каждый слой", "",
          "Пролог (4 реплики) — сцена без задания; затем игра выдаёт задание: карточка пишется прямо в краткосрочную "
          "память, сцена пролога уходит в архив. Токены по слоям — оценка по частям запроса, отмасштабированная "
          "к точному `prompt_tokens` первого запроса хода.", "",
          "| # | Реплика / событие | Записи в память | Вход | system | долговр. | рабочая | краткоср. | ход | "
          "Хвост / не отпр. |", "|---|---|---|---|---|---|---|---|---|---|"]
    by_turn: dict[str, list[str]] = {}
    for item in run["log"]:
        src = item.get("source") or ""
        if item["op"] in ("add", "update", "move", "delete"):
            by_turn.setdefault(src, []).append(md_escape(op_label(item)))
    stats = {x.get("user"): x for x in run["prologue"] + run["layer_stats"]}
    for item in run["played"]:
        if "mission" in item:
            card_ops = by_turn.get("игра (карточка задания)", [])
            L.append(f"| {item['n']} | 📜 задание «{MISSIONS[item['mission']].title}» | {'<br>'.join(card_ops) or '—'} | "
                     "| | | | | | |")
            continue
        s = stats.get(item["line"]) or {}
        tk = s.get("tokens") or {}
        ops = by_turn.get(turn_source(s.get("scene"), s.get("turn")), []) if s else []
        L.append(f"| {item['n']} | {md_escape(clip(item['line'], 70))} | {'<br>'.join(ops) or '—'} | "
                 f"{fmt(s.get('prompt'))} | {fmt(tk.get('system'))} | {fmt(tk.get('long'))} | {fmt(tk.get('working'))} | "
                 f"{fmt(tk.get('short'))} | {fmt(tk.get('current'))} | {s.get('tail_messages', '?')} / "
                 f"{s.get('dropped', '?')} |")
    other = [x for src, xs in by_turn.items() if src.startswith("догон") for x in xs]
    if other:
        L += ["", "Догон перед сменой задания и в конце (последний рассказ Мастера): " + "; ".join(other)]
    t, h = run["totals"], run["router"]
    total_cost = None if t["cost"] is None or h["cost"] is None else t["cost"] + h["cost"]
    L += ["", f"Сцена задания: вход {fmt(t['prompt'])} (из кэша {fmt(t['cached'])}), выход {fmt(t['completion'])}; "
              f"летописец за весь сценарий: {h['count']} запросов, вход {fmt(h['prompt'])}, выход {fmt(h['completion'])}. "
              f"Стоимость сцены задания и летописца: {money(t['cost'])} + {money(h['cost'])} = {money(total_cost)}.", ""]

    lay = run["layers"]
    L += ["### Содержимое слоёв после сценария", "", "**Долговременная** (`long_term.json`)", ""]
    for s in LONG_SECTIONS:
        items = lay["long"][s]
        L.append(f"- {SECTION_TITLES[s]}: " + ("; ".join(f"`{k}` = {md_escape(e['value'])}" for k, e in items.items())
                                              or "(пусто)"))
    L += ["", f"**Рабочая** (`working.json`): круг {lay['working']['id']} «{lay['working']['title']}»", ""]
    for s in WORKING_SECTIONS:
        items = lay["working"]["data"][s]
        L.append(f"- {SECTION_TITLES[s]}: " + ("; ".join(f"`{k}` = {md_escape(e['value'])}" for k, e in items.items())
                                              or "(пусто)"))
    ms = lay["short"]["mission"]
    L += ["", f"**Краткосрочная**: задание {ms['id']} «{ms['title']}» (`mission.json`), сцена `{lay['short']['file']}` — "
              f"{lay['short']['messages']} сообщений, в промпт уходят последние {lay['short']['tail']}.", ""]
    L += [f"- `{k}`: {md_escape(e['value'])} — _{e['source']}_" for k, e in ms["data"]["mission"].items()] or ["- (пусто)"]
    L.append("")

    exp = report["experiment"]
    runs = exp["runs"]
    L += ["## 3. Очистка слоёв: на какие вопросы Мастер перестаёт отвечать", "",
          "Каждая конфигурация — копия героя после сценария, в которой слои очищены настоящей командой `/clear` "
          "(файлы — в `experiment/<конфигурация>/`). Вопросы — вне игры, отдельными запросами, в сцену не попадают. "
          "Где лежит: К — краткосрочная (карточка задания или хвост сцены), к — в сцене, но вне хвоста, Р — рабочая, "
          "Д — долговременная. ✅ — ответ с деталью, ❔ — слова детали есть, но ответ звучит как отрицание, "
          "❌ — не знает или неверно.", "",
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
    for k, title in (("base", "без памяти: правила, мир, схема костей, вопрос"), ("long", "долговременная"),
                     ("working", "рабочая"), ("short", "краткосрочная: карточка задания + хвост сцены"),
                     ("all", "всё вместе")):
        if k in lt:
            L.append(f"| {title} | {fmt(lt[k]['exact'])} | {fmt(lt[k]['estimate'])} |")
    L.append("")

    sw = report.get("switch")
    if sw:
        L += ["## 4. Смена задания и круга", "",
              "B: `/mission new escort` (первое задание откладывается, новая сцена) → одна реплика → вопросы. "
              f"A: `/mission switch {sw['mission_a']}` → вопросы. C: `/circle next` (круг — в архив, снаряжение "
              f"переносится) → вопросы. Итог: {sw['passed']}/{sw['total']} проверок, утечек: {sw['leaks']}.", "",
              f"Реплика во втором задании: «{ESCORT_TURN}»", ""]
        for s in sw["steps"]:
            if s["step"] != "probe":
                continue
            L += [f"**{s['phase']}. {s['title']}** — задание «{s['mission']}», круг «{s['circle']}». Карточка: "
                  + ("; ".join(f"`{k}` = {md_escape(v)}" for k, v in s["card"].items()) or "(пусто)"), "",
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
    L += ["## Сцена задания: ответы Мастера", ""]
    for x in run["scene_turns"]:
        L.append(f"- **{x['n']}. {md_escape(clip(x['user'], 120))}** {md_escape(clip(x['answer'] or x['error'], 600))}")
    req = report["requests"]
    L += ["", f"Запросов к модели: {req['main']['requests']}, повторов {req['main']['retries']}, "
              f"отказов {req['main']['failed']}"
          + (f"; отдельный клиент летописца: {req['helper']['requests']}" if req.get("helper") else "") + "."]
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {md_path.relative_to(ROOT)}, {json_path.name}")
    return md_path


# ---------------------------------------------------------------- запуск


def main() -> int:
    parser = argparse.ArgumentParser(description="День 11: уся-РПГ — Мастер игры и три слоя памяти героя: "
                                                 "долговременная, рабочая, краткосрочная.")
    parser.add_argument("prompt", nargs="?", help="один ход и выход; без него — игра в консоли")
    parser.add_argument("--hero", default="default", help="герой: memory/day11/<имя>/")
    parser.add_argument("--new", action="store_true", help="начать заново: папка героя уходит в архив")
    parser.add_argument("--keep", type=int, default=6, help="краткосрочная: сколько последних сообщений сцены в промпт")
    parser.add_argument("--long-budget", type=int, default=1500, help="долговременная: бюджет токенов на особенности "
                                                                       "и связи")
    parser.add_argument("--card-budget", type=int, default=CARD_BUDGET,
                        help="краткосрочная: бюджет токенов на записи летописца в карточке задания")
    parser.add_argument("--helper-model", help="модель летописца (по умолчанию та же)")
    parser.add_argument("--import-facts", nargs="?", const="", metavar="ФАЙЛ",
                        help="разложить факты «ключ → значение» по слоям (по умолчанию — легенда Фэн Ся; "
                             "подходит results/day10_*.json)")
    parser.add_argument("--demo", action="store_true", help="импорт, сценарий, очистка слоёв, смена задания и круга")
    parser.add_argument("--no-import", action="store_true", help="для --demo: без импорта легенды")
    parser.add_argument("--no-switch", action="store_true", help="для --demo: без смены задания и круга")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--max-steps", type=int, default=8, help="предел запросов к модели за один ход")
    args = parser.parse_args()

    if not NAME_RE.fullmatch(args.hero):
        print("Ошибка: имя героя — буквы, цифры, дефис, подчёркивание.", file=sys.stderr)
        return 1
    if args.keep < 0 or args.long_budget < 0 or args.card_budget < 0:
        print("Ошибка: --keep, --long-budget, --card-budget ≥ 0.", file=sys.stderr)
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
                    not args.no_switch, args.card_budget)

    root = DAY_DIR / args.hero
    if args.new and root.exists():
        target = root.with_name(f"{root.name}_{stamp_now()}")
        os.replace(root, target)
        print(f"Прошлый герой сохранён в {target.relative_to(ROOT)}.")
    helper = (LLMClient(settings.base_url, settings.api_key, args.helper_model, settings.timeout,
                        retries=settings.retries) if args.helper_model else client)
    agent = make_agent(client, helper, root, args.keep, args.long_budget, args.max_steps, card=args.card_budget)
    try:
        info = agent.load()
    except StoreError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print_restore(info, agent.store, agent)
    for note in agent.notes:
        print(f"⚠️  {note}")
    print(f"🧠 Память героя: задание «{agent.mission.title}» ({agent.mission.count()} зап.), круг «{agent.circle.title}» "
          f"({agent.circle.count()} зап.), постоянное — {agent.long.count()} зап. /rules — правила слоёв.\n")

    if args.import_facts is not None:
        try:
            print_import(import_facts(agent, Path(args.import_facts) if args.import_facts else None))
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
        print(f"\nМастер: {reply.text.strip()}\n")
        print_turn(agent, agent.ledger[-1])
        return 0
    return chat(agent)


if __name__ == "__main__":
    from run_timer import run_timed
    sys.exit(run_timed(main))
