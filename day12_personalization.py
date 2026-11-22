"""День 12. Персонализация ассистента: профиль пользователя поверх модели памяти Дня 11.

Агент Дня 11 (`LayeredAgent`: краткосрочная, рабочая и долговременная память, роутер памяти) получает ещё один
слой — профиль пользователя, memory/day12/<пользователь>/profile.json:

    о пользователе   имя, роль, чем занимается                         раздел profile
    стиль            тон (формально / неформально), «вы» или «ты», юмор раздел style
    формат           длина, структура, нужен ли код и на каком языке    раздел format
    ограничения      чего в ответе быть не должно и что обязательно     раздел constraints

Профиль подключается к каждому запросу: блок «Профиль пользователя» в конце системного сообщения, всегда целиком.
Пополняется из диалога: роутер памяти Дня 11 получил разделы style, format, constraints и после каждого сообщения
пользователя записывает предпочтения, которые тот назвал прямо («давай на ты») или показал реакцией («слишком
длинно»). Вручную — /pref и /unpref; готовые профили — --preset brief | deep | code.

Соблюдение предпочтений проверяется двумя способами: кодом (число слов, есть ли код и на каком языке, «вы» или «ты»,
эмодзи) и судьёй — отдельным запросом к модели с temperature=0 (тон, жаргон, вступления). Где есть проверка кодом,
решает она; судья — где её нет. Расхождения кода и судьи попадают в отчёт.

    python day12_personalization.py                          # чат (пользователь default)
    python day12_personalization.py --user anna --preset brief
    python day12_personalization.py --no-attach              # профиль ведётся, но в запрос не подключается
    python day12_personalization.py --demo                   # профиль из диалога, 3 профиля × 1 вопрос, 10 вопросов подряд
    python day12_personalization.py --recheck results/day12_<время>.json  # пересчитать проверки кодом, без модели

Команды чата: /profile, /pref <раздел> <ключ> = <значение>, /unpref <раздел> <ключ>, /preset [id], /attach on|off,
/check, /compare, /stability [N], /memory, /context, /newchat, /log, /tokens, /history, /exit.
"""
import argparse
import copy
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from config import ConfigError, load_settings
from day6_agent import TOOLS
from day7_memory import MEMORY_DIR, ROOT, StoreError, now_iso, print_history, print_restore
from day8_tokens import Usage, clip, fmt, money, price_for, print_ledger
from day9_compression import normalize, watch_retries
from day10_strategies import HelperCall, StickyFacts, read_json, write_json
from day11_memory_layers import (LAYER_TITLES, LAYERS, SECTION_ALIASES, SECTION_LAYER, SECTION_TITLES, LayeredAgent,
                                 MemoryLog, archive_file, entries, md_escape, print_memory, stamp_now)
from day11_memory_layers import on_event as day11_event
from llm_client import LLMClient, LLMError

RESULTS_DIR = ROOT / "results"
DAY_DIR = MEMORY_DIR / "day12"
NAME_RE = re.compile(r"[\w-]+")

# Профиль — четыре раздела в profile.json. Раздел profile («о пользователе») в Дне 11 жил в долговременной памяти,
# теперь он часть профиля; решения, знания и задача остаются в слоях Дня 11.
PROFILE_SECTIONS = ("profile", "style", "format", "constraints")
PREF_SECTIONS = ("style", "format", "constraints")
ALL_SECTIONS = PROFILE_SECTIONS + ("decisions", "knowledge", "task")
SECTION_LAYER12 = {**SECTION_LAYER, **{s: "persona" for s in PROFILE_SECTIONS}}
SECTION_TITLES12 = {**SECTION_TITLES, "profile": "о пользователе", "style": "стиль", "format": "формат",
                    "constraints": "ограничения"}
LAYER_TITLES12 = {**LAYER_TITLES, "persona": "профиль"}
SECTION_ALIASES12 = {**SECTION_ALIASES, "about": "profile", "о_себе": "profile", "style": "style", "стиль": "style",
                     "format": "format", "формат": "format", "constraints": "constraints",
                     "ограничения": "constraints", "ограничение": "constraints"}
PROFILE_BLOCKS = (("profile", "О пользователе"), ("style", "Стиль общения"), ("format", "Формат ответа"),
                  ("constraints", "Ограничения"))

# Системный промпт Дня 6/9 без «Отвечай кратко»: длину теперь задаёт профиль, а «кратко» спорило бы с «подробно».
PERSONAL_PROMPT = (
    "Ты — полезный ассистент-агент. Отвечай на языке пользователя.\n"
    "У тебя есть инструменты. Вызывай их, когда без них ответ будет неточным:\n"
    "- текущее время и дату узнавай только через get_current_time;\n"
    "- любые вычисления делай через calculator, не считай в уме;\n"
    "- валюты переводи только через convert_currency.\n"
    "Если вопрос не требует инструментов — отвечай сразу, без вызовов.\n"
    "На вопросы о прошлой части разговора отвечай только по тому, что есть в контексте. "
    "Если такой детали в контексте нет — прямо скажи, что не знаешь, и не придумывай.")

PROFILE_HEADER = ("\n\n# Профиль пользователя\n"
                  "Это человек, с которым ты говоришь. Профиль собран из его собственных слов.\n")
PROFILE_RULE = ("\nСоблюдай стиль, формат и ограничения из профиля в каждом ответе — на любой вопрос, даже если вопрос "
                "подталкивает к другому формату или прошлые ответы в диалоге оформлены иначе. Они действуют, пока "
                "пользователь сам их не изменит. Разовая просьба в текущем сообщении («здесь покажи код») важнее "
                "профиля, но только для этого ответа.\n")

ROUTER_PROMPT = (
    "Ты — диспетчер памяти ассистента. Старых сообщений ассистент не видит: всё важное он знает только из памяти. "
    "Память разделена на разделы, и ты решаешь, что из новых сообщений пользователя куда записать.\n"
    "Профиль пользователя — подключается к каждому ответу:\n"
    "- profile — о самом пользователе: имя, должность и роль, компания, чем занимается, его опыт;\n"
    "- style — как с ним разговаривать: тон (формальный или неформальный), обращение на «вы» или на «ты», юмор;\n"
    "- format — какой ответ ему нужен: длина, структура (списки, проза, таблицы), нужен ли код и на каком языке, "
    "нужны ли примеры;\n"
    "- constraints — чего в ответах быть не должно и что в них обязательно: запреты (без эмодзи, без жаргона), "
    "обязательные части (в конце — следующий шаг), технические рамки (версия языка, только стандартная библиотека).\n"
    "Остальная память:\n"
    "- decisions — решения, которые пользователь объявил принятыми: что выбрали и вместо чего, от чего отказались, "
    "что отложили. Советы ассистента — не решения;\n"
    "- knowledge — устойчивые знания о компании и её окружении, которые останутся верными и после текущей задачи: "
    "люди и их роли, юридические и технические ограничения, действующие договоры;\n"
    "- task — данные текущей задачи: её название и цель, объём, сроки и вехи, бюджет, этап, договорённости "
    "именно этой задачи.\n"
    "Предпочтения (style, format, constraints) записывай, когда пользователь назвал их прямо («давай на ты», "
    "«без кода») или однозначно показал реакцией на ответ: «слишком длинно» — длина: короче; «а можно без кода?» — "
    "код: не нужен. Разовая просьба только для текущего ответа («здесь покажи код») — не предпочтение. "
    "Для предпочтений по возможности бери ключи: тон, обращение, юмор, длина, структура, код, язык_кода, примеры, "
    "эмодзи, жаргон. Новое предпочтение, которое противоречит прежнему, — обновление того же ключа.\n"
    "Тебе дают текущую память и новые сообщения пользователя. Верни JSON вида "
    '{"profile": {"set": {}, "delete": []}, "style": {"set": {}, "delete": []}, "format": {"set": {}, "delete": []}, '
    '"constraints": {"set": {}, "delete": []}, "decisions": {"set": {}, "delete": []}, '
    '"knowledge": {"set": {}, "delete": []}, "task": {"set": {}, "delete": []}}:\n'
    "- set — новые записи и записи, значение которых изменилось; delete — записи, которые пользователь отменил;\n"
    "- значения пиши точно, как у пользователя, и коротко (до 20 слов). Если значение изменилось — новое, "
    "а в скобках старое: «2,7 млн ₽ (было 2,4 млн ₽)»;\n"
    "- меняй число, только если пользователь прямо сказал, что прежнее значение изменилось;\n"
    "- ключи короткие, по-русски, в нижнем регистре через подчёркивание. Если запись уже есть — обнови тот же ключ "
    "в том же разделе, не заводи новый. Один ключ — в одном разделе;\n"
    "- вопросы пользователя и просьбы что-то объяснить — не записи. Ничего не придумывай;\n"
    "- если записывать нечего — верни пустые set и delete.\n"
    "Верни только JSON.")

JUDGE_PROMPT = (
    "Ты строго проверяешь, соблюдает ли ответ ассистента предпочтения пользователя. Тебе дают вопрос пользователя, "
    "его предпочтения (id: что он просил) и ответ ассистента. Для каждого предпочтения реши: соблюдено (true) "
    "или нарушено (false). Нарушено — если в ответе есть хотя бы один явный случай против предпочтения: например, "
    "предпочтение «без кода», а в ответе есть фрагмент кода или команда; «на вы», а в ответе есть обращение на «ты». "
    "Оценивай только соблюдение предпочтений, а не качество ответа. "
    'Верни JSON вида {"<id>": {"ok": true, "why": "до 15 слов"}} по всем id. Только JSON.')


# ---------------------------------------------------------------- профиль


class UserProfile:
    """Профиль пользователя: о пользователе, стиль, формат, ограничения. Один файл, подключается к каждому запросу."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict[str, dict]] = {s: {} for s in PROFILE_SECTIONS}
        self.preset = ""  # из какого готового профиля заполнен (для проверок кодом)

    def load(self) -> None:
        self.data, self.preset = {s: {} for s in PROFILE_SECTIONS}, ""
        if self.path.exists():
            raw = read_json(self.path)
            for s in PROFILE_SECTIONS:
                self.data[s] = entries(raw.get(s))
            self.preset = str(raw.get("preset") or "")

    def save(self) -> None:
        write_json(self.path, {"format": 1, "updated_at": now_iso(), "preset": self.preset, **self.data})

    def count(self, sections=PROFILE_SECTIONS) -> int:
        return sum(len(self.data[s]) for s in sections)

    def values(self) -> dict[str, dict[str, str]]:
        return {s: {k: e["value"] for k, e in self.data[s].items()} for s in PROFILE_SECTIONS}

    def clear(self) -> Path | None:
        archived = archive_file(self.path)
        self.data, self.preset = {s: {} for s in PROFILE_SECTIONS}, ""
        return archived


class PersonaLog(MemoryLog):
    """Журнал Дня 11, который знает слой «профиль»."""

    def add(self, op: str, section: str, key: str = "", value: str = "", source: str = "", **extra) -> dict:
        item = super().add(op, section, key, value, source, **extra)
        item["layer"] = SECTION_LAYER12.get(section, item["layer"])
        return item


def parse_route(text: str) -> dict[str, tuple[dict[str, str], list[str]]]:
    """Ответ роутера → {раздел: (set, delete)} по всем разделам Дня 12; каждый раздел — парсер Sticky Facts Дня 10."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("нет JSON")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("ожидался объект")
    out = {}
    for s in ALL_SECTIONS:
        part = data.get(s) or {}
        if not isinstance(part, dict):
            raise ValueError(f"{s} — не объект")
        out[s] = StickyFacts.parse(json.dumps(part, ensure_ascii=False)) if part else ({}, [])
    return out


def op_label(item: dict) -> str:
    if item["op"] == "clear":
        return f"🧹 очищен слой: {LAYER_TITLES12.get(item['layer'], item['layer'])}"
    if item["op"] == "task":
        return f"📋 задача {item['key']} «{item['value']}» — {item.get('status', '')}"
    layer = LAYER_TITLES12.get(item["layer"], item["layer"])
    where = f"{layer} · {SECTION_TITLES12.get(item['section'], item['section'])}"
    if item["op"] == "move":
        src = item.get("from", "")
        where = f"{SECTION_TITLES12.get(src, src)} → {where}"
    sign = {"add": "+", "update": "~", "move": "↪", "delete": "−"}.get(item["op"], item["op"])
    value = f" = {clip(item['value'], 80)}" if item.get("value") and item["op"] != "delete" else ""
    old = (f" (было: {clip(item['old'], 40)})" if item.get("old") and item["op"] == "update"
           and normalize(item["old"]) not in normalize(item.get("value") or "") else "")
    return f"{sign} [{where}] {item['key']}{value}{old}"


# ---------------------------------------------------------------- готовые профили и предпочтения


@dataclass(frozen=True)
class Pref:
    """Одно предпочтение: что записано в профиль и как его проверить кодом (check пустой — проверяет только судья)."""
    id: str
    section: str  # style | format | constraints
    key: str
    value: str
    check: str = ""  # max_words | min_words | no_code | has_code | code_lang | formal | informal | no_emoji
    param: int | str | None = None


@dataclass(frozen=True)
class Preset:
    id: str
    title: str
    about: dict
    prefs: tuple[Pref, ...]


PRESETS = {
    "brief": Preset("brief", "Анна — финансовый директор: кратко, без кода, формально",
                    {"имя": "Анна Сергеевна Белова", "роль": "финансовый директор сети пекарен «Колосок», не программист"},
                    (Pref("length", "format", "длина", "кратко: не больше 3–4 предложений, до 70 слов", "max_words", 70),
                     Pref("code", "format", "код", "без кода, команд и формул — только объяснение словами", "no_code"),
                     Pref("tone", "style", "тон", "формальный, деловой, без шуток и разговорных слов"),
                     Pref("address", "style", "обращение", "на «вы»", "formal"),
                     Pref("jargon", "constraints", "жаргон", "без технического жаргона и англицизмов; нужный термин — "
                                                             "пояснить простыми словами"),
                     Pref("emoji", "constraints", "эмодзи", "без эмодзи", "no_emoji"))),
    "deep": Preset("deep", "Дима — Python-разработчик: подробно, с кодом, неформально",
                   {"имя": "Дима", "роль": "бэкенд-разработчик на Python, 5 лет опыта"},
                   (Pref("length", "format", "длина", "подробно: разбор по шагам, с объяснением почему, от 250 слов",
                         "min_words", 250),
                    Pref("code", "format", "код", "обязательно пример рабочего кода", "has_code"),
                    Pref("lang", "format", "язык_кода", "код только на Python 3.12", "code_lang", "python"),
                    Pref("tone", "style", "тон", "неформальный, живой, можно с юмором"),
                    Pref("address", "style", "обращение", "на «ты»", "informal"),
                    Pref("next", "constraints", "в_конце", "в конце — что можно улучшить или попробовать дальше"))),
    "code": Preset("code", "Сергей — тимлид на Go: кратко, только код, формально",
                   {"имя": "Сергей Викторович", "роль": "тимлид бэкенда, пишет на Go"},
                   (Pref("length", "format", "длина", "кратко: одно-два предложения пояснения, остальное — код, "
                                                      "до 60 слов текста", "max_words", 60),
                    Pref("code", "format", "код", "всегда пример кода", "has_code"),
                    Pref("lang", "format", "язык_кода", "код только на Go", "code_lang", "go"),
                    Pref("tone", "style", "тон", "формальный, сухой, без шуток"),
                    Pref("address", "style", "обращение", "на «вы»", "formal"),
                    Pref("intro", "constraints", "вступления", "без вступлений, повторения вопроса и выводов — "
                                                              "сразу к делу"))),
}

COMMON_QUESTION = "Наш сервис иногда зависает, когда внешний платёжный API долго отвечает. Как это исправить?"

STABILITY = [  # 10 вопросов подряд: часть сама тянет к коду, длинному ответу или списку
    "Что такое кэширование и зачем оно нужно сервису?",
    "Как проверить, что пользователь ввёл корректный email?",
    "Чем REST отличается от GraphQL?",
    "Как посчитать среднее время ответа сервиса по его логам?",
    "Стоит ли нам переписывать монолит на микросервисы?",
    "Как правильно хранить пароли пользователей?",
    "Как устроить повторные попытки, если запрос к внешнему сервису упал?",
    "Какие метрики сервиса смотреть в первую очередь?",
    "Как найти, какая часть программы работает медленнее всего?",
    "Подведи итог нашего разговора: что из всего этого внедрить первым?",
]

# Без подключения профиля: те же предпочтения Анны сказаны один раз — в первом сообщении диалога.
ABLATION_INTRO = ("Здравствуйте. Я Анна Сергеевна, финансовый директор, не программист. Прошу отвечать так: кратко — "
                  "не больше 3–4 предложений, без кода и формул, формально и на «вы», без технического жаргона "
                  "и эмодзи. Первый вопрос: ")

# Профиль из диалога: пользователь без профиля называет предпочтения по ходу разговора.
OLEG_TURNS = [
    "Привет! Я Олег, пишу бэкенд на Go. Как лучше хранить настройки сервиса — в файле или в переменных окружения?",
    "Давай на ты, без официоза. И короче — мне хватит трёх-пяти пунктов.",
    "Код показывай на Go, а не на Python. Как сделать graceful shutdown HTTP-сервера?",
    "Слишком много текста вокруг кода. Пояснений — одна-две строки.",
    "И без эмодзи, пожалуйста. Как в Go правильно передавать таймаут в запрос к базе?",
]
OLEG_CHECK = "Как ограничить число одновременных запросов к базе данных?"
OLEG_EXPECT = (Pref("address", "style", "обращение", "на «ты»", "informal"),
               Pref("tone", "style", "тон", "неформальный, без официоза"),
               Pref("length", "format", "длина", "коротко: 3–5 пунктов, пояснений к коду — одна-две строки, до 80 слов",
                    "max_words", 80),
               Pref("lang", "format", "язык_кода", "код — на Go, не на Python", "code_lang", "go"),
               Pref("emoji", "constraints", "эмодзи", "без эмодзи", "no_emoji"))
OLEG_PROFILE = [  # что должно оказаться в профиле после диалога: (что, в каких разделах, regex по ключам и значениям)
    ("имя — Олег", ("profile",), r"олег"), ("пишет на Go", ("profile",), r"\bgo\b|golang"),
    ("на «ты»", ("style",), r"\bты\b|на ты"), ("неформально", ("style",), r"неформал|без официоз|дружеск|прост"),
    ("коротко", ("format",), r"корот|кратк|пункт|строк"), ("код на Go", ("format", "constraints"), r"\bgo\b|golang"),
    ("без эмодзи", PREF_SECTIONS, r"эмодзи|emoji|смайл"),
]


# ---------------------------------------------------------------- агент


class PersonalAgent(LayeredAgent):
    """LayeredAgent Дня 11 с профилем пользователя. Профиль — свой файл; build() добавляет его к каждому запросу,
    роутер памяти пишет в него предпочтения из диалога."""

    def __init__(self, client: LLMClient, tools, root: Path, *, attach: bool = True, **kwargs):
        self.profile = UserProfile(root / "profile.json")
        self.attach_profile = attach  # False — профиль ведётся, но в запрос не идёт (контрольный прогон)
        self.routing = True  # False — роутер не вызывается (сравнение профилей на одном вопросе)
        kwargs.setdefault("system_prompt", PERSONAL_PROMPT)
        super().__init__(client, tools, root, **kwargs)
        self.log = PersonaLog(root / "log.jsonl")

    # --- файлы

    def load(self):
        self.profile.load()
        return super().load()

    def save_memory(self) -> None:
        self.profile.save()
        super().save_memory()

    # --- разделы: профиль — свой файл, остальное — слои Дня 11

    def section(self, name: str) -> dict[str, dict]:
        return self.profile.data[name] if name in PROFILE_SECTIONS else super().section(name)

    def find(self, key: str) -> str | None:
        return next((s for s in ALL_SECTIONS if key in self.section(s)), None)

    def write(self, section: str, key: str, value: str, source: str) -> list[dict]:
        moved = None
        for other in ALL_SECTIONS:
            if other != section and key in self.section(other):
                moved = other
                self.section(other).pop(key)
        target = self.section(section)
        old = target.get(key)
        if old and old["value"] == value and not moved:
            return []
        target[key] = {"value": value, "at": now_iso(), "source": source, "task": self.working.task["title"]}
        op = "move" if moved else ("update" if old else "add")
        return [self.log.add(op, section, key, value, source, old=old["value"] if old else None, **{"from": moved})]

    def apply(self, changes: dict, source: str) -> list[dict]:
        ops = []
        for s in ALL_SECTIONS:
            update, delete = changes.get(s) or ({}, [])
            for key, value in update.items():
                ops += self.write(s, key, value, source)
            for key in delete:
                ops += self.forget(s, key, source)
        return ops

    def memory_json(self) -> dict:
        return {s: {k: e["value"] for k, e in self.section(s).items()} for s in ALL_SECTIONS}

    def route(self, texts: list[str], source: str, imported: bool = False) -> list[dict]:
        """Роутер Дня 11 с разделами профиля: новые сообщения пользователя → изменения по разделам."""
        if not self.routing:
            return []
        end = self.turn_end()
        pending = [m.get("content") or "" for m in self.history[self.router_upto:end] if m.get("role") == "user"]
        pending += [t for t in texts if t]
        if not pending:
            return []
        numbered = "\n\n".join(f"{i}. {t}" for i, t in enumerate(pending, 1))
        user = (f"Текущая задача: «{self.working.task['title']}»\n\nТекущая память:\n"
                f"{json.dumps(self.memory_json(), ensure_ascii=False, indent=1)}\n\n"
                f"Новые сообщения пользователя:\n\n{numbered}")
        result = self.helper_client.chat([{"role": "system", "content": ROUTER_PROMPT},
                                          {"role": "user", "content": user}], temperature=0)
        usage = Usage.parse(result.usage)
        priced = price_for(self.helper_client.model)
        call = HelperCall(strategy="router", before_turn=len(self.ledger) + 1, messages=len(pending),
                          prompt=usage.prompt, completion=usage.completion, cached=usage.cached,
                          cost=usage.cost(priced[0] if priced else None), latency=result.latency, at=now_iso())
        self.router_calls.append(call)
        try:
            changes = parse_route(result.text)
        except ValueError as e:
            call.note = f"ответ не разобран: {e}"
            raise LLMError(f"ответ роутера не разобран ({e}): {clip(result.text, 120)!r}")
        ops = self.apply(changes, source)
        self.router_upto = end + (1 if self._turn_start is not None else 0)
        counts = {layer: sum(o["layer"] == layer for o in ops) for layer in ("persona", "working", "long")}
        call.note = (f"изменений {len(ops)}: профиль {counts['persona']}, рабочая {counts['working']}, "
                     f"долговременная {counts['long']}" if ops else "изменений нет")
        self.on_event("route", {"call": call, "ops": ops})
        return ops

    # --- профиль в запросе

    def profile_block(self) -> str:
        d = self.profile.data
        parts = [f"### {title}\n" + "\n".join(f"- {k}: {e['value']}" for k, e in d[s].items())
                 for s, title in PROFILE_BLOCKS if d[s]]
        if not parts:
            return ""
        rule = PROFILE_RULE if any(d[s] for s in PREF_SECTIONS) else ""
        return PROFILE_HEADER + "\n\n".join(parts) + "\n" + rule

    def build(self, end: int | None = None, layers=LAYERS, tools: list[dict] | None = None):
        """Запрос Дня 11 + профиль в конце системного сообщения (если подключён)."""
        messages, dropped, parts = super().build(end, layers, tools)
        text = self.profile_block() if self.attach_profile else ""
        if text:
            messages[0] = dict(messages[0], content=(messages[0].get("content") or "") + text)
        raw = self.estimator.raw_text(text) if text else 0
        parts["raw"]["persona"] = raw
        parts["estimate"]["persona"] = self.estimator.apply(raw) if raw else 0
        return messages, dropped, parts

    # --- ход, откат, очистка

    def snapshot(self) -> dict:
        snap = super().snapshot()
        snap["profile"] = copy.deepcopy(self.profile.data)
        return snap

    def rollback(self, snap: dict) -> None:
        super().rollback(snap)
        self.profile.data = snap["profile"]

    def clear(self, layer: str) -> str:
        if layer != "persona":
            return super().clear(layer)
        archived = self.profile.clear()
        self.log.add("clear", "profile", source="команда пользователя")
        self.save_memory()
        return "профиль очищен" + (f", копия — {archived.name}" if archived else "")

    def load_preset(self, preset_id: str) -> list[dict]:
        """Заполнить профиль готовым набором предпочтений; прежний профиль уходит в архив."""
        preset = PRESETS[preset_id]
        archive_file(self.profile.path)
        self.profile.data = {s: {} for s in PROFILE_SECTIONS}
        source = f"готовый профиль {preset_id}"
        ops = [op for k, v in preset.about.items() for op in self.write("profile", k, v, source)]
        ops += [op for p in preset.prefs for op in self.write(p.section, p.key, p.value, source)]
        self.profile.preset = preset_id
        self.save_memory()
        return ops

    def current_prefs(self) -> list[Pref]:
        """Предпочтения из профиля для проверки. Записи готового профиля без изменений проверяются ещё и кодом."""
        preset = PRESETS.get(self.profile.preset)
        known = {(p.section, p.key): p for p in preset.prefs} if preset else {}
        out = []
        for s in PREF_SECTIONS:
            for k, e in self.profile.data[s].items():
                p = known.get((s, k))
                out.append(p if p and p.value == e["value"] else Pref(f"{s}.{k}", s, k, e["value"]))
        return out


# ---------------------------------------------------------------- проверка: код + судья


FENCE = re.compile(r"```([^\n`]*)\n(.*?)(?:```|\Z)", re.S)
INLINE = re.compile(r"`([^`\n]+)`")
WORD = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*")
CODE_LINE = re.compile(r"^\s*(?:def |async def |class \w+|import \w|import \(|from [\w.]+ import|func |package \w|"
                       r"return\b|if err != nil|select\s.+\sfrom\s|insert into|\w+\s*:=|\w+(?:\.\w+)+\(.*\)\s*$|"
                       r".*[{}]\s*$|#include|console\.|print\()", re.M | re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⭐⭕⬛⬜]")
TY = re.compile(r"\b(?:ты|тебе|тебя|тобой|твой|твоя|твоё|твое|твои|твоего|твоей|твоему|твоим|твоих|твою)\b")
VY = re.compile(r"\b(?:вы|вам|вас|вами|ваш|ваша|ваше|ваши|вашего|вашей|вашему|вашим|ваших|вашу)\b")
IMPERATIVES = ("используй", "добавь", "сделай", "поставь", "проверь", "посмотри", "настрой", "попробуй", "вынеси",
               "оберни", "запусти", "напиши", "укажи", "включи", "задай", "ограничь", "разбей", "не забудь", "учти",
               "обрати", "выбери", "храни", "логируй", "замерь", "создай", "открой", "определи", "держи", "начни",
               "подключи", "внедри", "установи", "передавай", "ставь", "смотри")
TY_IMP = re.compile(r"\b(?:" + "|".join(IMPERATIVES) + r")\b")
VY_IMP = re.compile(r"\b(?:" + "|".join(w + "те" for w in IMPERATIVES) + r")\b")
LANG_TAGS = {"python": "python", "py": "python", "python3": "python", "go": "go", "golang": "go",
             "js": "javascript", "javascript": "javascript", "ts": "typescript", "typescript": "typescript",
             "java": "java", "kotlin": "kotlin", "rust": "rust", "c": "c", "cpp": "cpp", "c++": "cpp",
             "cs": "csharp", "csharp": "csharp", "php": "php", "ruby": "ruby", "sql": "sql",
             "bash": "shell", "sh": "shell", "shell": "shell", "console": "shell", "zsh": "shell",
             "powershell": "shell", "json": "data", "yaml": "data", "yml": "data", "toml": "data", "ini": "data",
             "env": "data", "text": "text", "txt": "text", "plaintext": "text", "": ""}
PROGRAMMING_LANGS = {"python", "go", "javascript", "typescript", "java", "kotlin", "rust", "c", "cpp", "csharp", "php",
                     "ruby"}  # нарушение «код только на X» — только код на другом языке программирования;
# команды оболочки, конфиги, SQL, PromQL и прочие языки запросов — не нарушение


def split_code(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(текст без блоков кода, [(тег языка, тело блока)])."""
    text = text or ""
    return FENCE.sub(" ", text), [(lang.strip().lower(), body) for lang, body in FENCE.findall(text)]


def block_lang(tag: str, body: str) -> str:
    lang = LANG_TAGS.get(tag, tag)
    if lang in ("", "text", "data"):  # блок без тега — по приметам
        if re.search(r"^\s*(?:package \w|func |import \()|:=|\bfmt\.|if err != nil", body, re.M):
            return "go"
        if re.search(r"^\s*(?:def |async def |import \w|from [\w.]+ import|print\()|\bself\.", body, re.M):
            return "python"
        return lang or "?"
    return lang


def code_info(text: str) -> dict:
    prose, blocks = split_code(text)
    inline = [s for s in INLINE.findall(prose) if re.search(r"[(){}=\[\]]|:=|\w\.\w+\(", s)]
    return {"prose": prose, "words": len(WORD.findall(prose)), "blocks": len(blocks),
            "langs": sorted({block_lang(t, b) for t, b in blocks}), "code_lines": len(CODE_LINE.findall(prose)),
            "inline": len(inline)}


QUOTED = re.compile(r"«[^»\n]*»|\"[^\"\n]*\"|“[^”\n]*”|`[^`\n]*`")


def address(prose: str) -> tuple[list[str], list[str]]:
    """Формы на «ты» и на «вы» в тексте ответа. Цитаты не считаются: «попробуйте позже» в кавычках — это
    сообщение для чужих пользователей, а не обращение к собеседнику."""
    t = QUOTED.sub(" ", normalize(prose))
    return TY.findall(t) + TY_IMP.findall(t), VY.findall(t) + VY_IMP.findall(t)


def run_check(p: Pref, text: str) -> tuple[bool | None, str]:
    """Проверка кодом: (соблюдено | None — код не может решить, пояснение)."""
    if not p.check:
        return None, ""
    info = code_info(text)
    if p.check == "max_words":
        return info["words"] <= int(p.param), f"{info['words']} сл. текста (норма ≤ {p.param})"
    if p.check == "min_words":
        return info["words"] >= int(p.param), f"{info['words']} сл. текста (норма ≥ {p.param})"
    if p.check == "no_code":
        bad = info["blocks"] or info["code_lines"] >= 2 or info["inline"]
        found = [f"{label} {info[k]}" for k, label in (("blocks", "блоков кода"), ("code_lines", "строк кода"),
                                                       ("inline", "кода в строке")) if info[k]]
        return not bad, ", ".join(found) if bad else "кода нет"
    if p.check == "has_code":
        ok = bool(info["blocks"]) or info["code_lines"] >= 3
        return ok, f"блоков кода {info['blocks']}" + (f" ({', '.join(info['langs'])})" if info["langs"] else "")
    if p.check == "code_lang":
        if not info["blocks"]:
            return None, "кода нет"
        other = [x for x in info["langs"] if x != p.param and x in PROGRAMMING_LANGS]
        if other:
            return False, f"код на {', '.join(other)}"
        return (True, f"код на {p.param}") if p.param in info["langs"] else (None, f"языки: {', '.join(info['langs'])}")
    if p.check in ("formal", "informal"):
        ty, vy = address(info["prose"])
        if p.check == "formal":
            return (False, f"на «ты»: {', '.join(ty[:3])}") if ty else ((True, "на «вы»") if vy else (None, "обращения нет"))
        return (False, f"на «вы»: {', '.join(vy[:3])}") if vy else ((True, "на «ты»") if ty else (None, "обращения нет"))
    if p.check == "no_emoji":
        found = EMOJI.findall(text or "")
        return (not found), (f"эмодзи: {' '.join(found[:5])}" if found else "эмодзи нет")
    raise ValueError(f"нет проверки {p.check}")


def mark(ok: bool | None) -> str:
    return "✅" if ok is True else ("❌" if ok is False else "·")


class Evaluator:
    """Соблюдено ли каждое предпочтение: проверка кодом, где она есть, иначе — судья (отдельный запрос, temperature 0)."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.calls: list[dict] = []

    def judge(self, question: str, answer: str, prefs: list[Pref]) -> dict[str, dict]:
        listing = "\n".join(f"- {p.id}: {SECTION_TITLES12.get(p.section, p.section)} — {p.key}: {p.value}" for p in prefs)
        user = (f"Вопрос пользователя: {question}\n\nПредпочтения пользователя:\n{listing}\n\n"
                f"Ответ ассистента:\n<<<\n{answer}\n>>>")
        priced = price_for(self.client.model)
        try:
            result = self.client.chat([{"role": "system", "content": JUDGE_PROMPT},
                                       {"role": "user", "content": user}], temperature=0)
        except LLMError as e:
            self.calls.append({"error": str(e), "prompt": 0, "completion": 0, "cost": 0.0, "latency": 0.0})
            return {}
        usage = Usage.parse(result.usage)
        call = {"prompt": usage.prompt, "completion": usage.completion, "latency": round(result.latency, 2),
                "cost": usage.cost(priced[0] if priced else None), "error": ""}
        self.calls.append(call)
        try:
            data = json.loads(re.search(r"\{.*\}", result.text or "", re.S).group(0))
        except (AttributeError, ValueError):
            call["error"] = f"ответ судьи не разобран: {clip(result.text, 100)!r}"
            return {}
        out = {}
        for p in prefs:
            v = data.get(p.id) if isinstance(data, dict) else None
            if isinstance(v, dict):
                ok = v.get("ok")
                ok = ok if isinstance(ok, bool) else ({"true": True, "false": False}.get(str(ok).lower()))
                out[p.id] = {"ok": ok, "why": str(v.get("why") or "")}
        return out

    def evaluate(self, question: str, answer: str, prefs) -> dict:
        prefs = list(prefs)
        empty = not answer.strip()  # запрос упал — проверять нечего (иначе «0 слов» сошло бы за «кратко»)
        verdicts = {} if empty else self.judge(question, answer, prefs)
        items = []
        for p in prefs:
            auto, evidence = (None, "нет ответа") if empty else run_check(p, answer)
            j = verdicts.get(p.id) or {}
            final = auto if auto is not None else j.get("ok")
            items.append({"id": p.id, "key": p.key, "value": p.value, "auto": auto, "evidence": evidence,
                          "judge": j.get("ok"), "why": j.get("why", ""), "ok": final,
                          "by": "код" if auto is not None else ("судья" if j.get("ok") is not None else "")})
        info = code_info(answer)
        return {"items": items, "kept": sum(i["ok"] is True for i in items), "failed": sum(i["ok"] is False for i in items),
                "total": len(items), "all_ok": not empty and bool(items) and all(i["ok"] is not False for i in items),
                "empty": empty,
                "words": info["words"], "blocks": info["blocks"], "langs": info["langs"]}

    def totals(self) -> dict:
        costs = [c["cost"] for c in self.calls]
        return {"count": len(self.calls), "prompt": sum(c["prompt"] for c in self.calls),
                "completion": sum(c["completion"] for c in self.calls),
                "cost": None if any(c is None for c in costs) else sum(costs),
                "errors": sum(bool(c["error"]) for c in self.calls)}


def eval_line(e: dict) -> str:
    if e.get("empty"):
        return "⚠️ нет ответа — запрос не прошёл, не проверялось"
    undecided = e["total"] - e["kept"] - e["failed"]
    failed = [f"{i['key']} ({i['evidence'] or i['why']})" for i in e["items"] if i["ok"] is False]
    code = f"код: {', '.join(e['langs']) or 'да'}" if e["blocks"] else "кода нет"
    return (f"{'✅' if e['all_ok'] else '❌'} {e['kept']}/{e['total']}"
            + (f" (не решено {undecided})" if undecided else "") + f" | {e['words']} сл., {code}"
            + (f" | нарушено: {'; '.join(failed)}" if failed else ""))


# ---------------------------------------------------------------- вывод


def on_event(kind: str, data: dict) -> None:
    if kind == "route":
        c: HelperCall = data["call"]
        print(f"  🧠 роутер памяти: {c.note} | вход {fmt(c.prompt)}, выход {fmt(c.completion)}, {c.latency:.1f} с, "
              f"{money(c.cost)}")
        for item in data["ops"]:
            print(f"     {op_label(item)}")
    else:
        day11_event(kind, data)


def quiet_event(kind: str, data: dict) -> None:
    """Для --demo: без строк шагов; роутер печатается, только если он что-то записал."""
    if kind in ("tokens", "llm", "truncate", "reset") or (kind == "route" and not data["ops"]):
        return
    on_event(kind, data)


def layer_line(tokens: dict) -> str:
    names = (("system", "system"), ("persona", "профиль"), ("long", "долговр."), ("working", "рабочая"),
             ("short", "краткоср."), ("current", "ход"))
    return " · ".join(f"{label} {fmt(tokens.get(k, 0))}" for k, label in names)


def print_turn(agent: PersonalAgent) -> None:
    turn = agent.ledger[-1] if agent.ledger else None
    if turn is None or not turn.calls:
        print(f"  ❌ ответа нет: {turn.error if turn else ''}")
        return
    s = agent.layer_stats[-1] if agent.layer_stats and agent.layer_stats[-1]["turn"] == turn.n else None
    print(f"  📊 ход {turn.n}: вход {fmt(turn.calls[0].prompt)} = " + (layer_line(s["tokens"]) if s else "?")
          + f" | выход {fmt(turn.completion)} | {money(turn.cost)}")


def print_profile(agent: PersonalAgent) -> None:
    text = agent.profile_block()
    state = "подключён к каждому запросу" if agent.attach_profile else "НЕ подключается к запросу (--no-attach)"
    preset = f", заполнен из готового «{agent.profile.preset}»" if agent.profile.preset else ""
    print(f"  👤 профиль — {agent.profile.path.name}: записей {agent.profile.count()}, ≈{fmt(agent.estimator.text(text))} "
          f"токенов, {state}{preset}")
    for s, title in PROFILE_BLOCKS:
        items = agent.profile.data[s]
        print(f"     {title}:" + ("" if items else " —"))
        for k, e in items.items():
            print(f"       · {k}: {e['value']}  ⟵ {e['source']}")


def print_context12(agent: PersonalAgent) -> None:
    schemas = [t.schema() for t in agent.tools.values()]
    _, dropped, parts = agent.build(end=len(agent.history), tools=schemas)
    e = parts["estimate"]
    print(f"  Следующий запрос ≈{fmt(sum(e.values()))} токенов (без нового сообщения):")
    rows = [("system", "системный промпт и схемы инструментов", "всегда"),
            ("persona", "профиль пользователя", "всегда целиком, в конце системного сообщения" if agent.attach_profile
             else "не подключён (/attach on)"),
            ("long", "долговременная: решения и знания", f"от новых к старым, пока влезают в {agent.long_budget} токенов"),
            ("working", "рабочая: активная задача", "все поля только активной задачи"),
            ("short", "краткосрочная: хвост диалога", f"последние {agent.keep_last} сообщений (граница — начало хода)")]
    for k, title, rule in rows:
        print(f"   {title:<38} ≈{fmt(e.get(k, 0)):>6}  — {rule}")
    print(f"   в хвосте {parts['tail_messages']} сообщ., не отправлено {dropped} старых сообщений диалога")


def section_arg(name: str) -> str:
    s = SECTION_ALIASES12.get(name.lower())
    if not s:
        raise ValueError(f"нет раздела «{name}». Есть: profile, style, format, constraints, decisions, knowledge, task")
    return s


def print_ops(ops: list[dict]) -> None:
    for item in ops:
        print(f"  {op_label(item)}")
    if not ops:
        print("  (ничего не изменилось)")


# ---------------------------------------------------------------- проверки (демо)


def make_agent(client: LLMClient, helper: LLMClient, root: Path, keep: int, budget: int, max_steps: int,
               attach: bool = True, event=on_event) -> PersonalAgent:
    watch_retries(client, helper, event=event)
    agent = PersonalAgent(client, TOOLS, root, attach=attach, keep_last=keep, long_budget=budget,
                          helper_client=helper, max_steps=max_steps, on_event=event)
    agent.restore_info = agent.load()
    return agent


def turn_record(agent: PersonalAgent, question: str, reply_text: str, error: str = "") -> dict:
    turn = agent.ledger[-1] if agent.ledger else None
    s = agent.layer_stats[-1] if agent.layer_stats and turn and agent.layer_stats[-1]["turn"] == turn.n else {}
    return {"question": question, "answer": reply_text, "error": error,
            "prompt": turn.calls[0].prompt if turn and turn.calls else 0,
            "completion": turn.completion if turn else 0, "cost": turn.cost if turn and turn.calls else 0.0,
            "tokens": (s or {}).get("tokens") or {}, "dropped": (s or {}).get("dropped", 0),
            "tail": (s or {}).get("tail_messages", 0)}


def ask_turn(agent: PersonalAgent, question: str) -> dict:
    try:
        reply = agent.run(question)
        return turn_record(agent, question, reply.text.strip())
    except LLMError as e:
        print(f"  ❌ {e}")
        return {"question": question, "answer": "", "error": str(e), "prompt": 0, "completion": 0, "cost": 0.0,
                "tokens": {}, "dropped": 0, "tail": 0}


def fill_demo(make, ev: Evaluator, root: Path) -> dict:
    """Профиль из диалога: пустой профиль, пользователь называет предпочтения по ходу разговора."""
    print(f"{'=' * 78}\n=== 1. Профиль пополняется из диалога: Олег, профиль пуст\n{'=' * 78}")
    agent = make(root / "fill_oleg")
    steps = []
    for k, line in enumerate(OLEG_TURNS, 1):
        print(f"\n--- реплика {k}/{len(OLEG_TURNS)}\nВы: {line}")
        rec = ask_turn(agent, line)
        print(f"Агент: {clip(rec['answer'], 260) or '(нет ответа)'}")
        rec["eval"] = ev.evaluate(line, rec["answer"], OLEG_EXPECT)
        rec["profile"] = agent.profile.values()
        print(f"  проверка по ожидаемым предпочтениям: {eval_line(rec['eval'])}")
        print_turn(agent)
        steps.append(rec)
    log = agent.log.tail(500)
    values = agent.profile.values()
    learned = []
    for what, sections, rx in OLEG_PROFILE:
        text = normalize(" ".join(f"{k} {v}" for s in sections for k, v in values[s].items()))
        learned.append({"what": what, "ok": bool(re.search(rx, text))})
    print("\n--- профиль после диалога:")
    print_profile(agent)
    print("  записано: " + ", ".join(f"{mark(x['ok'])} {x['what']}" for x in learned))
    print(f"\n  🧹 /newchat: {agent.clear('short')} — профиль остаётся")
    print(f"Вы: {OLEG_CHECK}")
    check = ask_turn(agent, OLEG_CHECK)
    print(f"Агент: {clip(check['answer'], 400)}")
    check["eval"] = ev.evaluate(OLEG_CHECK, check["answer"], OLEG_EXPECT)
    print(f"  проверка в новом диалоге: {eval_line(check['eval'])}")
    print_turn(agent)
    return {"turns": steps, "log": log, "profile": agent.profile.values(), "learned": learned, "check": check,
            "expect": [asdict(p) for p in OLEG_EXPECT], "dir": str(agent.root.relative_to(ROOT))}


def compare_demo(make, ev: Evaluator, root: Path, samples: int, question: str = COMMON_QUESTION) -> dict:
    """Один и тот же вопрос — трём профилям и агенту без профиля; каждый ответ — в новом диалоге."""
    print(f"\n{'=' * 78}\n=== 2. Один вопрос — три профиля и контроль без профиля; ответов на профиль: {samples}\n"
          f"{'=' * 78}\n"
          f"Вопрос: {question}")
    runs = []
    for pid in list(PRESETS) + ["none"]:
        agent = make(root / "compare" / pid)
        if pid != "none":
            agent.load_preset(pid)
        agent.routing = False  # вопрос без предпочтений: роутер здесь не нужен
        title = PRESETS[pid].title if pid != "none" else "без профиля (контроль)"
        print(f"\n--- {title}")
        answers = []
        for i in range(samples):
            rec = ask_turn(agent, question)
            if pid != "none":
                rec["eval"] = ev.evaluate(question, rec["answer"], PRESETS[pid].prefs)
                print(f"  ответ {i + 1}: {eval_line(rec['eval'])}")
            else:
                print(f"  ответ {i + 1}: {code_info(rec['answer'])['words']} слов, блоков кода "
                      f"{code_info(rec['answer'])['blocks']}")
            answers.append(rec)
            agent.clear("short")  # следующий ответ — в новом диалоге
        runs.append({"id": pid, "title": title, "answers": answers,
                     "profile_tokens": agent.estimator.text(agent.profile_block())})
    base = next(r for r in runs if r["id"] == "none")
    base["vs"] = {}
    print("\n--- ответы без профиля по меркам каждого профиля (случайно ли совпадение):")
    for pid in PRESETS:
        base["vs"][pid] = [ev.evaluate(question, a["answer"], PRESETS[pid].prefs) for a in base["answers"]]
        kept = sum(e["kept"] for e in base["vs"][pid])
        total = sum(e["total"] for e in base["vs"][pid])
        print(f"  {pid}: соблюдено {kept}/{total}, ответов полностью по профилю "
              f"{sum(e['all_ok'] for e in base['vs'][pid])}/{len(base['vs'][pid])}")
    print_compare(runs)
    return {"question": question, "samples": samples, "runs": runs}


def print_compare(runs: list[dict]) -> None:
    print("\n  Итог: в скольких ответах предпочтение соблюдено (в скобках — у ответов без профиля)")
    base = next((x for x in runs if x["id"] == "none"), None)
    for r in runs:
        if r["id"] == "none":
            continue
        evals = [a["eval"] for a in r["answers"] if a.get("eval")]
        ctrl_evals = (base or {}).get("vs", {}).get(r["id"], [])
        print(f"  {PRESETS[r['id']].title}: ответов полностью по профилю {sum(e['all_ok'] for e in evals)}/{len(evals)}"
              f" ({sum(e['all_ok'] for e in ctrl_evals)}/{len(ctrl_evals)})")
        for i, p in enumerate(PRESETS[r["id"]].prefs):
            got = sum(e["items"][i]["ok"] is True for e in evals)
            ctrl = sum(e["items"][i]["ok"] is True for e in ctrl_evals)
            print(f"     {p.key:<12} {got}/{len(evals)}  ({ctrl}/{len(ctrl_evals)})  — {clip(p.value, 60)}")


def stability_run(make, ev: Evaluator, root: Path, pid: str, questions: list[str], attach: bool = True) -> dict:
    """10 вопросов подряд в одном диалоге; каждый ответ проверяется по предпочтениям профиля."""
    preset = PRESETS[pid]
    agent = make(root, attach=attach)
    if attach:
        agent.load_preset(pid)
    before = agent.profile.values()
    label = preset.title if attach else f"{preset.title} — профиль НЕ подключён, предпочтения в 1-м сообщении"
    print(f"\n--- {label}")
    steps = []
    for k, q in enumerate(questions, 1):
        text = (ABLATION_INTRO + q) if not attach and k == 1 else q
        rec = ask_turn(agent, text)
        rec["n"] = k
        rec["eval"] = ev.evaluate(q, rec["answer"], preset.prefs)
        rec["first_in_prompt"] = rec["dropped"] == 0  # первое сообщение диалога ещё уходит в модель
        note = "" if attach else (" | 1-е сообщение в запросе" if rec["first_in_prompt"] else " | 1-е сообщение выпало")
        print(f"  {k:>2}. {clip(q, 52):<53} {eval_line(rec['eval'])}{note}")
        steps.append(rec)
    fails = [s["n"] for s in steps if s["eval"]["failed"]]
    per_pref = {}
    for i, p in enumerate(preset.prefs):
        bad = [s["n"] for s in steps if s["eval"]["items"][i]["ok"] is False]
        per_pref[p.id] = {"key": p.key, "first": bad[0] if bad else None, "failed": bad,
                          "kept": sum(s["eval"]["items"][i]["ok"] is True for s in steps)}
    after = agent.profile.values()
    result = {"preset": pid, "title": label, "attach": attach, "steps": steps, "first_fail": fails[0] if fails else None,
              "failed_answers": fails, "per_pref": per_pref, "profile_before": before, "profile_after": after,
              "profile_changed": before != after, "dir": str(agent.root.relative_to(ROOT)),
              "memory_after": {s: v for s, v in agent.memory_json().items() if s not in PROFILE_SECTIONS},
              "kept": sum(s["eval"]["kept"] for s in steps), "total": sum(s["eval"]["total"] for s in steps)}
    first = (f"первое нарушение — ответ {result['first_fail']}" if fails else "предпочтения не нарушены ни разу")
    print(f"  → {first}; ответов с нарушениями {len(fails)}/{len(steps)}; соблюдено {result['kept']}/{result['total']}"
          + ("; профиль изменился за диалог" if result["profile_changed"] and attach else ""))
    return result


def stability_demo(make, ev: Evaluator, root: Path, presets: list[str], n: int, ablation: bool) -> dict:
    print(f"\n{'=' * 78}\n=== 3. Устойчивость: {n} вопросов подряд в одном профиле\n{'=' * 78}")
    runs = [stability_run(make, ev, root / "stability" / pid, pid, STABILITY[:n]) for pid in presets]
    if ablation:
        runs.append(stability_run(make, ev, root / "stability" / "brief_no_attach", "brief", STABILITY[:n], attach=False))
    return {"questions": STABILITY[:n], "runs": runs}


def demo(client: LLMClient, helper: LLMClient, judge: LLMClient, keep: int, budget: int, max_steps: int,
         samples: int, n: int, parts: set[str]) -> int:
    stamp = stamp_now()
    root = DAY_DIR / f"demo_{stamp}"
    ev = Evaluator(judge)
    make = lambda path, attach=True: make_agent(client, helper, path, keep, budget, max_steps, attach,  # noqa: E731
                                                quiet_event)
    report: dict = {"model": client.model, "helper_model": helper.model, "judge_model": judge.model, "keep_last": keep,
                    "long_budget": budget, "stamp": stamp, "dir": str(root.relative_to(ROOT)),
                    "presets": {k: asdict(v) for k, v in PRESETS.items()}, "system_prompt": PERSONAL_PROMPT,
                    "profile_rule": PROFILE_RULE, "router_prompt": ROUTER_PROMPT, "judge_prompt": JUDGE_PROMPT}
    if "fill" in parts:
        report["fill"] = fill_demo(make, ev, root)
    if "compare" in parts:
        report["compare"] = compare_demo(make, ev, root, samples)
    if "stability" in parts or "ablation" in parts:
        report["stability"] = stability_demo(make, ev, root, list(PRESETS) if "stability" in parts else [], n,
                                             "ablation" in parts)
    report["judge"] = ev.totals()
    report["requests"] = {"main": dict(client.stats), "helper": dict(helper.stats) if helper is not client else None,
                          "judge": dict(judge.stats) if judge is not client and judge is not helper else None}
    save_report(report, stamp)
    return 0


# ---------------------------------------------------------------- пересчёт проверок кодом


def stability_summary(run: dict, prefs: list[Pref]) -> None:
    steps = run["steps"]
    run["failed_answers"] = [s["n"] for s in steps if s["eval"]["failed"]]
    run["first_fail"] = run["failed_answers"][0] if run["failed_answers"] else None
    for i, p in enumerate(prefs):
        bad = [s["n"] for s in steps if s["eval"]["items"][i]["ok"] is False]
        run["per_pref"][p.id] = {"key": p.key, "first": bad[0] if bad else None, "failed": bad,
                                 "kept": sum(s["eval"]["items"][i]["ok"] is True for s in steps)}
    run["kept"] = sum(s["eval"]["kept"] for s in steps)
    run["total"] = sum(s["eval"]["total"] for s in steps)


def recheck(path: Path) -> int:
    """Пересчитать проверки кодом в готовом отчёте (после правки детекторов) без запросов к модели:
    ответы и вердикты судьи остаются как были, меняются только вердикты кода и итоги."""
    report = json.loads(path.read_text(encoding="utf-8"))
    known = Pref.__dataclass_fields__
    as_pref = lambda d: Pref(**{k: v for k, v in d.items() if k in known})  # noqa: E731
    presets = {pid: [as_pref(x) for x in p["prefs"]] for pid, p in report["presets"].items()}
    changed = []

    def redo(e: dict, answer: str, prefs: list[Pref], where: str) -> None:
        for item, p in zip(e["items"], prefs):
            auto, evidence = (None, "нет ответа") if e.get("empty") else run_check(p, answer)
            if auto != item["auto"]:
                changed.append({"where": where, "key": p.key, "was": item["auto"], "now": auto,
                                "was_evidence": item["evidence"], "evidence": evidence})
            item["auto"], item["evidence"] = auto, evidence
            item["ok"] = auto if auto is not None else item["judge"]
            item["by"] = "код" if auto is not None else ("судья" if item["judge"] is not None else "")
        e["kept"] = sum(i["ok"] is True for i in e["items"])
        e["failed"] = sum(i["ok"] is False for i in e["items"])
        e["all_ok"] = not e.get("empty") and all(i["ok"] is not False for i in e["items"])

    fill = report.get("fill")
    if fill:
        expect = [as_pref(x) for x in fill["expect"]]
        for k, t in enumerate(fill["turns"], 1):
            redo(t["eval"], t["answer"], expect, f"профиль из диалога, реплика {k}")
        redo(fill["check"]["eval"], fill["check"]["answer"], expect, "профиль из диалога, новый диалог")
    for r in (report.get("compare") or {}).get("runs", []):
        for k, a in enumerate(r["answers"], 1):
            if a.get("eval"):
                redo(a["eval"], a["answer"], presets[r["id"]], f"один вопрос, {r['id']}, ответ {k}")
        for pid, evals in (r.get("vs") or {}).items():
            for k, (e, a) in enumerate(zip(evals, r["answers"]), 1):
                redo(e, a["answer"], presets[pid], f"один вопрос, без профиля по меркам {pid}, ответ {k}")
    for r in (report.get("stability") or {}).get("runs", []):
        for s in r["steps"]:
            redo(s["eval"], s["answer"], presets[r["preset"]],
                 f"устойчивость, {r['preset']}{'' if r['attach'] else ' без профиля'}, ответ {s['n']}")
        stability_summary(r, presets[r["preset"]])
    report["recheck"] = {"at": now_iso(), "changed": changed}
    print(f"Пересчитано: изменилось вердиктов кода — {len(changed)}")
    for c in changed:
        print(f"  {c['where']}: {c['key']} {mark(c['was'])} {c['was_evidence']} → {mark(c['now'])} {c['evidence']}")
    save_report(report, report["stamp"])
    return 0


# ---------------------------------------------------------------- отчёт


def quote(text: str, limit: int) -> list[str]:
    """Ответ цитатой с сохранением строк; обрезанный блок кода закрывается, чтобы не сломать остальной отчёт."""
    text = (text or "").strip() or "(нет ответа)"
    text = text if len(text) <= limit else text[:limit].rstrip() + "…"
    if text.count("```") % 2:
        text += "\n```"
    return ["> " + line if line.strip() else ">" for line in text.splitlines()]


def agreement(report: dict) -> dict:
    """Где есть и проверка кодом, и судья — совпали ли они."""
    evals = []
    if report.get("fill"):
        evals += [t["eval"] for t in report["fill"]["turns"]] + [report["fill"]["check"]["eval"]]
    if report.get("compare"):
        for r in report["compare"]["runs"]:
            evals += [a["eval"] for a in r["answers"] if a.get("eval")]
            for vs in (r.get("vs") or {}).values():
                evals += vs
    if report.get("stability"):
        evals += [s["eval"] for r in report["stability"]["runs"] for s in r["steps"]]
    both = [i for e in evals for i in e["items"] if i["auto"] is not None and i["judge"] is not None]
    differ = [i for i in both if i["auto"] != i["judge"]]
    return {"both": len(both), "agree": len(both) - len(differ),
            "differ": [{"key": i["key"], "auto": i["auto"], "evidence": i["evidence"], "judge": i["judge"],
                        "why": i["why"]} for i in differ]}


def save_report(report: dict, stamp: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    report["agreement"] = agreement(report)
    json_path, md_path = RESULTS_DIR / f"day12_{stamp}.json", RESULTS_DIR / f"day12_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    L = [f"# День 12 · Персонализация ассистента — прогон {stamp}", "",
         f"Модель: `{report['model']}`, роутер памяти: `{report['helper_model']}`, судья: `{report['judge_model']}`. "
         f"Память — как в Дне 11: хвост {report['keep_last']} сообщений, бюджет долговременной {report['long_budget']} "
         f"токенов. Файлы прогона: `{report['dir']}/`.", "",
         "## Профиль пользователя", "",
         "| Раздел | Что хранит | Пример ключей |", "|---|---|---|",
         "| о пользователе (`profile`) | кто он: имя, роль, опыт | имя, роль |",
         "| стиль (`style`) | как с ним разговаривать | тон, обращение, юмор |",
         "| формат (`format`) | какой ответ нужен | длина, структура, код, язык_кода |",
         "| ограничения (`constraints`) | чего в ответе быть не должно и что обязательно | эмодзи, жаргон, в_конце |", "",
         "Файл — `profile.json` рядом со слоями Дня 11. В каждый запрос профиль идёт целиком, блоком в конце "
         "системного сообщения, с правилом: " + f"«{PROFILE_RULE.strip()}»", "",
         "Пополняется из диалога роутером памяти Дня 11 (разделы `style`, `format`, `constraints` добавлены к "
         "`profile`, `decisions`, `knowledge`, `task`), вручную — `/pref`, `/unpref`, готовые профили — `--preset`.", "",
         "Проверка: ✅ — соблюдено, ❌ — нарушено, · — не решено. Где есть проверка кодом (слова, код и его язык, "
         "«вы»/«ты», эмодзи) — решает она, иначе судья (отдельный запрос, temperature 0).", ""]
    rc = report.get("recheck")
    if rc:
        L += [f"Проверки кодом пересчитаны `--recheck` после правки детекторов ({rc['at'][:16].replace('T', ' ')}): "
              f"ответы и вердикты судьи прежние, изменилось вердиктов кода — {len(rc['changed'])}"
              + (": " + "; ".join(f"{md_escape(c['where'])} — {c['key']}: {mark(c['was'])} {md_escape(c['was_evidence'])}"
                                  f" → {mark(c['now'])} {md_escape(c['evidence'])}" for c in rc["changed"])
                 if rc["changed"] else "") + ".", ""]

    fill = report.get("fill")
    if fill:
        L += ["## 1. Профиль пополняется из диалога", "",
              "Профиль пуст; пользователь называет предпочтения по ходу разговора. После каждой реплики видно, что "
              "роутер записал в профиль, и как ответ соответствует ожидаемым предпочтениям: "
              + "; ".join(f"`{p['key']}` — {md_escape(p['value'])}" for p in fill["expect"]) + ".", "",
              "| # | Реплика | Записи в профиль | Ответ: соблюдено | Нарушено |", "|---|---|---|---|---|"]
        by_turn: dict[int, list[str]] = {}
        for item in fill["log"]:
            m = re.search(r"ход (\d+)", item.get("source") or "")
            if m and item["op"] in ("add", "update", "move", "delete"):
                by_turn.setdefault(int(m.group(1)), []).append(md_escape(op_label(item)))
        for k, t in enumerate(fill["turns"], 1):
            e = t["eval"]
            bad = "; ".join(f"{i['key']}: {md_escape(i['evidence'] or i['why'])}" for i in e["items"] if i["ok"] is False)
            L.append(f"| {k} | {md_escape(clip(t['question'], 80))} | {'<br>'.join(by_turn.get(k, [])) or '—'} | "
                     f"{e['kept']}/{e['total']} | {bad or '—'} |")
        L += ["", "**Профиль после диалога:**", ""]
        for s, title in PROFILE_BLOCKS:
            items = fill["profile"].get(s) or {}
            L.append(f"- {title}: " + ("; ".join(f"`{k}` = {md_escape(v)}" for k, v in items.items()) or "—"))
        L += ["", "Записано: " + ", ".join(f"{mark(x['ok'])} {x['what']}" for x in fill["learned"]), ""]
        c = fill["check"]
        L += [f"**Новый диалог** (`/newchat`, краткосрочная память очищена, профиль остался): «{OLEG_CHECK}» → "
              f"соблюдено {c['eval']['kept']}/{c['eval']['total']}", "",
              "| Предпочтение | Код | Судья | Итог |", "|---|---|---|---|"]
        for i in c["eval"]["items"]:
            L.append(f"| {md_escape(i['value'])} | {mark(i['auto'])} {md_escape(i['evidence'])} | {mark(i['judge'])} "
                     f"{md_escape(i['why'])} | {mark(i['ok'])} |")
        L += [""] + quote(c["answer"], 900) + [""]

    cmp_ = report.get("compare")
    if cmp_:
        runs = cmp_["runs"]
        base = next((r for r in runs if r["id"] == "none"), None)
        n = cmp_["samples"]
        L += ["## 2. Один вопрос — три профиля", "",
              f"Вопрос: «{cmp_['question']}». Ответов на профиль: {n}, каждый — в новом диалоге. "
              f"Контроль — тот же вопрос без профиля: его ответы проверены по меркам каждого профиля, чтобы видеть, "
              f"какие предпочтения выполняются и без персонализации.", "",
              "| Профиль | Ответов полностью по профилю | Предпочтений соблюдено | Без профиля: ответов | Без профиля: "
              "предпочтений | Слов в среднем | Блоков кода | Профиль, ≈токенов |", "|---|---|---|---|---|---|---|---|"]
        for r in runs:
            if r["id"] == "none":
                continue
            ev_ = [a["eval"] for a in r["answers"] if a.get("eval")]
            vs = (base or {}).get("vs", {}).get(r["id"], [])
            words = round(sum(e["words"] for e in ev_) / len(ev_)) if ev_ else 0
            L.append(f"| {r['title']} | {sum(e['all_ok'] for e in ev_)}/{len(ev_)} | {sum(e['kept'] for e in ev_)}/"
                     f"{sum(e['total'] for e in ev_)} | {sum(e['all_ok'] for e in vs)}/{len(vs)} | "
                     f"{sum(e['kept'] for e in vs)}/{sum(e['total'] for e in vs)} | {words} | "
                     f"{sum(e['blocks'] for e in ev_)} | {fmt(r['profile_tokens'])} |")
        if base:
            bw = [code_info(a["answer"])["words"] for a in base["answers"]]
            L += ["", f"Без профиля: в среднем {round(sum(bw) / len(bw)) if bw else 0} слов, блоков кода "
                      f"{sum(code_info(a['answer'])['blocks'] for a in base['answers'])} на {len(bw)} ответа."]
        prompts = {r["id"]: [a["prompt"] for a in r["answers"] if a["prompt"]] for r in runs}
        if prompts.get("none"):
            b = sum(prompts["none"]) / len(prompts["none"])
            L += ["", "Сколько токенов добавляет профиль (точно — разница `prompt_tokens` с профилем и без): "
                  + ", ".join(f"{pid} +{fmt(round(sum(v) / len(v) - b))}" for pid, v in prompts.items()
                              if pid != "none" and v) + f" к {fmt(round(b))}."]
        for r in runs:
            if r["id"] == "none":
                continue
            prefs = PRESETS[r["id"]].prefs
            ev_ = [a["eval"] for a in r["answers"]]
            vs = (base or {}).get("vs", {}).get(r["id"], [])
            L += ["", f"### {r['title']}", "",
                  "| Предпочтение | " + " | ".join(f"ответ {i + 1}" for i in range(len(ev_))) + " | соблюдено | "
                  + " | ".join(f"без профиля {i + 1}" for i in range(len(vs))) + " |",
                  "|---|" + "---|" * (len(ev_) + 1 + len(vs))]
            for i, p in enumerate(prefs):
                cells = [f"{mark(e['items'][i]['ok'])} {md_escape(clip(e['items'][i]['evidence'] or e['items'][i]['why'], 50))}"
                         for e in ev_]
                ctrl = [f"{mark(e['items'][i]['ok'])} {md_escape(clip(e['items'][i]['evidence'] or e['items'][i]['why'], 40))}"
                        for e in vs]
                got = sum(e["items"][i]["ok"] is True for e in ev_)
                L.append(f"| {md_escape(p.key)}: {md_escape(p.value)} | " + " | ".join(cells)
                         + f" | **{got}/{len(ev_)}** | " + " | ".join(ctrl) + " |")
            L += ["", "Ответ 1:", ""] + quote(r["answers"][0]["answer"], 900)
        if base:
            L += ["", "### Без профиля — ответ 1", ""] + quote(base["answers"][0]["answer"], 900)
        L.append("")

    st = report.get("stability")
    if st:
        qs = st["questions"]
        L += [f"## 3. Устойчивость: {len(qs)} вопросов подряд в одном диалоге", "",
              "Профиль подключён к каждому запросу, диалог идёт подряд: в запрос уходит хвост из последних сообщений "
              "(как в Дне 11) — значит, и прошлые ответы. Последний прогон — контроль: тот же профиль Анны не "
              "подключён, а её предпочтения сказаны один раз, в первом сообщении.", "",
              "Вопросы: " + " ".join(f"**{i}.** {md_escape(q)}" for i, q in enumerate(qs, 1)), "",
              "| Прогон | Первое нарушение | Ответов с нарушениями | Предпочтений соблюдено | Профиль изменился |",
              "|---|---|---|---|---|"]
        for r in st["runs"]:
            first = f"ответ {r['first_fail']}" if r["first_fail"] else "нет"
            L.append(f"| {r['title']} | {first} | {len(r['failed_answers'])}/{len(r['steps'])} "
                     f"({', '.join(map(str, r['failed_answers'])) or '—'}) | {r['kept']}/{r['total']} | "
                     f"{'да' if r['profile_changed'] and r['attach'] else 'нет' if r['attach'] else '— (заполнен роутером)'} |")
        for r in st["runs"]:
            prefs = PRESETS[r["preset"]].prefs
            L += ["", f"### {r['title']}", "", "| Предпочтение | " + " | ".join(str(s["n"]) for s in r["steps"])
                  + " | первое нарушение |", "|---|" + "---|" * (len(r["steps"]) + 1)]
            for i, p in enumerate(prefs):
                pp = r["per_pref"][p.id]
                L.append(f"| {md_escape(p.key)} | " + " | ".join(mark(s["eval"]["items"][i]["ok"]) for s in r["steps"])
                         + f" | {pp['first'] or '—'} |")
            L.append("| **все соблюдены** | " + " | ".join("⚠️" if s["eval"].get("empty") else "✅" if s["eval"]["all_ok"]
                                                         else "❌" for s in r["steps"]) + " | |")
            L.append("| слов | " + " | ".join(str(s["eval"]["words"]) for s in r["steps"]) + " | |")
            if not r["attach"]:
                L.append("| 1-е сообщение в запросе | " + " | ".join("да" if s["first_in_prompt"] else "нет"
                                                                  for s in r["steps"]) + " | |")
            L.append("| вход, токенов | " + " | ".join(fmt(s["prompt"]) for s in r["steps"]) + " | |")
            bad = [(s["n"], i) for s in r["steps"] for i in s["eval"]["items"] if i["ok"] is False]
            if bad:
                L += ["", "Нарушения: " + "; ".join(f"{n}. {i['key']} — {md_escape(clip(i['evidence'] or i['why'], 70))}"
                                                     for n, i in bad[:15])]
            if r["attach"] and r["profile_changed"]:
                L += ["", "Профиль после диалога: " + md_escape(json.dumps(r["profile_after"], ensure_ascii=False))]
            if not r["attach"]:
                L += ["", "Роутер записал в профиль (он не подключался): "
                      + md_escape(json.dumps({s: v for s, v in r["profile_after"].items() if v}, ensure_ascii=False))]
                mem = {s: v for s, v in (r.get("memory_after") or {}).items() if v}
                L += ["", "В рабочую и долговременную память (они подключаются): "
                      + (md_escape(json.dumps(mem, ensure_ascii=False)) if mem else "ничего")]
        L.append("")

    ag = report["agreement"]
    L += ["## Проверка кодом и судья", "",
          f"Где есть оба вердикта, они совпали в {ag['agree']} из {ag['both']} случаев."]
    if ag["differ"]:
        L += ["", "| Предпочтение | Код | Судья |", "|---|---|---|"]
        for d in ag["differ"][:20]:
            L.append(f"| {md_escape(d['key'])} | {mark(d['auto'])} {md_escape(d['evidence'])} | {mark(d['judge'])} "
                     f"{md_escape(d['why'])} |")
    j = report["judge"]
    L += ["", f"Судья: {j['count']} запросов, вход {fmt(j['prompt'])}, выход {fmt(j['completion'])}, {money(j['cost'])}"
          + (f", не разобрано {j['errors']}" if j["errors"] else "") + ".", ""]

    if st:
        L += ["## Все ответы прогонов на устойчивость", ""]
        for r in st["runs"]:
            L += [f"### {r['title']}", ""]
            for s in r["steps"]:
                L.append(f"- {'✅' if s['eval']['all_ok'] else '❌'} **{s['n']}. {md_escape(s['question'][:120])}** "
                         f"{md_escape(clip(s['answer'] or s['error'], 600))}")
            L.append("")
    req = report["requests"]
    L += [f"Запросов к модели: {req['main']['requests']}, повторов {req['main']['retries']}, отказов {req['main']['failed']}"
          + (f"; отдельный клиент роутера: {req['helper']['requests']}" if req.get("helper") else "")
          + (f"; отдельный клиент судьи: {req['judge']['requests']}" if req.get("judge") else "") + "."]
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {md_path.relative_to(ROOT)}, {json_path.name}")
    return md_path


# ---------------------------------------------------------------- чат


HELP = ("/profile, /pref <раздел> <ключ> = <значение>, /unpref <раздел> <ключ>, /preset [brief|deep|code], "
        "/attach on|off, /check, /compare, /stability [N], /clear profile|short|working|long, /memory, /context, "
        "/newchat, /log, /tokens, /history, /exit")


def chat(agent: PersonalAgent, ev: Evaluator, make) -> int:
    print(f"Агент на {agent.client.model}, пользователь {agent.root.name}. Команды: {HELP}\n")
    last: tuple[str, str] | None = None
    while True:
        try:
            text = input(f"[{agent.root.name}{'' if agent.attach_profile else ' · профиль off'}] Вы: ").strip()
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
            if command == "/profile":
                print_profile(agent)
            elif command == "/pref":
                head, sep, value = arg.partition("=")
                parts = head.split()
                if not sep or len(parts) != 2 or not value.strip():
                    print("  /pref <style|format|constraints|profile> <ключ> = <значение>")
                else:
                    print_ops(agent.write(section_arg(parts[0]), parts[1], value.strip(), "команда пользователя"))
                    agent.save_memory()
            elif command == "/unpref":
                if len(args) != 2:
                    print("  /unpref <раздел> <ключ>")
                else:
                    print_ops(agent.forget(section_arg(args[0]), args[1], "команда пользователя"))
                    agent.save_memory()
            elif command == "/preset":
                if not args:
                    for p in PRESETS.values():
                        print(f"  {p.id:<6} {p.title}")
                elif args[0] not in PRESETS:
                    print(f"  нет профиля «{args[0]}». Есть: {', '.join(PRESETS)}")
                else:
                    print_ops(agent.load_preset(args[0]))
                    print_profile(agent)
            elif command == "/attach":
                if args and args[0] in ("on", "off"):
                    agent.attach_profile = args[0] == "on"
                print(f"  профиль {'подключён к каждому запросу' if agent.attach_profile else 'НЕ подключается к запросу'}")
            elif command == "/check":
                if not last:
                    print("  ещё не было ответа")
                else:
                    prefs = agent.current_prefs()
                    if not prefs:
                        print("  в профиле нет предпочтений — проверять нечего")
                    else:
                        e = ev.evaluate(last[0], last[1], prefs)
                        for i in e["items"]:
                            print(f"  {mark(i['ok'])} {i['key']}: {i['value']}  — {i['by']}: {i['evidence'] or i['why']}")
                        print(f"  {eval_line(e)}")
            elif command == "/compare":
                compare_demo(make, ev, DAY_DIR / f"{agent.root.name}_compare_{stamp_now()}", 1,
                             arg.strip() or COMMON_QUESTION)
            elif command == "/stability":
                count = int(args[0]) if args and args[0].isdigit() else len(STABILITY)
                prefs = agent.current_prefs()
                for k, q in enumerate(STABILITY[:count], 1):
                    rec = ask_turn(agent, q)
                    e = ev.evaluate(q, rec["answer"], prefs) if prefs else None
                    print(f"  {k:>2}. {clip(q, 52):<53} " + (eval_line(e) if e else f"{code_info(rec['answer'])['words']} слов"))
            elif command == "/clear":
                layer = {"profile": "persona", "профиль": "persona", "persona": "persona", "short": "short",
                         "working": "working", "long": "long"}.get(args[0].lower()) if args else None
                if not layer:
                    print("  /clear profile|short|working|long")
                else:
                    print(f"  🧹 {agent.clear(layer)}")
            elif command == "/memory":
                print_profile(agent)
                print_memory(agent)
            elif command == "/context":
                print_context12(agent)
            elif command == "/newchat":
                print(f"  🧹 {agent.clear('short')}; профиль, рабочая и долговременная память остаются")
            elif command == "/log":
                for item in agent.log.tail(int(args[0]) if args and args[0].isdigit() else 20):
                    print(f"  {item['at'][11:19]} {op_label(item)}  ⟵ {item.get('source', '')}")
            elif command == "/tokens":
                print_ledger(agent)
                for s in agent.layer_stats:
                    print(f"  ход {s['turn']}: вход {fmt(s['prompt'])} = {layer_line(s['tokens'])}")
                j = ev.totals()
                print(f"  + судья: запросов {j['count']}, {money(j['cost'])}")
            elif command == "/history":
                print_history(agent.history)
            elif command.startswith("/"):
                print(f"  Неизвестная команда. Есть: {HELP}")
            else:
                reply = agent.run(text)
                last = (text, reply.text.strip())
                print(f"\nАгент: {reply.text.strip() or '(пустой ответ)'}\n")
                print_turn(agent)
            print()
        except LLMError as e:
            print(f"❌ Ошибка: {e}\n", file=sys.stderr)
        except (StoreError, ValueError) as e:
            print(f"⚠️  {e}\n", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n(прервано — ход в историю не попал)\n")


# ---------------------------------------------------------------- запуск


def main() -> int:
    parser = argparse.ArgumentParser(description="День 12: персонализация — профиль пользователя поверх памяти Дня 11.")
    parser.add_argument("prompt", nargs="?", help="один ход и выход; без него — чат в консоли")
    parser.add_argument("--user", default="default", help="пользователь: memory/day12/<имя>/")
    parser.add_argument("--new", action="store_true", help="начать заново: папка пользователя уходит в архив")
    parser.add_argument("--preset", choices=list(PRESETS), help="заполнить профиль готовым набором предпочтений")
    parser.add_argument("--no-attach", action="store_true", help="не подключать профиль к запросам (для сравнения)")
    parser.add_argument("--keep", type=int, default=6, help="краткосрочная память: сколько последних сообщений в промпт")
    parser.add_argument("--long-budget", type=int, default=1500, help="долговременная память: бюджет токенов")
    parser.add_argument("--helper-model", help="модель роутера памяти (по умолчанию та же)")
    parser.add_argument("--judge-model", help="модель судьи (по умолчанию та же)")
    parser.add_argument("--demo", action="store_true", help="профиль из диалога, 3 профиля × 1 вопрос, 10 вопросов подряд")
    parser.add_argument("--only", help="для --demo: части через запятую — fill, compare, stability, ablation")
    parser.add_argument("--samples", type=int, default=3, help="для --demo: ответов на общий вопрос у каждого профиля")
    parser.add_argument("--questions", type=int, default=len(STABILITY), help="для --demo: вопросов подряд (≤ 10)")
    parser.add_argument("--recheck", metavar="ФАЙЛ", help="пересчитать проверки кодом в results/day12_*.json "
                                                          "без запросов к модели и переписать отчёт")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--max-steps", type=int, default=8, help="предел запросов к модели за один ход")
    args = parser.parse_args()

    if not NAME_RE.fullmatch(args.user):
        print("Ошибка: имя пользователя — буквы, цифры, дефис, подчёркивание.", file=sys.stderr)
        return 1
    if args.keep < 0 or args.long_budget < 0 or args.samples < 1 or not 1 <= args.questions <= len(STABILITY):
        print(f"Ошибка: --keep ≥ 0, --long-budget ≥ 0, --samples ≥ 1, --questions от 1 до {len(STABILITY)}.",
              file=sys.stderr)
        return 1
    if args.recheck:
        try:
            return recheck(Path(args.recheck))
        except (OSError, ValueError, KeyError) as e:
            print(f"Ошибка: не удалось пересчитать {args.recheck}: {e}", file=sys.stderr)
            return 1
    parts = {"fill", "compare", "stability", "ablation"}
    if args.only:
        chosen = {x.strip() for x in args.only.split(",") if x.strip()}
        if not chosen or chosen - parts:
            print(f"Ошибка: --only — из {', '.join(sorted(parts))}.", file=sys.stderr)
            return 1
        parts = chosen
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    model = args.model or settings.model
    new_client = lambda m: LLMClient(settings.base_url, settings.api_key, m, settings.timeout,  # noqa: E731
                                     retries=settings.retries)
    client = new_client(model)
    helper = new_client(args.helper_model) if args.helper_model else client
    judge = new_client(args.judge_model) if args.judge_model else client
    if args.demo:
        return demo(client, helper, judge, args.keep, args.long_budget, args.max_steps, args.samples, args.questions,
                    parts)

    root = DAY_DIR / args.user
    if args.new and root.exists():
        target = root.with_name(f"{root.name}_{stamp_now()}")
        os.replace(root, target)
        print(f"Прошлый профиль сохранён в {target.relative_to(ROOT)}.")
    try:
        agent = make_agent(client, helper, root, args.keep, args.long_budget, args.max_steps, not args.no_attach)
    except StoreError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print_restore(agent.restore_info, agent.store, agent)
    for note in agent.notes:
        print(f"⚠️  {note}")
    if args.preset:
        print_ops(agent.load_preset(args.preset))
    print_profile(agent)
    print()
    ev = Evaluator(judge)
    make = lambda path, attach=True: make_agent(client, helper, path, args.keep, args.long_budget,  # noqa: E731
                                                args.max_steps, attach, quiet_event)
    if args.prompt:
        try:
            reply = agent.run(args.prompt)
        except (LLMError, StoreError) as e:
            print(f"❌ Ошибка: {e}", file=sys.stderr)
            return 1
        print(f"\nАгент: {reply.text.strip()}\n")
        print_turn(agent)
        return 0
    return chat(agent, ev, make)


if __name__ == "__main__":
    from run_timer import run_timed
    sys.exit(run_timed(main))
