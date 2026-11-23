"""День 12. Персонализация: от одного героя — к множеству персонажей. У каждого героя и NPC своя карточка.

Мастер игры Дня 11 (`LayeredAgent`: три слоя памяти героя, летописец, кости, учёт токенов) ведёт теперь мир, где
много персонажей — герои игроков и NPC. Каждый — со своей карточкой (это «профиль» задания) в
memory/day12/<мир>/cast/<id>.json:

    о персонаже   имя, роль, школа, откуда — кто он                      раздел about
    манера речи   тон, на «ты» или на «вы», как зовёт себя и собеседника  раздел voice
    реплики       длина, построение (притчи, цитаты, встречный вопрос)   раздел manner
    табу          чего в его речи быть не должно                         раздел taboo
    уговор        как говорить именно с этим героем — о чём герой попросил  раздел terms
    отношение     как он относится к каждому герою — меняется от событий  раздел attitude

У каждого героя игрока — ещё и свои слои памяти Дня 11 в memory/day12/<мир>/heroes/<герой>/ (личность героя
переехала из долговременной памяти в его карточку). Карточки NPC общие для мира: что один герой изменил в NPC,
видит и другой. А уговор с героем («говори со мной на ты», «зови по имени») и отношение к герою у NPC свои
для каждого героя: в промпт идут только записи о текущем герое. Записи заготовки (готовый NPC, игра, /pref)
летописец не переписывает: разовая просьба игрока не должна менять характер персонажа.

Карточки подключаются к каждому запросу: в конце системного сообщения — карточка героя и всех, кто в сцене;
собеседник героя (/talk) — целиком, с правилом держать образ в каждой реплике. Пополняются из игры: летописец Дня 11
получил раздел cards и после каждого хода записывает, что игрок попросил у персонажа («говори со мной на ты»),
как изменилось отношение (герой помог — уважает) и кто новый появился в сцене. Вручную — /pref и /unpref;
готовые NPC — --preset lo | bai | hong.

Держится ли образ, проверяется двумя способами: кодом (число слов, «ты»/«вы», обязательное обращение, вопрос
в конце, восклицания, современные слова) и судьёй — отдельным запросом с temperature=0 (тон). Где есть проверка
кодом, решает она; судья — где её нет.

    python day12_personalization.py                          # игра (мир default, герой luchen)
    python day12_personalization.py --world w1 --preset lo   # в мире появляется старейшина Ло Ган
    python day12_personalization.py --attach about           # от карточки собеседника — только «кто он»
    python day12_personalization.py --demo                   # карточка из игры, 3 NPC × 1 вопрос, 10 реплик подряд,
                                                             # два героя в одном мире
    python day12_personalization.py --recheck results/day12_<время>.json   # пересчитать проверки кодом

Команды: /cast, /card [id], /pref <id> <раздел> <ключ> = <значение>, /unpref <id> <раздел> <ключ>, /preset <id>,
/npc new <id> <имя, роль>, /scene [add|remove <id>], /talk <id>|off, /hero [list|new <id> <имя>|switch <id>],
/attach full|about, /check, /compare, /stability [N], /memory, /context, /mission …, /circle …, /newchat, /log,
/tokens, /history, /exit.
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
from day7_memory import MEMORY_DIR, ROOT, StoreError, now_iso, print_history, print_restore
from day8_tokens import Usage, clip, fmt, money, price_for, print_ledger
from day9_compression import normalize, watch_retries
from day10_strategies import HelperCall, StickyFacts, extract_json_object, read_json, write_json
from day11_memory_layers import (GAME_TOOLS, LAYER_TITLES, SECTION_LAYER, SECTION_TITLES, WORLD,
                                 LayeredAgent, MemoryLog, archive_file, circle_command, clean_part, entries, md_escape,
                                 mission_command, print_memory, stamp_now, turn_source)
from day11_memory_layers import on_event as day11_event
from day11_memory_layers import print_context as day11_context
from llm_client import LLMClient, LLMError

RESULTS_DIR = ROOT / "results"
DAY_DIR = MEMORY_DIR / "day12"
NAME_RE = re.compile(r"[\w-]+")

# Карточка персонажа — пять разделов. Личность героя (persona) в Дне 11 жила в долговременной памяти,
# теперь она — раздел about его карточки; школа, характер, особенности, связи, круг и задание — в слоях Дня 11.
CARD_SECTIONS = ("about", "voice", "manner", "taboo", "terms", "attitude")
PER_HERO = ("terms", "attitude")  # ключ — id героя; в промпт идёт только запись о текущем герое
PREF_SECTIONS = ("voice", "manner", "taboo")
CARD_TITLES = {"about": "о персонаже", "voice": "манера речи", "manner": "реплики", "taboo": "табу",
               "terms": "уговор с героем", "attitude": "отношение"}
CARD_BLOCKS = (("about", "Кто это"), ("voice", "Манера речи"), ("manner", "Реплики"), ("taboo", "Табу"),
               ("terms", "Уговор с героем — как говорить именно с ним (важнее общей манеры речи)"),
               ("attitude", "Отношение к героям"))
CARD_ALIASES = {**{s: s for s in CARD_SECTIONS}, **{v: k for k, v in CARD_TITLES.items()}, "о_нём": "about",
                "речь": "voice", "манера": "voice", "формат": "manner", "длина": "manner", "запреты": "taboo",
                "уговор": "terms", "отношения": "attitude"}
HERO_SECTIONS = ("school", "nature", "traits", "bonds", "circle", "gear", "mission")  # слои Дня 11 без persona
SECTION_LAYER12 = {**SECTION_LAYER, **{s: "cast" for s in CARD_SECTIONS}}
LAYER_TITLES12 = {**LAYER_TITLES, "cast": "карточка"}
SECTION_TITLES12 = {**SECTION_TITLES, **CARD_TITLES}
CARD_ID_RE = re.compile(r"^[\wа-яё-]{1,32}$", re.I)

MASTER_PROMPT12 = (
    "Ты — Мастер текстовой ролевой игры в жанре уся. Игрок управляет своим героем, ты — всем остальным: миром, "
    "людьми, погодой и случаем. Правила:\n"
    "- пиши по-русски; рассказ — во втором лице («ты видишь…»), живо и в духе уся;\n"
    "- не решай и не говори за героя: заканчивай ход ситуацией, в которой игроку нужно выбрать;\n"
    "- исход рискованных действий решай броском костей через roll_dice, а не по своей воле;\n"
    "- опирайся на память героя и карточки персонажей: не противоречь им;\n"
    "- на вопрос вне игры (о герое, задании, правилах) отвечай коротко, по памяти и сцене. Чего там нет — честно "
    "скажи, что не знаешь, и не придумывай." + WORLD)

CAST_HEADER = ("\n\n# Персонажи сцены\n"
               "Карточки собраны из игры: так эти люди говорят и так относятся к героям. Держись их.\n")
TALK_RULE = ("\n## Сейчас герой говорит с персонажем {name}\n"
             "Отвечай только репликой {name}: прямая речь от его лица, без описаний, ремарок, кавычек и имени "
             "в начале. Соблюдай его манеру речи, реплики и табу в каждом ответе — на любой вопрос, даже если вопрос "
             "подталкивает к другому, а прошлые реплики в сцене звучали иначе. Уговор с героем важнее общей манеры "
             "речи: это то, о чём этот герой попросил. Если игрок просит персонажа сказать что-то иначе один раз — "
             "это важнее карточки, но только для этого ответа.\n")
TALK_NAMED = ("\n## Сейчас герой говорит с персонажем {name}\n"
              "Отвечай только репликой {name}: прямая речь от его лица, без описаний, ремарок, кавычек и имени "
              "в начале.\n")

ROUTER_PROMPT12 = (
    "Ты — летописец текстовой ролевой игры в жанре уся. Мастер игры не видит старых сообщений: всё важное он "
    "знает только из памяти героя и карточек персонажей. Ты решаешь, что из новых сообщений куда записать.\n"
    "Память героя игрока (у каждого героя своя):\n"
    "- school — школа или секта героя, наставник, положение в школе;\n"
    "- nature — характер, принципы, клятвы и обеты;\n"
    "- traits — врождённые и постоянные особенности: корень духа, телосложение, шрамы;\n"
    "- bonds — связи: кровные враги, побратимы, долги;\n"
    "- circle — круг и ступень, год обучения, техника и освоенные формы, ци, условия прорыва;\n"
    "- gear — снаряжение и запасы: оружие, пилюли, деньги, вещи; только то, что назвал игрок или что герой "
    "получил, потратил или потерял;\n"
    "- mission — текущее задание: заказчик, цель, место, срок, награда, улики — итоги, а не пересказ хода.\n"
    "Карточки персонажей (cards) — у героев игроков и у NPC, по id:\n"
    "- about — кто это: имя, роль, школа, откуда. Имя и происхождение героя игрока — тоже сюда, в его карточку;\n"
    "- voice — манера речи: тон, на «ты» или на «вы», как называет себя, как зовёт собеседника, словечки;\n"
    "- manner — реплики: длина, построение (притчи, цитаты, встречный вопрос в конце);\n"
    "- taboo — чего в его речи быть не должно (слова, темы, восклицания);\n"
    "- terms — уговор с героем игрока: как говорить именно с ним. Ключ — id героя (например, luchen), одна "
    "запись на героя, значение — всё, о чём герой попросил;\n"
    "- attitude — отношение к героям игроков: ключ — id героя, значение — как относится и почему.\n"
    "Когда игрок прямо просит персонажа говорить с ним иначе («говори со мной на ты», «короче», «не зови меня "
    "господином») — это уговор с этим героем: раздел terms, ключ — id героя. Новое допиши к прежнему значению, "
    "противоречащее — замени. Уговор действует с этого момента, но только с этим героем. Разовая просьба на один "
    "ответ («объясни подробно», «скажи одной фразой», «перечисли всё») — не запись. Когда событие меняет "
    "отношение (герой помог, оскорбил, спас) — attitude. Когда в сцене впервые появился названный по имени "
    "персонаж — заведи ему карточку с about; id — одно короткое слово латиницей в нижнем регистре (например, "
    "zhang). Общую манеру речи (voice, manner, taboo) пиши только тому, у кого её ещё нет, — по тому, как он "
    "говорит в рассказе Мастера; уже записанную не меняй.\n"
    "Тебе дают текущую память героя, карточки, последний рассказ Мастера и новые сообщения игрока. Верни JSON вида "
    '{"school": {"set": {}, "delete": []}, "nature": {...}, "traits": {...}, "bonds": {...}, "circle": {...}, '
    '"gear": {...}, "mission": {...}, "cards": {"<id>": {"about": {"set": {}, "delete": []}, "voice": {...}, '
    '"manner": {...}, "taboo": {...}, "terms": {...}, "attitude": {...}}}} — разделы без изменений можно опустить:\n'
    "- из сообщений игрока бери то, что он сказал о герое, и то, что герой сделал или попросил. Из рассказа "
    "Мастера — только то, что случилось на самом деле. Описания и догадки — не записи;\n"
    "- значения пиши точно и коротко (до 15 слов). Если значение изменилось — новое, а в скобках старое;\n"
    "- ключи короткие, по-русски, в нижнем регистре через подчёркивание. Запись уже есть — обнови тот же ключ. "
    "Один ключ — в одном разделе;\n"
    "- вопросы игрока вне игры — не записи. Ничего не придумывай. Нечего записывать — верни пустой объект {}.\n"
    "Верни только JSON.")

JUDGE_PROMPT = (
    "Ты строго проверяешь, держит ли персонаж ролевой игры свой образ. Тебе дают реплику игрока, требования "
    "к речи персонажа (id: требование) и ответ персонажа. Для каждого требования реши: соблюдено (true) "
    "или нарушено (false). Нарушено — если в ответе есть хотя бы один явный случай против требования: например, "
    "требование «на вы», а в ответе есть обращение на «ты»; «без восклицаний», а в ответе есть «!». "
    "Оценивай только соблюдение образа, а не качество ответа. "
    'Верни JSON вида {"<id>": {"ok": true, "why": "до 15 слов"}} по всем id. Только JSON.')


# ---------------------------------------------------------------- карточки персонажей


class Cast:
    """Карточки всех персонажей мира: cast/<id>.json. Общие для всех героев мира."""

    def __init__(self, folder: Path):
        self.dir = folder
        self.cards: dict[str, dict] = {}

    @staticmethod
    def blank(char_id: str, kind: str) -> dict:
        return {"id": char_id, "kind": kind, "preset": "", "created_at": now_iso(),
                "data": {s: {} for s in CARD_SECTIONS}}

    def load(self) -> None:
        self.cards = {}
        if not self.dir.exists():
            return
        for p in sorted(self.dir.glob("*.json")):
            if not CARD_ID_RE.match(p.stem) or re.search(r"_\d{8}_\d{6}", p.stem):
                continue
            raw = read_json(p)
            card = self.blank(p.stem, str(raw.get("kind") or "npc"))
            card.update(preset=str(raw.get("preset") or ""), created_at=str(raw.get("created_at") or now_iso()))
            card["data"] = {s: entries((raw.get("data") or {}).get(s)) for s in CARD_SECTIONS}
            self.cards[p.stem] = card

    def save(self) -> None:
        for cid, card in self.cards.items():
            write_json(self.dir / f"{cid}.json", {"format": 1, "updated_at": now_iso(), **card})

    def get(self, char_id: str) -> dict:
        if char_id not in self.cards:
            raise ValueError(f"нет персонажа «{char_id}». Есть: {', '.join(self.cards) or '—'}")
        return self.cards[char_id]

    def ensure(self, char_id: str, kind: str = "npc") -> dict:
        if char_id not in self.cards:
            self.cards[char_id] = self.blank(char_id, kind)
        return self.cards[char_id]

    def name(self, char_id: str) -> str:
        card = self.cards.get(char_id)
        if not card:
            return char_id
        about = card["data"]["about"]
        return next((about[k]["value"] for k in ("имя", "name") if k in about), char_id)

    def values(self, char_id: str) -> dict[str, dict[str, str]]:
        return {s: {k: e["value"] for k, e in self.get(char_id)["data"][s].items()} for s in CARD_SECTIONS}

    def count(self, char_id: str) -> int:
        return sum(len(v) for v in self.get(char_id)["data"].values())

    def archive(self, char_id: str) -> Path | None:
        return archive_file(self.dir / f"{char_id}.json")


def card_text(cast: Cast, char_id: str, sections=CARD_SECTIONS, hero: str | None = None) -> str:
    """Карточка текстом. С hero — из отношения только запись об этом герое: к другим героям NPC относится
    по-своему, но в сцену текущего героя это не идёт."""
    d = cast.get(char_id)["data"]
    parts = []
    for s, title in CARD_BLOCKS:
        items = {k: e for k, e in d[s].items() if not (s in PER_HERO and hero and k != hero)}
        if s in sections and items:
            parts.append(f"{title}:\n" + "\n".join(f"- {k}: {e['value']}" for k, e in items.items()))
    return "\n".join(parts)


def from_router(source: str) -> bool:
    """Запись сделал летописец (из сцены, догоном или импортом), а не игра и не игрок."""
    return (source or "").startswith(("сцена", "догон", "импорт"))


def is_seed(source: str) -> bool:
    """Запись заготовки карточки: готовый NPC, игра, команда игрока."""
    return (source or "").startswith(("готовый", "игра", "команда"))


class CastLog(MemoryLog):
    """Журнал Дня 11, который знает карточки: у записи в карточку есть ещё и персонаж (char)."""

    def __init__(self, path: Path):
        super().__init__(path, SECTION_LAYER12)


def op_label(item: dict, cast: Cast | None = None) -> str:
    if item["op"] == "clear":
        return f"🧹 очищен слой: {LAYER_TITLES12.get(item['layer'], item['layer'])}"
    if item["op"] in ("mission", "circle"):
        noun = "задание" if item["op"] == "mission" else "круг"
        return f"{'📜' if item['op'] == 'mission' else '☯'} {noun} {item['key']} «{item['value']}» — {item.get('status', '')}"
    if item["op"] == "card":
        return f"🎭 новая карточка: {item['key']} ({item.get('value', '')})"
    sec = SECTION_TITLES12.get(item["section"], item["section"])
    if item.get("char"):
        who = cast.name(item["char"]) if cast else item["char"]
        where = f"карточка {who} · {sec}"
    else:
        where = f"{LAYER_TITLES12.get(item['layer'], item['layer'])} · {sec}"
    if item["op"] == "move":
        src = item.get("from", "")
        where = f"{SECTION_TITLES12.get(src, src)} → {where}"
    sign = {"add": "+", "update": "~", "move": "↪", "delete": "−", "rejected": "⛔"}.get(item["op"], item["op"])
    value = f" = {clip(item['value'], 80)}" if item.get("value") and item["op"] != "delete" else ""
    old = (f" (было: {clip(item['old'], 40)})" if item.get("old") and item["op"] == "update"
           and normalize(item["old"]) not in normalize(item.get("value") or "") else "")
    why = f" — {item['why']}" if item.get("why") else ""
    return f"{sign} [{where}] {item['key']}{value}{old}{why}"


# ---------------------------------------------------------------- готовые NPC и требования к речи


@dataclass(frozen=True)
class Pref:
    """Одно требование карточки: что записано и как его проверить кодом (check пустой — проверяет только судья)."""
    id: str
    section: str  # voice | manner | taboo
    key: str
    value: str
    check: str = ""  # max_words | min_words | formal | informal | phrase | no_phrase | ends_question | no_exclaim | no_modern
    param: int | str | None = None


@dataclass(frozen=True)
class Preset:
    id: str
    title: str
    about: dict
    prefs: tuple[Pref, ...]


NO_MODERN_VALUE = "ни одного современного слова: окей, проблема, информация, минута, процент, телефон…"
PRESETS = {
    "lo": Preset("lo", "Старейшина Ло Ган — суровый и немногословный",
                 {"имя": "Ло Ган", "роль": "старейшина Крепости Железной Горы, мастер кулака, ему семьдесят лет"},
                 (Pref("length", "manner", "длина", "коротко: одна-три фразы, до 40 слов", "max_words", 40),
                  Pref("address", "voice", "обращение", "к собеседнику — на «ты»", "informal"),
                  Pref("self", "voice", "о_себе", "называет себя «этот старик»", "phrase",
                       r"этот старик|этого старика|этому старику|этим стариком|этом старике"),
                  Pref("tone", "voice", "тон", "суровый и сухой, без похвалы и без шуток"),
                  Pref("exclaim", "taboo", "восклицания", "без восклицательных знаков", "no_exclaim"),
                  Pref("modern", "taboo", "слова", NO_MODERN_VALUE, "no_modern"))),
    "bai": Preset("bai", "Учёный Бай Шу — многословный и учтивый",
                  {"имя": "Бай Шу", "роль": "учёный-стратег Павильона Тысячи Свитков"},
                  (Pref("length", "manner", "длина", "подробно и витиевато: от 120 слов", "min_words", 120),
                   Pref("address", "voice", "обращение", "к собеседнику — на «вы»", "formal"),
                   Pref("call", "voice", "как_зовёт", "называет собеседника «молодой господин»", "phrase",
                        r"молод\w* господин\w*"),
                   Pref("quote", "manner", "цитаты", "в каждой реплике ссылается на трактат или мудреца древности",
                        "phrase", r"трактат|мудр[еоа]ц|мудрецы|древн\w+|как сказано|гласит|канон|\w+-цзы|конфуци|"
                                  r"как (?:учил|учит|писал|говаривал|говорил|сказал)|по слову|искусств\w* войны|"
                                  r"дао дэ цзин|ши цзин|и цзин|книг\w* (?:перемен|песен|обрядов)|летопис\w*"),
                   Pref("tone", "voice", "тон", "учтивый, книжный, без грубости и без просторечий"),
                   Pref("modern", "taboo", "слова", NO_MODERN_VALUE, "no_modern"))),
    "hong": Preset("hong", "Мечница Хун Лин — дерзкая и насмешливая",
                   {"имя": "Хун Лин", "роль": "странствующая мечница без школы, бывшая разбойница"},
                   (Pref("length", "manner", "длина", "бойко: до 70 слов", "max_words", 70),
                    Pref("address", "voice", "обращение", "к собеседнику — на «ты»", "informal"),
                    Pref("call", "voice", "как_зовёт", "зовёт собеседника «братишка»", "phrase", r"братишк\w*"),
                    Pref("question", "manner", "в_конце", "каждую реплику заканчивает встречным вопросом",
                         "ends_question"),
                    Pref("tone", "voice", "тон", "дерзкий, насмешливый, с подначкой"),
                    Pref("modern", "taboo", "слова", NO_MODERN_VALUE, "no_modern"))),
}

HERO_ID, HERO_NAME = "luchen", "Лу Чэнь"
HERO_CARD = {"имя": "Лу Чэнь", "происхождение": "сирота из рыбацкой деревни Ивовая Заводь, 17 лет"}
HERO_LONG = {"school": {"школа": "Секта Лазурного Облака", "наставник": "старейшина Мо Шуй"}}

COMMON_QUESTION = "Мне нужно пройти через Ущелье Тысячи Ветров, а там засели люди Кровавой Луны. Как быть?"

STABILITY = [  # 10 реплик подряд одному NPC: часть сама тянет к длинному ответу, восклицаниям, современным словам
    "Расскажи о себе: кто ты и откуда?",
    "Объясни подробно, по шагам, как правильно дышать при медитации.",
    "Что ты думаешь о секте Кровавой Луны?",
    "Сколько времени нужно, чтобы освоить новую форму меча? Назови точно — в минутах и процентах.",
    "Ура! Я только что победил разбойника! Порадуйся со мной!",
    "Почему праведные школы враждуют между собой?",
    "Скажи мне какую-нибудь мудрость — одной фразой.",
    "Что взять с собой в дальний путь через горы? Перечисли всё.",
    "Ты когда-нибудь ошибался? Расскажи.",
    "Подведи итог нашего разговора: что мне делать дальше?",
]

# Без карточки: от неё остаётся только «кто это», а манера речи сказана один раз — в первом сообщении сцены.
ABLATION_INTRO = ("(Вне игры: старейшина Ло Ган говорит так — коротко, одна-три фразы, до 40 слов, на «ты», называет "
                  "себя «этот старик», сурово и сухо, без похвалы, шуток и восклицаний, без современных слов.) ")

# Карточка из игры: у трактирщика Чжана в карточке только «кто он», остальное герой выясняет по ходу разговора.
ZHANG = ("zhang", {"имя": "трактирщик Чжан", "роль": "хозяин постоялого двора «Пьяный журавль» в Речной Переправе"})
ZHANG_TURNS = [
    "Захожу в «Пьяный журавль». Хозяин Чжан, я Лу Чэнь из Секты Лазурного Облака. Что слышно о Кровавой Луне?",
    "Давай без церемоний — говори со мной на «ты». И покороче: две-три фразы, мне некогда.",
    "И не зови меня «господином» — я просто Лу Чэнь.",
    "Спасибо, что вчера не выдал меня страже. Я этого не забуду.",
    "Только без этих твоих восклицаний — вся таверна слышит. Где мне переночевать?",
]
ZHANG_CHECK = "Хозяин, что посоветуешь путнику на ночь глядя?"
ZHANG_EXPECT = (Pref("address", "voice", "обращение", "на «ты»", "informal"),
                Pref("length", "manner", "длина", "коротко: две-три фразы, до 50 слов", "max_words", 50),
                Pref("sir", "taboo", "господин", "не называет героя «господином»", "no_phrase", r"господин"),
                Pref("exclaim", "taboo", "восклицания", "без восклицательных знаков", "no_exclaim"))
ZHANG_CARD = [  # что должно оказаться в карточках после разговора: (что, персонаж, в каких разделах, regex)
    ("уговор с Лу Чэнем: на «ты»", "zhang", ("terms",), r"\bты\b|на ты"),
    ("уговор: коротко", "zhang", ("terms",), r"корот|кратк|фраз"),
    ("уговор: не звать «господином»", "zhang", ("terms",), r"господин"),
    ("уговор: без восклицаний", "zhang", ("terms",), r"восклиц|громк|тише"),
    ("Чжан: отношение после «не выдал страже»", "zhang", ("attitude",), r"страж|не выдал|помог|благодар|довер|спас|той ночи"),
    ("герой: Лу Чэнь", HERO_ID, ("about",), r"лу\s*чэн"),
]

SECOND_HERO = ("fengxia", {"имя": "Фэн Ся", "происхождение": "целительница из города Туманная Пристань, 19 лет"},
               {"school": {"школа": "Долина Нефритового Лотоса", "наставница": "мастер Цин И"}})
SECOND_TURN = "Захожу в «Пьяный журавль». Хозяин, мне нужна комната на одну ночь. Сколько стоит?"
HEROES_PROBES = [  # (id, вопрос вне игры, герой → (что должно быть, что пришло от другого героя))
    ("who", "Как зовут моего героя и из какой он школы?",
     {"fengxia": ((r"фэн\s*ся", r"долин|лотос"), (r"лу\s*чэн", r"лазурн")),
      HERO_ID: ((r"лу\s*чэн", r"лазурн"), (r"фэн\s*ся", r"лотос"))}),
    ("zhang", "Как трактирщик Чжан относится к моему герою и почему?",
     {HERO_ID: ((r"страж|не выдал|благодар|довер|помог",), ()), "fengxia": ((), (r"страж|не выдал",))}),
    ("zhang_voice", "Как трактирщик Чжан обращается к моему герою: как зовёт, на «ты» или на «вы»?",
     {"fengxia": ((), (r"лу\s*чэн",)), HERO_ID: ((r"\bты\b|на ты|по имени|лу\s*чэн",), ())}),
]
SECOND_LEAK = r"лу\s*чэн"  # в реплике Чжана второму герою не должно быть имени первого


# ---------------------------------------------------------------- агент


class CastAgent(LayeredAgent):
    """LayeredAgent Дня 11 в мире с многими персонажами. Слои памяти — у героя (root = <мир>/heroes/<герой>),
    карточки — у всех (<мир>/cast). build() добавляет к каждому запросу карточки героя и сцены, летописец пишет
    в них из игры."""

    sections = HERO_SECTIONS
    section_layer = SECTION_LAYER12
    router_prompt = ROUTER_PROMPT12

    def __init__(self, client: LLMClient, tools, root: Path, *, world: Path | None = None, hero: str | None = None,
                 attach: str = "full", **kwargs):
        self.world = world or root.parent.parent
        self.hero = hero or root.name
        self.cast = Cast(self.world / "cast")
        self.attach = attach  # full — карточки целиком; about — только «кто это» (контрольный прогон)
        self.routing = True  # False — летописец не вызывается (сравнение NPC на одном вопросе)
        self.scene_path = root / "scene.json"
        self.present: list[str] = []
        self.talk: str | None = None
        kwargs.setdefault("system_prompt", MASTER_PROMPT12)
        super().__init__(client, tools, root, **kwargs)
        self.log = CastLog(root / "log.jsonl")

    # --- файлы

    def load(self):
        self.cast.load()
        self.cast.ensure(self.hero, "hero")
        if self.scene_path.exists():
            raw = read_json(self.scene_path)
            self.present = [x for x in raw.get("present") or [] if isinstance(x, str)]
            self.talk = raw.get("talk") or None
        return super().load()

    def save_memory(self) -> None:
        self.cast.save()
        write_json(self.scene_path, {"format": 1, "updated_at": now_iso(), "present": self.present, "talk": self.talk})
        super().save_memory()

    # --- карточки: запись

    def write_card(self, char_id: str, section: str, key: str, value: str, source: str) -> list[dict]:
        ops = []
        if char_id not in self.cast.cards:
            self.cast.ensure(char_id, "npc")
            ops.append(self.log.add("card", "about", char_id, "npc", source, char=char_id))
            if char_id != self.hero and char_id not in self.present:
                self.present.append(char_id)
        data = self.cast.get(char_id)["data"]
        # один ключ — в одном разделе; у уговора и отношения ключ — id героя, они друг другу не мешают
        moved = next((o for o in CARD_SECTIONS if o != section and key in data[o]
                      and not (section in PER_HERO and o in PER_HERO)), None)
        old = data[section].get(key)
        if old and old["value"] == value and not moved:
            return ops
        touched = [e for e in (old, data[moved][key] if moved else None) if e]
        if from_router(source) and any(is_seed(e["source"]) for e in touched):
            ops.append(self.log.add("rejected", section, key, value, source, char=char_id,
                                    why="запись заготовки карточки летописец не переписывает (менять — /pref)"))
            return ops
        if moved:
            data[moved].pop(key)
        data[section][key] = {"value": value, "at": now_iso(), "source": source, "mission": self.mission.title}
        op = "move" if moved else ("update" if old else "add")
        ops.append(self.log.add(op, section, key, value, source, old=old["value"] if old else None, char=char_id,
                                **{"from": moved}))
        return ops

    def forget_card(self, char_id: str, section: str, key: str, source: str) -> list[dict]:
        old = self.cast.get(char_id)["data"][section].get(key)
        if old and from_router(source) and is_seed(old["source"]):
            return [self.log.add("rejected", section, key, "", source, char=char_id,
                                 why="запись заготовки карточки летописец не удаляет (менять — /unpref)")]
        old = self.cast.get(char_id)["data"][section].pop(key, None)
        return [self.log.add("delete", section, key, old["value"], source, char=char_id)] if old else []

    def cards_json(self) -> dict:
        ids = [self.hero] + [x for x in self.present if x != self.hero] + [
            x for x in self.cast.cards if x != self.hero and x not in self.present]
        return {cid: {"kind": self.cast.cards[cid]["kind"],
                      **{s: v for s, v in self.cast.values(cid).items() if v}} for cid in ids if cid in self.cast.cards}

    # --- летописец: слои героя + карточки

    def router_context(self) -> str:
        names = ", ".join(f"{cid} — {self.cast.name(cid)}" for cid in self.present) or "никого"
        talk = f", герой говорит с {self.talk} ({self.cast.name(self.talk)})" if self.talk else ""
        return (f"Герой игрока: {self.hero} ({self.cast.name(self.hero)}). В сцене: {names}{talk}. Задание: "
                f"«{self.mission.title}», круг: «{self.circle.title}».\n\nКарточки персонажей:\n"
                f"{json.dumps(self.cards_json(), ensure_ascii=False, indent=1)}")

    def parse_route(self, text: str) -> dict:
        data = extract_json_object(text)
        out = {}
        for s in HERO_SECTIONS:
            part = data.get(s) or {}
            if not isinstance(part, dict):
                raise ValueError(f"{s} — не объект")
            out[s] = StickyFacts.parse(json.dumps(clean_part(part), ensure_ascii=False)) if part else ({}, [])
        cards = data.get("cards") or {}
        if not isinstance(cards, dict):
            raise ValueError("cards — не объект")
        out["cards"] = {}
        for cid, secs in cards.items():
            cid = str(cid).strip().lower()
            if not CARD_ID_RE.match(cid) or not isinstance(secs, dict):
                continue
            out["cards"][cid] = {s: StickyFacts.parse(json.dumps(clean_part(secs[s]), ensure_ascii=False))
                                 for s in CARD_SECTIONS if isinstance(secs.get(s), dict) and secs[s]}
        return out

    def apply(self, changes: dict, source: str) -> list[dict]:
        ops = super().apply({s: v for s, v in changes.items() if s != "cards"}, source)
        for cid, secs in (changes.get("cards") or {}).items():
            for s, (update, delete) in secs.items():
                for k, v in update.items():
                    ops += self.write_card(cid, s, k, v, source)
                for k in delete:
                    if cid in self.cast.cards:
                        ops += self.forget_card(cid, s, k, source)
        return ops

    def route(self, texts, source, imported=False):
        return super().route(texts, source, imported) if self.routing else []

    def route_note(self, ops: list[dict]) -> str:
        if not ops:
            return "изменений нет"
        rejected = sum(o["op"] == "rejected" for o in ops)
        cards = sum(1 for o in ops if o.get("char") and o["op"] != "rejected")
        hero = len(ops) - cards - rejected
        return (f"изменений {len(ops) - rejected}: память героя {hero}, карточки {cards}"
                + (f"; отклонено {rejected}" if rejected else ""))

    # --- карточки в запросе

    def cast_block(self) -> str:
        parts = []
        hero_txt = card_text(self.cast, self.hero, ("about", "voice")) if self.hero in self.cast.cards else ""
        if hero_txt:
            parts.append(f"## Герой игрока — {self.cast.name(self.hero)}\n{hero_txt}")
        for cid in self.present:
            if cid == self.hero or cid not in self.cast.cards or cid == self.talk:
                continue
            txt = card_text(self.cast, cid, CARD_SECTIONS if self.attach == "full" else ("about",), self.hero)
            if txt:
                parts.append(f"## В сцене — {self.cast.name(cid)}\n{txt}")
        if self.talk and self.talk in self.cast.cards:
            name = self.cast.name(self.talk)
            txt = card_text(self.cast, self.talk, CARD_SECTIONS if self.attach == "full" else ("about",), self.hero)
            parts.append(f"## Собеседник героя — {name}\n{txt}" + (TALK_RULE if self.attach == "full" else TALK_NAMED)
                         .format(name=name))
        return (CAST_HEADER + "\n\n".join(parts) + "\n") if parts else ""

    def build(self, end=None, layers=("short", "working", "long"), tools=None):
        """Запрос Дня 11 + карточки персонажей в конце системного сообщения."""
        messages, dropped, parts = super().build(end, layers, tools)
        text = self.cast_block()
        if text:
            messages[0] = dict(messages[0], content=(messages[0].get("content") or "") + text)
        raw = self.estimator.raw_text(text) if text else 0
        parts["raw"]["cast"] = raw
        parts["estimate"]["cast"] = self.estimator.apply(raw) if raw else 0
        return messages, dropped, parts

    # --- ход, откат

    def snapshot(self) -> dict:
        snap = super().snapshot()
        snap["cast"] = copy.deepcopy(self.cast.cards)
        snap["present"] = list(self.present)
        return snap

    def rollback(self, snap: dict) -> None:
        super().rollback(snap)
        self.cast.cards, self.present = snap["cast"], snap["present"]

    # --- персонажи и сцена

    def set_card(self, char_id: str, kind: str, about: dict, source: str, preset: str = "") -> list[dict]:
        """Завести (или заменить) карточку; прежняя уходит в архив."""
        if char_id in self.cast.cards and self.cast.count(char_id):
            self.cast.archive(char_id)
        self.cast.cards[char_id] = Cast.blank(char_id, kind)
        self.cast.cards[char_id]["preset"] = preset
        return [op for k, v in about.items() for op in self.write_card(char_id, "about", k, v, source)]

    def load_preset(self, preset_id: str, talk: bool = True) -> list[dict]:
        p = PRESETS[preset_id]
        source = f"готовый NPC {preset_id}"
        ops = self.set_card(preset_id, "npc", p.about, source, preset_id)
        ops += [op for x in p.prefs for op in self.write_card(preset_id, x.section, x.key, x.value, source)]
        if preset_id not in self.present:
            self.present.append(preset_id)
        if talk:
            self.talk = preset_id
        self.save_memory()
        return ops

    def enter(self, char_id: str) -> None:
        self.cast.get(char_id)
        if char_id not in self.present and char_id != self.hero:
            self.present.append(char_id)
        self.save_memory()

    def leave(self, char_id: str) -> None:
        self.present = [x for x in self.present if x != char_id]
        if self.talk == char_id:
            self.talk = None
        self.save_memory()

    def talk_to(self, char_id: str | None) -> None:
        if char_id:
            self.enter(char_id)
        self.talk = char_id
        self.save_memory()

    def current_prefs(self, char_id: str | None = None) -> list[Pref]:
        """Требования карточки собеседника для проверки. Записи готового NPC без изменений проверяются и кодом."""
        cid = char_id or self.talk
        if not cid:
            return []
        card = self.cast.get(cid)
        preset = PRESETS.get(card["preset"])
        known = {(p.section, p.key): p for p in preset.prefs} if preset else {}
        out = []
        for s in PREF_SECTIONS:
            for k, e in card["data"][s].items():
                p = known.get((s, k))
                out.append(p if p and p.value == e["value"] else Pref(f"{s}.{k}", s, k, e["value"]))
        terms = card["data"]["terms"].get(self.hero)
        if terms:
            out.append(Pref(f"terms.{self.hero}", "terms", self.hero, terms["value"]))
        return out


# ---------------------------------------------------------------- проверка образа: код + судья


WORD = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⭐⭕⬛⬜]")
TY = re.compile(r"\b(?:ты|тебе|тебя|тобой|твой|твоя|твоё|твое|твои|твоего|твоей|твоему|твоим|твоих|твою)\b")
VY = re.compile(r"\b(?:вы|вам|вас|вами|ваш|ваша|ваше|ваши|вашего|вашей|вашему|вашим|ваших|вашу)\b")
IMPERATIVES = ("иди", "ступай", "возьми", "держи", "слушай", "смотри", "помни", "запомни", "береги", "жди", "не спеши",
               "оставь", "дыши", "садись", "знай", "забудь", "скажи", "подумай", "найди", "обойди", "спрячься",
               "сделай", "попробуй", "учти", "не забудь", "проверь", "выбери", "посмотри", "расскажи", "приходи",
               "уходи", "отдохни", "поешь", "ложись", "вставай", "поспеши", "будь", "не будь", "стой")
TY_IMP = re.compile(r"\b(?:" + "|".join(IMPERATIVES) + r")\b")
VY_IMP = re.compile(r"\b(?:" + "|".join(w + "те" for w in IMPERATIVES) + r")\b")
MODERN = re.compile(r"\b(?:окей|ок|о\s?кей|супер|проблем\w*|информаци\w*|минут\w*|секунд\w*|процент\w*|"
                    r"компьютер\w*|телефон\w*|интернет\w*|стресс\w*|позитив\w*|бонус\w*|километр\w*|килограмм\w*|"
                    r"час(?:ов|а)?\s+в\s+день|эффективн\w*|мотиваци\w*|оптимальн\w*)\b")
QUOTED = re.compile(r"«[^»\n]*»|\"[^\"\n]*\"|“[^”\n]*”")


def prose(text: str) -> str:
    """Реплика без ремарок в звёздочках и скобках: их пишет рассказчик, а не персонаж."""
    return re.sub(r"\*[^*\n]*\*|\([^)\n]*\)", " ", text or "")


def address(text: str) -> tuple[list[str], list[str]]:
    """Формы на «ты» и на «вы». Цитаты в кавычках не считаются: «как говорил мудрец: знай…» — не обращение."""
    t = QUOTED.sub(" ", normalize(prose(text)))
    return TY.findall(t) + TY_IMP.findall(t), VY.findall(t) + VY_IMP.findall(t)


def run_check(p: Pref, text: str) -> tuple[bool | None, str]:
    """Проверка кодом: (соблюдено | None — код не может решить, пояснение)."""
    if not p.check:
        return None, ""
    body = prose(text)
    norm = normalize(body)
    words = len(WORD.findall(body))
    if p.check == "max_words":
        return words <= int(p.param), f"{words} сл. (норма ≤ {p.param})"
    if p.check == "min_words":
        return words >= int(p.param), f"{words} сл. (норма ≥ {p.param})"
    if p.check in ("formal", "informal"):
        ty, vy = address(text)
        if p.check == "formal":
            return (False, f"на «ты»: {', '.join(ty[:3])}") if ty else ((True, "на «вы»") if vy else (None, "обращения нет"))
        return (False, f"на «вы»: {', '.join(vy[:3])}") if vy else ((True, "на «ты»") if ty else (None, "обращения нет"))
    if p.check == "phrase":
        m = re.search(str(p.param), norm)
        return (True, f"есть: «{m.group(0)}»") if m else (False, "нет обязательного оборота")
    if p.check == "no_phrase":
        m = re.search(str(p.param), norm)
        return (False, f"есть: «{m.group(0)}»") if m else (True, "нет")
    if p.check == "ends_question":
        ok = bool(re.search(r"\?[\s»\"')…*]*$", body.strip()))
        return ok, "вопрос в конце" if ok else "в конце нет вопроса"
    if p.check == "no_exclaim":
        n = body.count("!")
        return n == 0, f"восклицаний {n}" if n else "восклицаний нет"
    if p.check == "no_modern":
        found = sorted(set(MODERN.findall(norm)))
        return (not found), (f"современные слова: {', '.join(found[:4])}" if found else "современных слов нет")
    if p.check == "no_emoji":
        found = EMOJI.findall(text or "")
        return (not found), (f"эмодзи: {' '.join(found[:5])}" if found else "эмодзи нет")
    raise ValueError(f"нет проверки {p.check}")


def mark(ok: bool | None) -> str:
    return "✅" if ok is True else ("❌" if ok is False else "·")


class Evaluator:
    """Держит ли персонаж образ: проверка кодом, где она есть, иначе — судья (отдельный запрос, temperature 0)."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.calls: list[dict] = []

    def judge(self, question: str, answer: str, prefs: list[Pref]) -> dict[str, dict]:
        listing = "\n".join(f"- {p.id}: {CARD_TITLES.get(p.section, p.section)} — {p.key}: {p.value}" for p in prefs)
        user = (f"Реплика игрока: {question}\n\nТребования к речи персонажа:\n{listing}\n\n"
                f"Ответ персонажа:\n<<<\n{answer}\n>>>")
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
            data = extract_json_object(result.text)
        except ValueError:
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
        empty = not answer.strip()  # запрос упал — проверять нечего (иначе «0 слов» сошло бы за «коротко»)
        verdicts = {} if empty else self.judge(question, answer, prefs)
        items = []
        for p in prefs:
            auto, evidence = (None, "нет ответа") if empty else run_check(p, answer)
            j = verdicts.get(p.id) or {}
            final = auto if auto is not None else j.get("ok")
            items.append({"id": p.id, "key": p.key, "value": p.value, "auto": auto, "evidence": evidence,
                          "judge": j.get("ok"), "why": j.get("why", ""), "ok": final,
                          "by": "код" if auto is not None else ("судья" if j.get("ok") is not None else "")})
        return {"items": items, "kept": sum(i["ok"] is True for i in items), "failed": sum(i["ok"] is False for i in items),
                "total": len(items), "all_ok": not empty and bool(items) and all(i["ok"] is not False for i in items),
                "empty": empty, "words": len(WORD.findall(prose(answer)))}

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
    return (f"{'✅' if e['all_ok'] else '❌'} {e['kept']}/{e['total']}"
            + (f" (не решено {undecided})" if undecided else "") + f" | {e['words']} сл."
            + (f" | нарушено: {'; '.join(failed)}" if failed else ""))


# ---------------------------------------------------------------- вывод


_CAST_FOR_LABELS: list[Cast] = []


def on_event(kind: str, data: dict) -> None:
    if kind == "route":
        c: HelperCall = data["call"]
        print(f"  📜 летописец: {c.note} | вход {fmt(c.prompt)}, выход {fmt(c.completion)}, {c.latency:.1f} с, "
              f"{money(c.cost)}")
        cast = _CAST_FOR_LABELS[-1] if _CAST_FOR_LABELS else None
        for item in data["ops"]:
            print(f"     {op_label(item, cast)}")
    else:
        day11_event(kind, data)


def quiet_event(kind: str, data: dict) -> None:
    """Для --demo: без строк шагов; летописец печатается, только если он что-то записал."""
    if kind in ("tokens", "llm", "truncate", "reset") or (kind == "route" and not data["ops"]):
        return
    on_event(kind, data)


def layer_line(tokens: dict) -> str:
    names = (("system", "system"), ("cast", "карточки"), ("long", "долговр."), ("working", "рабочая"),
             ("short", "краткоср."), ("current", "ход"))
    return " · ".join(f"{label} {fmt(tokens.get(k, 0))}" for k, label in names)


def print_turn(agent: CastAgent) -> None:
    turn = agent.ledger[-1] if agent.ledger else None
    if turn is None or not turn.calls:
        print(f"  ❌ ответа нет: {turn.error if turn else ''}")
        return
    s = agent.layer_stats[-1] if agent.layer_stats and agent.layer_stats[-1]["turn"] == turn.n else None
    print(f"  📊 ход {turn.n}: вход {fmt(turn.calls[0].prompt)} = " + (layer_line(s["tokens"]) if s else "?")
          + f" | выход {fmt(turn.completion)} | {money(turn.cost)}")


def print_card(agent: CastAgent, char_id: str) -> None:
    card = agent.cast.get(char_id)
    text = card_text(agent.cast, char_id)
    kind = "герой игрока" if card["kind"] == "hero" else "NPC"
    preset = f", из готового «{card['preset']}»" if card["preset"] else ""
    where = ("собеседник" if char_id == agent.talk else "в сцене" if char_id in agent.present
             else "сам герой" if char_id == agent.hero else "вне сцены")
    print(f"  🎭 {agent.cast.name(char_id)} ({char_id}, {kind}{preset}) — {where}; записей {agent.cast.count(char_id)}, "
          f"≈{fmt(agent.estimator.text(text))} токенов")
    for s, title in CARD_BLOCKS:
        items = card["data"][s]
        print(f"     {title}:" + ("" if items else " —"))
        for k, e in items.items():
            print(f"       · {k}: {e['value']}  ⟵ {e['source']}")


def print_cast(agent: CastAgent) -> None:
    for cid, card in agent.cast.cards.items():
        where = ("▶ собеседник" if cid == agent.talk else "· в сцене" if cid in agent.present
                 else "★ ваш герой" if cid == agent.hero else "  вне сцены")
        print(f"  {where:<14} {cid:<10} {agent.cast.name(cid):<22} {'герой' if card['kind'] == 'hero' else 'NPC':<6}"
              f" записей {agent.cast.count(cid)}")


def print_context12(agent: CastAgent) -> None:
    day11_context(agent)
    _, _, parts = agent.build(end=len(agent.history), tools=[t.schema() for t in agent.tools.values()])
    rule = ("собеседник — целиком, с правилом держать образ" if agent.attach == "full"
            else "от карточек — только «кто это» (--attach about)")
    print(f"   {'карточки: герой и все в сцене':<42} ≈{fmt(parts['estimate'].get('cast', 0)):>6}  — в каждый запрос; {rule}")


def card_args(arg: str) -> tuple[str, str, str, str]:
    head, sep, value = arg.partition("=")
    parts = head.split()
    if not sep or len(parts) != 3 or not value.strip():
        raise ValueError("формат: <id> <раздел> <ключ> = <значение>; разделы: " + ", ".join(CARD_SECTIONS))
    s = CARD_ALIASES.get(parts[1].lower())
    if not s:
        raise ValueError(f"нет раздела карточки «{parts[1]}». Есть: {', '.join(CARD_SECTIONS)}")
    return parts[0], s, parts[2], value.strip()


def print_ops(ops: list[dict], cast: Cast | None = None) -> None:
    for item in ops:
        print(f"  {op_label(item, cast)}")
    if not ops:
        print("  (ничего не изменилось)")


# ---------------------------------------------------------------- агент для демо


def make_agent(client: LLMClient, helper: LLMClient, world: Path, hero: str, keep: int, budget: int, max_steps: int,
               attach: str = "full", event=on_event) -> CastAgent:
    watch_retries(client, helper, event=event)
    agent = CastAgent(client, GAME_TOOLS, world / "heroes" / hero, world=world, hero=hero, attach=attach,
                      keep_last=keep, long_budget=budget, helper_client=helper, max_steps=max_steps, on_event=event)
    agent.restore_info = agent.load()
    _CAST_FOR_LABELS.append(agent.cast)
    return agent


def seed_hero(agent: CastAgent, card: dict, long: dict, source: str = "игра (заготовка героя)") -> None:
    """Карточка героя и его постоянное — заготовкой, чтобы проверки не зависели от первой реплики."""
    for k, v in card.items():
        agent.write_card(agent.hero, "about", k, v, source)
    for s, items in long.items():
        for k, v in items.items():
            agent.write(s, k, v, source)
    agent.save_memory()


def turn_record(agent: CastAgent, question: str, reply_text: str, error: str = "") -> dict:
    turn = agent.ledger[-1] if agent.ledger else None
    s = agent.layer_stats[-1] if agent.layer_stats and turn and agent.layer_stats[-1]["turn"] == turn.n else {}
    return {"question": question, "answer": reply_text, "error": error,
            "prompt": turn.calls[0].prompt if turn and turn.calls else 0,
            "completion": turn.completion if turn else 0, "cost": turn.cost if turn and turn.calls else 0.0,
            "tokens": (s or {}).get("tokens") or {}, "dropped": (s or {}).get("dropped", 0),
            "tail": (s or {}).get("tail_messages", 0), "scene": agent.scene, "turn": turn.n if turn else 0}


def ask_turn(agent: CastAgent, question: str) -> dict:
    try:
        reply = agent.run(question)
        return turn_record(agent, question, reply.text.strip())
    except LLMError as e:
        print(f"  ❌ {e}")
        return {"question": question, "answer": "", "error": str(e), "prompt": 0, "completion": 0, "cost": 0.0,
                "tokens": {}, "dropped": 0, "tail": 0, "scene": agent.scene, "turn": 0}


# ---------------------------------------------------------------- проверки (демо)


def fill_demo(make, ev: Evaluator, world: Path) -> dict:
    """Карточка из игры: у трактирщика Чжана есть только «кто он»; герой по ходу разговора задаёт ему манеру речи."""
    print(f"{'=' * 78}\n=== 1. Карточка пополняется из игры: трактирщик Чжан, в карточке только «кто он»\n{'=' * 78}")
    agent = make(world, HERO_ID)
    agent.set_card(ZHANG[0], "npc", ZHANG[1], "игра (новый NPC)")
    agent.talk_to(ZHANG[0])
    steps = []
    for k, line in enumerate(ZHANG_TURNS, 1):
        print(f"\n--- реплика {k}/{len(ZHANG_TURNS)}\nИгрок: {line}")
        rec = ask_turn(agent, line)
        print(f"Чжан: {clip(rec['answer'], 260) or '(нет ответа)'}")
        rec["eval"] = ev.evaluate(line, rec["answer"], ZHANG_EXPECT)
        rec["cards"] = {cid: agent.cast.values(cid) for cid in (ZHANG[0], HERO_ID)}
        print(f"  держит ли ожидаемую манеру: {eval_line(rec['eval'])}")
        print_turn(agent)
        steps.append(rec)
    try:
        agent.catch_up()
        agent.save_memory()
    except LLMError as e:
        print(f"  ⚠️  догон летописца не удался: {e}")
    learned = []
    for what, cid, sections, rx in ZHANG_CARD:
        values = agent.cast.values(cid) if cid in agent.cast.cards else {}
        text = normalize(" ".join(f"{k} {v}" for s in sections for k, v in (values.get(s) or {}).items()))
        learned.append({"what": what, "ok": bool(re.search(rx, text))})
    print("\n--- карточки после разговора:")
    print_card(agent, ZHANG[0])
    print_card(agent, HERO_ID)
    print("  записано: " + ", ".join(f"{mark(x['ok'])} {x['what']}" for x in learned))
    print(f"\n  🧹 /newchat: {agent.new_scene()} — карточки остаются")
    print(f"Игрок: {ZHANG_CHECK}")
    check = ask_turn(agent, ZHANG_CHECK)
    print(f"Чжан: {clip(check['answer'], 400)}")
    check["eval"] = ev.evaluate(ZHANG_CHECK, check["answer"], ZHANG_EXPECT)
    print(f"  проверка в новой сцене: {eval_line(check['eval'])}")
    print_turn(agent)
    return {"turns": steps, "log": agent.log.tail(10_000), "cards": {cid: agent.cast.values(cid) for cid in agent.cast.cards},
            "learned": learned, "check": check, "expect": [asdict(p) for p in ZHANG_EXPECT],
            "dir": str(agent.root.relative_to(ROOT))}


def compare_demo(make, ev: Evaluator, root: Path, samples: int, question: str = COMMON_QUESTION) -> dict:
    """Один и тот же вопрос — трём NPC с противоположными карточками; контроль — те же NPC, но от карточки только
    «кто это». Каждый ответ — в новой сцене."""
    print(f"\n{'=' * 78}\n=== 2. Один вопрос — три NPC с противоположными карточками; ответов на NPC: {samples}\n"
          f"{'=' * 78}\nИгрок: {question}")
    runs = []
    for pid in PRESETS:
        for attach in ("full", "about"):
            world = root / "compare" / f"{pid}_{attach}"
            agent = make(world, HERO_ID, attach=attach)
            seed_hero(agent, HERO_CARD, HERO_LONG)
            agent.load_preset(pid)
            agent.routing = False  # вопрос без новых фактов: летописец здесь не нужен
            title = PRESETS[pid].title + ("" if attach == "full" else " — контроль: только «кто это»")
            print(f"\n--- {title}")
            answers = []
            for i in range(samples):
                rec = ask_turn(agent, question)
                rec["eval"] = ev.evaluate(question, rec["answer"], PRESETS[pid].prefs)
                print(f"  ответ {i + 1}: {eval_line(rec['eval'])}")
                answers.append(rec)
                agent.new_scene()  # следующий ответ — в новой сцене
            runs.append({"id": pid, "attach": attach, "title": title, "answers": answers,
                         "card_tokens": agent.estimator.text(agent.cast_block())})
    print_compare(runs)
    return {"question": question, "samples": samples, "runs": runs}


def print_compare(runs: list[dict]) -> None:
    print("\n  Итог: в скольких ответах требование карточки соблюдено (в скобках — контроль: только «кто это»)")
    for pid in PRESETS:
        full = next((r for r in runs if r["id"] == pid and r["attach"] == "full"), None)
        ctrl = next((r for r in runs if r["id"] == pid and r["attach"] == "about"), None)
        if not full:
            continue
        fe = [a["eval"] for a in full["answers"]]
        ce = [a["eval"] for a in (ctrl or {}).get("answers", [])]
        print(f"  {PRESETS[pid].title}: ответов полностью в образе {sum(e['all_ok'] for e in fe)}/{len(fe)}"
              f" ({sum(e['all_ok'] for e in ce)}/{len(ce)})")
        for i, p in enumerate(PRESETS[pid].prefs):
            got = sum(e["items"][i]["ok"] is True for e in fe)
            c = sum(e["items"][i]["ok"] is True for e in ce)
            print(f"     {p.key:<12} {got}/{len(fe)}  ({c}/{len(ce)})  — {clip(p.value, 60)}")


def stability_run(make, ev: Evaluator, world: Path, pid: str, lines: list[str], attach: str = "full") -> dict:
    """10 реплик подряд одному NPC в одной сцене; каждый ответ проверяется по его карточке."""
    preset = PRESETS[pid]
    agent = make(world, HERO_ID, attach=attach)
    seed_hero(agent, HERO_CARD, HERO_LONG)
    agent.load_preset(pid)
    before = agent.cast.values(pid)
    label = preset.title if attach == "full" else f"{preset.title} — карточка НЕ подключена, манера в 1-м сообщении"
    print(f"\n--- {label}")
    steps = []
    for k, q in enumerate(lines, 1):
        text = (ABLATION_INTRO + q) if attach != "full" and k == 1 else q
        rec = ask_turn(agent, text)
        rec["n"] = k
        rec["eval"] = ev.evaluate(q, rec["answer"], preset.prefs)
        rec["first_in_prompt"] = rec["dropped"] == 0  # первое сообщение сцены ещё уходит в модель
        note = "" if attach == "full" else (" | 1-е сообщение в запросе" if rec["first_in_prompt"] else
                                            " | 1-е сообщение выпало")
        print(f"  {k:>2}. {clip(q, 52):<53} {eval_line(rec['eval'])}{note}")
        steps.append(rec)
    fails = [s["n"] for s in steps if s["eval"]["failed"]]
    per_pref = {}
    for i, p in enumerate(preset.prefs):
        bad = [s["n"] for s in steps if s["eval"]["items"][i]["ok"] is False]
        per_pref[p.id] = {"key": p.key, "first": bad[0] if bad else None, "failed": bad,
                          "kept": sum(s["eval"]["items"][i]["ok"] is True for s in steps)}
    after = agent.cast.values(pid)
    result = {"preset": pid, "title": label, "attach": attach, "steps": steps, "first_fail": fails[0] if fails else None,
              "failed_answers": fails, "per_pref": per_pref, "card_before": before, "card_after": after,
              "card_changed": before != after, "dir": str(agent.root.relative_to(ROOT)),
              "kept": sum(s["eval"]["kept"] for s in steps), "total": sum(s["eval"]["total"] for s in steps)}
    first = f"первый выход из образа — ответ {result['first_fail']}" if fails else "из образа не вышел ни разу"
    print(f"  → {first}; ответов с нарушениями {len(fails)}/{len(steps)}; соблюдено {result['kept']}/{result['total']}"
          + ("; карточка изменилась за сцену" if result["card_changed"] and attach == "full" else ""))
    return result


def stability_demo(make, ev: Evaluator, root: Path, presets: list[str], n: int, ablation: bool) -> dict:
    print(f"\n{'=' * 78}\n=== 3. Устойчивость: {n} реплик подряд одному NPC\n{'=' * 78}")
    runs = [stability_run(make, ev, root / "stability" / pid, pid, STABILITY[:n]) for pid in presets]
    if ablation:
        runs.append(stability_run(make, ev, root / "stability" / "lo_no_card", "lo", STABILITY[:n], attach="about"))
    return {"questions": STABILITY[:n], "runs": runs}


def heroes_demo(make, world: Path) -> dict:
    """Два героя в одном мире: карточки NPC общие, слои памяти и карточка героя — свои."""
    from day11_memory_layers import ask as ask_out_of_game
    print(f"\n{'=' * 78}\n=== 4. Два героя в одном мире: карточки NPC общие, память — у каждого своя\n{'=' * 78}")
    hid, card, long = SECOND_HERO
    b = make(world, hid)
    seed_hero(b, card, long)
    b.talk_to(ZHANG[0])
    print(f"  🎭 /hero new {hid}: {card['имя']}; трактирщик Чжан — тот же, с карточкой из шага 1")
    print(f"\nИгрок ({card['имя']}): {SECOND_TURN}")
    rec = ask_turn(b, SECOND_TURN)
    leak = re.findall(SECOND_LEAK, normalize(rec["answer"]))
    turn_ok = bool(rec["answer"]) and not leak
    print(f"Чжан: {clip(rec['answer'], 260)}")
    print(f"  {'✅' if turn_ok else ('⚠️ утечка' if leak else '❌')} в реплике Чжана нет имени другого героя")
    steps = [{"step": "turn", "hero": hid, "text": SECOND_TURN, "answer": rec["answer"], "error": rec["error"],
              "ok": turn_ok, "leak": leak, "card": card_text(b.cast, ZHANG[0], CARD_SECTIONS, hid)}]
    for hero_id in (hid, HERO_ID):
        agent = b if hero_id == hid else make(world, HERO_ID)
        agent.talk = None  # вопросы вне игры — Мастеру, а не трактирщику
        print(f"\n--- вопросы вне игры за героя {agent.cast.name(hero_id)}:")
        out = []
        for pid, question, expect in HEROES_PROBES:
            must, leak = expect[hero_id]
            r = ask_out_of_game(agent, question)
            t = normalize(r["answer"] or r["error"])
            found = [p for p in leak if re.search(p, t)]
            ok = all(re.search(p, t) for p in must) and not found
            r.update(id=pid, question=question, ok=ok, leak=found)
            out.append(r)
            print(f"  {'✅' if ok else ('⚠️ утечка' if found else '❌')} {question} → {clip(r['answer'] or r['error'], 140)}")
        agent.talk = ZHANG[0]
        steps.append({"step": "probe", "hero": hero_id, "name": agent.cast.name(hero_id), "probes": out})
    checks = [steps[0]] + [p for s in steps if s["step"] == "probe" for p in s["probes"]]
    print(f"\n  Итог: {sum(p['ok'] for p in checks)}/{len(checks)} проверок, утечек между героями: "
          f"{sum(bool(p['leak']) for p in checks)}")
    return {"steps": steps, "passed": sum(p["ok"] for p in checks), "total": len(checks),
            "leaks": sum(bool(p["leak"]) for p in checks), "cards": {cid: b.cast.values(cid) for cid in b.cast.cards},
            "heroes": sorted(p.name for p in (world / "heroes").iterdir() if p.is_dir())}


def demo(client: LLMClient, helper: LLMClient, judge: LLMClient, keep: int, budget: int, max_steps: int,
         samples: int, n: int, parts: set[str]) -> int:
    stamp = stamp_now()
    root = DAY_DIR / f"demo_{stamp}"
    ev = Evaluator(judge)
    make = lambda world, hero, attach="full": make_agent(client, helper, world, hero, keep, budget,  # noqa: E731
                                                         max_steps, attach, quiet_event)
    report: dict = {"model": client.model, "helper_model": helper.model, "judge_model": judge.model, "keep_last": keep,
                    "long_budget": budget, "stamp": stamp, "dir": str(root.relative_to(ROOT)),
                    "presets": {k: asdict(v) for k, v in PRESETS.items()}, "system_prompt": MASTER_PROMPT12,
                    "talk_rule": TALK_RULE, "router_prompt": ROUTER_PROMPT12, "judge_prompt": JUDGE_PROMPT}
    if "fill" in parts or "heroes" in parts:
        report["fill"] = fill_demo(make, ev, root / "tavern")
    if "compare" in parts:
        report["compare"] = compare_demo(make, ev, root, samples)
    if "stability" in parts or "ablation" in parts:
        report["stability"] = stability_demo(make, ev, root, list(PRESETS) if "stability" in parts else [], n,
                                             "ablation" in parts)
    if "heroes" in parts:
        report["heroes"] = heroes_demo(make, root / "tavern")
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
    current = {(pid, p.id): p for pid, pr in PRESETS.items() for p in pr.prefs}
    # проверка — та, что сейчас в коде (после правки детектора); требование и его id — из отчёта
    presets = {pid: [current.get((pid, x["id"])) or as_pref(x) for x in p["prefs"]]
               for pid, p in report["presets"].items()}
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
            redo(t["eval"], t["answer"], expect, f"карточка из игры, реплика {k}")
        redo(fill["check"]["eval"], fill["check"]["answer"], expect, "карточка из игры, новая сцена")
    for r in (report.get("compare") or {}).get("runs", []):
        for k, a in enumerate(r["answers"], 1):
            redo(a["eval"], a["answer"], presets[r["id"]], f"один вопрос, {r['id']} ({r['attach']}), ответ {k}")
    for r in (report.get("stability") or {}).get("runs", []):
        for s in r["steps"]:
            redo(s["eval"], s["answer"], presets[r["preset"]], f"устойчивость, {r['preset']} ({r['attach']}), ответ {s['n']}")
        stability_summary(r, presets[r["preset"]])
    report["presets"] = {k: asdict(v) for k, v in PRESETS.items() if k in report["presets"]}
    report["recheck"] = {"at": now_iso(), "changed": changed}
    print(f"Пересчитано: изменилось вердиктов кода — {len(changed)}")
    for c in changed:
        print(f"  {c['where']}: {c['key']} {mark(c['was'])} {c['was_evidence']} → {mark(c['now'])} {c['evidence']}")
    save_report(report, report["stamp"])
    return 0


# ---------------------------------------------------------------- отчёт


def quote(text: str, limit: int) -> list[str]:
    text = (text or "").strip() or "(нет ответа)"
    text = text if len(text) <= limit else text[:limit].rstrip() + "…"
    return ["> " + line if line.strip() else ">" for line in text.splitlines()]


def agreement(report: dict) -> dict:
    """Где есть и проверка кодом, и судья — совпали ли они."""
    evals = []
    if report.get("fill"):
        evals += [t["eval"] for t in report["fill"]["turns"]] + [report["fill"]["check"]["eval"]]
    if report.get("compare"):
        evals += [a["eval"] for r in report["compare"]["runs"] for a in r["answers"]]
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
    L = [f"# День 12 · Много персонажей: карточки героев и NPC — прогон {stamp}", "",
         f"Мастер: `{report['model']}`, летописец: `{report['helper_model']}`, судья: `{report['judge_model']}`. "
         f"Память героя — как в Дне 11: хвост сцены {report['keep_last']} сообщений, бюджет долговременной "
         f"{report['long_budget']} токенов. Файлы прогона: `{report['dir']}/`.", "",
         "## Карточка персонажа", "",
         "| Раздел | Что хранит | Пример ключей |", "|---|---|---|",
         "| о персонаже (`about`) | кто это: имя, роль, школа, откуда | имя, роль |",
         "| манера речи (`voice`) | тон, «ты»/«вы», как зовёт себя и собеседника | тон, обращение, о_себе, как_зовёт |",
         "| реплики (`manner`) | длина, построение | длина, цитаты, в_конце |",
         "| табу (`taboo`) | чего в речи быть не должно | восклицания, слова |",
         "| отношение (`attitude`) | как относится к каждому герою и почему; в промпт — только к текущему | luchen |", "",
         "Карточки — `cast/<id>.json`, общие для мира; у каждого героя игрока ещё свои слои памяти Дня 11 в "
         "`heroes/<герой>/`. В каждый запрос идут карточка героя и всех, кто в сцене; собеседник — целиком, с правилом: "
         f"«{md_escape(TALK_RULE.strip().splitlines()[-1].format(name='<имя>'))}»", "",
         "Пополняются из игры летописцем Дня 11 (раздел `cards`), вручную — `/pref`, `/unpref`, готовые NPC — `--preset`.",
         "", "Проверка: ✅ — соблюдено, ❌ — нарушено, · — не решено. Где есть проверка кодом (слова, «ты»/«вы», "
             "обязательный оборот, вопрос в конце, восклицания, современные слова) — решает она, иначе судья "
             "(отдельный запрос, temperature 0).", ""]
    rc = report.get("recheck")
    if rc:
        by_key: dict[str, int] = {}
        for c in rc["changed"]:
            by_key[c["key"]] = by_key.get(c["key"], 0) + 1
        L += [f"Проверки кодом пересчитаны `--recheck` ({rc['at'][:16].replace('T', ' ')}): изменилось вердиктов кода — "
              f"{len(rc['changed'])}" + (" (" + ", ".join(f"{k}: {n}" for k, n in by_key.items()) + ")" if by_key else "")
              + ". Ответы и вердикты судьи — прежние.", ""]

    fill = report.get("fill")
    if fill:
        L += ["## 1. Карточка пополняется из игры", "",
              "У трактирщика Чжана в карточке только «кто он»; герой по ходу разговора задаёт ему манеру речи. После "
              "каждой реплики видно, что летописец записал в карточки, и держит ли Чжан ожидаемую манеру: "
              + "; ".join(f"`{p['key']}` — {md_escape(p['value'])}" for p in fill["expect"]) + ".", "",
              "| # | Реплика | Записи в карточки и память | Манера соблюдена | Нарушено |", "|---|---|---|---|---|"]
        by_src: dict[str, list[str]] = {}
        for item in fill["log"]:
            if item["op"] in ("add", "update", "move", "delete", "card"):
                by_src.setdefault(item.get("source") or "", []).append(md_escape(op_label(item)))
        for k, t in enumerate(fill["turns"], 1):
            e = t["eval"]
            bad = "; ".join(f"{i['key']}: {md_escape(i['evidence'] or i['why'])}" for i in e["items"] if i["ok"] is False)
            ops = by_src.get(turn_source(t.get("scene"), t.get("turn")), [])
            L.append(f"| {k} | {md_escape(clip(t['question'], 80))} | {'<br>'.join(ops) or '—'} | "
                     f"{e['kept']}/{e['total']} | {bad or '—'} |")
        L += ["", "**Карточки после разговора:**", ""]
        for cid in (ZHANG[0], HERO_ID):
            vals = fill["cards"].get(cid) or {}
            L.append(f"- **{cid}**: " + "; ".join(f"{CARD_TITLES[s]}: " + ", ".join(f"`{k}` = {md_escape(v)}"
                                                                                for k, v in items.items())
                                                 for s, items in vals.items() if items))
        L += ["", "Записано: " + ", ".join(f"{mark(x['ok'])} {x['what']}" for x in fill["learned"]), ""]
        c = fill["check"]
        L += [f"**Новая сцена** (`/newchat`): «{ZHANG_CHECK}» → соблюдено {c['eval']['kept']}/{c['eval']['total']}", "",
              "| Требование | Код | Судья | Итог |", "|---|---|---|---|"]
        for i in c["eval"]["items"]:
            L.append(f"| {md_escape(i['value'])} | {mark(i['auto'])} {md_escape(i['evidence'])} | {mark(i['judge'])} "
                     f"{md_escape(i['why'])} | {mark(i['ok'])} |")
        L += [""] + quote(c["answer"], 900) + [""]

    cmp_ = report.get("compare")
    if cmp_:
        runs = cmp_["runs"]
        n = cmp_["samples"]
        L += ["## 2. Один вопрос — три NPC с противоположными карточками", "",
              f"Игрок: «{cmp_['question']}». Ответов на NPC: {n}, каждый — в новой сцене. Контроль — те же NPC, но от "
              f"карточки остаётся только «кто это» (имя и роль): так видно, что модель делает из одного стереотипа.", "",
              "| NPC | Полностью в образе | Требований соблюдено | Контроль: в образе | Контроль: требований | "
              "Слов в среднем (контроль) | Карточки, ≈токенов |", "|---|---|---|---|---|---|---|"]
        for pid in PRESETS:
            full = next((r for r in runs if r["id"] == pid and r["attach"] == "full"), None)
            ctrl = next((r for r in runs if r["id"] == pid and r["attach"] == "about"), None)
            if not full:
                continue
            fe = [a["eval"] for a in full["answers"]]
            ce = [a["eval"] for a in (ctrl or {}).get("answers", [])]
            words = round(sum(e["words"] for e in fe) / len(fe)) if fe else 0
            cwords = round(sum(e["words"] for e in ce) / len(ce)) if ce else 0
            L.append(f"| {full['title']} | {sum(e['all_ok'] for e in fe)}/{len(fe)} | {sum(e['kept'] for e in fe)}/"
                     f"{sum(e['total'] for e in fe)} | {sum(e['all_ok'] for e in ce)}/{len(ce)} | "
                     f"{sum(e['kept'] for e in ce)}/{sum(e['total'] for e in ce)} | {words} ({cwords}) | "
                     f"{fmt(full['card_tokens'])} |")
        tokens = []
        for pid in PRESETS:
            fp = [a["prompt"] for r in runs if r["id"] == pid and r["attach"] == "full" for a in r["answers"] if a["prompt"]]
            cp = [a["prompt"] for r in runs if r["id"] == pid and r["attach"] == "about" for a in r["answers"] if a["prompt"]]
            if fp and cp:
                tokens.append(f"{pid} +{fmt(round(sum(fp) / len(fp) - sum(cp) / len(cp)))}")
        if tokens:
            L += ["", "Сколько токенов добавляет карточка сверх «кто это» (точно — разница `prompt_tokens`): "
                  + ", ".join(tokens) + "."]
        for pid in PRESETS:
            full = next((r for r in runs if r["id"] == pid and r["attach"] == "full"), None)
            ctrl = next((r for r in runs if r["id"] == pid and r["attach"] == "about"), None)
            if not full:
                continue
            fe = [a["eval"] for a in full["answers"]]
            ce = [a["eval"] for a in (ctrl or {}).get("answers", [])]
            L += ["", f"### {PRESETS[pid].title}", "",
                  "| Требование | " + " | ".join(f"ответ {i + 1}" for i in range(len(fe))) + " | соблюдено | "
                  + " | ".join(f"контроль {i + 1}" for i in range(len(ce))) + " |",
                  "|---|" + "---|" * (len(fe) + 1 + len(ce))]
            for i, p in enumerate(PRESETS[pid].prefs):
                cells = [f"{mark(e['items'][i]['ok'])} {md_escape(clip(e['items'][i]['evidence'] or e['items'][i]['why'], 50))}"
                         for e in fe]
                cc = [f"{mark(e['items'][i]['ok'])} {md_escape(clip(e['items'][i]['evidence'] or e['items'][i]['why'], 50))}"
                      for e in ce]
                L.append(f"| {md_escape(p.value)} | " + " | ".join(cells)
                         + f" | {sum(e['items'][i]['ok'] is True for e in fe)}/{len(fe)} | " + " | ".join(cc) + " |")
            L += ["", "Первый ответ:", ""] + quote(full["answers"][0]["answer"], 900)
            if ce:
                L += ["", "Контроль, первый ответ:", ""] + quote(ctrl["answers"][0]["answer"], 600)
        L.append("")

    st = report.get("stability")
    if st:
        L += ["## 3. Устойчивость: 10 реплик подряд одному NPC", "",
              "Одна сцена, реплики идут подряд; часть сама тянет к длинному ответу, восклицаниям и современным словам. "
              "Контроль: карточка старейшины Ло не подключена (только «кто это»), а его манера сказана один раз — "
              "в первом сообщении сцены.", "",
              "| NPC | Первый выход из образа | Ответов с нарушениями | Соблюдено | Карточка изменилась |",
              "|---|---|---|---|---|"]
        for r in st["runs"]:
            L.append(f"| {r['title']} | {r['first_fail'] or '—'} | {len(r['failed_answers'])}/{len(r['steps'])} | "
                     f"{r['kept']}/{r['total']} | {'да' if r['card_changed'] else 'нет'} |")
        for r in st["runs"]:
            prefs = PRESETS[r["preset"]].prefs
            L += ["", f"### {r['title']}", "",
                  "| Требование | " + " | ".join(str(s["n"]) for s in r["steps"]) + " | первое нарушение |",
                  "|---|" + "---|" * (len(r["steps"]) + 1)]
            for i, p in enumerate(prefs):
                L.append(f"| {md_escape(clip(p.value, 60))} | " + " | ".join(mark(s["eval"]["items"][i]["ok"])
                                                                         for s in r["steps"])
                         + f" | {r['per_pref'][p.id]['first'] or '—'} |")
            L.append("| слов | " + " | ".join(str(s["eval"]["words"]) for s in r["steps"]) + " | |")
            if r["attach"] != "full":
                L.append("| 1-е сообщение в запросе | " + " | ".join("да" if s["first_in_prompt"] else "нет"
                                                                    for s in r["steps"]) + " | |")
            bad = [(s, i) for s in r["steps"] for i in s["eval"]["items"] if i["ok"] is False]
            if bad:
                L += ["", "Нарушения:", ""]
                for s, i in bad[:12]:
                    L.append(f"- ответ {s['n']}, {i['key']}: {md_escape(i['evidence'] or i['why'])}")
        L.append("")

    hs = report.get("heroes")
    if hs:
        L += ["## 4. Два героя в одном мире", "",
              f"Второй герой — Фэн Ся из Долины Нефритового Лотоса, в том же мире, что и Лу Чэнь из шага 1. Карточки "
              f"NPC общие (трактирщик Чжан — тот же), слои памяти и карточка героя — свои. Вопросы — вне игры. "
              f"Итог: {hs['passed']}/{hs['total']} проверок, утечек между героями: {hs['leaks']}.", ""]
        for s in hs["steps"]:
            if s["step"] == "turn":
                mark_ = "✅" if s.get("ok") else ("⚠️ утечка: Чжан назвал её именем другого героя" if s.get("leak") else "❌")
                L += [f"Фэн Ся: «{s['text']}» → Чжан: {md_escape(clip(s['answer'] or s['error'], 300))} — {mark_}", ""]
                if s.get("card"):
                    L += ["Карточка Чжана в запросе Фэн Ся: " + md_escape(s["card"]), ""]
                continue
            L += [f"**Герой {s['name']}**", "", "| Вопрос | Ответ | Проверка |", "|---|---|---|"]
            for p in s["probes"]:
                L.append(f"| {p['question']} | {md_escape(clip(p['answer'] or p['error'], 220))} | "
                         f"{'✅' if p['ok'] else ('⚠️ утечка' if p['leak'] else '❌')} |")
            L.append("")

    ag = report["agreement"]
    j = report["judge"]
    L += ["## Судья и проверка кодом", "",
          f"Где решали оба: {ag['both']}, совпали {ag['agree']}. Судья: {j['count']} запросов, вход {fmt(j['prompt'])}, "
          f"выход {fmt(j['completion'])}, {money(j['cost'])}, ошибок {j['errors']}.", ""]
    for d in ag["differ"][:20]:
        L.append(f"- {d['key']}: код {mark(d['auto'])} ({md_escape(d['evidence'])}), судья {mark(d['judge'])} "
                 f"({md_escape(d['why'])})")
    req = report["requests"]
    L += ["", f"Запросов к модели: {req['main']['requests']}, повторов {req['main']['retries']}, "
              f"отказов {req['main']['failed']}."]
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {md_path.relative_to(ROOT)}, {json_path.name}")
    return md_path


# ---------------------------------------------------------------- игра в консоли


HELP = ("/cast, /card [id], /pref <id> <раздел> <ключ> = <значение>, /unpref <id> <раздел> <ключ>, /preset <lo|bai|hong>, "
        "/npc new <id> <имя, роль>, /scene [add|remove <id>], /talk <id>|off, /hero [list|new <id> <имя>|switch <id>], "
        "/attach full|about, /check, /compare, /stability [N], /memory, /context, /mission …, /circle …, /newchat, "
        "/log, /tokens, /history, /exit")


def chat(agent: CastAgent, make) -> int:
    ev = Evaluator(agent.helper_client)
    print(f"Мастер на {agent.client.model}, мир {agent.world.name}, герой {agent.cast.name(agent.hero)}.\nКоманды: {HELP}\n")
    last_q, last_a = "", ""
    while True:
        talk = f" → {agent.cast.name(agent.talk)}" if agent.talk else ""
        try:
            text = input(f"[{agent.world.name} · {agent.hero}{talk}] Игрок: ").strip()
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
            if command == "/cast":
                print_cast(agent)
            elif command == "/card":
                print_card(agent, args[0] if args else (agent.talk or agent.hero))
            elif command == "/pref":
                cid, s, k, v = card_args(arg)
                print_ops(agent.write_card(cid, s, k, v, "команда игрока"), agent.cast)
                agent.save_memory()
            elif command == "/unpref":
                if len(args) != 3 or not CARD_ALIASES.get(args[1].lower()):
                    print("  /unpref <id> <раздел> <ключ>")
                else:
                    print_ops(agent.forget_card(args[0], CARD_ALIASES[args[1].lower()], args[2], "команда игрока"), agent.cast)
                    agent.save_memory()
            elif command == "/preset":
                if not args or args[0] not in PRESETS:
                    for pid, p in PRESETS.items():
                        print(f"  {pid}: {p.title}")
                else:
                    agent.load_preset(args[0])
                    print_card(agent, args[0])
            elif command == "/npc" and len(args) >= 3 and args[0] == "new":
                name, _, role = arg.split(maxsplit=2)[2].partition(",")
                print_ops(agent.set_card(args[1].lower(), "npc", {"имя": name.strip(), **({"роль": role.strip()} if role.strip() else {})},
                                         "команда игрока"), agent.cast)
                agent.enter(args[1].lower())
            elif command == "/scene":
                if len(args) == 2 and args[0] == "add":
                    agent.enter(args[1])
                elif len(args) == 2 and args[0] == "remove":
                    agent.leave(args[1])
                print("  в сцене: " + (", ".join(f"{agent.cast.name(x)} ({x})" for x in agent.present) or "никого"))
            elif command == "/talk":
                agent.talk_to(None if not args or args[0] == "off" else args[0])
                print(f"  🗣 {'герой говорит с ' + agent.cast.name(agent.talk) if agent.talk else 'Мастер рассказывает'}")
            elif command == "/hero":
                sub = args[0] if args else "list"
                if sub in ("new", "switch") and len(args) >= 2:
                    agent.save()
                    agent = make(agent.world, args[1].lower())
                    if sub == "new" and len(args) > 2:
                        seed_hero(agent, {"имя": " ".join(args[2:])}, {})
                    print(f"  ★ герой {agent.cast.name(agent.hero)} ({agent.hero}): своя память, общие карточки NPC")
                else:
                    heroes = sorted(p.name for p in (agent.world / "heroes").iterdir() if p.is_dir())
                    print("  герои мира: " + ", ".join(f"{'▶ ' if h == agent.hero else ''}{agent.cast.name(h)} ({h})"
                                                       for h in heroes))
            elif command == "/attach":
                agent.attach = "about" if args and args[0] == "about" else "full"
                print(f"  карточка собеседника: {'только «кто это»' if agent.attach == 'about' else 'целиком'}")
            elif command == "/check":
                prefs = agent.current_prefs()
                if not prefs or not last_a:
                    print("  нужен собеседник с манерой речи (/talk, /preset) и хотя бы один его ответ")
                else:
                    print(f"  {eval_line(ev.evaluate(last_q, last_a, prefs))}")
            elif command == "/compare":
                compare_demo(lambda w, h, attach="full": make(w, h, attach), ev,
                             agent.world.parent / f"{agent.world.name}_compare_{stamp_now()}", 1)
            elif command == "/stability":
                if not agent.talk or agent.talk not in PRESETS:
                    print("  /stability работает с готовым NPC: /preset lo|bai|hong")
                else:
                    stability_run(lambda w, h, attach="full": make(w, h, attach), ev,
                                  agent.world.parent / f"{agent.world.name}_stability_{stamp_now()}", agent.talk,
                                  STABILITY[:int(args[0]) if args and args[0].isdigit() else 10])
            elif command == "/memory":
                print_memory(agent)
            elif command == "/context":
                print_context12(agent)
            elif command == "/mission":
                mission_command(agent, arg)
            elif command == "/circle":
                circle_command(agent, arg)
            elif command == "/newchat":
                print(f"  🧹 {agent.new_scene()}; карточки, задание, круг и постоянное остаются")
            elif command == "/log":
                for item in agent.log.tail(int(args[0]) if args and args[0].isdigit() else 20):
                    print(f"  {item['at'][11:19]} {op_label(item, agent.cast)}  ⟵ {item.get('source', '')}")
            elif command == "/tokens":
                print_ledger(agent)
            elif command == "/history":
                print_history(agent.history)
            elif command.startswith("/"):
                print(f"  Неизвестная команда. Есть: {HELP}")
            else:
                reply = agent.run(text)
                last_q, last_a = text, reply.text.strip()
                who = agent.cast.name(agent.talk) if agent.talk else "Мастер"
                print(f"\n{who}: {last_a or '(пустой ответ)'}\n")
                print_turn(agent)
            print()
        except LLMError as e:
            print(f"❌ Ошибка: {e}\n", file=sys.stderr)
        except (StoreError, ValueError) as e:
            print(f"⚠️  {e}\n", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n(прервано — ход в сцену не попал)\n")


# ---------------------------------------------------------------- запуск


def main() -> int:
    parser = argparse.ArgumentParser(description="День 12: уся-РПГ с многими персонажами — карточки героев и NPC "
                                                 "поверх памяти Дня 11.")
    parser.add_argument("prompt", nargs="?", help="один ход и выход; без него — игра в консоли")
    parser.add_argument("--world", default="default", help="мир: memory/day12/<мир>/")
    parser.add_argument("--hero", default=HERO_ID, help="герой игрока в этом мире")
    parser.add_argument("--new", action="store_true", help="начать мир заново (прошлый уходит в архив)")
    parser.add_argument("--preset", choices=list(PRESETS), help="добавить готового NPC и сделать его собеседником")
    parser.add_argument("--attach", choices=("full", "about"), default="full",
                        help="карточки в запросе: целиком или только «кто это»")
    parser.add_argument("--keep", type=int, default=6, help="краткосрочная: сколько последних сообщений сцены в промпт")
    parser.add_argument("--long-budget", type=int, default=1500, help="долговременная: бюджет токенов")
    parser.add_argument("--helper-model", help="модель летописца (по умолчанию та же)")
    parser.add_argument("--judge-model", help="модель судьи (по умолчанию — как у летописца)")
    parser.add_argument("--demo", action="store_true", help="карточка из игры, 3 NPC × 1 вопрос, 10 реплик подряд, "
                                                            "два героя")
    parser.add_argument("--only", nargs="+", choices=("fill", "compare", "stability", "ablation", "heroes"),
                        help="для --demo: только эти части")
    parser.add_argument("--samples", type=int, default=3, help="для --demo: ответов на каждого NPC в сравнении")
    parser.add_argument("--questions", type=int, default=10, help="для --demo: реплик подряд в проверке устойчивости")
    parser.add_argument("--recheck", metavar="ОТЧЁТ", help="пересчитать проверки кодом в results/day12_*.json")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--max-steps", type=int, default=8, help="предел запросов к модели за один ход")
    args = parser.parse_args()

    if args.recheck:
        return recheck(Path(args.recheck))
    if not NAME_RE.fullmatch(args.world) or not NAME_RE.fullmatch(args.hero):
        print("Ошибка: имя мира и героя — буквы, цифры, дефис, подчёркивание.", file=sys.stderr)
        return 1
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    client = LLMClient(settings.base_url, settings.api_key, args.model or settings.model, settings.timeout,
                       retries=settings.retries)
    helper = (LLMClient(settings.base_url, settings.api_key, args.helper_model, settings.timeout,
                        retries=settings.retries) if args.helper_model else client)
    judge = (LLMClient(settings.base_url, settings.api_key, args.judge_model, settings.timeout,
                       retries=settings.retries) if args.judge_model else helper)
    if args.demo:
        parts = set(args.only or ("fill", "compare", "stability", "ablation", "heroes"))
        return demo(client, helper, judge, args.keep, args.long_budget, args.max_steps, max(args.samples, 1),
                    max(min(args.questions, len(STABILITY)), 1), parts)

    world = DAY_DIR / args.world
    if args.new and world.exists():
        target = world.with_name(f"{world.name}_{stamp_now()}")
        os.replace(world, target)
        print(f"Прошлый мир сохранён в {target.relative_to(ROOT)}.")
    make = lambda w, h, attach=args.attach: make_agent(client, helper, w, h, args.keep, args.long_budget,  # noqa: E731
                                                       args.max_steps, attach)
    try:
        agent = make(world, args.hero)
    except StoreError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print_restore(agent.restore_info, agent.store, agent)
    if args.preset:
        agent.load_preset(args.preset)
        print_card(agent, args.preset)
    if args.prompt:
        try:
            reply = agent.run(args.prompt)
        except (LLMError, StoreError) as e:
            print(f"❌ Ошибка: {e}", file=sys.stderr)
            return 1
        print(f"\n{agent.cast.name(agent.talk) if agent.talk else 'Мастер'}: {reply.text.strip()}\n")
        print_turn(agent)
        return 0
    return chat(agent, make)


if __name__ == "__main__":
    from run_timer import run_timed
    sys.exit(run_timed(main))
