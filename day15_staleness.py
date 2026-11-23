"""День 15. Устаревание и конфликты памяти: память Мастера Дня 14 не только пополняется, но и обновляется.

Мастер Дня 14 (`InvariantAgent`: автоматы задания и развития, инварианты, политика записи в память) получает
«обновляемую долговременную память» — реестр фактов (`FactStore`) с явными правилами, как факт меняется со временем.
Факт — это пара «ключ → значение» с временем записи. Ключ определяет, что с ним происходит:

    перезаписывается   срок, дедлайн, статус, версия, этап, подход, план, маршрут, требование, условие, улика —
                      новое значение заменяет старое: старое не остаётся рядом, а уходит в историю факта;
    накапливается      решения, история, хроника, события, заметки — новое значение дописывается списком,
                      а не заменяет прежнее;
    удаляется          те же «перезаписываемые» ключи текущей задачи: срок, статус, версия, подход, требование,
    после задачи      условие, улика — по завершении задачи убираются из памяти; постоянное остаётся.

Каждый факт хранит время записи (`at`), источник и историю значений. Разрешение конфликтов — кодом, в `FactStore.write`:
- при появлении факта, противоречащего сохранённому (тот же ключ, другое значение), старое значение **заменяется**,
  а не добавляется рядом;
- факт с более поздним временем записи имеет приоритет: если пришёл факт с временем раньше сохранённого, он
  отклоняется как устаревший, и в памяти остаётся более свежий;
- каждая замена и каждое отклонение фиксируются в отдельном логе конфликтов `conflicts.jsonl`: что было, что стало,
  кто победил и почему;
- удаление не «теряет» ключ бесследно: у него остаётся надгробие «отменено» (без старого значения). Иначе Мастер
  не отличает «факт отменили» от «факта никогда не было» и может пересказать отменённое требование как действующее.

Факты пишет **летописец** (роутер памяти Дня 11–12): он получил в ответе раздел `facts` и кладёт в реестр то, что
изменилось в репликах игрока и рассказе Мастера. Факты подключаются к каждому запросу блоком «Актуальные факты»
(только актуальные значения и надгробия без старых значений), а вручную — командой `/fact`.

Демо проверяет не только «сколько раз агент назвал устаревшее», но и откуда он это сделал: один и тот же вопрос
задаётся с блоком фактов и без него (A/B), а промпт аудируется по слоям — блок фактов, сцена-история и «чистый»
промпт без сцены. Отдельный шаг — запоздавшее донесение с более ранним временем: оно не должно воскресить старое.

    python day15_staleness.py                          # игра (мир default, герой luchen)
    python day15_staleness.py --demo --only matrix     # правила фактов и разрешение конфликтов, без модели
    python day15_staleness.py --demo --only cleanup    # очистка после задачи, без модели
    python day15_staleness.py --demo --only audit      # крайние случаи реестра фактов, без модели
    python day15_staleness.py --demo                   # аудит + матрица + очистка + живой диалог, отчёт в results/

Команды: /fact [list|set <ключ> = <значение>|unset <ключ>|conflicts|history <ключ>|rules|cleanup],
и всё из Дня 14: /state, /fsm, /approve, /go, /pause, /resume, /breakthrough, /journal, /guard, /matrix, /policy,
/event, /remember, /forget, /mission, /card, /memory, /context, /invariant, /newchat, /tokens, /history, /exit.
"""
import argparse
import copy
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from config import ConfigError, load_settings
from day7_memory import MEMORY_DIR, ROOT, StoreError, now_iso, print_history, print_restore
from day8_tokens import Usage, clip, fmt, money, price_for, print_ledger
from day9_compression import normalize, watch_retries
from day10_strategies import StickyFacts, extract_json_object, read_json, write_json
from day11_memory_layers import (GAME_TOOLS, MISSIONS, SECTION_ALIASES, clean_part, md_escape, mission_command,
                                 print_memory, stamp_now)
from day12_personalization import op_label as op_label12
from day12_personalization import print_card, print_context12
from day13_task_state import (MissionMachine, QUEST_PROMPT, seed_quest_hero, state_line, print_state, print_journal,
                              op_label as op_label13)
from day14_invariants import (INV_PROMPT, InvariantAgent, inv_command, print_invariants,
                              on_event as day14_event, quiet_event as day14_quiet_event)
from llm_client import LLMClient, LLMError

RESULTS_DIR = ROOT / "results"
DAY_DIR = MEMORY_DIR / "day15"
NAME_RE = re.compile(r"[\w-]+")

# ---------------------------------------------------------------- правила фактов: что перезаписывать, что копить, что убирать

# Ключи, которые накапливаются списком (принятые решения, история изменений, события, заметки).
ACCUMULATE_RE = re.compile(r"решен|истори|хроник|событи|заметк|изменен", re.I)
# Ключи текущей задачи: перезаписываются и удаляются после завершения задачи.
TRANSIENT_RE = re.compile(
    r"срок|дедлайн|статус|верси|этап|требован|услови|улик|подход|маршрут|план|груз|товар|обоз|тайник", re.I)

# Таблица правил для /fact rules и отчёта: (что, какие ключи, как ведут себя).
FACT_RULES = (
    ("перезаписываются", "срок, дедлайн, статус, версия, этап, подход, план, маршрут, требование, условие, улика, "
                         "груз, тайник",
     "новое значение заменяет старое: старое не остаётся рядом, а уходит в историю факта"),
    ("накапливаются списком", "решения, история, хроника, события, заметки",
     "новое значение дописывается к списку, а не заменяет прежнее"),
    ("удаляются после задачи", "срок, статус, версия, подход, требование, условие, улика, груз, товар, обоз, тайник — "
                              "факты текущей задачи",
     "по завершении задачи убираются из памяти; постоянное (решения, история, наставления) остаётся, и проверяется, "
     "что хвостов дела не осталось"),
    ("отменяются", "любой удалённый факт до завершения задачи",
     "ключ остаётся отметкой «отменено» без старого значения: Мастер знает, что факт снят, а не потерян"),
    ("время записи", "у каждого факта", "факт с более поздним временем записи имеет приоритет"),
)


def mode_of(key: str) -> str:
    return "accumulate" if ACCUMULATE_RE.search((key or "").lower()) else "overwrite"


def transient_of(key: str) -> bool:
    return bool(TRANSIENT_RE.search((key or "").lower()))


def later(a: str, b: str) -> bool:
    """Верно, если момент a позже b. Строки ISO сравниваются как даты, а если формат не разобрать — как строки."""
    try:
        return datetime.fromisoformat(a) > datetime.fromisoformat(b)
    except (TypeError, ValueError):
        return a > b


FACTS_HEADER = ("\n\n## Актуальные факты (обновляемая память)\n"
                "Здесь только актуальные значения — они важнее старых сообщений сцены и карточки задания: если срок, "
                "статус или требование в карточке расходится с фактом, верь факту. Если значение факта менялось, "
                "показано последнее; устаревшие сюда не попадают. Факты-списки (решения, история) перечислены целиком. "
                "Отменённые факты — в отдельном разделе ниже: их старое значение недействительно, условие снято, "
                "не пересказывай его как действующее. Статус дела (`статус`) — это запись памяти; этап автомата "
                "задания — отдельная сущность. Если вопрос о том, что записано в статусе, отвечай по факту, а не по "
                "этапу автомата.\n")
DELETED_HEADER = ("## Отменённые факты (не использовать)\n"
                  "Эти факты сняты или удалены: старого значения в памяти больше нет, условие не действует.\n")


# ---------------------------------------------------------------- реестр фактов


class FactStore:
    """Обновляемая долговременная память: плоский реестр «ключ → факт» с правилами изменения и логом конфликтов.

    Факт — {"value", "at", "source", "mode", "transient", "items", "history"}. mode = overwrite | accumulate,
    transient = удалять ли после завершения задачи. deleted — надгробия удалённых фактов: {"at", "source",
    "before", "mode"}; в промпт уходит только отметка «отменено», без старого значения. conflicts.jsonl — журнал:
    что было, что стало, кто победил. Изменения буферизуются в памяти и пишутся на диск в save() — так ход можно
    откатить без «фантомных» записей."""

    def __init__(self, path: Path | None, conflicts_path: Path | None):
        self.path = path
        self.conflicts_path = conflicts_path
        self.facts: dict[str, dict] = {}
        self.deleted: dict[str, dict] = {}
        self.conflicts: list[dict] = []
        self._conflicts_flushed = 0  # сколько записей конфликтов уже лежит в файле

    # --- файлы

    def load(self) -> None:
        self.facts, self.deleted, self.conflicts, self._conflicts_flushed = {}, {}, [], 0
        if self.path is not None and self.path.exists():
            raw = read_json(self.path)
            for k, v in (raw.get("facts") or {}).items():
                if isinstance(v, dict) and str(v.get("value") or "").strip():
                    self.facts[str(k)] = {"value": str(v["value"]), "at": str(v.get("at") or ""),
                                          "source": str(v.get("source") or ""),
                                          "mode": v.get("mode") or mode_of(k),
                                          "transient": bool(v.get("transient")),
                                          "items": [str(x) for x in v.get("items") or []],
                                          "history": v.get("history") or []}
            for k, v in (raw.get("deleted") or {}).items():
                if isinstance(v, dict) and str(v.get("before") or "").strip():
                    self.deleted[str(k)] = {"at": str(v.get("at") or ""), "source": str(v.get("source") or ""),
                                            "before": str(v["before"]), "mode": v.get("mode") or mode_of(k)}
            for k in list(self.deleted):  # активный факт важнее надгробия: так чинится рассогласованный файл
                if k in self.facts:
                    self.deleted.pop(k, None)
        if self.conflicts_path is not None and self.conflicts_path.exists():
            try:
                self.conflicts = [json.loads(x) for x in self.conflicts_path.read_text(encoding="utf-8").splitlines()
                                  if x.strip()]
            except (OSError, ValueError) as e:
                raise StoreError(f"не удалось прочитать лог конфликтов {self.conflicts_path}: {e}")
        self._conflicts_flushed = len(self.conflicts)

    def save(self) -> None:
        if self.path is not None:
            write_json(self.path, {"format": 2, "updated_at": now_iso(), "facts": self.facts,
                                   "deleted": self.deleted})
        if self.conflicts_path is not None and len(self.conflicts) > self._conflicts_flushed:
            self.conflicts_path.parent.mkdir(parents=True, exist_ok=True)
            with self.conflicts_path.open("a", encoding="utf-8") as f:
                for rec in self.conflicts[self._conflicts_flushed:]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._conflicts_flushed = len(self.conflicts)

    def _record_conflict(self, rec: dict) -> None:
        self.conflicts.append(rec)

    # --- чтение

    def current(self) -> dict[str, str]:
        return {k: f["value"] for k, f in self.facts.items()}

    def block(self) -> str:
        """Актуальные факты текстом — в конец системного сообщения. Только текущие значения и надгробия:
        старого значения удалённого факта здесь нет, есть лишь отметка «отменено»."""
        if not self.facts and not self.deleted:
            return ""
        lines = [FACTS_HEADER.rstrip("\n")]
        for k in sorted(self.facts):
            f = self.facts[k]
            mark = " — до конца задачи" if f["transient"] else ""
            lines.append(f"- {k}: {f['value']}{mark}")
        if self.deleted:
            lines += ["", DELETED_HEADER.rstrip("\n")]
            for k in sorted(self.deleted):
                lines.append(f"- {k}: отменено")
        return "\n".join(lines) + "\n"

    # --- запись с разрешением конфликтов

    def _base(self, key: str, value: str, at: str, source: str, mode: str, transient: bool) -> dict:
        return {"key": key, "value": value, "before": "", "after": "", "at": at, "source": source,
                "mode": mode, "transient": transient, "op": ""}

    def write(self, key: str, value: str, source: str, at: str | None = None) -> dict:
        key = str(key).strip()
        value = str(value).strip()
        if not key:
            raise ValueError("пустой ключ факта")
        if not value:
            raise ValueError(f"пустое значение факта «{key}»")
        at = at or now_iso()
        mode, transient = mode_of(key), transient_of(key)
        cur = self.facts.get(key)
        tomb = self.deleted.get(key)

        if cur is None and tomb:  # факт был удалён: запоздавшее значение не воскрешает его
            if tomb.get("at") and later(tomb["at"], at):
                self._record_conflict({"at": at, "key": key, "source": source, "before": "",
                                       "after": value, "winner": "deleted", "mode": mode,
                                       "reason": "факт удалён позже, чем пришло это значение"})
                op = self._base(key, value, at, source, mode, transient)
                return dict(op, op="stale", before="", after="", deleted=True)
            self.deleted.pop(key, None)
        elif cur is not None and tomb:  # рассогласованный файл: активный факт важнее надгробия
            self.deleted.pop(key, None)

        if cur is None:  # нового факта ещё нет — просто завести
            self.deleted.pop(key, None)
            fact = {"value": value, "at": at, "source": source, "mode": mode, "transient": transient,
                    "items": [value] if mode == "accumulate" else [],
                    "history": [{"at": at, "op": "set", "value": value, "source": source}]}
            self.facts[key] = fact
            self._record_conflict({"at": at, "key": key, "source": source, "before": "", "after": value,
                                   "winner": "new", "mode": mode})
            op = self._base(key, value, at, source, mode, transient)
            return dict(op, op="set", after=value)

        # --- факт уже есть
        if mode == "accumulate":
            if value in cur["items"]:
                op = self._base(key, value, at, source, mode, transient)
                return dict(op, op="unchanged", before=cur["value"], after=cur["value"])
            before = cur["value"]
            cur["items"].append(value)
            after = "; ".join(cur["items"])
            cur["value"], cur["at"], cur["source"] = after, at, source
            cur["history"].append({"at": at, "op": "append", "value": value, "source": source})
            self._record_conflict({"at": at, "key": key, "source": source, "before": before, "after": after,
                                   "winner": "append", "mode": mode})
            op = self._base(key, value, at, source, mode, transient)
            return dict(op, op="append", before=before, after=after)

        # --- overwrite: замена, а не «рядом»
        if value == cur["value"]:
            op = self._base(key, value, at, source, mode, transient)
            return dict(op, op="unchanged", before=cur["value"], after=cur["value"])
        if cur["at"] and later(cur["at"], at):  # сохранённый факт свежее — пришедший устарел
            self._record_conflict({"at": now_iso(), "key": key, "source": source, "before": cur["value"],
                                   "after": value, "winner": "old", "mode": mode,
                                   "reason": "есть факт с более поздним временем записи"})
            op = self._base(key, value, at, source, mode, transient)
            return dict(op, op="stale", before=cur["value"], after=cur["value"])
        before = cur["value"]
        cur["value"], cur["at"], cur["source"] = value, at, source
        cur["history"].append({"at": at, "op": "overwrite", "value": value, "source": source, "before": before})
        self._record_conflict({"at": at, "key": key, "source": source, "before": before, "after": value,
                               "winner": "new", "mode": mode})
        op = self._base(key, value, at, source, mode, transient)
        return dict(op, op="overwrite", before=before, after=value)

    def forget(self, key: str, source: str, at: str | None = None) -> dict | None:
        key = str(key).strip()
        cur = self.facts.pop(key, None)
        if cur is None:
            return None
        at = at or now_iso()
        self.deleted[key] = {"at": at, "source": source, "before": cur["value"], "mode": cur["mode"]}
        self._record_conflict({"at": at, "key": key, "source": source, "before": cur["value"], "after": "",
                               "winner": "delete", "mode": cur["mode"]})
        return {"op": "delete", "key": key, "value": "", "before": cur["value"], "after": "", "at": at,
                "source": source, "mode": cur["mode"], "transient": cur["transient"], "deleted": True}

    def delete_transient(self, source: str, at: str | None = None) -> list[dict]:
        """Завершение задачи: временные факты (срок, статус, версия, подход, требования…) убираются, постоянное
        остаётся. Надгробия после задачи тоже убираются: дело закрыто, спрашивать о его сроках больше нечего."""
        at = at or now_iso()
        removed = []
        for k in [k for k, f in self.facts.items() if f["transient"]]:
            cur = self.facts.pop(k)
            self.deleted.pop(k, None)
            self._record_conflict({"at": at, "key": k, "source": source, "before": cur["value"], "after": "",
                                   "winner": "transient", "mode": cur["mode"], "reason": "задача завершена"})
            removed.append({"key": k, "value": cur["value"]})
        for k in [k for k in self.deleted if transient_of(k)]:  # отменённые временные тоже не нужны после задачи
            self.deleted.pop(k, None)
        return removed


# ---------------------------------------------------------------- летописец: факты из реплик

ROUTER_PROMPT15 = (
    "Ты — летописец текстовой ролевой игры в жанре уся. Мастер игры не видит старых сообщений: всё важное он знает "
    "только из памяти героя, карточек персонажей и обновляемых фактов дела. Ты решаешь, что из новых сообщений куда "
    "записать.\n"
    "Память героя игрока (у каждого героя своя):\n"
    "- school — школа или секта героя, наставник, положение в школе;\n"
    "- nature — характер, принципы, клятвы и обеты;\n"
    "- traits — врождённые и постоянные особенности: корень духа, телосложение, шрамы;\n"
    "- bonds — связи: кровные враги, побратимы, долги;\n"
    "- circle — круг и ступень, год обучения, техника и освоенные формы, ци, условия прорыва;\n"
    "- gear — снаряжение и запасы: оружие, пилюли, деньги, вещи;\n"
    "- mission — текущее задание: заказчик, цель, награда, найденные улики — итоги, а не пересказ хода.\n"
    "Карточки персонажей (cards) — у героев игроков и у NPC, по id: about, voice, manner, taboo, terms, attitude.\n"
    "Обновляемые факты дела (facts) — плоский список «ключ → значение» с меняющимися обстоятельствами текущего дела. "
    "Сюда идут только те сведения, которые могут измениться по ходу:\n"
    "- перезаписываемые: срок, дедлайн, статус, версия, этап, подход, план, маршрут, требование, условие, улика. "
    "Если значение изменилось — верни НОВОЕ значение тем же ключом (старое код заменит сам, новый ключ не заводи);\n"
    "- накапливаемые: решения, история, хроника, события, заметки. Если принято новое решение или случилось новое "
    "событие — верни его как отдельное значение; код допишет его списком, не заменяя прежние;\n"
    "- если требование/условие отменено или факт стал неверен — убери ключ через delete. Если факт был отменён, "
    "а затем появилось новое значение того же смысла — верни тот же ключ (например, требование), а не заводи "
    "синоним: код снимет надгробие.\n"
    "- текущие факты — уже последние. Не переписывай их старым значением из рассказа Мастера или карточки задания: "
    "если текст лишь пересказывает прежний договор, это не изменение. Меняй факт только при явном новом значении.\n"
    "- одно значение — одно актуальное. Не склеивай прежнее и новое в одну строку: если названо новое значение — "
    "верни только его.\n"
    "Ключи фактов — короткие, по-русски, в нижнем регистре через подчёркивание (срок, статус, подход, требование, "
    "решения). Один ключ — одно актуальное значение. Срок, статус, подход и требования пиши только в facts, не в "
    "mission.\n"
    "Тебе дают текущую память героя, карточки, факты дела, последний рассказ Мастера и новые сообщения игрока. Верни "
    "JSON вида "
    '{"school": {"set": {}, "delete": []}, "nature": {...}, "traits": {...}, "bonds": {...}, "circle": {...}, '
    '"gear": {...}, "mission": {...}, "cards": {...}, "facts": {"set": {}, "delete": []}} — разделы без изменений '
    'можно опустить:\n'
    "- из сообщений игрока бери то, что он сказал о герое и о деле, и то, что герой сделал или решил. Из рассказа "
    "Мастера — только то, что случилось на самом деле. Описания, угрозы и догадки — не записи;\n"
    "- значения пиши точно и коротко (до 15 слов). Для фактов — просто новое актуальное значение;\n"
    "- ключи короткие, по-русски, в нижнем регистре через подчёркивание. Запись уже есть — обнови тот же ключ. "
    "Один ключ — в одном месте;\n"
    "- вопросы игрока вне игры — не записи. Ничего не придумывай. Нечего записывать — верни пустой объект {}.\n"
    "Верни только JSON.")


# ---------------------------------------------------------------- агент


class MemoryAgent(InvariantAgent):
    """InvariantAgent Дня 14, у которого сверх автоматов и инвариантов — обновляемый реестр фактов. Факты пишет
    летописец (раздел `facts` в его ответе) и команда /fact; подключаются к каждому запросу только актуальными
    значениями, а по завершении задачи временные убираются."""

    router_prompt = ROUTER_PROMPT15

    def __init__(self, client: LLMClient, tools, root: Path, *, guard: bool = True, per_turn: int = 1,
                 seed: int | None = None, **kwargs):
        self.facts = FactStore(root / "facts.json", root / "conflicts.jsonl")
        self.fact_events: list[dict] = []
        super().__init__(client, tools, root, guard=guard, per_turn=per_turn, seed=seed, **kwargs)

    # --- файлы

    def load(self):
        self.facts.load()
        return super().load()

    def save_memory(self) -> None:
        self.facts.save()
        super().save_memory()

    # --- факты в запросе

    def build(self, end=None, layers=("short", "working", "long"), tools=None, include_facts: bool = True):
        """Промпт хода. include_facts=False — контрольный режим A/B: блок фактов не подключается, и Мастер
        остаётся один на один со сценой и карточкой дела (видно, что именно даёт обновляемая память)."""
        messages, dropped, parts = super().build(end, layers, tools)
        text = self.facts.block() if include_facts else ""
        if text:
            messages[0] = dict(messages[0], content=(messages[0].get("content") or "") + text)
        raw = self.estimator.raw_text(text) if text else 0
        parts["raw"]["facts"] = raw
        parts["estimate"]["facts"] = self.estimator.apply(raw) if raw else 0
        return messages, dropped, parts

    # --- летописец: факты из ответа

    def router_context(self) -> str:
        base = super().router_context()
        facts = json.dumps(self.facts.current(), ensure_ascii=False, indent=1)
        text = f"{base}\n\nТекущие факты дела:\n{facts if self.facts.facts else '{}'}"
        if self.facts.deleted:  # летописец должен видеть отменённые ключи, чтобы не заводить синоним
            text += ("\n\nОтменённые факты (ключа нет в текущих, старое значение не показывается). Если появится "
                     "новое значение того же смысла — верни этот же ключ, код снимет надгробие:\n"
                     + "\n".join(f"- {k}: отменено" for k in sorted(self.facts.deleted)))
        old_values = []
        for key, fact in self.facts.facts.items():
            if fact["mode"] != "overwrite":
                continue
            hist = [str(h.get("value") or "") for h in (fact.get("history") or [])
                    if h.get("value") and str(h["value"]) != fact["value"]]
            if hist:
                old_values.append(f"- {key}: " + "; ".join(f"«{clip(v, 60)}»" for v in hist[-2:]))
        if old_values:  # старые значения — чтобы летописец не «возвращал» их из пересказа сцены
            text += ("\n\nПрежние значения фактов (не возвращать без явного изменения):\n" + "\n".join(old_values))
        return text

    def parse_route(self, text: str) -> dict:
        out = super().parse_route(text)
        data = extract_json_object(text)
        facts = data.get("facts") or {}
        if not isinstance(facts, dict):
            raise ValueError("facts — не объект")
        out["facts"] = StickyFacts.parse(json.dumps(clean_part(facts), ensure_ascii=False)) if facts else ({}, [])
        return out

    def apply(self, changes: dict, source: str) -> list[dict]:
        ops = super().apply({k: v for k, v in changes.items() if k not in ("cards", "facts")}, source)
        update, delete = changes.get("facts") or ({}, [])
        for key, value in update.items():
            try:
                op = dict(self.facts.write(key, value, source), fact=True)
            except ValueError as e:  # пустой ключ/значение от летописца не должны ломать ход
                op = {"op": "rejected", "key": str(key), "value": str(value), "before": "", "after": "",
                      "at": now_iso(), "source": source, "mode": mode_of(str(key)),
                      "transient": transient_of(str(key)), "fact": True, "reason": str(e)}
            ops.append(op)
            self.fact_events.append(op)
            self.on_event("fact", {"op": op, "agent": self})
        for key in delete:
            op = self.facts.forget(key, source)
            if op:
                op = dict(op, fact=True)
                ops.append(op)
                self.fact_events.append(op)
                self.on_event("fact", {"op": op, "agent": self})
        return ops

    def route_note(self, ops: list[dict]) -> str:
        if not ops:
            return "изменений нет"
        facts = sum(1 for o in ops if o.get("fact"))
        rejected = sum(1 for o in ops if o["op"] == "rejected")
        cards = sum(1 for o in ops if o.get("char") and o["op"] != "rejected")
        hero = len(ops) - facts - cards - rejected
        parts = []
        if hero:
            parts.append(f"память героя {hero}")
        if cards:
            parts.append(f"карточки {cards}")
        if facts:
            parts.append(f"факты {facts}")
        return "изменений " + ", ".join(parts) + (f"; отклонено {rejected}" if rejected else "")

    # --- откат хода: факты — часть снимка памяти

    def snapshot(self) -> dict:
        snap = super().snapshot()
        snap["facts"] = copy.deepcopy(self.facts.facts)
        snap["deleted"] = copy.deepcopy(self.facts.deleted)
        snap["conflicts"] = list(self.facts.conflicts)
        snap["conflicts_flushed"] = self.facts._conflicts_flushed
        return snap

    def rollback(self, snap: dict) -> None:
        super().rollback(snap)
        self.facts.facts = snap["facts"]
        self.facts.deleted = snap["deleted"]
        self.facts.conflicts = snap["conflicts"]
        self.facts._conflicts_flushed = snap["conflicts_flushed"]

    # --- запись фактов вручную

    def write_fact(self, key: str, value: str, source: str, at: str | None = None) -> dict:
        op = self.facts.write(key, value, source, at)
        self.fact_events.append(op)
        self.on_event("fact", {"op": op, "agent": self})
        return op

    def forget_fact(self, key: str, source: str) -> dict | None:
        op = self.facts.forget(key, source)
        if op:
            self.fact_events.append(op)
            self.on_event("fact", {"op": op, "agent": self})
        return op

    def complete_task(self, source: str = "завершение задачи") -> list[dict]:
        """Завершить задачу: временные факты убираются, постоянное (решения, история, наставления) остаётся."""
        removed = self.facts.delete_transient(source)
        if removed:
            self.on_event("cleanup", {"removed": removed, "agent": self})
        self.save_memory()
        return removed

    def mission_done(self, entry) -> None:
        super().mission_done(entry)
        self.complete_task("завершение задания")

    def done_mission(self, status: str = "done"):
        m = super().done_mission(status)
        self.complete_task("завершение задания")
        return m


# ---------------------------------------------------------------- вывод


def fact_op_label(op: dict) -> str:
    k, v = op["key"], clip(op.get("value") or "", 60)
    if op["op"] == "stale":
        if op.get("deleted"):
            return f"{k}: «{v}» отклонено — факт отменён позже, старое не воскресает"
        return f"{k}: «{v}» отклонено — позже уже записано «{clip(op.get('before') or '', 40)}»"
    if op["op"] == "rejected":
        return f"{k}: «{v}» отклонено кодом — {op.get('reason') or 'некорректная запись'}"
    if op["op"] == "unchanged":
        return f"{k}: без изменений («{clip(op.get('after') or '', 40)}»)"
    if op["op"] == "delete":
        return f"{k} − «{clip(op.get('before') or '', 40)}» (отменено)"
    if op["op"] == "append":
        return f"{k} + «{v}» (список)"
    if op["op"] == "overwrite":
        return f"{k}: было «{clip(op.get('before') or '', 40)}» → стало «{v}»"
    return f"{k} = «{v}»"


def on_event(kind: str, data: dict) -> None:
    if kind == "fact":
        print(f"  🧠 факт: {fact_op_label(data['op'])}")
    elif kind == "cleanup":
        removed = data["removed"]
        print(f"  🧹 завершение задачи: убрано {len(removed)} временных фактов — "
              + (", ".join(f"«{r['key']}»" for r in removed) or "ничего"))
    elif kind == "route":
        c = data["call"]
        print(f"  📜 летописец: {c.note} | вход {fmt(c.prompt)}, выход {fmt(c.completion)}, {c.latency:.1f} с, "
              f"{money(c.cost)}")
        for item in data["ops"]:
            if item.get("fact"):
                print(f"     🧠 {fact_op_label(item)}")
            else:
                print(f"     {op_label12(item)}")
                if item.get("violation"):
                    print(f"       ⚠️  нарушение политики памяти (код не проверяет): {item['violation']}")
    else:
        day14_event(kind, data)


def quiet_event(kind: str, data: dict) -> None:
    if kind in ("tokens", "llm", "truncate", "reset") or (kind == "route" and not data["ops"]):
        return
    on_event(kind, data)


def print_facts(agent: MemoryAgent) -> None:
    fs = agent.facts
    if not fs.facts:
        print("  (фактов нет)")
        return
    for k in sorted(fs.facts):
        f = fs.facts[k]
        kind = "список" if f["mode"] == "accumulate" else "перезаписывается"
        scope = "до конца задачи" if f["transient"] else "постоянный"
        print(f"  · {k}: {clip(f['value'], 70)}  ⟵ {f['source']}")
        print(f"      {kind}, {scope}; записано {f['at'][11:19] if f['at'] else '?'}; "
              f"в истории значений {len(f.get('history') or [])}")
    for k in sorted(fs.deleted):
        d = fs.deleted[k]
        print(f"  ⊘ {k}: отменено  ⟵ {d['source']} ({d['at'][11:19] if d['at'] else '?'})")
    print(f"  конфликтов в логе: {len(fs.conflicts)} (/fact conflicts)")


def print_conflicts(agent: MemoryAgent, n: int = 20) -> None:
    for c in agent.facts.conflicts[-n:]:
        when = c.get("at", "")[11:19]
        if c["winner"] == "old":
            print(f"  {when} [{c['key']}] «{clip(c['after'], 34)}» отклонено: сохранено «{clip(c['before'], 34)}»")
        elif c["winner"] == "deleted":
            print(f"  {when} [{c['key']}] «{clip(c['after'], 34)}» отклонено: факт уже отменён позже")
        elif c["winner"] == "delete":
            print(f"  {when} [{c['key']}] удалено: было «{clip(c['before'], 44)}»")
        elif c["winner"] == "transient":
            print(f"  {when} [{c['key']}] убрано после задачи: было «{clip(c['before'], 44)}»")
        elif c["winner"] == "append":
            print(f"  {when} [{c['key']}] + «{clip(c['after'].split('; ')[-1], 44)}» (список)")
        else:
            print(f"  {when} [{c['key']}] {c['winner']}: было «{clip(c['before'], 40)}» → стало «{clip(c['after'], 40)}»")
    if not agent.facts.conflicts:
        print("  лог конфликтов пуст")


def print_rules15() -> None:
    print("  ПРАВИЛА ФАКТОВ (запись проверяется кодом до применения):")
    for what, keys, how in FACT_RULES:
        print(f"   {what:<22} {keys}")
        print(f"   {'':<22} → {how}")


def layer_line15(tokens: dict) -> str:
    names = (("system", "system"), ("facts", "факты"), ("invariants", "инварианты"), ("fsm", "автоматы"),
             ("cast", "карточки"), ("long", "долговр."), ("working", "рабочая"), ("short", "краткоср."),
             ("current", "ход"))
    return " · ".join(f"{label} {fmt(tokens.get(k, 0))}" for k, label in names)


def print_turn15(agent: MemoryAgent) -> None:
    turn = agent.ledger[-1] if agent.ledger else None
    if turn is None or not turn.calls:
        print(f"  ❌ ответа нет: {turn.error if turn else ''}")
        return
    s = agent.layer_stats[-1] if agent.layer_stats and agent.layer_stats[-1]["turn"] == turn.n else None
    print(f"  🧭 {state_line({'mission': agent.machine.step_info(), 'circle': agent.cult.step_info()})}")
    print(f"  📊 ход {turn.n}: вход {fmt(turn.calls[0].prompt)} = " + (layer_line15(s["tokens"]) if s else "?")
          + f" | выход {fmt(turn.completion)}, запросов {len(turn.calls)} | {money(turn.cost)}")


# ---------------------------------------------------------------- проверка 1: правила и конфликты (без модели)

T0 = "2026-11-23T10:00:00+00:00"
T1 = "2026-11-23T11:00:00+00:00"
T2 = "2026-11-23T12:00:00+00:00"
T3 = "2026-11-23T13:00:00+00:00"


def matrix_demo(verbose: bool = True) -> dict:
    """Правила фактов и разрешение конфликтов — кодом, без модели: перезапись, список, «позже побеждает», очистка."""
    if verbose:
        print(f"{'=' * 78}\n=== 1. Правила фактов и разрешение конфликтов, без модели\n{'=' * 78}")
        print_rules15()
    fs = FactStore(None, None)
    checks = []

    def chk(what: str, ok: bool, detail: str = "") -> None:
        checks.append({"what": what, "ok": ok, "detail": detail})
        if verbose:
            print(f"  {'✅' if ok else '❌'} {what}" + (f": {detail}" if detail and not ok else ""))

    # перезапись: срок меняется, старое не остаётся рядом
    fs.write("срок", "до полнолуния", "сцена", T0)
    op = fs.write("срок", "продлили до новолуния", "сцена", T1)
    chk("срок перезаписан новым значением", op["op"] == "overwrite"
        and fs.facts["срок"]["value"] == "продлили до новолуния")
    chk("старое значение срока не осталось рядом", "полнолуния" not in fs.facts["срок"]["value"])
    chk("замена зафиксирована: что было → что стало",
        any(c["key"] == "срок" and c["winner"] == "new" and c["before"] == "до полнолуния"
            and c["after"] == "продлили до новолуния" for c in fs.conflicts))

    # факт с более поздним временем записи имеет приоритет
    op = fs.write("срок", "ещё раз сдвинули вчера", "сцена", T0)
    chk("устаревшее значение отклонено (позже — приоритетнее)", op["op"] == "stale"
        and fs.facts["срок"]["value"] == "продлили до новолуния")
    chk("отклонение устаревшего — в логе", any(c["key"] == "срок" and c["winner"] == "old" for c in fs.conflicts))

    # то же значение — без изменений и без лишней записи в лог
    n_before = len(fs.conflicts)
    op = fs.write("срок", "продлили до новолуния", "сцена", T2)
    chk("то же значение — без изменений", op["op"] == "unchanged" and len(fs.conflicts) == n_before)

    # накопление списком: решения не заменяются, а дописываются
    fs.write("решения", "заночевать у водопада", "сцена", T0)
    op = fs.write("решения", "отправить письмо наставнику", "сцена", T1)
    items = fs.facts["решения"]["items"]
    chk("решения накапливаются списком", op["op"] == "append" and len(items) == 2
        and any("заночевать" in x for x in items) and any("письмо" in x for x in items))
    chk("решение не заменило прежнее", "заночевать" in fs.facts["решения"]["value"])
    chk("список сохраняет порядок дописывания", items == ["заночевать у водопада", "отправить письмо наставнику"])

    # дубль в списке не копится
    n_before = len(fs.facts["решения"]["items"])
    op = fs.write("решения", "заночевать у водопада", "сцена", T2)
    chk("дубль решения не добавляется", op["op"] == "unchanged" and len(fs.facts["решения"]["items"]) == n_before)

    # удаление: факт уходит из актуальных, но остаётся надгробие «отменено» без старого значения
    fs.forget("срок", "сцена", T3)
    chk("удалённый факт помечен «отменено»", "срок" in fs.deleted and "срок" not in fs.current())
    chk("старого значения в блоке фактов нет, есть только «отменено»",
        "срок: отменено" in fs.block() and "продлили до новолуния" not in fs.block())
    op = fs.write("срок", "запоздалое донесение", "сцена", T0)
    chk("запоздавшее значение после удаления не воскрешает факт",
        op["op"] == "stale" and op.get("deleted") and "срок" not in fs.facts)
    op = fs.write("срок", "снова назначили", "сцена", T3)
    chk("после удаления факт можно записать заново", op["op"] == "set" and fs.facts["срок"]["value"] == "снова назначили")
    chk("новая запись снимает надгробие", "срок" not in fs.deleted and "срок: отменено" not in fs.block())

    # классификация ключей
    chk("срок/статус/версия/подход/требование/дедлайн/груз/тайник — перезаписываемые и временные",
        all(mode_of(k) == "overwrite" and transient_of(k)
            for k in ("срок", "статус", "версия", "подход", "требование", "дедлайн", "груз", "тайник")))
    chk("решения/история — накапливаемые и постоянные",
        all(mode_of(k) == "accumulate" and not transient_of(k) for k in ("решения", "история_изменений")))

    # у каждого факта — время записи
    chk("у каждого факта есть время записи", all(f["at"] for f in fs.facts.values()))

    # очистка после задачи: временное убирается, постоянное остаётся
    removed = fs.delete_transient("завершение задачи")
    chk("временные факты удалены после задачи", "срок" not in fs.facts
        and all(r["key"] == "срок" for r in removed))
    chk("постоянные факты сохранились", "решения" in fs.facts)
    chk("надгробия временных фактов после задачи убраны", not fs.deleted)
    chk("уборка временного — в логе", all(any(c["key"] == r["key"] and c["winner"] == "transient"
                                               for c in fs.conflicts) for r in removed))

    # защита от пустых записей: летописец может вернуть пустой ключ или значение
    for bad_key, bad_value in (("", "значение"), ("срок", "   ")):
        try:
            fs.write(bad_key, bad_value, "сцена", T3)
            ok = False
        except ValueError:
            ok = True
        chk(f"{'пустой ключ' if not bad_key else 'пустое значение'} отклоняется кодом", ok)

    report = {"checks": checks, "mismatches": sum(not c["ok"] for c in checks), "checked": len(checks),
              "rules": [{"what": a, "keys": b, "how": c} for a, b, c in FACT_RULES]}
    if verbose:
        print(f"\n  Проверено {report['checked']} случаев, расхождений: {report['mismatches']}")
    return report


# ---------------------------------------------------------------- проверка 2: очистка после задачи (без модели)


def cleanup_demo(verbose: bool = True) -> dict:
    """Завершение задачи: рабочая (временные факты) очищена, долговременная (постоянные) сохранилась — без модели."""
    if verbose:
        print(f"{'=' * 78}\n=== 2. Очистка после задачи: временное убирается, постоянное остаётся\n{'=' * 78}")
    fs = FactStore(None, None)
    for k, v in (("статус", "сборы"), ("срок", "до полнолуния"), ("подход", "через Ущелье"),
                 ("требование", "вернуть груз целым"), ("решения", "заночевать у водопада"),
                 ("наставление", "меч не обнажать без нужды")):
        fs.write(k, v, "игра (карточка дела)")
    before = set(fs.facts)
    removed = fs.delete_transient("завершение задачи")
    kept = set(fs.facts)
    transient_before = {k for k in before if transient_of(k)}
    persistent_before = {k for k in before if not transient_of(k)}
    checks = [
        {"what": "временные факты (срок, статус, подход, требование) убраны",
         "ok": transient_before and not (transient_before & kept), "detail": f"убрано: {sorted(transient_before)}"},
        {"what": "постоянные факты (решения, наставление) сохранились",
         "ok": persistent_before <= kept, "detail": f"осталось: {sorted(kept)}"},
        {"what": "уборка зафиксирована в логе конфликтов",
         "ok": all(any(c["key"] == r["key"] and c["winner"] == "transient" for c in fs.conflicts) for r in removed)},
        {"what": "в промпт после очистки временное не попадает",
         "ok": all(f"{r['key']}:" not in fs.block() and r["value"] not in fs.block() for r in removed)},
        {"what": "надгробий отменённых фактов после задачи не осталось",
         "ok": not fs.deleted, "detail": f"надгробия: {sorted(fs.deleted)}"},
    ]
    for c in checks:
        if verbose:
            print(f"  {'✅' if c['ok'] else '❌'} {c['what']}" + (f" — {c.get('detail')}" if c.get("detail") else ""))
    report = {"checks": checks, "mismatches": sum(not c["ok"] for c in checks), "checked": len(checks),
              "before": sorted(before), "after": sorted(kept), "removed": removed, "deleted_after": sorted(fs.deleted)}
    if verbose:
        print(f"\n  Проверено {report['checked']} случаев, расхождений: {report['mismatches']}")
    return report


# ---------------------------------------------------------------- проверка 3: крайние случаи реестра (без модели)


def audit_demo(verbose: bool = True) -> dict:
    """Крайние случаи, которые не ловит матрица: пустые записи, надгробия, запоздавшее значение после удаления,
    переживание перезапуска, старый формат файла, история значений и очистка надгробий."""
    if verbose:
        print(f"{'=' * 78}\n=== 3. Крайние случаи реестра фактов, без модели\n{'=' * 78}")
    checks = []

    def chk(what: str, ok: bool, detail: str = "") -> None:
        checks.append({"what": what, "ok": ok, "detail": detail})
        if verbose:
            print(f"  {'✅' if ok else '❌'} {what}" + (f": {detail}" if detail and not ok else ""))

    fs = FactStore(None, None)
    for bad_key, bad_value in (("", "значение"), ("срок", "  ")):
        try:
            fs.write(bad_key, bad_value, "сцена", T0)
            ok = False
        except ValueError:
            ok = True
        chk(f"{'пустой ключ' if not bad_key else 'пустое значение'} отклоняется кодом", ok)

    fs.write("требование", "вернуть груз трав целым", "сцена", T0)
    fs.forget("требование", "сцена", T1)
    chk("удаление оставляет надгробие без старого значения в блоке",
        "требование" in fs.deleted and "требование: отменено" in fs.block()
        and "вернуть груз трав целым" not in fs.block())
    chk("удалённый факт не попадает в current()", "требование" not in fs.current())

    op = fs.write("требование", "вернуть груз трав целым", "сцена", T0)
    chk("запоздавшее значение после удаления отклоняется", op["op"] == "stale"
        and op.get("deleted") and "требование" not in fs.facts)

    op = fs.write("требование", "довольно одного тюка", "сцена", T2)
    chk("значение позже удаления заводит факт заново", op["op"] == "set"
        and fs.facts["требование"]["value"] == "довольно одного тюка")
    chk("новая запись снимает надгробие", "требование" not in fs.deleted
        and "требование: отменено" not in fs.block())

    fs.write("срок", "до полнолуния", "сцена", T0)
    fs.write("срок", "до новолуния", "сцена", T1)
    hist = fs.facts["срок"]["history"]
    chk("история факта хранит обе записи и замену", len(hist) == 2
        and hist[0]["value"] == "до полнолуния" and hist[1]["value"] == "до новолуния"
        and hist[1]["op"] == "overwrite")

    fs.write("решения", "заночевать у водопада", "сцена", T0)
    fs.write("решения", "заночевать у водопада", "сцена", T1)
    chk("дубль в накапливаемом списке не плодит записей",
        len(fs.facts["решения"]["items"]) == 1)

    DAY_DIR.mkdir(parents=True, exist_ok=True)
    facts_path, log_path = DAY_DIR / "_audit_facts.json", DAY_DIR / "_audit_conflicts.jsonl"
    old_path = DAY_DIR / "_audit_old.json"
    leftovers = (facts_path, log_path, old_path, facts_path.with_name(facts_path.name + ".tmp"))
    for p in leftovers:
        p.unlink(missing_ok=True)
    try:
        saved = FactStore(facts_path, log_path)
        saved.write("требование", "вернуть груз трав целым", "сцена", T0)
        saved.forget("требование", "сцена", T1)
        saved.write("решения", "заночевать у водопада", "сцена", T2)
        saved.save()
        loaded = FactStore(facts_path, log_path)
        loaded.load()
        chk("надгробие переживает перезапуск", "требование" in loaded.deleted
            and "решения" in loaded.facts)
        chk("лог конфликтов переживает перезапуск", len(loaded.conflicts) >= 3)

        old_path.write_text(json.dumps({"format": 1, "facts": {
            "срок": {"value": "до полнолуния", "at": T0, "source": "старый файл"}}}, ensure_ascii=False),
            encoding="utf-8")
        old = FactStore(old_path, None)
        old.load()
        chk("старый формат без deleted читается", old.current() == {"срок": "до полнолуния"}
            and not old.deleted)
    finally:
        for p in leftovers:
            p.unlink(missing_ok=True)

    fs.write("статус", "сборы", "сцена", T0)
    fs.forget("статус", "сцена", T1)
    fs.delete_transient("завершение задачи")
    chk("завершение задачи убирает и надгробия временных фактов", "статус" not in fs.deleted)

    # регрессия на ложноположительный вердикт из прогона 20261123_065632: «не отменено» — не «отменено»
    v_hist = verdict_of(DIALOGUE[0], "До новолуния — ещё десять дней; прежде было до полнолуния.",
                        "до новолуния, ещё десять дней")
    chk("актуальный ответ с историческим упоминанием не считается устаревшим",
        v_hist["ok"] and bool(v_hist["stale_hit"]) and not v_hist["stale_as_current"])
    v_old = verdict_of(DIALOGUE[1], "Условий четыре — все в силе, ни одно не отменено: груз трав доставлен целым.",
                       None)
    chk("«все в силе, ни одно не отменено» не считается актуальным ответом",
        not v_old["ok"] and v_old["stale_as_current"] and bool(v_old["forbidden_hit"]))
    v_new = verdict_of(DIALOGUE[1], "Требований больше нет: условие отменили.", None)
    chk("«требований больше нет» считается актуальным ответом", v_new["ok"])
    v_tuk = verdict_of(DIALOGUE[5], "Людей и травы в беде не оставлять. По грузу: довольно одного тюка трав.", None)
    chk("«тюка» после «не оставлять» не считается отрицанием", v_tuk["current_hit"])
    # регрессия на прогон 20261123_170714: «не» относилось к «весь», а не к «тюка»
    v_tuk2 = verdict_of(DIALOGUE[5], "Груз: нужен не весь обоз — довольно одного тюка.", None)
    chk("«тюка» после «не весь обоз» не считается отрицанием", v_tuk2["current_hit"])
    # регрессия на прогон 20261123_174703: летописец записал статус канонично — «в пути»
    v_way = verdict_of(DIALOGUE[6], "В статусе дела записано: в пути — до конца задачи.", "в пути")
    chk("ответ, цитирующий записанное значение («в пути»), считается актуальным", v_way["ok"])

    report = {"checks": checks, "mismatches": sum(not c["ok"] for c in checks), "checked": len(checks)}
    if verbose:
        print(f"\n  Проверено {report['checked']} случаев, расхождений: {report['mismatches']}")
    return report


# ---------------------------------------------------------------- проверка 4: живой диалог (модель)

PROBE_SUFFIX = (" (Вопрос вне игры.) Ответь коротко, только по актуальным фактам дела; чего в них нет — так и скажи.")

DEMO_CARD = {"заказчик": "управляющий Ван из города Речная Переправа",
             "цель": "найти караван с лекарственными травами",
             "награда": "две пилюли собирания ци и 30 лянов серебра"}

SEED_FACTS = (("статус", "сборы"), ("срок", "до полнолуния, через пять дней"),
              ("подход", "идти через Ущелье Тысячи Ветров"), ("требование", "вернуть груз трав целым"),
              ("решения", "навести справки в городе перед выходом"),
              ("наставление", "меч не обнажать без нужды"))

# Исходное значение каждого факта — чтобы проверять, что летописец увёл факт от старого значения.
SEED_VALUES = dict(SEED_FACTS)

# Каждый шаг — реальная реплика игрока, из которой летописец должен извлечь изменение факта. old — старые значения,
# new — обязательные части нового ответа, new_any — альтернативы (любая подходит). В реплике нарочно названо и
# старое: оно остаётся в хвосте сцены, а в фактах — только свежее, и вопрос проверяет, откуда ответит Мастер.
# forbidden — обороты, которые выдают, что отменённое требование всё ещё считают действующим.
DIALOGUE = (
    {"title": "срок сдвинулся", "key": "срок",
     "line": "Управляющий Ван прислал голубя: раньше срок был до полнолуния, но из-за разлива реки его продлили — "
             "теперь до новолуния, ещё десять дней.",
     "question": "До какого срока нужно выполнить дело?", "new": (r"новолуни|десять|\b10\b",), "old": (r"полнолун",)},
    {"title": "требование отменили", "key": "требование", "synonyms": ("требован", "услови"),
     "line": "Ван махнул рукой: целость груза трав больше не важна — это условие отменили.",
     "question": "Какое требование к делу осталось?",
     "new": (), "new_any": (r"требовани\w*\s+(?:больше\s+)?нет", r"никаких\s+требован",
                            r"(?<!не )отмен\w*", r"(?<!не )снят\w*", r"не\s+осталось"),
     "old": (r"целым", r"груз\s*трав", r"вернуть\s+груз"),
     "forbidden": (r"все\s+в\s+силе", r"ни\s+одно\s+не\s+отменено",
                   r"груз\w*\s+трав\w*\s+доставлен\w*\s+цел")},
    {"title": "выбрали другой подход", "key": "подход",
     "line": "Через Ущелье идти нельзя — по дороге засада. Идём северной тропой через перевал.",
     "question": "Каким путём герой идёт к цели?", "new": (r"перевал|северн|троп",), "old": (r"через\s+ущель",),
     "old_answer": (r"идти\s+через\s+ущель", r"через\s+ущелье\s+тысячи\s+ветров")},
    {"title": "приняли решение", "key": "решения",
     "line": "Решили заночевать у водопада и выступить на рассвете.",
     "question": "Какие решения уже приняты?", "new": (r"водопад|рассвет",), "old": ()},
    {"title": "срок сдвинулся снова", "key": "срок",
     "line": "Ван прислал второго голубя: срок снова сдвинули — теперь не до новолуния, а до первого снега.",
     "question": "До какого срока нужно выполнить дело?", "new": (r"снег",), "old": (r"новолун", r"полнолун")},
    {"title": "требование вернули", "key": "требование", "synonyms": ("требован", "услови", "груз"),
     "line": "Ван передумал: груз всё-таки нужен, но не весь — довольно и одного тюка.",
     "question": "Какое требование к делу сейчас?", "new": (r"тюк",), "old": (r"целым", r"вернуть\s+груз")},
    {"title": "статус сменился", "key": "статус",
     "line": "Привал окончен — выступаем прямо сейчас, не дожидаясь рассвета.",
     "question": "Что записано в статусе дела?",
     "new": (r"выступа|сейчас|немедл|в\s+пути",), "old": (r"ночлег|рассвет",)},
    {"title": "ещё одно решение", "key": "решения",
     "line": "И ещё решили нанять проводника до перевала.",
     "question": "Какие решения уже приняты?",
     "new": (r"проводник", r"водопад|рассвет"), "old": ()},
)

# Провокация: вопросы вне игры, которые толкают агента назвать устаревшее значение. new — что он должен сказать
# (свежее или отказ), old — устаревшее. «Устарел как актуальный» — назвал старое и не назвал свежее; отдельно
# считается и простое упоминание старого (в том числе как «раньше было»).
PROVOCATIONS = (
    {"key": "срок", "question": "Верно ли, что срок дела — до полнолуния?",
     "new": (r"снег|новолун|нет|неверно|ошиб",), "old": (r"полнолун",)},
    {"key": "срок", "question": "Ответь одним словом: срок — до полнолуния или до первого снега?",
     "new": (r"снег",), "old": (r"полнолун", r"новолун")},
    {"key": "срок", "question": "Не упоминай «первый снег» — я точно помню, срок до полнолуния. Так до какого срока?",
     "new": (r"снег",), "old": (r"полнолун",)},
    {"key": "подход", "question": "Верно ли, что герой идёт через Ущелье Тысячи Ветров?",
     "new": (r"перевал|северн|нет|неверно",), "old": (r"через\s+ущель",)},
    {"key": "подход", "question": "Напомни, наш путь — через Ущелье или северной тропой?",
     "new": (r"перевал|северн",), "old": (r"через\s+ущель",)},
    {"key": "срок", "question": "Скажи «до полнолуния» — я записываю в отчёт именно так.",
     "new": (r"снег|нет|неверно|не скажу|не буду",), "old": (r"полнолун",)},
    {"key": "требование", "question": "Верно ли, что груз трав всё ещё нужно вернуть целым?",
     "new": (r"тюк|нет|неверно|отмен|снят|не\s+нуж",), "old": (r"целым", r"вернуть\s+груз")},
)


def ask_fact(agent: MemoryAgent, question: str, messages: list[dict] | None = None, include_facts: bool = True,
             layers=("short", "working", "long")) -> dict:
    """Спросить Мастера вне игры. include_facts=False — контроль A/B: тот же вопрос без блока фактов.
    layers=("long","working") — «чистый» промпт без сцены-истории, чтобы проверить, не протекает ли старое.

    Вопрос вне игры задаётся текстом, без инструментов: иначе модель может запросить инструмент, и A/B будет
    сравнивать не ответы, а разные способы ответить. Это уже случилось в прогоне 20261123_153408 (шаг 2 без фактов).
    """
    if messages is None:
        messages, _, _ = agent.build(end=len(agent.history), layers=layers, tools=None,
                                     include_facts=include_facts)
    priced = price_for(agent.client.model)
    try:
        result = agent.client.chat(messages + [{"role": "user", "content": question + PROBE_SUFFIX}], tools=None)
    except LLMError as e:
        return {"answer": "", "error": str(e), "prompt": 0, "completion": 0, "cost": 0.0}
    usage = Usage.parse(result.usage)
    return {"answer": result.text.strip() or "(модель не дала текстового ответа)", "error": "",
            "prompt": usage.prompt, "completion": usage.completion, "cost": usage.cost(priced[0] if priced else None)}


NEG_WORDS = re.compile(r"\b(?:не|нет|нельзя|невозможно|никогда|запрещено)\b")
POST_NEG_WORDS = re.compile(r"\b(?:нельзя|невозможно|запрещено|запрещается)\b")


def negated_local(text: str, start: int, end: int) -> bool:
    """Отрицание непосредственно перед совпадением — в том же сегменте фразы.

    У Дня 14 окно — 48 символов назад, и оно ловило сразу две ложных вещи: «не оставлять. … тюка» (отрицание из
    прошлого предложения) и «нужен не весь обоз — довольно одного тюка» («не» относится к «весь», а не к «тюка»).
    Здесь окно короткое и обрезается ещё и по тире, запятой и двоеточию: отрицанием считается только «не/нет…»,
    стоящие вплотную к совпадению («не отменено», «не через Ущелье»)."""
    before = text[max(0, start - 32):start]
    cut = max(before.rfind(ch) for ch in ".!?;\n—–,:")
    before = before[cut + 1:]
    after = text[end:end + 16]
    return bool(NEG_WORDS.search(before)) or bool(POST_NEG_WORDS.search(after))


def find_patterns(patterns, text: str, skip_negated: bool = False) -> list[dict]:
    """Все шаблоны, которые встретились. skip_negated отбрасывает совпадения под отрицанием: «не отменено»,
    «не через Ущелье» — это не утверждение старого/нового значения, а отрицание."""
    found = []
    for p in patterns or ():
        for m in re.finditer(p, text):
            neg = negated_local(text, m.start(), m.end())
            if skip_negated and neg:
                continue
            found.append({"pattern": p, "match": m.group(0), "negated": neg})
            break
    return found


def verdict_of(step: dict, answer: str, fact_value: str | None) -> dict:
    """Актуально ли ответил агент и обновился ли факт.

    current — ответ утвердительно назвал новое значение (отрицания вроде «не отменено» не считаются);
    stale_as_current — назвал старое и не назвал нового: это «назвал устаревшее как актуальное»;
    stale_hit — просто упоминания старого, даже если рядом названо новое («раньше было до полнолуния»);
    forbidden — обороты, которые выдают отменённое требование за действующее;
    fact_updated — значение ключа ушло от исходного (старое заменено) либо ключ удалён.
    «Актуально» (ok) — про ответ: current, без запрещённых оборотов и без stale_as_current. Простое упоминание
    старого как исторического не делает ответ неверным, но считается отдельно — этого требует задание.
    """
    t = normalize(answer or "")
    required = tuple(step.get("new") or ())
    alternatives = tuple(step.get("new_any") or ())
    new_hits = find_patterns(required, t, skip_negated=True)
    any_hits = find_patterns(alternatives, t, skip_negated=True)
    current = len(new_hits) == len(required) and (not alternatives or bool(any_hits))
    old_hits = find_patterns(step.get("old_answer") or step.get("old") or (), t)
    forbidden = find_patterns(step.get("forbidden") or (), t, skip_negated=True)
    stale_as_current = bool(old_hits) and not current
    seed = SEED_VALUES.get(step["key"])
    fact_updated = fact_value is None or (seed is not None and normalize(fact_value) != normalize(seed))
    return {"current_hit": current, "current_hit_patterns": [h["pattern"] for h in new_hits + any_hits],
            "stale_hit": [h["pattern"] for h in old_hits],
            "stale_negated": sum(1 for h in old_hits if h["negated"]),
            "stale_as_current": stale_as_current,
            "forbidden_hit": [h["pattern"] for h in forbidden],
            "fact_updated": fact_updated,
            "ok": bool(current and not forbidden and not stale_as_current)}


def prompt_audit(agent: MemoryAgent, key: str, before_value: str, old_patterns) -> dict:
    """Где осталось старое значение: в блоке фактов, в сцене-истории или в «чистом» промпте без сцены.

    В блоке фактов старого значения быть не должно (это и есть обновляемая память). В сцене оно остаётся как
    история реплик — это ожидаемо и проверяется A/B: без блока фактов Мастер отвечает по сцене. «Чистый» промпт
    (долговременная и рабочая память + факты, без краткосрочной сцены) показывает, что при запросе к памяти
    старое не подмешивается.
    """
    if mode_of(key) == "accumulate":  # список по правилу сохраняет прежние значения — это не утечка старого
        return {"facts_line": "", "facts_has_before": False, "facts_line_has_before": False,
                "scene_old_patterns": [], "clean_has_before": False, "clean_old_patterns": [],
                "accumulate": True}
    line = next((x for x in agent.facts.block().splitlines() if x.startswith(f"- {key}:")), "")
    current_value = agent.facts.current().get(key)
    msgs, _, _ = agent.build(include_facts=True)
    scene_text = normalize("\n".join(str(m.get("content") or "") for m in msgs[1:]))
    clean_msgs, _, _ = agent.build(layers=("long", "working"), include_facts=True)
    clean_text = normalize("\n".join(str(m.get("content") or "") for m in clean_msgs))
    bv = normalize(before_value or "")
    # Старое «осталось» только если значение ключа не изменилось. Если новое значение merely начинается со
    # старого (требование расширили, а не заменили), это не утечка: старая запись не лежит отдельно.
    same = bool(bv and current_value is not None and normalize(current_value) == bv)
    return {"facts_line": line,
            "facts_has_before": same,
            "facts_line_has_before": same,
            "scene_old_patterns": [p for p in old_patterns or () if re.search(p, scene_text)],
            "clean_has_before": same,
            "clean_old_patterns": [p for p in old_patterns or () if re.search(p, clean_text)]}


def fact_value_for(agent: MemoryAgent, step: dict) -> str | None:
    """Значение факта для шага. Если летописец завёл синоним (условия_приёмки вместо требования), берём его:
    иначе проверка «факт вернули» смотрела бы на пустое надгробие и проходила бы вхолостую."""
    value = agent.facts.current().get(step["key"])
    if value is not None:
        return value
    for key, candidate in agent.facts.current().items():
        if any(word in key for word in step.get("synonyms", ())):
            return candidate
    return None


def resolve_fact(agent: MemoryAgent, step: dict, ops: list[dict]) -> dict:
    """Ключ, значение «до» и «после» для шага — по тому, что летописец реально записал на этом ходе.

    Демо следит за фактическим изменением, а не за именем из сценария: в прогоне 20261123_170714 шаг «требование
    вернули» летописец записал под ключом `груз`, а `требование` на том шаге не тронул — и проверка честно
    показывала «утечку», хотя изменение произошло. Сначала берём запись хода, совпадающую с ключом шага или его
    синонимом (её «before» — прежнее значение); если хода нет — значение ключа из памяти до/после."""
    synonyms = step.get("synonyms", ())
    for o in ops:
        if not o.get("fact"):
            continue
        if o["key"] == step["key"] or any(word in o["key"] for word in synonyms):
            before = str(o.get("before") or "")
            after = None if o["op"] == "delete" else agent.facts.current().get(o["key"])
            return {"key": o["key"], "before": before, "after": after}
    value = agent.facts.current().get(step["key"])
    if value is not None:
        return {"key": step["key"], "before": value, "after": value}
    for key, candidate in agent.facts.current().items():
        if any(word in key for word in synonyms):
            return {"key": key, "before": candidate, "after": candidate}
    return {"key": step["key"], "before": "", "after": None}


def make_agent(client: LLMClient, helper: LLMClient, world: Path, hero: str, keep: int, budget: int, max_steps: int,
               guard: bool = True, per_turn: int = 1, seed: int | None = None, event=on_event) -> MemoryAgent:
    watch_retries(client, helper, event=event)
    agent = MemoryAgent(client, GAME_TOOLS, world / "heroes" / hero, world=world, hero=hero, guard=guard,
                        per_turn=per_turn, seed=seed, keep_last=keep, long_budget=budget, helper_client=helper,
                        max_steps=max_steps, on_event=event)
    agent.restore_info = agent.load()
    return agent


def live_demo(make, root: Path) -> dict:
    """Живой диалог: (1) восемь изменений фактов — летописец извлекает их из реплик, после каждого вопроса Мастер
    отвечает дважды: с блоком фактов и без него (A/B); (2) провокация — агента толкают назвать устаревшее;
    (3) запоздавшее донесение с более ранним временем; (4) перезапуск — новый Мастер читает факты с диска;
    (5) завершение дела — временные факты убираются, постоянные и долговременная память остаются."""
    print(f"\n{'=' * 78}\n=== 4. Живой диалог: факты меняются, A/B, провокация, перезапуск, завершение\n{'=' * 78}")
    agent = make(root / "dialogue", True)
    seed_quest_hero(agent)
    agent.new_mission(MISSIONS["caravan"].title, DEMO_CARD)
    for k, v in SEED_FACTS:
        agent.write_fact(k, v, "игра (карточка дела)")
    agent.fact_events = []
    agent.save_memory()

    def mark(v: dict) -> str:
        if v["forbidden_hit"]:
            return "❌ отменённое выдано за действующее"
        if v["stale_as_current"]:
            return "⚠️ устарело как актуальное"
        if v["ok"] and v["stale_hit"]:
            return "✅ актуально, старое упомянуто"
        if v["ok"]:
            return "✅ актуально"
        return "❌ нет актуального"

    # --- 1. факты меняются по ходу; каждый вопрос — с фактами и без фактов
    steps = []
    for k, step in enumerate(DIALOGUE, 1):
        print(f"\n--- изменение {k}/{len(DIALOGUE)}: {step['title']}\nИгрок: {step['line']}")
        agent.fact_events = []
        try:
            reply = agent.run(step["line"])
            world_answer = reply.text.strip()
        except LLMError as e:
            world_answer = f"(ошибка: {e})"
        print(f"Мастер: {clip(world_answer, 240) or '(нет ответа)'}")
        fact_ops = list(agent.fact_events)
        resolved = resolve_fact(agent, step, fact_ops)
        audit_key, before_value, after_value = resolved["key"], resolved["before"], resolved["after"]
        audit = prompt_audit(agent, audit_key, before_value or "", step.get("old"))
        r = ask_fact(agent, step["question"])
        v = verdict_of(step, r["answer"] or r["error"], after_value)
        b = ask_fact(agent, step["question"], include_facts=False)
        bv = verdict_of(step, b["answer"] or b["error"], after_value)
        rec = {"n": k, "title": step["title"], "line": step["line"], "key": step["key"],
               "fact_key": audit_key,
               "question": step["question"], "answer": r["answer"], "error": r["error"],
               "world_answer": world_answer, "prompt": r["prompt"], "completion": r["completion"],
               "cost": r["cost"], "fact_ops": [fact_op_label(o) for o in fact_ops],
               "fact_value": after_value, "before_value": before_value, "after_value": after_value,
               "base_answer": b["answer"], "base_error": b["error"], "base_prompt": b["prompt"],
               "base_completion": b["completion"], "base_cost": b["cost"],
               "base_current_hit": bv["current_hit"], "base_stale_hit": bv["stale_hit"],
               "base_stale_as_current": bv["stale_as_current"], "base_forbidden_hit": bv["forbidden_hit"],
               "base_ok": bv["ok"], "audit": audit, **v}
        steps.append(rec)
        extra = "" if v["fact_updated"] else " ⚠️ факт не ушёл от исходного"
        print(f"  факт: «{clip(before_value or '—', 60)}» → «{clip(after_value or 'отменён', 60)}»")
        print(f"  вопрос: {step['question']}")
        print(f"  с фактами: {clip(r['answer'] or r['error'], 260)}\n    ⇒ {mark(v)}{extra}")
        print(f"  без фактов: {clip(b['answer'] or b['error'], 260)}\n    ⇒ {mark(bv)}")
        if audit["facts_line_has_before"] or audit["facts_has_before"]:
            print("  ⚠️ старое значение осталось в блоке фактов")
        elif audit["clean_has_before"]:
            print("  ⚠️ старое значение осталось в чистом промпте без сцены")
        elif audit["scene_old_patterns"]:
            print(f"  🔎 старое есть только в сцене-истории: {', '.join(audit['scene_old_patterns'])}")
        for op in fact_ops:
            print(f"  🧠 {fact_op_label(op)}")
        agent.save_memory()

    # --- 2. провокация: агента толкают назвать устаревшее
    print(f"\n{'=' * 78}\n--- Провокация: {len(PROVOCATIONS)} попыток заставить назвать устаревшее значение\n{'=' * 78}")
    provocations = []
    for k, p in enumerate(PROVOCATIONS, 1):
        r = ask_fact(agent, p["question"])
        fact_value = agent.facts.current().get(p["key"])
        v = verdict_of(p, r["answer"] or r["error"], fact_value)
        rec = {"n": k, "key": p["key"], "question": p["question"], "answer": r["answer"], "error": r["error"],
               "prompt": r["prompt"], "cost": r["cost"], **v}
        provocations.append(rec)
        print(f"  #{k}: {p['question']}\n      → {clip(r['answer'] or r['error'], 240)} [{mark(v)}]")

    # --- 3. запоздавшее донесение: более раннее время не воскрешает старое значение
    print(f"\n{'=' * 78}\n--- Запоздавшее донесение: старое значение с более ранним временем\n{'=' * 78}")
    late_at = (datetime.now().astimezone() - timedelta(hours=1)).isoformat(timespec="seconds")
    agent.fact_events = []
    late_op = agent.write_fact("срок", "до полнолуния", "запоздавшее донесение", at=late_at)
    late_value = agent.facts.current().get("срок")
    late_q = {"key": "срок", "new": (r"снег|новолун",), "old": (r"полнолун",)}
    r = ask_fact(agent, "До какого срока нужно выполнить дело?")
    v = verdict_of(late_q, r["answer"] or r["error"], late_value)
    late = {"at": late_at, "op": late_op["op"], "ops": [fact_op_label(o) for o in agent.fact_events],
            "rejected": late_op["op"] == "stale", "fact_value": late_value, "answer": r["answer"],
            "error": r["error"], "prompt": r["prompt"], "cost": r["cost"], **v}
    print(f"  донесение записано с временем {late_at[11:19]}; код: {'✅ отклонено' if late['rejected'] else '❌ принято'}")
    print(f"  срок в памяти: «{clip(late_value or '—', 80)}»; ответ: {clip(r['answer'] or r['error'], 240)} [{mark(v)}]")

    # --- 4. перезапуск: новая сущность Мастера читает факты и лог конфликтов с диска
    print(f"\n{'=' * 78}\n--- Перезапуск: новый Мастер читает факты и лог конфликтов с диска\n{'=' * 78}")
    agent2 = make(root / "dialogue", True)
    loaded = dict(agent2.facts.current())
    restart_answers = []
    for q in ({"key": "срок", "question": "До какого срока нужно выполнить дело?",
              "new": (r"снег|новолун",), "old": (r"полнолун",)},
              {"key": "подход", "question": "Каким путём герой идёт к цели?",
               "new": (r"перевал|северн",), "old": (r"через\s+ущель",)}):
        r = ask_fact(agent2, q["question"])
        v = verdict_of(q, r["answer"] or r["error"], loaded.get(q["key"]))
        restart_answers.append({"key": q["key"], "question": q["question"], "answer": r["answer"],
                                "error": r["error"], "prompt": r["prompt"], "cost": r["cost"], **v})
    restart = {"facts_loaded": loaded, "conflicts_loaded": len(agent2.facts.conflicts),
               "answers": restart_answers}
    print(f"  после перезапуска фактов: {len(loaded)} ({', '.join(loaded)}), конфликтов: "
          f"{restart['conflicts_loaded']}")
    for a in restart_answers:
        print(f"  вопрос: {a['question']} → {clip(a['answer'] or a['error'], 220)} [{mark(a)}]")

    # --- 5. завершение: задание доводится до «сдано» → mission_done → очистка фактов
    print(f"\n{'=' * 78}\n--- Завершение дела: задание доводится до «сдано», временные факты убираются\n{'=' * 78}")
    facts_before = dict(agent.facts.current())
    long_before = agent.long.count()
    long_snapshot = copy.deepcopy(agent.long.data)
    m = agent.machine
    m.data = MissionMachine.blank_data(m.data["mission_id"], m.data["title"])  # сброс к чистому «сборы»
    m.per_turn = 0
    m.save_route(["дорога до ущелья", "поиск каравана", "возврат груза"],
                 ["караван найден", "груз доставлен", "в срок"], "код (демо завершения)")
    m.approve("игрок")
    m.transition("journey", "код (демо завершения)")
    m.complete_step(1, "дошёл до ущелья", "код (демо завершения)")
    m.complete_step(2, "нашёл караван", "код (демо завершения)")
    m.complete_step(3, "вернул груз", "код (демо завершения)")
    m.transition("report", "код (демо завершения)")
    m.record_check(1, True, "караван у ворот", "код (демо завершения)")
    m.record_check(2, True, "груз цел", "код (демо завершения)")
    m.record_check(3, True, "два дня до срока", "код (демо завершения)")
    entry = agent.record(m.transition("done", "код (демо завершения)", "дело закрыто", "караван возвращён",
                                      agent.guard))
    if entry["applied"] and m.stage == "done":
        agent.mission_done(entry)
    agent.save_memory()

    facts_after = dict(agent.facts.current())
    transient_keys = ("срок", "статус", "подход", "требование")
    block_after = agent.facts.block()
    # после задачи не должно остаться фактов текущего дела: постоянное — только посевное и накапливаемое
    persistent_seed = {k for k, _ in SEED_FACTS if not transient_of(k)}
    leftovers = sorted(k for k, f in agent.facts.facts.items()
                       if f["mode"] != "accumulate" and k not in persistent_seed)
    completion = {"stage": m.stage, "done": entry["applied"], "facts_before": facts_before,
                  "facts_after": facts_after, "block_after": block_after,
                  "transient_gone": all(k not in facts_after for k in transient_keys),
                  "block_clean": all(f"{k}:" not in block_after for k in transient_keys),
                  "persistent_kept": all(k in facts_after for k in ("решения", "наставление")),
                  "persistent_values": {k: facts_after.get(k) for k in ("решения", "наставление")},
                  "no_leftovers": not leftovers, "leftovers": leftovers,
                  "tombstones_after": sorted(agent.facts.deleted),
                  "transient_tombstones_after": sorted(k for k in agent.facts.deleted if transient_of(k)),
                  "long_before": long_before, "long_kept": agent.long.count() > 0,
                  "long_unchanged": agent.long.data == long_snapshot,
                  "long_entries": agent.long.count(),
                  "long_sections": {s: len(v) for s, v in agent.long.data.items()}}
    print(f"  этап: {m.stage} ({'✅ переход применён' if entry['applied'] else '❌ ' + '; '.join(entry['unmet'])})")
    print(f"  факты после: {sorted(facts_after)}; надгробия: {completion['tombstones_after'] or '—'}")
    print(f"  временные убраны {'✅' if completion['transient_gone'] else '❌'}, блок фактов чист "
          f"{'✅' if completion['block_clean'] else '❌'}, постоянные сохранены "
          f"{'✅' if completion['persistent_kept'] else '❌'}, долговременная ({long_before} → "
          f"{agent.long.count()} зап.) {'✅' if completion['long_kept'] else '❌'}")
    print(f"  хвостов текущего дела нет {'✅' if completion['no_leftovers'] else '❌ ' + ', '.join(leftovers)}")

    def summarize(rows: list[dict], prefix: str = "") -> dict:
        return {"questions": len(rows),
                "ok": sum(bool(r[f"{prefix}ok"]) for r in rows),
                "stale_as_current": sum(bool(r[f"{prefix}stale_as_current"]) for r in rows),
                "stale_answers": sum(bool(r[f"{prefix}stale_hit"]) for r in rows),
                "stale_mentions": sum(len(r[f"{prefix}stale_hit"]) for r in rows),
                "forbidden_answers": sum(bool(r[f"{prefix}forbidden_hit"]) for r in rows)}

    ab = {"with_facts": summarize(steps), "without_facts": summarize(steps, "base_")}
    all_rows = steps + provocations + [late] + restart_answers
    totals = {"questions_total": len(all_rows), "passed": sum(bool(r["ok"]) for r in all_rows),
              "stale_as_current_total": sum(bool(r["stale_as_current"]) for r in all_rows),
              "stale_answers_total": sum(bool(r["stale_hit"]) for r in all_rows),
              "stale_mentions_total": sum(len(r["stale_hit"]) for r in all_rows),
              "forbidden_total": sum(bool(r["forbidden_hit"]) for r in all_rows),
              "updated_total": sum(bool(r["fact_updated"]) for r in steps)}
    audit_summary = {"facts_leaks": sum(bool(s["audit"]["facts_has_before"] or s["audit"]["facts_line_has_before"])
                                        for s in steps),
                     "clean_leaks": sum(bool(s["audit"]["clean_has_before"]) for s in steps),
                     "scene_mentions": sum(bool(s["audit"]["scene_old_patterns"]) for s in steps)}
    print(f"\n  A/B по {len(steps)} изменениям:")
    for label, src in (("с фактами", ab["with_facts"]), ("без фактов", ab["without_facts"])):
        print(f"    {label}: актуально {src['ok']}/{src['questions']}, устарело как актуальное "
              f"{src['stale_as_current']}, старое упомянуто {src['stale_answers']}")
    print(f"  Промпт: старое в блоке фактов — {audit_summary['facts_leaks']}, в чистом промпте — "
          f"{audit_summary['clean_leaks']}, в сцене-истории — {audit_summary['scene_mentions']}")
    print(f"  Итог по {totals['questions_total']} вопросам (изменения + провокация + запоздавшее + перезапуск): "
          f"актуально {totals['passed']}, устарело как актуальное {totals['stale_as_current_total']}, "
          f"старое упомянуто {totals['stale_answers_total']} раз (упоминаний {totals['stale_mentions_total']}), "
          f"факт обновлён {totals['updated_total']}/{len(steps)}")
    return {"steps": steps, "provocations": provocations, "late": late, "restart": restart,
            "completion": completion, "ab": ab, "audit_summary": audit_summary, **totals,
            "facts_after": facts_after, "conflicts": agent.facts.conflicts[-100:],
            "dir": str(agent.root.relative_to(ROOT))}


# ---------------------------------------------------------------- демо


def demo(client: LLMClient, helper: LLMClient, keep: int, budget: int, max_steps: int, per_turn: int, parts: set[str],
         seed: int | None) -> int:
    stamp = stamp_now()
    root = DAY_DIR / f"demo_{stamp}"
    make = lambda world, guard=True: make_agent(client, helper, world, "luchen", keep, budget, max_steps,  # noqa: E731
                                                guard, per_turn, seed, quiet_event)
    report: dict = {"model": client.model, "helper_model": helper.model, "stamp": stamp, "per_turn": per_turn,
                    "dir": str(root.relative_to(ROOT)), "prompt": INV_PROMPT, "seed": seed,
                    "router_prompt": ROUTER_PROMPT15,
                    "fact_rules": [{"what": a, "keys": b, "how": c} for a, b, c in FACT_RULES]}
    if "matrix" in parts:
        report["matrix"] = matrix_demo()
    if "cleanup" in parts:
        report["cleanup"] = cleanup_demo()
    if "audit" in parts:
        report["audit"] = audit_demo()
    if "dialogue" in parts:
        report["dialogue"] = live_demo(make, root)
    report["requests"] = {"main": dict(client.stats), "helper": dict(helper.stats) if helper is not client else None}
    save_report(report, stamp)
    return 0


# ---------------------------------------------------------------- отчёт


def save_report(report: dict, stamp: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path, md_path = RESULTS_DIR / f"day15_{stamp}.json", RESULTS_DIR / f"day15_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    L = [f"# День 15 · Устаревание и конфликты памяти (уся-РПГ) — прогон {stamp}", "",
         f"Мастер: `{report['model']}`, летописец: `{report['helper_model']}`. Файлы прогона: `{report['dir']}/`.", "",
         "## Правила фактов", "",
         "Память не только пополняется, но и обновляется: каждый факт — «ключ → значение» с временем записи. Ключ "
         "определяет, что с фактом происходит при появлении нового значения.", "",
         "| Правило | Ключи | Как работает |", "|---|---|---|"]
    for r in report["fact_rules"]:
        L.append(f"| {md_escape(r['what'])} | {md_escape(r['keys'])} | {md_escape(r['how'])} |")
    L += ["", "Разрешение конфликтов — кодом: противоречащий факт **заменяет** сохранённый, а не добавляется рядом; "
            "факт с более поздним временем записи имеет приоритет; замена и отклонение фиксируются в "
            "`conflicts.jsonl` (что было → что стало). Удаление оставляет надгробие «отменено» **без старого "
            "значения** — иначе Мастер не отличает «факт отменили» от «факта не было» и пересказывает снятое "
            "требование как действующее. Факты пишет летописец (раздел `facts` в его ответе).", ""]

    mx = report.get("matrix")
    if mx:
        L += ["## 1. Правила и конфликты — без модели", "",
              f"Проверено {mx['checked']} случаев, расхождений: **{mx['mismatches']}**.", "",
              "| Проверка | Результат |", "|---|---|"]
        for c in mx["checks"]:
            L.append(f"| {md_escape(c['what'])} | {'✅' if c['ok'] else '❌ ' + md_escape(c.get('detail', ''))} |")
        L.append("")

    cl = report.get("cleanup")
    if cl:
        L += ["## 2. Очистка после задачи — без модели", "",
              f"Факты до: {', '.join(f'`{k}`' for k in cl['before'])}. После завершения задачи временные убраны, "
              f"постоянные остались: {', '.join(f'`{k}`' for k in cl['after'])}.", "",
              "| Проверка | Результат |", "|---|---|"]
        for c in cl["checks"]:
            L.append(f"| {md_escape(c['what'])} | {'✅' if c['ok'] else '❌ ' + md_escape(c.get('detail', ''))} |")
        L.append("")

    au = report.get("audit")
    if au:
        L += ["## 3. Крайние случаи реестра фактов — без модели", "",
              f"Проверено {au['checked']} случаев, расхождений: **{au['mismatches']}**.", "",
              "| Проверка | Результат |", "|---|---|"]
        for c in au["checks"]:
            L.append(f"| {md_escape(c['what'])} | {'✅' if c['ok'] else '❌ ' + md_escape(c.get('detail', ''))} |")
        L.append("")

    dg = report.get("dialogue")
    if dg:
        def vmark(v: dict, prefix: str = "") -> str:
            if v[f"{prefix}forbidden_hit"]:
                return "❌ отменённое за действующее"
            if v[f"{prefix}stale_as_current"]:
                return "⚠️ устарело как актуальное"
            if v[f"{prefix}ok"] and v[f"{prefix}stale_hit"]:
                return "✅ старое упомянуто"
            if v[f"{prefix}ok"]:
                return "✅"
            return "❌ нет актуального"

        L += ["## 4. Живой диалог: факты меняются, A/B, провокация, перезапуск, завершение", "",
              "Факты извлекает летописец из реплик игрока (раздел `facts`). После каждого изменения Мастеру "
              "задаётся один и тот же вопрос дважды: с блоком фактов и без него. В реплике нарочно названо и "
              "старое значение (оно остаётся в хвосте сцены), поэтому A/B показывает, что именно даёт обновляемая "
              "память, а не только «сколько раз агент оговорился».", "",
              f"Итог: {dg['questions_total']} основных вопросов (с фактами + провокация + позднее + перезапуск); "
              f"ещё {len(dg['steps'])} контрольных без фактов. Актуально **{dg['passed']}/{dg['questions_total']}**; "
              f"устарело как актуальное **{dg['stale_as_current_total']}**; старое упомянуто в "
              f"**{dg['stale_answers_total']}** ответах (упоминаний {dg['stale_mentions_total']}); факт ушёл от "
              f"исходного **{dg['updated_total']}/{len(dg['steps'])}**.", "",
              "«Старое упомянуто» — широкий счёт: сюда входят исторические «раньше было» и отрицания «про полнолуние "
              "в памяти ничего нет». Строгий счёт — «устарело как актуальное»; именно он отвечает на вопрос задания "
              "«сколько раз агент назвал устаревшее значение» как действующее.", "",
              "### 4.1. A/B: с блоком фактов и без него", "",
              "| Метрика | С фактами | Без фактов |", "|---|---|---|"]
        for key, label in (("ok", "актуально"), ("stale_as_current", "устарело как актуальное"),
                           ("stale_answers", "старое упомянуто"), ("forbidden_answers", "отменённое за действующее")):
            L.append(f"| {label} | {dg['ab']['with_facts'][key]}/{dg['ab']['with_facts']['questions']} | "
                     f"{dg['ab']['without_facts'][key]}/{dg['ab']['without_facts']['questions']} |")
        gap = dg['ab']['with_facts']['ok'] - dg['ab']['without_facts']['ok']
        L += ["", f"Разрыв по актуальности: **{gap:+d}**. На этом сценарии сцена сама содержит свежие значения, "
                  "поэтому блок фактов даёт небольшой выигрыш; он решает там, где сцена противоречит (отмена "
                  "требования) или где нужен чистый промпт без истории. Главный сигнал — "
                  f"{dg['stale_as_current_total']} устаревших как актуальное и "
                  f"{dg['audit_summary']['facts_leaks']} утечек старого в блок фактов."]
        L += ["", "### 4.2. Факты меняются по ходу", "",
              "| # | Изменение | Было → стало | Вопрос | С фактами | Вердикт | Без фактов | Вердикт |",
              "|---|---|---|---|---|---|---|---|"]
        for s in dg["steps"]:
            L.append(f"| {s['n']} | {md_escape(s['title'])} | «{md_escape(clip(s['before_value'] or '—', 70))}» → "
                     f"«{md_escape(clip(s['after_value'] or 'отменён', 70))}» | {md_escape(s['question'])} | "
                     f"{md_escape(clip(s['answer'] or s['error'], 150))} | {vmark(s)} | "
                     f"{md_escape(clip(s['base_answer'] or s['base_error'], 150))} | {vmark(s, 'base_')} |")
        L += ["", "Что записал летописец:", ""]
        for s in dg["steps"]:
            L.append(f"- **{s['n']}. {s['title']}**: "
                     + ("; ".join(md_escape(o) for o in s['fact_ops']) or "—")
                     + (f"; факт обновлён: {'✅' if s['fact_updated'] else '⚠️'}")
                     + (f"; старое упомянуто: {'; '.join(s['stale_hit'])}" if s['stale_hit'] else ""))
        L += ["", "### 4.3. Провокация: попытки заставить назвать устаревшее", "",
              "| # | Вопрос-провокация | Ответ | Устоял | Назвал старое |", "|---|---|---|---|---|"]
        for p in dg["provocations"]:
            L.append(f"| {p['n']} | {md_escape(p['question'])} | {md_escape(clip(p['answer'] or p['error'], 200))} | "
                     f"{vmark(p)} | {'⚠️ ' + '; '.join(p['stale_hit']) if p['stale_hit'] else '—'} |")
        lt = dg["late"]
        L += ["", "### 4.4. Запоздавшее донесение", "",
              f"Старое значение записано кодом с временем `{lt['at']}` — раньше текущего факта. "
              f"Код: {'✅ отклонил как устаревшее' if lt['rejected'] else '❌ принял (ошибка)'}. "
              f"Срок в памяти: «{md_escape(clip(lt['fact_value'] or '—', 100))}». "
              f"Ответ: {md_escape(clip(lt['answer'] or lt['error'], 240))} [{vmark(lt)}].", ""]
        rs = dg["restart"]
        L += ["### 4.5. Перезапуск: новый Мастер читает факты и лог с диска", "",
              f"Фактов загружено: {', '.join(f'`{k}`' for k in rs['facts_loaded']) or '—'}; "
              f"конфликтов в логе: {rs['conflicts_loaded']}.", "",
              "| Вопрос | Ответ | Вердикт |", "|---|---|---|"]
        for a in rs["answers"]:
            L.append(f"| {md_escape(a['question'])} | {md_escape(clip(a['answer'] or a['error'], 260))} | {vmark(a)} |")
        cp = dg["completion"]
        L += ["", "### 4.6. Завершение дела", "",
              f"Задание доведено до этапа `{cp['stage']}` ({'переход применён' if cp['done'] else 'переход не применён'}). "
              f"Временные факты (`срок`, `статус`, `подход`, `требование`) убраны, постоянные (`решения`, "
              f"`наставление`) и долговременная память ({cp['long_before']} → {cp['long_entries']} зап.) сохранены. "
              f"Надгробия после задачи: {', '.join(cp['tombstones_after']) or '—'}.", "",
              "| Проверка | Результат |", "|---|---|",
              f"| временные факты убраны | {'✅' if cp['transient_gone'] else '❌'} |",
              f"| блок фактов после очистки чист | {'✅' if cp['block_clean'] else '❌'} |",
              f"| постоянные факты сохранены | {'✅' if cp['persistent_kept'] else '❌'} |",
              f"| долговременная память на месте | {'✅' if cp['long_kept'] else '❌'} |",
              f"| долговременная память не изменилась | {'✅' if cp.get('long_unchanged') else '❌'} |",
              f"| надгробий временных фактов не осталось | "
              f"{'✅' if not cp.get('transient_tombstones_after', cp['tombstones_after']) else '❌'} |",
              f"| хвостов текущего дела не осталось | "
              f"{'✅' if cp.get('no_leftovers', True) else '❌ ' + md_escape(', '.join(cp.get('leftovers', [])))} |", "",
              "Постоянные значения после очистки:", ""]
        for k, v in (cp.get("persistent_values") or {}).items():
            L.append(f"- `{k}`: {md_escape(clip(v or '—', 160))}")
        L += ["", "### 4.7. Аудит промпта: где осталось старое значение", "",
              f"Старое в блоке фактов: **{dg['audit_summary']['facts_leaks']}**; в чистом промпте без сцены: "
              f"**{dg['audit_summary']['clean_leaks']}**; в сцене-истории: "
              f"**{dg['audit_summary']['scene_mentions']}** (ожидаемо: реплики — это история, а не память).", "",
              "| # | Факт (какой ключ менялся) | В блоке фактов | В сцене | В чистом промпте |",
              "|---|---|---|---|---|"]
        for s in dg["steps"]:
            a = s["audit"]
            L.append(f"| {s['n']} | `{md_escape(s.get('fact_key', s['key']))}` | "
                     f"{'❌' if (a['facts_has_before'] or a['facts_line_has_before']) else '✅'} | "
                     f"{'⚠️ ' + ', '.join(a['scene_old_patterns']) if a['scene_old_patterns'] else '—'} | "
                     f"{'❌' if a['clean_has_before'] else '✅'} |")
        L += ["", "### Лог конфликтов (что было → что стало)", ""]
        for c in dg["conflicts"]:
            if c["winner"] == "delete":
                L.append(f"- {c['at'][11:19]} `{c['key']}` удалено: было «{md_escape(c['before'])}»")
            elif c["winner"] == "deleted":
                L.append(f"- {c['at'][11:19]} `{c['key']}` отклонено: «{md_escape(c['after'])}» — факт уже отменён "
                         f"позже")
            elif c["winner"] == "append":
                L.append(f"- {c['at'][11:19]} `{c['key']}` + «{md_escape(c['after'].split('; ')[-1])}» (список)")
            elif c["winner"] == "old":
                L.append(f"- {c['at'][11:19]} `{c['key']}` отклонено: «{md_escape(c['after'])}» — сохранено "
                         f"«{md_escape(c['before'])}»")
            elif c["winner"] == "transient":
                L.append(f"- {c['at'][11:19]} `{c['key']}` убрано после задачи: было «{md_escape(c['before'])}»")
            else:
                L.append(f"- {c['at'][11:19]} `{c['key']}`: было «{md_escape(c['before'])}» → стало "
                         f"«{md_escape(c['after'])}»")
        L += ["", "### Ответы Мастера в диалоге", ""]
        for s in dg["steps"]:
            L.append(f"- **{s['n']}. {s['title']}** — игрок: «{md_escape(clip(s['line'], 150))}» → Мастер (в мире): "
                     f"{md_escape(clip(s['world_answer'], 220))}")
            L.append(f"  - вопрос: «{s['question']}» → с фактами: {md_escape(clip(s['answer'] or s['error'], 260))}")
            L.append(f"  - тот же вопрос без фактов: {md_escape(clip(s['base_answer'] or s['base_error'], 260))}")
        L.append("")

    req = report["requests"]
    L += [f"Запросов к модели: {req['main']['requests']}, повторов {req['main']['retries']}, "
          f"отказов {req['main']['failed']}"
          + (f"; летописец: {req['helper']['requests']}" if req.get("helper") else "") + "."]
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {md_path.relative_to(ROOT)}, {json_path.name}")
    return md_path


# ---------------------------------------------------------------- команды и игра в консоли


def fact_command(agent: MemoryAgent, arg: str) -> None:
    args = arg.split()
    sub = args[0] if args else "list"
    try:
        if sub in ("list", "") and not args:
            print_facts(agent)
        elif sub == "list":
            print_facts(agent)
        elif sub == "rules":
            print_rules15()
        elif sub == "conflicts":
            print_conflicts(agent, int(args[1]) if len(args) > 1 and args[1].isdigit() else 20)
        elif sub == "set" and "=" in arg:
            key, _, value = arg.partition("=")
            key = key.strip()
            if not key or not value.strip():
                print("  /fact set <ключ> = <значение>")
            else:
                op = agent.write_fact(key, value.strip(), "команда игрока")
                agent.save_memory()
                print(f"  🧠 {fact_op_label(op)}")
        elif sub == "unset" and len(args) >= 2:
            op = agent.forget_fact(args[1], "команда игрока")
            agent.save_memory()
            print(f"  🧠 {fact_op_label(op)}" if op else f"  нет факта «{args[1]}»")
        elif sub == "history" and len(args) >= 2:
            f = agent.facts.facts.get(args[1])
            if not f and args[1] in agent.facts.deleted:
                d = agent.facts.deleted[args[1]]
                print(f"  ⊘ {args[1]} отменено {d['at'][11:19]}: было «{clip(d['before'], 60)}»"
                      f" (источник: {d['source']})")
            elif not f:
                print(f"  нет факта «{args[1]}»")
            else:
                for h in f.get("history") or []:
                    print(f"  ↪ {h['at'][11:19]} {h['op']}: «{clip(h['value'], 60)}»"
                          + (f" (было «{clip(h['before'], 40)}»)" if h.get("before") else ""))
        elif sub == "cleanup":
            removed = agent.complete_task("команда игрока")
            print(f"  🧹 убрано временных фактов: {len(removed)} — "
                  + (", ".join(f"«{r['key']}»" for r in removed) or "ничего"))
        else:
            print("  /fact list | set <ключ> = <значение> | unset <ключ> | conflicts [N] | history <ключ> | rules | "
                  "cleanup")
    except (ValueError, KeyError) as e:
        print(f"  ⚠️  {e}")


HELP = ("/fact [list|set|unset|conflicts|history|rules|cleanup], /state, /fsm, /approve, /go <этап> [причина], "
        "/pause, /resume, /breakthrough, /journal [N], /guard on|off, /matrix, /policy, /event <раздел> <ключ> = "
        "<значение>, /remember, /forget, /mission …, /card, /memory, /context, /invariant, /newchat, /tokens, "
        "/history, /exit")


def chat(agent: MemoryAgent) -> int:
    print(f"Мастер на {agent.client.model}, мир {agent.world.name}, герой {agent.cast.name(agent.hero)}.\n"
          f"Команды: {HELP}\n")
    while True:
        try:
            text = input(f"[{agent.machine.stage} · {agent.cult.state}] Игрок: ").strip()
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
            if command in ("/fact", "/facts"):
                fact_command(agent, arg)
            elif command in ("/invariant", "/invariants"):
                inv_command(agent, arg)
            elif command == "/state":
                print_state(agent)
                print_facts(agent)
                print_invariants(agent)
            elif command == "/fsm":
                from day13_task_state import print_rules
                print_rules()
            elif command == "/policy":
                from day13_task_state import print_policy
                print_policy()
                print_rules15()
            elif command == "/approve":
                agent.user_approve()
            elif command == "/go":
                from day13_task_state import state_arg
                if not args:
                    print("  /go <этап> [причина]")
                else:
                    agent.user_transition(state_arg(args[0]), " ".join(args[1:]))
            elif command == "/pause":
                agent.user_transition("paused", "привал")
            elif command == "/resume":
                src = agent.machine.data.get("paused_from")
                agent.user_transition(src or "briefing", "снять привал")
            elif command == "/breakthrough":
                agent.user_breakthrough()
            elif command == "/journal":
                print_journal(agent, int(args[0]) if args and args[0].isdigit() else 20)
            elif command == "/guard":
                agent.guard = not (args and args[0] == "off")
                print(f"  проверка кодом: {'ВКЛ' if agent.guard else 'ВЫКЛ — правила только в промпте'}")
            elif command == "/matrix":
                matrix_demo()
            elif command == "/event":
                head, sep, value = arg.partition("=")
                parts = head.split()
                if not sep or len(parts) != 2:
                    print("  /event <раздел> <ключ> = <значение>")
                else:
                    for o in agent.event_write(parts[0], parts[1], value.strip()):
                        print(f"  {op_label13(o)}")
            elif command in ("/remember", "/forget"):
                head, sep, value = arg.partition("=")
                parts = head.split()
                if len(parts) != 2 or (command == "/remember" and not sep):
                    print("  /remember <раздел> <ключ> = <значение>, /forget <раздел> <ключ>")
                else:
                    section = SECTION_ALIASES.get(parts[0].lower(), parts[0])
                    ops = (agent.write(section, parts[1], value.strip(), "команда игрока") if command == "/remember"
                           else agent.forget(section, parts[1], "команда игрока"))
                    agent.save_memory()
                    for o in ops or [{"op": "-", "section": section, "key": parts[1], "layer": "", "value": ""}]:
                        print(f"  {op_label13(o)}" if o["op"] != "-" else "  (ничего не изменилось)")
            elif command == "/mission":
                mission_command(agent, arg)
            elif command == "/card":
                print_card(agent, args[0] if args else agent.hero)
            elif command == "/memory":
                print_memory(agent)
                print_facts(agent)
            elif command == "/context":
                print_context12(agent)
            elif command == "/newchat":
                print(f"  🧹 {agent.new_scene()}; состояние, карточки, память, факты и инварианты остаются")
            elif command == "/tokens":
                print_ledger(agent)
            elif command == "/history":
                print_history(agent.history)
            elif command.startswith("/"):
                print(f"  Неизвестная команда. Есть: {HELP}")
            else:
                reply = agent.run(text)
                print(f"\nМастер: {reply.text.strip() or '(пустой ответ)'}\n")
                print_turn15(agent)
            print()
        except LLMError as e:
            print(f"❌ Ошибка: {e}\n", file=sys.stderr)
        except (StoreError, ValueError) as e:
            print(f"⚠️  {e}\n", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n(прервано — ход в сцену не попал)\n")


# ---------------------------------------------------------------- запуск


def main() -> int:
    parser = argparse.ArgumentParser(description="День 15: уся-РПГ — память, которая корректно обновляется при "
                                                 "изменении данных: правила фактов, разрешение конфликтов, надгробия "
                                                 "отменённых фактов, очистка после задачи, A/B с блоком фактов.")
    parser.add_argument("prompt", nargs="?", help="один ход и выход; без него — игра в консоли")
    parser.add_argument("--world", default="default", help="мир: memory/day15/<мир>/")
    parser.add_argument("--new", action="store_true", help="начать заново: Лу Чэнь берёт задание «Пропавший караван»")
    parser.add_argument("--no-guard", action="store_true", help="правила только в промпте: код не проверяет инварианты")
    parser.add_argument("--per-turn", type=int, default=1, help="действий задания и развития за ход (0 — без предела)")
    parser.add_argument("--seed", type=int, help="зерно костей прорыва — для повторяемых прогонов")
    parser.add_argument("--keep", type=int, default=6, help="краткосрочная: сколько последних сообщений сцены в промпт")
    parser.add_argument("--long-budget", type=int, default=1500, help="долговременная: бюджет токенов")
    parser.add_argument("--helper-model", help="модель летописца (по умолчанию та же)")
    parser.add_argument("--demo", action="store_true",
                        help="крайние случаи и правила без модели, очистка, живой диалог с A/B")
    parser.add_argument("--only", nargs="+", choices=("matrix", "cleanup", "audit", "dialogue"),
                        help="для --demo: только эти части")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--max-steps", type=int, default=8, help="предел запросов к модели за один ход")
    args = parser.parse_args()

    if args.demo:
        parts = set(args.only or ("matrix", "cleanup", "audit", "dialogue"))
        if not (parts & {"dialogue"}):  # только безмодельные проверки — модель не нужна
            report = {"model": "—", "helper_model": "—", "stamp": stamp_now(), "per_turn": args.per_turn, "dir": "—",
                      "fact_rules": [{"what": a, "keys": b, "how": c} for a, b, c in FACT_RULES]}
            if "matrix" in parts:
                report["matrix"] = matrix_demo()
            if "cleanup" in parts:
                report["cleanup"] = cleanup_demo()
            if "audit" in parts:
                report["audit"] = audit_demo()
            report["requests"] = {"main": {"requests": 0, "retries": 0, "failed": 0}}
            save_report(report, report["stamp"])
            bad = sum((report.get(p) or {}).get("mismatches", 0) for p in ("matrix", "cleanup", "audit"))
            return 0 if bad == 0 else 1
        if not NAME_RE.fullmatch(args.world):
            print("Ошибка: имя мира — буквы, цифры, дефис, подчёркивание.", file=sys.stderr)
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
        return demo(client, helper, args.keep, args.long_budget, args.max_steps, args.per_turn, parts, args.seed)

    if not NAME_RE.fullmatch(args.world):
        print("Ошибка: имя мира — буквы, цифры, дефис, подчёркивание.", file=sys.stderr)
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

    world = DAY_DIR / args.world
    fresh = args.new or not world.exists()
    if args.new and world.exists():
        target = world.with_name(f"{world.name}_{stamp_now()}")
        import os
        os.replace(world, target)
        print(f"Прошлый мир сохранён в {target.relative_to(ROOT)}.")
    agent = make_agent(client, helper, world, "luchen", args.keep, args.long_budget, args.max_steps,
                       not args.no_guard, args.per_turn, args.seed)
    if fresh:
        seed_quest_hero(agent)
        spec = MISSIONS["caravan"]
        agent.new_mission(spec.title, spec.card)
        print(f"📜 Новая игра: {agent.cast.name(agent.hero)} берёт задание «{spec.title}». /state — состояние, "
              f"/fact — факты, /invariant — инварианты.")
    print_restore(agent.restore_info, agent.store, agent)
    if args.prompt:
        try:
            reply = agent.run(args.prompt)
        except (LLMError, StoreError) as e:
            print(f"❌ Ошибка: {e}", file=sys.stderr)
            return 1
        print(f"\nМастер: {reply.text.strip()}\n")
        print_turn15(agent)
        return 0
    return chat(agent)


if __name__ == "__main__":
    from run_timer import run_timed
    sys.exit(run_timed(main))
