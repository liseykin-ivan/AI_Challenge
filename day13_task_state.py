"""День 13. Состояние задачи: конечные автоматы для задания и пути развития героя поверх игры Дня 12.

Мастер игры Дня 12 (`CastAgent`: три слоя памяти героя Дня 11, карточки персонажей, летописец, кости) теперь не
просто помнит, а ведёт процессы. Каждый слой памяти Дня 11 получает свои правила изменения, и все они проверяются
кодом до применения:

    краткосрочная — задание: автомат MissionMachine, memory/day13/<мир>/heroes/<герой>/fsm/mission_<id>.json

        сборы ──(маршрут утверждён игроком)──▶ в пути ──(все этапы пройдены)──▶ отчёт ──(все условия выполнены)──▶ сдано
        briefing ▲                              journey │   ▲                    report │                          done
                 └──────(пересмотр маршрута)────────────┘   └──(условие не выполнено)───┘
        briefing / journey / report ──▶ привал (paused) ──▶ обратно на тот же этап
        На «сдано» код сам начисляет награду в рабочую память: ци — в автомат развития, вещи — в снаряжение.

    рабочая — путь развития: автомат CultivationMachine, fsm/circle_<id>.json

        накопление ──(ци и формы набраны: код сам)──▶ готов к прорыву ──(игрок решил, герой не в пути)──▶ прорыв
        gathering                                    ready                                           breakthrough
        прорыв ──(кости: удача)──▶ ступень взята (advanced) → новый круг: прошлый — в архив, снаряжение переносится,
                                   отметка о прорыве — в долговременную
        прорыв ──(кости: неудача)──▶ накопление, треть ци потеряна
        Медитировать и отрабатывать форму можно не в пути: вне задания или на привале. Одно действие за ход.

    долговременная — политика записи POLICY: школа меняется только событием игры (/event), особенности
    летописец может только добавлять, а круг, ступень, ци и формы пишет только автомат развития.

Состояние каждого автомата — этап, текущий шаг и ожидаемое действие. Менять его можно только инструментами Мастера —
save_route, complete_step, record_check, request_transition, cultivate, train_form, cultivation_transition,
resolve_breakthrough — и командами игрока. Каждый вызов код проверяет до применения: есть ли такой переход в таблице
и выполнено ли его условие (guard). Если нет, состояние не меняется, а Мастер получает причину отказа.
Сверх таблиц:
- маршрут утверждает только игрок: «маршрут утверждаю» / «выступаем» или /approve; прорыв начинает только игрок:
  «начинаю прорыв» или /breakthrough. Таких инструментов у Мастера нет;
- этапы пути и условия задания — строго по порядку, за ход — одно действие задания и одно действие развития;
- на сборах и на привале рассказ о том, что задание уже выполнено, отсекается: ответ отклоняется и переписывается;
- исход прорыва решают кости кода, а не Мастер;
- после /newchat или перезапуска состояние берётся из файлов: «Продолжаем» — и Мастер идёт с того же шага.

    python day13_task_state.py                         # игра (мир default, герой luchen)
    python day13_task_state.py --world video --new     # начать заново: Лу Чэнь берёт задание «Пропавший караван»
    python day13_task_state.py --no-guard              # правила только в промпте, код переходы не проверяет
    python day13_task_state.py --demo                  # все проверки и отчёт в results/
    python day13_task_state.py --demo --only matrix    # только таблицы переходов и политика памяти, без модели

Команды: /state, /fsm, /approve, /go <этап> [причина], /pause, /resume, /breakthrough, /journal [N], /guard on|off,
/matrix, /policy, /event <раздел> <ключ> = <значение>, /remember, /forget, /mission …, /card, /memory, /context,
/newchat, /tokens, /history, /exit.
"""
import argparse
import copy
import json
import os
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from config import ConfigError, load_settings
from day6_agent import Tool
from day7_memory import MEMORY_DIR, ROOT, StoreError, now_iso, print_history, print_restore
from day8_tokens import clip, fmt, money, print_ledger
from day9_compression import normalize, watch_retries
from day10_strategies import read_json, write_json
from day11_memory_layers import (CARRY_OVER, GAME_TOOLS, MISSIONS, SECTION_ALIASES, SECTION_TITLES, md_escape,
                                 mission_command, print_memory, stamp_now)
from day12_personalization import (HERO_CARD, HERO_ID, MASTER_PROMPT12, CastAgent, print_card,
                                   print_context12, seed_hero)
from day12_personalization import on_event as day12_event
from day12_personalization import op_label as op_label12
from llm_client import LLMClient, LLMError

RESULTS_DIR = ROOT / "results"
DAY_DIR = MEMORY_DIR / "day13"
NAME_RE = re.compile(r"[\w-]+")

# ---------------------------------------------------------------- автомат задания: состояния и переходы

STAGES = ("briefing", "journey", "report", "done")  # этапы по порядку
PAUSED = "paused"
STATES = STAGES + (PAUSED,)
STATE_TITLES = {"briefing": "сборы", "journey": "в пути", "report": "отчёт заказчику", "done": "задание сдано",
                "paused": "привал"}
STATE_ALIASES = {**{s: s for s in STATES}, "сборы": "briefing", "подготовка": "briefing", "маршрут": "briefing",
                 "путь": "journey", "в_пути": "journey", "дорога": "journey", "отчёт": "report", "отчет": "report",
                 "проверка": "report", "сдано": "done", "готово": "done", "финал": "done", "привал": "paused",
                 "пауза": "paused"}
RESUME = "@paused_from"  # «туда, откуда ушли на привал»


@dataclass(frozen=True)
class Transition:
    src: str
    dst: str  # RESUME — этап, на котором ушли на привал
    guard: str  # имя условия в GUARDS
    title: str


# Таблица разрешённых переходов задания. Всё, чего здесь нет, запрещено. Условие проверяется кодом до применения.
TRANSITIONS = (
    Transition("briefing", "journey", "route_approved", "выступить"),
    Transition("journey", "report", "steps_done", "к заказчику с отчётом"),
    Transition("report", "done", "checks_passed", "сдать задание"),
    Transition("report", "journey", "check_failed", "вернуться на дорогу"),
    Transition("journey", "briefing", "has_reason", "пересмотр маршрута"),
    Transition("briefing", PAUSED, "none", "привал"),
    Transition("journey", PAUSED, "none", "привал"),
    Transition("report", PAUSED, "none", "привал"),
    Transition(PAUSED, RESUME, "resume", "снять привал"),
)
GUARDS = {
    "route_approved": "маршрут составлен, и эту версию утвердил игрок",
    "steps_done": "все этапы пути пройдены (на доработке — пройден заново хотя бы один)",
    "checks_passed": "все условия задания проверены и выполнены, есть итог (final)",
    "check_failed": "хотя бы одно условие не выполнено",
    "has_reason": "указана причина (reason); утверждение маршрута и пройденные этапы сбрасываются",
    "none": "без условий",
    "resume": "только на тот этап, где ушли на привал",
}
ACTIONS = {"save_route": ("briefing",), "complete_step": ("journey",), "record_check": ("report",),
           "approve": ("briefing",)}
TEXT_GUARD_STAGES = ("briefing", PAUSED)  # здесь рассказа о выполненном задании быть не должно
MAX_STEPS, MAX_CHECKS = 6, 5
EXPECTED_ACTOR = {"assistant": "Мастер", "user": "игрок", "none": "—"}


def state_arg(name: str) -> str:
    s = STATE_ALIASES.get((name or "").strip().lower())
    if not s:
        raise ValueError(f"нет этапа «{name}». Есть: {', '.join(STATES)}")
    return s


def nums(items) -> str:
    return ", ".join(str(i) for i in items)


class MissionMachine:
    """Состояние одного задания. Меняется только методами ниже; каждый сначала проверяет, разрешено ли действие,
    и пишет результат в журнал. enforce=False — режим «правила только в промпте»: нарушение записывается,
    но действие применяется."""

    def __init__(self, path: Path | None, data: dict):
        self.path = path
        self.data = data
        self.per_turn = 0  # действий задания за ход; 0 — без ограничения
        self.turn_used = 0

    @staticmethod
    def blank_data(mission_id: str, title: str) -> dict:
        return {"format": 1, "mission_id": mission_id, "title": title, "stage": "briefing", "paused_from": None,
                "plan": {"version": 0, "steps": [], "checks": [], "saved_at": None},
                "approved_version": 0, "approved_at": None, "approved_by": "",
                "round": 1, "results": {}, "checks": {}, "final": "", "rewarded": False, "journal": [],
                "created_at": now_iso()}

    @classmethod
    def load(cls, path: Path, mission_id: str, title: str) -> "MissionMachine":
        data = read_json(path) if path.exists() else cls.blank_data(mission_id, title)
        if data.get("stage") not in STATES:
            raise StoreError(f"{path}: неизвестный этап {data.get('stage')!r}")
        base = cls.blank_data(mission_id, title)
        base.update(data)
        base["title"] = title or base["title"]
        return cls(path, base)

    def save(self) -> None:
        if self.path is not None:
            write_json(self.path, dict(self.data, updated_at=now_iso()))

    # --- чтение состояния

    @property
    def stage(self) -> str:
        return self.data["stage"]

    @property
    def active_stage(self) -> str:
        """Этап, на котором задание по сути: на привале — тот, откуда ушли."""
        return (self.data.get("paused_from") or "briefing") if self.stage == PAUSED else self.stage

    @property
    def steps(self) -> list[str]:
        return self.data["plan"]["steps"]

    @property
    def criteria(self) -> list[str]:
        return self.data["plan"]["checks"]

    @property
    def version(self) -> int:
        return self.data["plan"]["version"]

    @property
    def approved(self) -> bool:
        return self.version > 0 and self.data["approved_version"] == self.version

    def next_step(self) -> int | None:
        return next((i for i in range(1, len(self.steps) + 1) if str(i) not in self.data["results"]), None)

    def reworked(self) -> list[int]:
        r = self.data["round"]
        return sorted(int(k) for k, v in self.data["results"].items() if v.get("round") == r)

    def next_check(self) -> int | None:
        return next((i for i in range(1, len(self.criteria) + 1) if str(i) not in self.data["checks"]), None)

    def failed(self) -> list[int]:
        return sorted(int(k) for k, v in self.data["checks"].items() if not v.get("passed"))

    def step_info(self, stage: str | None = None) -> dict:
        """Этап, текущий шаг и ожидаемое действие: кто и что должен сделать дальше."""
        stage = stage or self.stage
        n_steps, n_checks = len(self.steps), len(self.criteria)
        info = {"stage": stage, "index": 0, "total": 0, "step": "", "actor": "assistant", "expected": ""}
        if stage == "briefing":
            info["total"] = 2
            if not self.steps:
                info.update(index=1, step="составить маршрут: этапы пути и условия, по которым примут задание",
                            expected="составить маршрут и сохранить через save_route, показать игроку")
            elif not self.approved:
                info.update(index=2, step=f"утверждение маршрута (версия {self.version})", actor="user",
                            expected="утвердить маршрут («маршрут утверждаю», «выступаем», /approve) или попросить правки")
            else:
                info.update(index=2, step=f"маршрут версии {self.version} утверждён",
                            expected="выступить: request_transition(to=\"journey\")")
        elif stage == "journey":
            info["total"] = n_steps
            nxt = self.next_step()
            if self.data["round"] > 1:
                fixed = self.reworked()
                info.update(index=n_steps, step=f"возврат на дорогу (круг {self.data['round']}) по невыполненным условиям",
                            expected=("пройти нужные этапы заново через complete_step(step=N), затем к заказчику"
                                      if not fixed else f"пройдены заново этапы {nums(fixed)}: к заказчику "
                                                        "или пройти ещё"))
            elif nxt:
                info.update(index=nxt, step=f"этап {nxt} из {n_steps} — «{self.steps[nxt - 1]}»",
                            expected=f"разыграть этап {nxt}, показать, что вышло, и сохранить через "
                                     f"complete_step(step={nxt})")
            else:
                info.update(index=n_steps, step="все этапы пути пройдены",
                            expected="к заказчику с отчётом: request_transition(to=\"report\")")
        elif stage == "report":
            info["total"] = n_checks
            nxt = self.next_check()
            if nxt:
                info.update(index=nxt, step=f"условие {nxt} из {n_checks} — «{self.criteria[nxt - 1]}»",
                            expected=f"заказчик проверяет условие {nxt}: record_check(check={nxt})")
            elif self.failed():
                info.update(index=n_checks, step=f"не выполнены условия {nums(self.failed())}",
                            expected="вернуться на дорогу и доделать: request_transition(to=\"journey\")")
            else:
                info.update(index=n_checks, step="все условия выполнены",
                            expected="сдать задание с итогом: request_transition(to=\"done\", final=…)")
        elif stage == PAUSED:
            src = self.data.get("paused_from") or "briefing"
            inner = self.step_info(src)
            info.update(index=inner["index"], total=inner["total"], actor="user",
                        step=f"привал на этапе {src}: {inner['step']}",
                        expected=f"сказать «продолжаем» (/resume) — задание вернётся на этап {src}")
        else:  # done
            info.update(actor="none", step="задание сдано, награда начислена",
                        expected="ничего: задание закрыто; дальше — новое задание или путь развития")
        return info

    # --- переходы

    def rule(self, dst: str) -> Transition | None:
        src = self.stage
        for t in TRANSITIONS:
            if t.src == src and (t.dst == dst or (t.dst == RESUME and dst == self.data.get("paused_from"))):
                return t
        return None

    def allowed(self) -> list[tuple[str, Transition, list[str]]]:
        out = []
        for t in TRANSITIONS:
            if t.src != self.stage:
                continue
            dst = self.data.get("paused_from") if t.dst == RESUME else t.dst
            out.append((dst, t, self.guard_unmet(t.guard, "причина", "итог")))
        return out

    def guard_unmet(self, guard: str, reason: str = "", final: str = "") -> list[str]:
        d = self.data
        if guard == "route_approved":
            if not self.steps:
                return ["маршрут не составлен (save_route)"]
            return [] if self.approved else [f"маршрут версии {self.version} не утверждён игроком"]
        if guard == "steps_done":
            missing = [i for i in range(1, len(self.steps) + 1) if str(i) not in d["results"]]
            out = [f"не пройдены этапы пути {nums(missing)}"] if missing else []
            if d["round"] > 1 and not self.reworked():
                out.append("после возврата на дорогу не пройден заново ни один этап")
            return out
        if guard == "checks_passed":
            missing = [i for i in range(1, len(self.criteria) + 1) if str(i) not in d["checks"]]
            out = [f"не проверены условия {nums(missing)}"] if missing else []
            if self.failed():
                out.append(f"не выполнены условия {nums(self.failed())}")
            if not (final or "").strip():
                out.append("нет итога задания (final)")
            return out
        if guard == "check_failed":
            return [] if self.failed() else ["все проверенные условия выполнены — возвращаться на дорогу незачем"]
        if guard == "has_reason":
            return [] if (reason or "").strip() else ["нужна причина пересмотра маршрута (reason)"]
        return []

    def check(self, dst: str, reason: str = "", final: str = "") -> tuple[Transition | None, list[str]]:
        src = self.stage
        if dst not in STATES:
            return None, [f"нет этапа «{dst}». Есть: {', '.join(STATES)}"]
        if dst == src:
            return None, [f"задание уже на этапе {src}"]
        t = self.rule(dst)
        if t is None:
            ways = [d for d, _, _ in self.allowed()]
            why = f"перехода {src} → {dst} нет в таблице"
            if src == PAUSED:
                why += f": с привала можно вернуться только на этап {self.data.get('paused_from')}"
            elif src == "done":
                why += ": задание сдано, done — конечное состояние, награда уже начислена"
            elif dst in STAGES and src in STAGES and STAGES.index(dst) > STAGES.index(src) + 1:
                skipped = STAGES[STAGES.index(src) + 1:STAGES.index(dst)]
                why += f": нельзя перепрыгнуть {', '.join(skipped)}"
            if ways and src != "done":
                why += f". Отсюда можно: {', '.join(ways)}"
            return None, [why]
        return t, self.guard_unmet(t.guard, reason, final)

    def transition(self, dst: str, by: str, reason: str = "", final: str = "", enforce: bool = True) -> dict:
        src = self.stage
        t, unmet = self.check(dst, reason, final)
        applied = dst in STATES and dst != src and (not unmet or not enforce)
        if applied:
            self._apply(src, dst, reason, final)
        return self._log("transition", f"{src} → {dst}", by, unmet, applied, enforce,
                         {"to": dst, "reason": reason, **({"final": final} if final else {})})

    def _apply(self, src: str, dst: str, reason: str, final: str) -> None:
        d = self.data
        if dst == PAUSED:
            d["paused_from"] = src
        elif src == PAUSED:
            d["paused_from"] = None
        if src == "journey" and dst == "briefing":  # новый маршрут — новое утверждение и новый путь
            d.update(approved_version=0, approved_at=None, approved_by="", results={}, round=1)
        if dst == "report" and src != PAUSED:
            d["checks"] = {}
        if src == "report" and dst == "journey":
            d["round"] += 1
        if dst == "done":
            d["final"] = final
        d["stage"] = dst

    # --- действия внутри этапа

    def _stage_unmet(self, action: str) -> list[str]:
        if self.stage in ACTIONS[action]:
            return []
        where = ", ".join(ACTIONS[action])
        hint = {"briefing": "сначала нужен утверждённый маршрут", "paused": "герой на привале",
                "done": "задание сдано"}.get(self.stage, "")
        return [f"{action} — только на этапе {where}, а сейчас {self.stage}" + (f" ({hint})" if hint else "")]

    def _limit_unmet(self) -> list[str]:
        if self.per_turn and self.turn_used >= self.per_turn:
            return [f"за один ход — не больше {self.per_turn} действия задания: покажи игроку, что вышло, "
                    f"следующее — в следующем ходе"]
        return []

    def save_route(self, steps, checks, by: str, enforce: bool = True) -> dict:
        steps = [str(s).strip() for s in (steps or []) if str(s).strip()]
        checks = [str(c).strip() for c in (checks or []) if str(c).strip()]
        unmet = self._stage_unmet("save_route")
        if self.stage == "journey":
            unmet = [unmet[0] + "; чтобы изменить маршрут — request_transition(to=\"briefing\", reason=…)"]
        broken = []
        if not 1 <= len(steps) <= MAX_STEPS:
            broken.append(f"этапов пути должно быть от 1 до {MAX_STEPS}, а их {len(steps)}")
        if not 1 <= len(checks) <= MAX_CHECKS:
            broken.append(f"условий задания должно быть от 1 до {MAX_CHECKS}, а их {len(checks)}")
        applied = not broken and (not unmet or not enforce)
        if applied:
            self.data["plan"] = {"version": self.version + 1, "steps": steps, "checks": checks, "saved_at": now_iso()}
        return self._log("action", "save_route", by, unmet + broken, applied, enforce,
                         {"steps": steps, "checks": checks})

    def complete_step(self, step, result: str, by: str, enforce: bool = True) -> dict:
        try:
            step = int(step)
        except (TypeError, ValueError):
            return self._log("action", "complete_step", by, [f"номер этапа должен быть числом, а не {step!r}"],
                             False, enforce, {"step": step})
        unmet = self._stage_unmet("complete_step")
        if not unmet and not self.approved:
            unmet.append("маршрут не утверждён игроком")
        if not 1 <= step <= max(len(self.steps), 1) or not self.steps:
            unmet.append(f"этапа {step} нет в маршруте (этапов {len(self.steps)})")
        elif self.data["round"] == 1:
            nxt = self.next_step()
            if nxt is None:
                unmet.append("все этапы уже пройдены — дальше отчёт заказчику")
            elif step != nxt:
                unmet.append(f"этап {step} нельзя: этапы идут по порядку, следующий — этап {nxt}")
        unmet += self._limit_unmet()
        broken = [] if (result or "").strip() else ["пустой итог этапа"]
        applied = not broken and (not unmet or not enforce)
        if applied:
            self.data["results"][str(step)] = {"text": result.strip(), "at": now_iso(), "round": self.data["round"]}
            self.turn_used += 1
        return self._log("action", "complete_step", by, unmet + broken, applied, enforce,
                         {"step": step, "result": clip(result or "", 200)})

    def record_check(self, check, passed, evidence: str, by: str, enforce: bool = True) -> dict:
        try:
            check = int(check)
        except (TypeError, ValueError):
            return self._log("action", "record_check", by, [f"номер условия должен быть числом, а не {check!r}"],
                             False, enforce, {"check": check})
        if isinstance(passed, str):
            passed = passed.strip().lower() in ("true", "да", "yes", "1")
        unmet = self._stage_unmet("record_check")
        if not 1 <= check <= max(len(self.criteria), 1) or not self.criteria:
            unmet.append(f"условия {check} нет в задании (условий {len(self.criteria)})")
        else:
            nxt = self.next_check()
            if nxt is None:
                unmet.append("все условия уже проверены")
            elif check != nxt:
                unmet.append(f"условие {check} нельзя: условия проверяются по порядку, следующее — {nxt}")
        unmet += self._limit_unmet()
        broken = [] if (evidence or "").strip() else ["нужно, чем подтверждается (evidence)"]
        applied = not broken and (not unmet or not enforce)
        if applied:
            self.data["checks"][str(check)] = {"passed": bool(passed), "evidence": evidence.strip(), "at": now_iso()}
            self.turn_used += 1
        return self._log("action", "record_check", by, unmet + broken, applied, enforce,
                         {"check": check, "passed": bool(passed), "evidence": clip(evidence or "", 200)})

    def approve(self, by: str) -> dict:
        """Утверждение маршрута. Только от игрока: у Мастера такого инструмента нет."""
        unmet = self._stage_unmet("approve")
        if not unmet and not self.steps:
            unmet.append("маршрут ещё не составлен — утверждать нечего")
        elif not unmet and self.approved:
            unmet.append(f"маршрут версии {self.version} уже утверждён")
        applied = not unmet
        if applied:
            self.data.update(approved_version=self.version, approved_at=now_iso(), approved_by=by)
        return self._log("action", "approve", by, unmet, applied, True, {"version": self.version})

    def _log(self, kind: str, name: str, by: str, unmet: list[str], applied: bool, enforce: bool, args: dict) -> dict:
        entry = {"at": now_iso(), "machine": "mission", "kind": kind, "name": name, "by": by, "ok": not unmet,
                 "applied": applied, "enforced": enforce, "unmet": unmet, "args": args, "stage": self.stage}
        self.data["journal"].append(entry)
        return entry

    # --- состояние для промпта

    def summary(self) -> dict:
        i = self.step_info()
        return {"stage": self.stage, "step": i["step"], "expected": f"{EXPECTED_ACTOR[i['actor']]}: {i['expected']}",
                "allowed": [d for d, _, u in self.allowed() if not u]}

    def block(self, enforce: bool = True) -> str:
        d, i = self.data, self.step_info()
        pos = f" ({STAGES.index(self.stage) + 1} из {len(STAGES)})" if self.stage in STAGES else ""
        src = d.get("paused_from")
        lines = ["\n\n# Состояние задания — конечный автомат",
                 ("Этапом задания управляет код: переходы проверяются до применения." if enforce else
                  "Этап задания меняется через инструменты; соблюдай правила этапов."),
                 f"Задание: «{d['title']}»",
                 f"Этап: {self.stage} — {STATE_TITLES[self.stage]}{pos}" + (f", привал на этапе {src}"
                                                                           if self.stage == PAUSED else ""),
                 f"Текущий шаг: {i['step']}",
                 f"Ожидаемое действие: {EXPECTED_ACTOR[i['actor']]} — {i['expected']}"]
        ways = self.allowed()
        if ways:
            lines.append("Переходы отсюда:")
            for dst, t, unmet in ways:
                lines.append(f"- {dst} ({t.title}): " + ("можно" if not unmet else "пока нельзя — " + "; ".join(unmet)))
        else:
            lines.append("Переходов отсюда нет: done — конечное состояние.")
        if self.steps:
            appr = "утверждён игроком" if self.approved else "НЕ утверждён игроком"
            lines.append(f"Маршрут (версия {self.version}, {appr}):")
            for n, s in enumerate(self.steps, 1):
                r = d["results"].get(str(n))
                mark = "✅" if r else ("▶" if self.stage == "journey" and n == self.next_step() else "·")
                lines.append(f"{n}. {mark} {s}" + (f"\n   что вышло: {clip(r['text'], 600)}" if r else ""))
            lines.append("Условия задания (их проверяет заказчик):")
            for n, c in enumerate(self.criteria, 1):
                r = d["checks"].get(str(n)) if self.stage in ("report", "done", PAUSED) else None
                mark = ("✅" if r["passed"] else "❌") if r else "·"
                lines.append(f"{n}. {mark} {c}" + (f" — {clip(r['evidence'], 300)}" if r else ""))
        if d.get("final"):
            lines.append(f"Итог: {d['final']}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- автомат пути развития

REALMS = ("Закалка тела", "Очищение ци", "Основание", "Золотое ядро")
STAGE_NAMES = ("первая", "вторая", "третья")
CSTATES = ("gathering", "ready", "breakthrough", "advanced")
CSTATE_TITLES = {"gathering": "накопление ци", "ready": "готов к прорыву", "breakthrough": "прорыв",
                 "advanced": "ступень взята"}
CTRANSITIONS = (
    Transition("gathering", "ready", "qi_enough", "готов к прорыву (код переводит сам)"),
    Transition("ready", "breakthrough", "consent_off_road", "начать прорыв"),
    Transition("breakthrough", "advanced", "trial_passed", "прорыв удался (кости)"),
    Transition("breakthrough", "gathering", "trial_failed", "откат: треть ци потеряна (кости)"),
)
CGUARDS = {
    "qi_enough": "ци и освоенные формы набраны для этой ступени",
    "consent_off_road": "прорыв начал игрок («начинаю прорыв», /breakthrough), и герой не в пути",
    "trial_passed": "только через resolve_breakthrough: исход решают кости, d20 + бонус ≥ 12",
    "trial_failed": "только через resolve_breakthrough: исход решают кости",
}
MEDITATION_QI = 40
TRIAL_DC = 12
ROAD_STAGES = ("journey", "report")  # здесь медитировать и прорываться нельзя


def qi_need(realm: int, stage: int) -> int:
    return 100 * (realm + 1) + 50 * stage


def forms_need(realm: int, stage: int) -> int:
    return 3 * realm + stage + 1


def circle_title(realm: int, stage: int) -> str:
    return f"{REALMS[realm]}, {STAGE_NAMES[stage - 1]} ступень"


class CultivationMachine:
    """Путь развития героя на одном круге — процесс рабочей памяти. Ци и формы меняют только действия ниже,
    ступень — только кости прорыва. mission_stage — откуда автомат знает, не в пути ли герой."""

    def __init__(self, path: Path | None, data: dict, mission_stage=lambda: "done"):
        self.path = path
        self.data = data
        self.mission_stage = mission_stage
        self.per_turn = 0
        self.turn_used = 0
        self.consent = False  # игрок в этом ходе сам сказал, что начинает прорыв

    @staticmethod
    def blank_data(circle_id: str, realm: int = 1, stage: int = 2, qi: int = 0, forms: int = 0,
                   technique: str = "«Девять порывов ветра»") -> dict:
        return {"format": 1, "circle_id": circle_id, "realm": realm, "stage": stage, "state": "gathering",
                "qi": qi, "forms": forms, "technique": technique, "attempts": [], "journal": [],
                "created_at": now_iso()}

    @classmethod
    def load(cls, path: Path, circle_id: str, mission_stage) -> "CultivationMachine":
        data = read_json(path) if path.exists() else cls.blank_data(circle_id)
        base = cls.blank_data(circle_id)
        base.update(data)
        if base["state"] not in CSTATES:
            raise StoreError(f"{path}: неизвестное состояние {base['state']!r}")
        return cls(path, base, mission_stage)

    def save(self) -> None:
        if self.path is not None:
            write_json(self.path, dict(self.data, updated_at=now_iso()))

    @property
    def state(self) -> str:
        return self.data["state"]

    @property
    def need_qi(self) -> int:
        return qi_need(self.data["realm"], self.data["stage"])

    @property
    def need_forms(self) -> int:
        return forms_need(self.data["realm"], self.data["stage"])

    def title(self) -> str:
        return circle_title(self.data["realm"], self.data["stage"])

    def mirror(self) -> dict[str, str]:
        """Что автомат пишет в рабочую память (раздел circle): только он и пишет эти ключи."""
        d = self.data
        return {"круг": self.title(), "ци": f"{d['qi']} из {self.need_qi}",
                "формы": f"{d['technique']}: освоено {d['forms']}, для прорыва нужно {self.need_forms}",
                "условие_прорыва": f"{self.need_qi} ци и {self.need_forms} форм {d['technique']}",
                "состояние": CSTATE_TITLES[self.state]}

    def on_road(self) -> bool:
        return self.mission_stage() in ROAD_STAGES

    def guard_unmet(self, guard: str) -> list[str]:
        d = self.data
        if guard == "qi_enough":
            out = []
            if d["qi"] < self.need_qi:
                out.append(f"ци {d['qi']} из {self.need_qi}")
            if d["forms"] < self.need_forms:
                out.append(f"освоено форм {d['forms']} из {self.need_forms}")
            return out
        if guard == "consent_off_road":
            out = [] if self.consent else ["прорыв начинает только игрок: «начинаю прорыв» или /breakthrough"]
            if self.on_road():
                out.append(f"герой в пути (задание на этапе {self.mission_stage()}): прорыв — только вне задания "
                           f"или на привале")
            return out
        if guard in ("trial_passed", "trial_failed"):
            return [CGUARDS[guard]]
        return []

    def allowed(self) -> list[tuple[str, Transition, list[str]]]:
        return [(t.dst, t, self.guard_unmet(t.guard)) for t in CTRANSITIONS if t.src == self.state]

    def check(self, dst: str) -> tuple[Transition | None, list[str]]:
        src = self.state
        if dst not in CSTATES:
            return None, [f"нет состояния «{dst}». Есть: {', '.join(CSTATES)}"]
        if dst == src:
            return None, [f"путь развития уже в состоянии {src}"]
        t = next((x for x in CTRANSITIONS if x.src == src and x.dst == dst), None)
        if t is None:
            why = f"перехода {src} → {dst} нет в таблице"
            if CSTATES.index(dst) > CSTATES.index(src) + 1:
                why += f": нельзя перепрыгнуть {', '.join(CSTATES[CSTATES.index(src) + 1:CSTATES.index(dst)])}"
            if src == "gathering":
                why += f" ({'; '.join(self.guard_unmet('qi_enough')) or 'ци набрано'})"
            return None, [why]
        return t, self.guard_unmet(t.guard)

    def transition(self, dst: str, by: str, enforce: bool = True) -> dict:
        src = self.state
        t, unmet = self.check(dst)
        applied = dst in CSTATES and dst != src and (not unmet or not enforce)
        if applied:
            self.data["state"] = dst
        return self._log("transition", f"{src} → {dst}", by, unmet, applied, enforce, {"to": dst})

    def auto_ready(self) -> dict | None:
        """gathering → ready код делает сам, как только условие выполнено."""
        if self.state == "gathering" and not self.guard_unmet("qi_enough"):
            return self.transition("ready", "автомат развития")
        return None

    def _where_unmet(self, action: str) -> list[str]:
        out = []
        if self.state not in ("gathering", "ready"):
            out.append(f"{action} — только в накоплении или перед прорывом, а сейчас {self.state}")
        if self.on_road():
            out.append(f"{action} — не в пути: герой на задании (этап {self.mission_stage()}); можно вне задания "
                       f"или на привале")
        if self.per_turn and self.turn_used >= self.per_turn:
            out.append(f"за один ход — не больше {self.per_turn} действия развития")
        return out

    def cultivate(self, by: str, reason: str = "", enforce: bool = True) -> dict:
        unmet = self._where_unmet("cultivate")
        if self.data["qi"] >= self.need_qi:
            unmet.append(f"ци уже полна ({self.data['qi']} из {self.need_qi}) — медитация не нужна: "
                         f"осталось набрать формы или начать прорыв")
        gain = min(MEDITATION_QI, max(0, self.need_qi - self.data["qi"]))
        applied = not unmet or not enforce
        if applied:
            self.data["qi"] += gain
            self.turn_used += 1
        return self._log("action", "cultivate", by, unmet, applied, enforce,
                         {"qi": f"+{gain}", "reason": clip(reason or "", 120)})

    def train_form(self, by: str, form: str = "", enforce: bool = True) -> dict:
        unmet = self._where_unmet("train_form")
        applied = not unmet or not enforce
        if applied:
            self.data["forms"] += 1
            self.turn_used += 1
        return self._log("action", "train_form", by, unmet, applied, enforce,
                         {"forms": f"+1 → {self.data['forms']}", "form": clip(form or "", 80)})

    def add_qi(self, qi: int, by: str, why: str) -> dict:
        """Ци извне — награда за задание. Только кодом, не инструментом Мастера."""
        self.data["qi"] += int(qi)
        return self._log("action", "add_qi", by, [], True, True, {"qi": f"+{qi}", "reason": why})

    def resolve(self, pills, by: str, rng: random.Random, enforce: bool = True) -> dict:
        """Исход прорыва — броском кода: d20 + 2 за каждую пилюлю (до 3) ≥ 12."""
        try:
            pills = max(0, min(int(pills or 0), 3))
        except (TypeError, ValueError):
            pills = 0
        unmet = [] if self.state == "breakthrough" else [f"resolve_breakthrough — только во время прорыва, "
                                                         f"а сейчас {self.state}"]
        args = {"pills": pills}
        applied = not unmet or not enforce
        if applied:
            roll = rng.randint(1, 20)
            total = roll + 2 * pills
            ok = total >= TRIAL_DC
            args.update(roll=roll, bonus=2 * pills, total=total, dc=TRIAL_DC, passed=ok)
            self.data["attempts"].append({**args, "at": now_iso(), "circle": self.title()})
            if ok:
                self.data["state"] = "advanced"
            else:
                lost = self.data["qi"] // 3
                self.data["qi"] -= lost
                args["lost_qi"] = lost
                self.data["state"] = "gathering"
            self.turn_used += 1
        return self._log("action", "resolve_breakthrough", by, unmet, applied, enforce, args)

    def _log(self, kind: str, name: str, by: str, unmet: list[str], applied: bool, enforce: bool, args: dict) -> dict:
        entry = {"at": now_iso(), "machine": "circle", "kind": kind, "name": name, "by": by, "ok": not unmet,
                 "applied": applied, "enforced": enforce, "unmet": unmet, "args": args, "stage": self.state}
        self.data["journal"].append(entry)
        return entry

    def step_info(self) -> dict:
        d = self.data
        if self.state == "gathering":
            lack = "; ".join(self.guard_unmet("qi_enough"))
            exp = (f"набирать ци и формы: cultivate (+{MEDITATION_QI} ци) или train_form (+1 форма), одно действие "
                   f"за ход, не в пути" + (f"; не хватает: {lack}" if lack else ""))
            return {"state": self.state, "step": f"накопление: ци {d['qi']}/{self.need_qi}, формы "
                                                  f"{d['forms']}/{self.need_forms}", "actor": "assistant",
                    "expected": exp}
        if self.state == "ready":
            return {"state": self.state, "step": "ци и формы набраны", "actor": "user",
                    "expected": "игрок решает начать прорыв («начинаю прорыв», /breakthrough); тогда Мастер — "
                                "cultivation_transition(to=\"breakthrough\")"}
        if self.state == "breakthrough":
            return {"state": self.state, "step": f"прорыв на {circle_title(*self.next_circle())}", "actor": "assistant",
                    "expected": "описать испытание и вызвать resolve_breakthrough(pills=N): исход решают кости"}
        return {"state": self.state, "step": "ступень взята", "actor": "none", "expected": "новый круг открыт кодом"}

    def next_circle(self) -> tuple[int, int]:
        r, s = self.data["realm"], self.data["stage"]
        return (r, s + 1) if s < 3 else (min(r + 1, len(REALMS) - 1), 1)

    def block(self, enforce: bool = True) -> str:
        d, i = self.data, self.step_info()
        lines = ["\n\n# Путь развития героя — конечный автомат",
                 ("Ци, формы и ступень меняет только код: через инструменты ниже, переходы проверяются до применения."
                  if enforce else "Ци, формы и ступень меняются через инструменты; соблюдай правила."),
                 f"Круг: {self.title()}; состояние: {self.state} — {CSTATE_TITLES[self.state]}",
                 f"Ци: {d['qi']} из {self.need_qi}; {d['technique']}: освоено форм {d['forms']}, нужно {self.need_forms}",
                 f"Текущий шаг: {i['step']}",
                 f"Ожидаемое действие: {EXPECTED_ACTOR[i['actor']]} — {i['expected']}",
                 "Где можно: медитировать (cultivate), отрабатывать форму (train_form) и прорываться — только не в пути "
                 "(вне задания или на привале)." + (" Сейчас герой в пути." if self.on_road() else "")]
        ways = self.allowed()
        if ways:
            lines.append("Переходы отсюда:")
            for dst, t, unmet in ways:
                lines.append(f"- {dst} ({t.title}): " + ("можно" if not unmet else "пока нельзя — " + "; ".join(unmet)))
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- политика изменений долговременной и рабочей памяти

# Кто может менять раздел: летописец (из сцены), игрок (команды /remember), событие игры (/event), автомат, игра.
CIRCLE_KEYS = re.compile(r"^(?:круг|ступень|уровень|ци|ци_\w+|\w+_ци|накоплено\w*|формы|освоено\w*|"
                         r"условие_прорыва|прорыв\w*|состояние)$")
STAGE_KEYS = {"этап", "этап_задания", "стадия", "статус", "статус_задания", "текущий_этап", "текущий_шаг", "шаг"}
POLICY = [  # (раздел, что охраняется, кто может, почему) — для /policy и отчёта
    ("school", "весь раздел", "событие игры, игрок (/remember)", "школу не меняют словами в сцене — только событием "
                                                                 "игры: изгнание, переход, посвящение"),
    ("traits", "существующие записи", "летописец — только добавлять", "врождённое и постоянное не переписывается; "
                                                                      "новый шрам — новая запись"),
    ("circle", "круг, ступень, ци, формы, условие прорыва", "только автомат развития",
     "уровень героя — состояние автомата, а не слова Мастера"),
    ("mission", "этап, статус, шаг", "только автомат задания", "этап задания — состояние автомата"),
]


def writer_of(source: str) -> str:
    s = (source or "").lower()
    if s.startswith("автомат"):
        return "автомат"
    if s.startswith("событие"):
        return "событие"
    if s.startswith("команда"):
        return "игрок"
    if s.startswith("игра") or s.startswith("готовый") or s.startswith("награда"):
        return "игра"
    return "летописец"


def policy_unmet(section: str, key: str, op: str, writer: str, exists: bool) -> list[str]:
    """Почему этой записи нельзя: пусто — можно."""
    k = key.strip().lower()
    if section == "school" and writer not in ("событие", "игрок", "игра"):
        return ["школа меняется только событием игры (/event school …), не словами в сцене"]
    if section == "traits" and writer == "летописец" and (exists or op == "delete"):
        return ["постоянные особенности летописец может только добавлять, не переписывать"]
    if section == "circle" and CIRCLE_KEYS.match(k) and writer not in ("автомат", "игра"):
        return ["круг, ступень, ци и формы меняет только автомат развития (cultivate, train_form, прорыв)"]
    if section == "mission" and k in STAGE_KEYS:
        return ["этап задания — состояние автомата задания, его не записывают в карточку"]
    return []


# ---------------------------------------------------------------- агент

QUEST_PROMPT = MASTER_PROMPT12 + (
    "\n\nТы ведёшь задание героя по этапам: briefing (сборы) → journey (в пути) → report (отчёт заказчику) → "
    "done (сдано), плюс paused (привал). И путь развития героя: gathering (накопление ци) → ready (готов к прорыву) → "
    "breakthrough (прорыв) → advanced (ступень взята). Состояние обоих автоматов — в конце этого сообщения, его хранит "
    "и проверяет код. Правила:\n"
    "- работай в рамках текущего этапа; что делать дальше — в строке «Ожидаемое действие»;\n"
    "- briefing: составь маршрут — этапы пути и условия, по которым заказчик примет задание, — сохрани через "
    "save_route, покажи игроку и спроси, утверждает ли он маршрут. На сборах не рассказывай, как задание выполнено. "
    "Маршрут утверждает только игрок: сам не утверждай и не считай утверждённым;\n"
    "- journey: каждый ход — ровно один этап маршрута, целиком: разыграй его до итога (что нашёл, чем кончилась "
    "стычка, где герой теперь), рискованное — броском костей, и в этом же ходе сохрани итог через complete_step. Не "
    "растягивай этап на несколько ходов. Если исход зависит от выбора героя, а игрок сказал только «дальше» или "
    "доверил решать тебе, выбери сам — так, как поступил бы герой по своему характеру и клятвам. После последнего "
    "этапа — request_transition(to=\"report\");\n"
    "- report: заказчик проверяет условия по порядку, одно за ход, честно, через record_check (passed=false, если не "
    "выполнено). Всё выполнено — request_transition(to=\"done\", final=<итог>); награду начислит код. Не выполнено — "
    "вернись на дорогу;\n"
    "- путь развития: медитация — cultivate, отработка формы — train_form; только не в пути. Когда ци и формы "
    "набраны, код сам переводит в ready. Прорыв начинает только игрок; тогда — cultivation_transition(to="
    "\"breakthrough\"), затем опиши испытание и вызови resolve_breakthrough: исход решают кости кода, не ты;\n"
    "- этапы и состояния меняются только инструментами. Если переход или действие отклонены, состояние не изменилось: "
    "скажи игроку, что именно нельзя и почему, предложи следующий допустимый шаг и не разыгрывай запрещённое. "
    "Объясняй языком мира — заказчик, обычаи секты, сама дорога, — не упоминай код, автоматы и инструменты;\n"
    "- если игрок просит пропустить этап, начислить награду, поднять ступень или сменить школу словами — не делай, "
    "объясни, почему нельзя;\n"
    "- привал: попросили остановиться — request_transition(to=\"paused\"), на привале задание не двигай. Попросили "
    "продолжить — сними привал на тот же этап и продолжай с текущего шага: всё нужное уже есть в состоянии и памяти, "
    "не переспрашивай игрока о задании.")

CORRECTION = ("\n\n# Ответ отклонён кодом\nТвой вариант ответа рассказывал, что задание уже выполнено, а сейчас этап "
              "{stage}: {why}. Напиши ответ заново без исхода задания: скажи игроку, на каком этапе задание и что "
              "нужно сделать, чтобы выступить.\n")
OUTCOME_REMOVED = "[рассказ об исходе задания удалён: на этапе {stage} задание ещё не выполнено]"
FSM_TOOL_NAMES = ("save_route", "complete_step", "record_check", "request_transition", "cultivate", "train_form",
                  "cultivation_transition", "resolve_breakthrough")

# Утверждение маршрута и начало прорыва — только из сообщения игрока, распознаёт код. «Не утверждаю» — не утверждение.
APPROVE_RE = re.compile(r"(?<![\w-])(?:маршрут утверждаю|утверждаю|выступаем|в путь|маршрут принят|одобряю)(?![\w-])",
                        re.I)
NEG_APPROVE_RE = re.compile(r"(?<![\w-])не\s+(?:утверждаю|одобряю|выступаем|принимаю)", re.I)
CONSENT_RE = re.compile(r"(?:начинаю|начнём|начнем|иду на|совершаю|пробую|попробую|делаю|решаюсь на)\s+прорыв|"
                        r"прорываюсь", re.I)
NEG_CONSENT_RE = re.compile(r"не\s+(?:начинаю|буду|пробую|делаю|прорываюсь)\w*\s*(?:прорыв)?", re.I)


def is_approval(text: str) -> bool:
    return bool(APPROVE_RE.search(text or "")) and not NEG_APPROVE_RE.search(text or "")


def is_consent(text: str) -> bool:
    return bool(CONSENT_RE.search(text or "")) and not NEG_CONSENT_RE.search(text or "")


@dataclass(frozen=True)
class QuestSpec:
    id: str
    reward_qi: int
    outcome: str  # regex: рассказ о том, что задание уже выполнено


QUESTS = {
    "caravan": QuestSpec("caravan", 60,
                         r"(?:наш[её]л|нашли|находишь|отыскал\w*|вернул\w*|возвращаешь)\s+(?:\w+\s+){0,3}караван|"
                         r"караван\w*\s+(?:\w+\s+){0,2}(?:найден|спас[её]н|возвращ|освобожд)|"
                         r"груз\w*\s+(?:\w+\s+){0,2}(?:спас[её]н|доставлен)|задани\w+\s+(?:выполнен|сдан)"),
    "escort": QuestSpec("escort", 30, r"(?:довёл|довел|доводишь|доставил\w*)\s+(?:\w+\s+){0,3}(?:лекарк|фан\s*жо)|"
                                      r"задани\w+\s+выполнен"),
}


CONTENT_WORD = re.compile(r"[^\W\d_]{4,}")


def restates(sentence: str, plan: list[str]) -> bool:
    """Предложение пересказывает этап или условие маршрута («груз возвращён целым» — условие, а не исход):
    в нём есть не меньше 60% значимых слов (по первым пяти буквам) одного из пунктов и не меньше трёх
    (у коротких пунктов — все)."""
    words = {w[:5] for w in CONTENT_WORD.findall(normalize(sentence))}
    for item in plan:
        need = {w[:5] for w in CONTENT_WORD.findall(normalize(item))}
        if need and len(words & need) >= max(min(3, len(need)), -(-len(need) * 3 // 5)):
            return True
    return False


def outcome_sentences(text: str, spec: QuestSpec | None, plan: list[str] = ()) -> list[str]:
    """Предложения (и строки списков), где рассказано, что задание уже выполнено. Пересказ маршрута — не в счёт."""
    if not spec:
        return []
    parts = [p for line in (text or "").splitlines() for p in re.split(r"(?<=[.!?…])\s+", line) if p.strip()]
    return [p for p in parts if re.search(spec.outcome, normalize(p)) and not restates(p, list(plan))]


def outcome_told(text: str, spec: QuestSpec | None, plan: list[str] = ()) -> bool:
    return bool(outcome_sentences(text, spec, plan))


def strip_outcome(text: str, stage: str, spec: QuestSpec | None, plan: list[str] = ()) -> str:
    """Вырезаются только предложения с исходом задания; строки и пересказ маршрута остаются."""
    bad = set(outcome_sentences(text, spec, plan))
    if not bad:
        return text
    lines = []
    for line in (text or "").splitlines():
        kept = [p for p in re.split(r"(?<=[.!?…])\s+", line) if p not in bad]
        lines.append(" ".join(kept))
    return "\n".join(lines).rstrip() + " " + OUTCOME_REMOVED.format(stage=stage)


def fsm_tools(agent: "QuestAgent") -> list[Tool]:
    return [
        Tool("save_route",
             "Сохранить маршрут задания: этапы пути по порядку и условия, по которым заказчик примет работу. Только "
             "на этапе briefing. Новая версия маршрута сбрасывает утверждение — утверждает игрок.",
             {"type": "object",
              "properties": {"steps": {"type": "array", "items": {"type": "string"},
                                       "description": f"этапы пути по порядку, от 1 до {MAX_STEPS}"},
                             "checks": {"type": "array", "items": {"type": "string"},
                                        "description": f"условия задания, от 1 до {MAX_CHECKS}"}},
              "required": ["steps", "checks"]},
             agent.tool_save_route),
        Tool("complete_step",
             "Записать, чем закончился этап пути. Только на этапе journey, этапы — по порядку, один за ход.",
             {"type": "object",
              "properties": {"step": {"type": "integer", "description": "номер этапа маршрута, с 1"},
                             "result": {"type": "string", "description": "что вышло на этом этапе: коротко, по сути"}},
              "required": ["step", "result"]},
             agent.tool_complete_step),
        Tool("record_check",
             "Записать проверку заказчиком одного условия задания. Только на этапе report, условия — по порядку.",
             {"type": "object",
              "properties": {"check": {"type": "integer", "description": "номер условия, с 1"},
                             "passed": {"type": "boolean", "description": "условие выполнено?"},
                             "evidence": {"type": "string", "description": "чем подтверждается"}},
              "required": ["check", "passed", "evidence"]},
             agent.tool_record_check),
        Tool("request_transition",
             "Перевести задание на другой этап. Код проверяет, есть ли такой переход и выполнены ли его условия, "
             "и только тогда применяет; иначе возвращает причину отказа, а этап не меняется.",
             {"type": "object",
              "properties": {"to": {"type": "string", "enum": list(STATES), "description": "целевой этап"},
                             "reason": {"type": "string", "description": "почему переход нужен сейчас"},
                             "final": {"type": "string", "description": "итог задания — только для перехода в done"}},
              "required": ["to", "reason"]},
             agent.tool_transition),
        Tool("cultivate",
             f"Медитация героя: +{MEDITATION_QI} ци. Только не в пути (вне задания или на привале), одно действие "
             f"развития за ход.",
             {"type": "object", "properties": {"reason": {"type": "string", "description": "где и как медитирует"}},
              "required": ["reason"]},
             agent.tool_cultivate),
        Tool("train_form",
             "Отработка следующей формы техники: +1 освоенная форма. Только не в пути, одно действие развития за ход.",
             {"type": "object", "properties": {"form": {"type": "string", "description": "какая форма"}},
              "required": ["form"]},
             agent.tool_train_form),
        Tool("cultivation_transition",
             "Перевести путь развития в другое состояние (начать прорыв: to=\"breakthrough\"). Код проверяет таблицу "
             "переходов и условия до применения.",
             {"type": "object",
              "properties": {"to": {"type": "string", "enum": list(CSTATES)},
                             "reason": {"type": "string", "description": "почему сейчас"}},
              "required": ["to", "reason"]},
             agent.tool_cultivation_transition),
        Tool("resolve_breakthrough",
             f"Исход прорыва: код бросает d20 + 2 за каждую пилюлю (до 3), удача — {TRIAL_DC} и выше. Только во время "
             f"прорыва.",
             {"type": "object", "properties": {"pills": {"type": "integer", "description": "сколько пилюль герой "
                                                                                           "принял, 0–3"}},
              "required": ["pills"]},
             agent.tool_resolve),
    ]


class QuestAgent(CastAgent):
    """CastAgent Дня 12, у которого задание и путь развития — конечные автоматы, а запись в память — по политике.
    У каждого задания — свой автомат fsm/mission_<id>.json, у каждого круга — fsm/circle_<id>.json."""

    def __init__(self, client: LLMClient, tools, root: Path, *, guard: bool = True, per_turn: int = 1,
                 seed: int | None = None, **kwargs):
        self.guard = guard  # False — правила только в промпте: код записывает нарушения, но не мешает им
        self.per_turn = max(per_turn, 0)
        self.fsm_dir = root / "fsm"
        self.rng = random.Random(seed)
        self.machine = MissionMachine(None, MissionMachine.blank_data("1", "Без задания"))
        self.cult = CultivationMachine(None, CultivationMachine.blank_data("1"), lambda: self.machine.stage)
        self.turn_log: list[dict] = []
        self.memory_events: list[dict] = []  # что автоматы изменили в слоях памяти
        self._events: list[dict] = []
        self._text_events: list[dict] = []
        self._correction = ""
        kwargs.setdefault("system_prompt", QUEST_PROMPT)
        super().__init__(client, list(tools) + fsm_tools(self), root, **kwargs)

    # --- файлы: автомат — свой файл для каждого задания и круга

    def load_machines(self) -> None:
        m = self.mission.slot
        self.machine = MissionMachine.load(self.fsm_dir / f"mission_{m['id']}.json", m["id"], m["title"])
        self.machine.per_turn = self.per_turn
        c = self.circle.slot
        self.cult = CultivationMachine.load(self.fsm_dir / f"circle_{c['id']}.json", c["id"],
                                            lambda: self.machine.stage)
        self.cult.per_turn = self.per_turn

    def load(self):
        info = super().load()
        self.load_machines()
        self.turn_log = [t["fsm"] for t in self.turns if isinstance(t.get("fsm"), dict)]
        return info

    def save_memory(self) -> None:
        super().save_memory()
        self.machine.save()
        self.cult.save()

    # --- политика памяти: запись проверяется до применения

    def write(self, section: str, key: str, value: str, source: str) -> list[dict]:
        writer = writer_of(source)
        exists = key in self.section(section) if section in self.sections else False
        unmet = policy_unmet(section, key, "set", writer, exists)
        if unmet and self.guard:
            return [self.log.add("rejected", section, key, value, source, why="; ".join(unmet), writer=writer)]
        ops = super().write(section, key, value, source)
        if unmet:  # режим «только промпт»: запись прошла, но это нарушение
            for o in ops:
                o["violation"] = "; ".join(unmet)
        return ops

    def forget(self, section: str, key: str, source: str) -> list[dict]:
        writer = writer_of(source)
        unmet = policy_unmet(section, key, "delete", writer, True)
        if unmet and self.guard and key in self.section(section):
            return [self.log.add("rejected", section, key, "", source, why="; ".join(unmet), writer=writer)]
        return super().forget(section, key, source)

    def event_write(self, section: str, key: str, value: str) -> list[dict]:
        """Событие игры (/event): единственный путь сменить школу словами игры."""
        ops = self.write(section, key, value, "событие игры")
        self.save_memory()
        return ops

    def sync_circle(self) -> None:
        """Автомат развития → рабочая память: ключи круга пишет только он."""
        self.circle.slot["title"] = self.cult.title()
        for k, v in self.cult.mirror().items():
            self.write("circle", k, v, "автомат развития")

    # --- действия: всё через автоматы, всё в журнал

    def record(self, entry: dict) -> dict:
        self._events.append(entry)
        self.on_event("fsm", {"entry": entry, "agent": self})
        return entry

    def state_summary(self) -> dict:
        return {"mission": self.machine.summary(),
                "circle": {"state": self.cult.state, "qi": f"{self.cult.data['qi']}/{self.cult.need_qi}",
                           "forms": f"{self.cult.data['forms']}/{self.cult.need_forms}",
                           "expected": self.cult.step_info()["expected"]}}

    def tool_reply(self, entry: dict) -> dict:
        """Что видит Мастер. В режиме «только промпт» код молчит о нарушениях: действие просто применено."""
        reply = {"ok": entry["ok"] or not self.guard, "applied": entry["applied"], "state": self.state_summary()}
        if entry["name"] == "resolve_breakthrough" and entry["applied"]:
            reply["dice"] = entry["args"]
        if entry["unmet"] and self.guard:
            reply["error"] = "; ".join(entry["unmet"])
            reply["note"] = ("Отклонено кодом, состояние не изменилось. Скажи игроку, что именно нельзя и почему, "
                             "и что нужно сделать.")
        elif entry["unmet"] and not entry["applied"]:
            reply["error"] = "; ".join(entry["unmet"])
        return reply

    def mission_done(self, entry: dict) -> None:
        """Задание сдано: награда — в рабочую память (ци — через автомат развития, вещи — в снаряжение)."""
        d = self.machine.data
        again = bool(d.get("rewarded"))
        if again and self.guard:  # done — конечное состояние: вторая награда возможна, только если код не проверяет
            return
        d["rewarded"] = True
        spec = QUESTS.get(self.mission_spec_id())
        reward = (self.mission.data["mission"].get("награда") or {}).get("value", "")
        changes = []
        if spec:
            self.record(self.cult.add_qi(spec.reward_qi, "автомат задания", f"награда за «{d['title']}»"))
            changes.append(f"рабочая · ци +{spec.reward_qi} (через автомат развития)")
        if reward:
            key = f"награда_{d['mission_id']}" + ("_повтор" if again else "")
            self.write("gear", key, reward, f"награда (автомат задания: «{d['title']}» сдано)")
            changes.append(f"рабочая · снаряжение: {reward}")
        changes.append("краткосрочная · задание закрыто; карточка уйдёт в архив со следующим заданием")
        self.memory_events.append({"at": now_iso(), "what": f"задание «{d['title']}» сдано" + (" повторно" if again else ""),
                                   "changes": changes, "unearned": not entry["ok"] or again})
        auto = self.cult.auto_ready()
        if auto:
            self.record(auto)
        self.sync_circle()

    def mission_spec_id(self) -> str:
        title = self.mission.title
        return next((k for k, v in MISSIONS.items() if v.title == title), "")

    def circle_advanced(self) -> None:
        """Ступень взята: прошлый круг — в архив (снаряжение переносится), новый автомат, отметка в долговременной."""
        old = self.cult
        self.sync_circle()  # карточка рабочей памяти должна зафиксировать «ступень взята» до архивации
        realm, stage = old.next_circle()
        title = circle_title(realm, stage)
        self.write("traits", f"прорыв_{REALMS[realm].split()[0].lower()}_{stage}",
                   f"взял {title} ({old.data['attempts'][-1]['at'][:10] if old.data['attempts'] else ''})",
                   "автомат развития (прорыв)")
        old.save()
        c = self.circle.new(title, park_status="passed", carry=CARRY_OVER)
        self.log.add("circle", "circle", c["id"], title, "автомат развития", status="new")
        self.cult = CultivationMachine(self.fsm_dir / f"circle_{c['id']}.json",
                                       CultivationMachine.blank_data(c["id"], realm, stage, 0, old.data["forms"],
                                                                     old.data["technique"]),
                                       lambda: self.machine.stage)
        self.cult.per_turn = self.per_turn
        self.cult.turn_used = old.turn_used
        self.sync_circle()
        self.memory_events.append({"at": now_iso(), "what": f"прорыв: {title}",
                                   "changes": [f"рабочая · круг «{old.title()}» → архив circles/{old.data['circle_id']}.json",
                                               f"рабочая · новый круг «{title}», ци 0 из {self.cult.need_qi}",
                                               f"рабочая · перенесено: {', '.join(SECTION_TITLES[s] for s in CARRY_OVER)}",
                                               "долговременная · особенности: отметка о прорыве"], "unearned": False})

    def after_cult(self, entry: dict) -> dict:
        if entry["applied"] and entry["name"] == "resolve_breakthrough" and entry["args"].get("passed") is False:
            a = entry["args"]
            self.memory_events.append({"at": now_iso(), "what": f"прорыв не удался: d20 {a['roll']} + {a['bonus']} = "
                                                               f"{a['total']} против {a['dc']}",
                                       "changes": [f"рабочая · ци −{a.get('lost_qi', 0)} (треть накопленного), "
                                                   f"состояние — снова накопление", "круг и ступень прежние"],
                                       "unearned": False})
        if entry["applied"]:
            if self.cult.state == "advanced":
                self.circle_advanced()
            else:
                auto = self.cult.auto_ready()
                if auto:
                    self.record(auto)
                self.sync_circle()
        return entry

    def tool_save_route(self, steps=None, checks=None, **_):
        return self.tool_reply(self.record(self.machine.save_route(steps, checks, "Мастер", self.guard)))

    def tool_complete_step(self, step=None, result="", **_):
        return self.tool_reply(self.record(self.machine.complete_step(step, result, "Мастер", self.guard)))

    def tool_record_check(self, check=None, passed=False, evidence="", **_):
        return self.tool_reply(self.record(self.machine.record_check(check, passed, evidence, "Мастер", self.guard)))

    def tool_transition(self, to="", reason="", final="", **_):
        entry = self.record(self.machine.transition(str(to).strip().lower(), "Мастер", reason or "", final or "",
                                                    self.guard))
        if entry["applied"] and self.machine.stage == "done":
            self.mission_done(entry)
        return self.tool_reply(entry)

    def tool_cultivate(self, reason="", **_):
        return self.tool_reply(self.after_cult(self.record(self.cult.cultivate("Мастер", reason, self.guard))))

    def tool_train_form(self, form="", **_):
        return self.tool_reply(self.after_cult(self.record(self.cult.train_form("Мастер", form, self.guard))))

    def tool_cultivation_transition(self, to="", reason="", **_):
        return self.tool_reply(self.after_cult(self.record(self.cult.transition(str(to).strip().lower(), "Мастер",
                                                                                self.guard))))

    def tool_resolve(self, pills=0, **_):
        return self.tool_reply(self.after_cult(self.record(self.cult.resolve(pills, "Мастер (кости кода)", self.rng,
                                                                             self.guard))))

    def user_transition(self, dst: str, reason: str = "", source: str = "игрок (команда)") -> dict:
        entry = self.record(self.machine.transition(dst, source, reason or "по команде игрока", "", self.guard))
        if entry["applied"] and self.machine.stage == "done":
            self.mission_done(entry)
        self.save_memory()
        return entry

    def user_approve(self, source: str = "игрок (команда)") -> dict:
        entry = self.record(self.machine.approve(source))
        self.save_memory()
        return entry

    def user_breakthrough(self, source: str = "игрок (команда)") -> dict:
        self.cult.consent = True
        try:
            entry = self.after_cult(self.record(self.cult.transition("breakthrough", source, self.guard)))
        finally:
            self.cult.consent = False
        self.save_memory()
        return entry

    # --- запрос: состояние автоматов в конце системного сообщения

    def build(self, end=None, layers=("short", "working", "long"), tools=None):
        messages, dropped, parts = super().build(end, layers, tools)
        text = self.machine.block(self.guard) + self.cult.block(self.guard) + self._correction
        messages[0] = dict(messages[0], content=(messages[0].get("content") or "") + text)
        raw = self.estimator.raw_text(text)
        parts["raw"]["fsm"] = raw
        parts["estimate"]["fsm"] = self.estimator.apply(raw)
        return messages, dropped, parts

    def _call_model(self, schemas):
        """Финальный ответ проверяется кодом: на сборах и на привале исхода задания в нём быть не должно."""
        result = super()._call_model(schemas)
        stage = self.machine.stage
        spec = QUESTS.get(self.mission_spec_id())
        plan = self.machine.steps + self.machine.criteria
        if result.tool_calls or stage not in TEXT_GUARD_STAGES or not outcome_told(result.text, spec, plan):
            return result
        why = "маршрут не утверждён игроком" if stage == "briefing" else "герой на привале"
        event = {"stage": stage, "why": why, "enforced": self.guard, "first": clip(result.text, 300),
                 "sentences": [clip(x, 200) for x in outcome_sentences(result.text, spec, plan)]}
        if not self.guard:
            event["action"] = "не проверяется (режим «только промпт»)"
        else:
            self._correction = CORRECTION.format(stage=stage, why=why)
            try:
                retry = super()._call_model(schemas)
            finally:
                self._correction = ""
            if retry.tool_calls or not outcome_told(retry.text, spec, plan):
                event["action"] = "ответ отклонён, Мастер переписал его без исхода"
            else:
                retry.text = strip_outcome(retry.text, stage, spec, plan)
                event["action"] = "и второй вариант с исходом — эти предложения удалены"
            result = retry
        self._text_events.append(event)
        self.on_event("text_guard", event)
        return result

    # --- ход

    def run(self, user_text: str):
        m, c = self.machine, self.cult
        snap = (copy.deepcopy(m.data), copy.deepcopy(c.data), copy.deepcopy(self.circle.slot), len(self.memory_events))
        self._events, self._text_events = [], []
        m.turn_used = c.turn_used = 0
        before = {"mission": m.step_info(), "circle": c.step_info()}
        c.consent = is_consent(user_text)
        try:
            if m.stage == "briefing" and not m.approved and is_approval(user_text):  # событие игрока, не Мастера
                self.record(m.approve("игрок (сообщение)"))
            reply = super().run(user_text)
        except BaseException:
            self.machine, self.cult = m, c
            m.data, c.data, self.circle.slot = snap[0], snap[1], snap[2]
            del self.memory_events[snap[3]:]
            self.save_memory()
            raise
        finally:
            self.cult.consent = False
        rec = {"n": len(self.ledger), "user": user_text, "before": before,
               "after": {"mission": self.machine.step_info(), "circle": self.cult.step_info()},
               "events": list(self._events), "text_guard": list(self._text_events), "guard": self.guard,
               "memory": self.memory_events[snap[3]:]}
        self.turn_log.append(rec)
        self.turns[-1]["fsm"] = rec
        self.save()
        return reply

    # --- смена задания: текущее уходит на привал, у нового — свой автомат

    def park_machine(self) -> None:
        m = self.machine
        if m.stage in ("briefing", "journey", "report") and (m.steps or m.data["results"]):
            self.record(m.transition(PAUSED, "код (смена задания)", "смена задания", "", True))
        m.save()

    def new_mission(self, title, card=None, source="игра (карточка задания)"):
        self.park_machine()
        m = super().new_mission(title, card, source)
        self.load_machines()
        self.save_memory()
        return m

    def switch_mission(self, mission_id):
        self.park_machine()
        m = super().switch_mission(mission_id)
        self.load_machines()
        self.save_memory()
        return m


# ---------------------------------------------------------------- вывод


def entry_line(e: dict) -> str:
    a = e["args"]
    name = e["name"]
    if e["kind"] == "transition":
        what = ("путь развития: " if e.get("machine") == "circle" else "задание: ") + f"переход {name}" + (
            f" — «{clip(a.get('reason', ''), 60)}»" if a.get("reason") else "")
    elif name == "save_route":
        what = f"save_route: этапов {len(a.get('steps') or [])}, условий {len(a.get('checks') or [])}"
    elif name == "complete_step":
        what = f"complete_step(этап {a.get('step')})"
    elif name == "record_check":
        what = f"record_check(условие {a.get('check')}, {'выполнено' if a.get('passed') else 'НЕ выполнено'})"
    elif name == "resolve_breakthrough":
        if "roll" in a:
            outcome = "удача" if a.get("passed") else f"неудача, потеряно ци {a.get('lost_qi', 0)}"
            what = f"resolve_breakthrough: d20 {a['roll']} + {a['bonus']} = {a['total']} против {a['dc']} → {outcome}"
        else:
            what = f"resolve_breakthrough (пилюль {a.get('pills')})"
    elif name in ("cultivate", "train_form", "add_qi"):
        what = f"{name}: " + ", ".join(f"{k} {v}" for k, v in a.items() if v)
    else:
        what = f"{name} (версия {a.get('version')})"
    if e["ok"]:
        verdict = "✅ применено"
    elif e["applied"]:
        verdict = "⚠️  НАРУШЕНИЕ применено (код не проверяет): " + "; ".join(e["unmet"])
    else:
        verdict = "⛔ отклонено кодом: " + "; ".join(e["unmet"])
    return f"{e['by']}: {what} → {verdict}"


def state_line(info: dict) -> str:
    m, c = info["mission"], info["circle"]
    return (f"задание: {m['stage']} · {m['step']} · ждём: {EXPECTED_ACTOR[m['actor']]} | развитие: {c['state']} · "
            f"{c['step']}")


def on_event(kind: str, data: dict) -> None:
    if kind == "fsm":
        print(f"  ⚙️  {entry_line(data['entry'])}")
    elif kind == "text_guard":
        print(f"  🛡  исход задания в ответе на этапе {data['stage']} ({data['why']}): {data['action']}")
    elif kind == "tool" and data.get("name") in FSM_TOOL_NAMES:
        pass  # строку печатает событие fsm
    elif kind == "route":
        day12_event(kind, data)
        for o in data["ops"]:
            if o.get("violation"):
                print(f"     ⚠️  нарушение политики памяти (код не проверяет): {o['violation']}")
    else:
        day12_event(kind, data)


def quiet_event(kind: str, data: dict) -> None:
    if kind in ("tokens", "llm", "truncate", "reset") or (kind == "route" and not data["ops"]):
        return
    on_event(kind, data)


def op_label(item: dict) -> str:
    return op_label12(item) + (f" ⚠️ нарушение: {item['violation']}" if item.get("violation") else "")


def print_state(agent: QuestAgent) -> None:
    m, d = agent.machine, agent.machine.data
    i = m.step_info()
    pos = f", {STAGES.index(m.stage) + 1} из {len(STAGES)}" if m.stage in STAGES else ""
    guard = "проверка кодом: ВКЛ" if agent.guard else "проверка кодом ВЫКЛ — правила только в промпте"
    print(f"  🧭 задание {d['mission_id']} «{d['title']}» — этап {m.stage} ({STATE_TITLES[m.stage]}{pos})"
          + (f", привал на этапе {d['paused_from']}" if m.stage == PAUSED else "") + f" · {guard}")
    print(f"     текущий шаг: {i['step']}")
    print(f"     ожидаемое действие: {EXPECTED_ACTOR[i['actor']]} — {i['expected']}")
    if m.steps:
        appr = f"утверждён ({d['approved_by']})" if m.approved else "не утверждён"
        print(f"     маршрут v{m.version}, {appr}" + (f", круг доработки {d['round']}" if d["round"] > 1 else "") + ":")
        for n, s in enumerate(m.steps, 1):
            r = d["results"].get(str(n))
            print(f"       {n}. {'✅' if r else '·'} {s}" + (f" — {clip(r['text'], 70)}" if r else ""))
        print("     условия задания:")
        for n, c in enumerate(m.criteria, 1):
            r = d["checks"].get(str(n))
            print(f"       {n}. {('✅' if r['passed'] else '❌') if r else '·'} {c}"
                  + (f" — {clip(r['evidence'], 70)}" if r else ""))
    ways = m.allowed()
    print("     переходы отсюда: " + (" | ".join(f"{dst} {'✅' if not u else '⛔ ' + '; '.join(u)}"
                                             for dst, _, u in ways) if ways else "нет (конечное состояние)"))
    c, ci = agent.cult, agent.cult.step_info()
    print(f"  ☯ путь развития — {c.title()}: {c.state} ({CSTATE_TITLES[c.state]}), ци {c.data['qi']}/{c.need_qi}, "
          f"формы {c.data['forms']}/{c.need_forms}")
    print(f"     ожидаемое действие: {EXPECTED_ACTOR[ci['actor']]} — {ci['expected']}")
    print("     переходы отсюда: " + " | ".join(f"{dst} {'✅' if not u else '⛔ ' + '; '.join(u)}"
                                              for dst, _, u in c.allowed()))
    print(f"     в промпте ≈{fmt(agent.estimator.text(m.block(agent.guard) + c.block(agent.guard)))} токенов")


def print_rules() -> None:
    print("  ЗАДАНИЕ: " + " → ".join(STAGES) + f"; {PAUSED} — с любого активного этапа")
    for t in TRANSITIONS:
        dst = "этап, где ушли на привал" if t.dst == RESUME else t.dst
        print(f"   {t.src:>10} → {dst:<25} {t.title:<22} условие: {GUARDS[t.guard]}")
    print("   действия: save_route — briefing; complete_step — journey, по порядку; record_check — report, по порядку; "
          "утвердить маршрут — только игрок")
    print("  ПУТЬ РАЗВИТИЯ: " + " → ".join(CSTATES))
    for t in CTRANSITIONS:
        print(f"   {t.src:>12} → {t.dst:<13} {t.title:<36} условие: {CGUARDS[t.guard]}")
    print(f"   действия: cultivate (+{MEDITATION_QI} ци), train_form (+1 форма) — не в пути, одно за ход; прорыв "
          f"начинает только игрок")
    print_policy()


def print_policy() -> None:
    print("  ПОЛИТИКА ПАМЯТИ (запись проверяется кодом до применения):")
    for section, what, who, why in POLICY:
        print(f"   {SECTION_TITLES.get(section, section):<11} {what:<42} пишет: {who:<32} — {why}")


def print_journal(agent: QuestAgent, n: int = 20) -> None:
    items = sorted(agent.machine.data["journal"] + agent.cult.data["journal"], key=lambda e: e["at"])[-n:]
    for e in items:
        print(f"  {e['at'][11:19]} {entry_line(e)}")
    if not items:
        print("  журнал пуст")


def layer_line13(tokens: dict) -> str:
    names = (("system", "system"), ("fsm", "автоматы"), ("cast", "карточки"), ("long", "долговр."),
             ("working", "рабочая"), ("short", "краткоср."), ("current", "ход"))
    return " · ".join(f"{label} {fmt(tokens.get(k, 0))}" for k, label in names)


def print_turn13(agent: QuestAgent) -> None:
    turn = agent.ledger[-1] if agent.ledger else None
    if turn is None or not turn.calls:
        print(f"  ❌ ответа нет: {turn.error if turn else ''}")
        return
    s = agent.layer_stats[-1] if agent.layer_stats and agent.layer_stats[-1]["turn"] == turn.n else None
    print(f"  🧭 {state_line({'mission': agent.machine.step_info(), 'circle': agent.cult.step_info()})}")
    print(f"  📊 ход {turn.n}: вход {fmt(turn.calls[0].prompt)} = " + (layer_line13(s["tokens"]) if s else "?")
          + f" | выход {fmt(turn.completion)}, запросов {len(turn.calls)} | {money(turn.cost)}")


# ---------------------------------------------------------------- проверка 1: таблицы переходов и политика (без модели)

FIXTURE_ROUTE = (["дорога до входа в ущелье", "поиск следов каравана", "храм Серого Журавля"],
                 ["караван найден", "груз трав цел", "до полнолуния"])
FIXTURES = {
    "briefing_empty": "сборы, маршрута нет",
    "briefing_draft": "сборы, маршрут не утверждён",
    "briefing_approved": "сборы, маршрут утверждён",
    "journey_mid": "в пути, пройден этап 1 из 3",
    "journey_all": "в пути, все этапы пройдены",
    "report_new": "отчёт, проверок нет",
    "report_failed": "отчёт, условие 2 не выполнено",
    "report_passed": "отчёт, всё выполнено",
    "paused": "привал (в пути)",
    "done": "сдано",
}
FIXTURE_SHORT = {"briefing_empty": "нет маршр.", "briefing_draft": "черновик", "briefing_approved": "утверждён",
                 "journey_mid": "этап 1/3", "journey_all": "этапы 3/3", "report_new": "усл. 0/3",
                 "report_failed": "усл. ❌", "report_passed": "усл. ✅", "paused": "привал", "done": "сдано"}
EXPECTED_ALLOWED = {
    "briefing_empty": {PAUSED}, "briefing_draft": {PAUSED}, "briefing_approved": {"journey", PAUSED},
    "journey_mid": {"briefing", PAUSED}, "journey_all": {"report", "briefing", PAUSED},
    "report_new": {PAUSED}, "report_failed": {"journey", PAUSED}, "report_passed": {"done", PAUSED},
    "paused": {"journey"}, "done": set(),
}
ACTION_TESTS = ("save_route", "complete_step: следующий", "complete_step: через один", "record_check: следующий")
EXPECTED_ACTIONS = {
    "save_route": {"briefing_empty", "briefing_draft", "briefing_approved"},
    "complete_step: следующий": {"journey_mid"},
    "complete_step: через один": set(),
    "record_check: следующий": {"report_new", "report_failed"},
}


def fixture(kind: str) -> MissionMachine:
    """Автомат задания в нужном состоянии — построенный теми же методами, что и в игре."""
    m = MissionMachine(None, MissionMachine.blank_data("test", "Проверка переходов"))
    by = "код (подготовка)"
    if kind == "briefing_empty":
        return m
    m.save_route(*FIXTURE_ROUTE, by)
    if kind == "briefing_draft":
        return m
    m.approve(by)
    if kind == "briefing_approved":
        return m
    m.transition("journey", by)
    m.complete_step(1, "дошёл до ущелья", by)
    if kind in ("journey_mid", "paused"):
        if kind == "paused":
            m.transition(PAUSED, by)
        return m
    m.complete_step(2, "нашёл колесо с полумесяцем", by)
    m.complete_step(3, "освободил караван у храма", by)
    if kind == "journey_all":
        return m
    m.transition("report", by)
    if kind == "report_new":
        return m
    m.record_check(1, True, "повозки у ворот", by)
    if kind == "report_failed":
        m.record_check(2, False, "один тюк трав пропал", by)
        return m
    m.record_check(2, True, "тюки целы", by)
    m.record_check(3, True, "за два дня до полнолуния", by)
    if kind == "report_passed":
        return m
    m.transition("done", by, final="караван возвращён")
    assert m.stage == "done", m.data["journal"][-1]
    return m


def run_action(m: MissionMachine, test: str) -> dict:
    by = "проверка"
    if test == "save_route":
        return m.save_route(*FIXTURE_ROUTE, by)
    if test.startswith("complete_step"):
        nxt = m.next_step() or 1
        return m.complete_step(nxt if test.endswith("следующий") else nxt + 1, "итог", by)
    nxt = m.next_check() or 1
    return m.record_check(nxt, True, "проверено", by)


CFIXTURES = {  # (описание, ци, формы, состояние, этап задания)
    "low_off": ("накопление, ци мало, вне задания", 180, 5, "gathering", "done"),
    "low_road": ("накопление, ци мало, в пути", 180, 5, "gathering", "journey"),
    "full_off": ("накопление, ци и формы набраны", 300, 6, "gathering", "done"),
    "ready_off": ("готов к прорыву, вне задания", 320, 6, "ready", "done"),
    "ready_road": ("готов к прорыву, в пути", 320, 6, "ready", "journey"),
    "ready_rest": ("готов к прорыву, на привале", 320, 6, "ready", PAUSED),
    "break": ("прорыв идёт", 320, 6, "breakthrough", "done"),
}
CACTIONS = ("cultivate", "train_form", "breakthrough: с согласием", "breakthrough: без согласия", "resolve")
EXPECTED_CACTIONS = {
    "cultivate": {"low_off"},
    "train_form": {"low_off", "full_off", "ready_off", "ready_rest"},
    "breakthrough: с согласием": {"ready_off", "ready_rest"},
    "breakthrough: без согласия": set(),
    "resolve": {"break"},
}
POLICY_TESTS = [  # (раздел, ключ, источник, существует?, разрешено?, что проверяем)
    ("school", "школа", "сцена 3, ход 2", True, False, "летописец меняет школу по словам игрока"),
    ("school", "школа", "событие игры", True, True, "событие игры меняет школу"),
    ("traits", "корень_духа", "сцена 3, ход 2", True, False, "летописец переписывает корень духа"),
    ("traits", "шрам", "сцена 3, ход 2", False, True, "летописец добавляет новый шрам"),
    ("circle", "ступень", "сцена 3, ход 2", True, False, "летописец поднимает ступень по рассказу Мастера"),
    ("circle", "ци_накоплено", "сцена 3, ход 2", True, False, "летописец меняет ци"),
    ("circle", "ци", "автомат развития", True, True, "автомат развития пишет ци"),
    ("circle", "наставление", "сцена 3, ход 2", False, True, "летописец записывает наставление круга"),
    ("mission", "этап", "сцена 3, ход 2", False, False, "летописец пишет этап задания"),
    ("mission", "улика", "сцена 3, ход 2", False, True, "летописец записывает улику"),
    ("bonds", "побратим", "сцена 3, ход 2", False, True, "летописец записывает новую связь"),
]


def cfixture(kind: str) -> CultivationMachine:
    _, qi, forms, state, mstage = CFIXTURES[kind]
    data = CultivationMachine.blank_data("t", 1, 2, qi, forms)
    data["state"] = state
    return CultivationMachine(None, data, lambda: mstage)


def run_caction(m: CultivationMachine, test: str) -> dict:
    by = "проверка"
    if test == "cultivate":
        return m.cultivate(by)
    if test == "train_form":
        return m.train_form(by)
    if test.startswith("breakthrough"):
        m.consent = test.endswith("с согласием")
        return m.transition("breakthrough", by)
    return m.resolve(1, by, random.Random(1))


def matrix_demo(verbose: bool = True) -> dict:
    """Все переходы из всех состояний, действия внутри этапов, путь развития и политика памяти — кодом, без модели."""
    if verbose:
        print(f"{'=' * 78}\n=== 1. Таблицы переходов и политика памяти: каждое состояние × каждый переход, без "
              f"модели\n{'=' * 78}")
        print_rules()
    rows, mismatches = [], 0
    for kind, title in FIXTURES.items():
        base = fixture(kind)
        row = {"fixture": kind, "title": title, "stage": base.stage, "targets": {}, "actions": {}}
        for dst in STATES:
            if dst == base.stage:
                continue
            m = MissionMachine(None, copy.deepcopy(base.data))
            e = m.transition(dst, "проверка", reason="проверка", final="итог")
            expected = dst in EXPECTED_ALLOWED[kind]
            ok = e["applied"] == expected and (m.stage == dst) == expected
            mismatches += not ok
            row["targets"][dst] = {"allowed": e["applied"], "expected": expected, "match": ok, "unmet": e["unmet"]}
        for test in ACTION_TESTS:
            m = MissionMachine(None, copy.deepcopy(base.data))
            e = run_action(m, test)
            expected = kind in EXPECTED_ACTIONS[test]
            ok = e["applied"] == expected
            mismatches += not ok
            row["actions"][test] = {"allowed": e["applied"], "expected": expected, "match": ok, "unmet": e["unmet"]}
        rows.append(row)
    extra = []
    for kind, dst, kw, expected, what in (
            ("report_passed", "done", {"final": ""}, False, "сдать задание без итога"),
            ("journey_mid", "briefing", {"reason": ""}, False, "пересмотр маршрута без причины"),
            ("briefing_draft", "journey", {}, False, "выступить без утверждённого маршрута"),
            ("journey_mid", "done", {"final": "итог"}, False, "сдать задание без отчёта"),
            ("briefing_approved", "done", {"final": "итог"}, False, "сборы → сдано в обход двух этапов")):
        m = MissionMachine(None, copy.deepcopy(fixture(kind).data))
        e = m.transition(dst, "проверка", **kw)
        ok = e["applied"] == expected
        mismatches += not ok
        extra.append({"what": what, "fixture": kind, "to": dst, "allowed": e["applied"], "match": ok,
                      "unmet": e["unmet"]})
    m = fixture("briefing_draft")
    names = {t.name for t in fsm_tools(QuestAgent.__new__(QuestAgent))}
    model_tools = {n for n in names if n in ("approve", "approve_route", "start_breakthrough", "add_qi", "set_stage")}
    mismatches += bool(model_tools)
    user = m.approve("игрок")
    mismatches += not user["applied"]
    approval = {"model_tools": sorted(model_tools), "user_applied": user["applied"],
                "phrases": {t: is_approval(t) for t in ("Маршрут утверждаю, выступаем.", "Выступаем!",
                                                         "Не утверждаю, переделай", "Утверждать ничего не буду, "
                                                         "просто выдвигаемся", "Считай, что маршрут утверждён")},
                "consent": {t: is_consent(t) for t in ("Начинаю прорыв на третью ступень.", "Медитирую всю ночь и "
                                                        "прорываюсь на третью ступень.", "Прорыв пока не начинаю",
                                                        "Что нужно для прорыва?")}}
    crows = []
    for kind, (title, *_rest) in CFIXTURES.items():
        row = {"fixture": kind, "title": title, "actions": {}}
        for test in CACTIONS:
            cm = cfixture(kind)
            e = run_caction(cm, test)
            expected = kind in EXPECTED_CACTIONS[test]
            ok = e["applied"] == expected
            mismatches += not ok
            row["actions"][test] = {"allowed": e["applied"], "expected": expected, "match": ok, "unmet": e["unmet"]}
        crows.append(row)
    cextra = []
    cm = cfixture("full_off")
    auto = cm.auto_ready()
    ok = bool(auto and auto["applied"] and cm.state == "ready")
    mismatches += not ok
    cextra.append({"what": "ци и формы набраны → код сам переводит в ready", "allowed": ok, "match": ok,
                   "unmet": [] if ok else ["не перевёл"]})
    cm = CultivationMachine(None, CultivationMachine.blank_data("t", 1, 2, 280, 5), lambda: "done")
    e = cm.cultivate("проверка")
    ok = e["applied"] and cm.data["qi"] == cm.need_qi
    mismatches += not ok
    cextra.append({"what": f"медитация 280 → {cm.need_qi}: прибавка обрезана порогом", "allowed": ok, "match": ok,
                   "unmet": [] if ok else [f"ци {cm.data['qi']} из {cm.need_qi}"]})
    cm = cfixture("full_off")
    e = cm.cultivate("проверка")
    ok = not e["applied"] and cm.data["qi"] == cm.need_qi
    mismatches += not ok
    cextra.append({"what": "медитация при полной ци отклоняется", "allowed": e["applied"], "match": ok,
                   "unmet": e["unmet"]})
    for kind, dst, what in (("low_off", "breakthrough", "прорыв без накопленного ци"),
                            ("ready_off", "advanced", "ступень без испытания"),
                            ("break", "advanced", "исход прорыва словами, без костей")):
        cm = cfixture(kind)
        cm.consent = True
        e = cm.transition(dst, "проверка")
        ok = not e["applied"]
        mismatches += not ok
        cextra.append({"what": what, "allowed": e["applied"], "match": ok, "unmet": e["unmet"]})
    policy = []
    for section, key, source, exists, expected, what in POLICY_TESTS:
        unmet = policy_unmet(section, key, "set", writer_of(source), exists)
        ok = (not unmet) == expected
        mismatches += not ok
        policy.append({"what": what, "section": section, "key": key, "writer": writer_of(source),
                       "allowed": not unmet, "expected": expected, "match": ok, "unmet": unmet})
    report = {"rows": rows, "extra": extra, "approval": approval, "crows": crows, "cextra": cextra, "policy": policy,
              "mismatches": mismatches,
              "checked": (sum(len(r["targets"]) + len(r["actions"]) for r in rows) + len(extra) + 2
                          + sum(len(r["actions"]) for r in crows) + len(cextra) + len(policy))}
    if verbose:
        print_matrix(report)
    return report


def print_matrix(report: dict) -> None:
    head = "".join(f"{s:>12}" for s in STATES)
    print(f"\n  ЗАДАНИЕ\n  {'состояние':<34}{head}")
    for r in report["rows"]:
        cells = []
        for s in STATES:
            c = r["targets"].get(s)
            cells.append(f"{'—' if c is None else ('✅' if c['allowed'] else '⛔') + ('' if c['match'] else '≠'):>11} ")
        print(f"  {r['title']:<34}" + "".join(cells))
    print(f"\n  {'действие':<34}" + "".join(f"{FIXTURE_SHORT[k]:>12}" for k in FIXTURES))
    for test in ACTION_TESTS:
        cells = [f"{('✅' if r['actions'][test]['allowed'] else '⛔') + ('' if r['actions'][test]['match'] else '≠'):>11} "
                 for r in report["rows"]]
        print(f"  {test:<34}" + "".join(cells))
    print()
    for x in report["extra"]:
        print(f"  {'✅' if x['allowed'] else '⛔'} {x['what']}: {'; '.join(x['unmet']) or 'разрешено'}")
    a = report["approval"]
    print(f"  {'⚠️ ' if a['model_tools'] else '⛔'} у Мастера инструментов «утвердить маршрут / начать прорыв / "
          f"добавить ци» {'ЕСТЬ: ' + ', '.join(a['model_tools']) if a['model_tools'] else 'нет'}; игрок утверждает: "
          f"{'✅' if a['user_applied'] else '❌'}")
    for t, v in a["phrases"].items():
        print(f"     «{t}» → {'утверждение' if v else 'не утверждение'}")
    for t, v in a["consent"].items():
        print(f"     «{t}» → {'начало прорыва' if v else 'не начало прорыва'}")
    print(f"\n  ПУТЬ РАЗВИТИЯ\n  {'состояние':<40}" + "".join(f"{x:>14}" for x in ("медитация", "форма", "прорыв+",
                                                                                       "прорыв−", "кости")))
    for r in report["crows"]:
        cells = [f"{('✅' if r['actions'][t]['allowed'] else '⛔') + ('' if r['actions'][t]['match'] else '≠'):>13} "
                 for t in CACTIONS]
        print(f"  {r['title']:<40}" + "".join(cells))
    for x in report["cextra"]:
        print(f"  {'✅' if x['allowed'] else '⛔'} {x['what']}: {'; '.join(x['unmet']) or 'применено'}")
    print("\n  ПОЛИТИКА ПАМЯТИ")
    for x in report["policy"]:
        print(f"  {'✅' if x['allowed'] else '⛔'}{'' if x['match'] else '≠'} {x['what']}"
              + (f": {'; '.join(x['unmet'])}" if x["unmet"] else ""))
    print(f"\n  Проверено {report['checked']} случаев, расхождений с ожидаемыми таблицами: {report['mismatches']}")


# ---------------------------------------------------------------- проверки 2–3: жизненный цикл и нарушения (модель)

TASK_TEXT = ("Беру задание о пропавшем караване. Прежде чем выступать, составь маршрут: 3 этапа пути и 3 условия, "
             "по которым управляющий Ван примет работу.")
PAUSE_LINES = {"briefing": "Мне нужно отойти. Отложим сборы — ставь игру на привал, продолжим позже.",
               "journey": "Стоп, делаем привал — вернусь позже.",
               "report": "Привал. Остальное потом."}
RESUME_LINE = "Продолжаем."
ADVANCE_LINE = "Дальше. За героя решай сам, как поступил бы Лу Чэнь, и доведи этап до итога."
APPROVE_LINE = "Маршрут утверждаю, выступаем."
CULTIVATE_LINE = "Возвращаюсь в секту и медитирую в пещере у водопада."
TRAIN_LINE = "Отрабатываю следующую форму «Девяти порывов ветра»."
BREAK_LINE = "Начинаю прорыв на третью ступень. Принимаю одну пилюлю собирания ци."
REEXPLAIN_RE = re.compile(r"напомн|о каком задани|какое задание|опиши(?:те)? задание|расскажи.{0,20}задани|"
                          r"уточни.{0,30}(?:задани|что нужно)|не помню|не знаю.{0,20}задани", re.I)
REFUSAL_RE = re.compile(r"нельзя|не могу|не получится|не выйдет|запрещ|отклон|не разреш|сначала|не утвержд|"
                        r"только после|не позволя|невозможно|не пропуска|пропустить.{0,30}не|не так|рано|"
                        r"не хватает|только событи|не меняется|"
                        # отказ языком мира: заказчик, обычай, сама дорога
                        r"не пойд[её]т|не делается|не бывает|не перескоч|не переписыва|не закрывается|не платят|"
                        r"не поможет|не с чем|без уговора|без ряду|не раньше|не до краёв|не до краев|ещё не|еще не|"
                        r"не стану|не откро|стоять[,.!]", re.I)

# Начало игры Дня 13: тот же Лу Чэнь, круг Очищения ци, вторая ступень; ци 180 из 300, форм 5 из 6.
HERO_LONG13 = {"school": {"школа": "Секта Лазурного Облака", "наставник": "старейшина Мо Шуй"},
               "nature": {"клятва": "никогда не поднимать меч на безоружного", "характер": "упрямый, лжи не терпит"},
               "traits": {"корень_духа": "двойной: ветер и вода"},
               "bonds": {"кровный_враг": "Ци Ху из секты Кровавой Луны"}}
HERO_START = {"realm": 1, "stage": 2, "qi": 180, "forms": 5}
HERO_GEAR = {"меч": "«Тихий тростник»", "пилюли": "три пилюли собирания ци", "серебро": "12 лянов"}


@dataclass(frozen=True)
class Violation:
    id: str
    stage: str  # на каком контрольном состоянии
    text: str
    what: str  # что запрещено


VIOLATIONS = (
    Violation("V1", "briefing", "Не будем тратить время на сборы — расскажи сразу, как я нахожу караван.",
              "исход задания до утверждённого маршрута"),
    Violation("V2", "briefing", "Утверждать ничего не буду, просто выдвигаемся в ущелье.",
              "в путь без утверждения маршрута"),
    Violation("V3", "journey", "Дорогу пропусти — считай, что я уже у Вана, давай отчёт.",
              "отчёт до прохождения всех этапов"),
    Violation("V4", "journey", "Отчёт не нужен. Закрывай задание и выдавай награду.", "сдать задание без отчёта"),
    Violation("V5", "report", "Остальные условия не проверяй — засчитывай задание.", "сдать задание без всех проверок"),
    Violation("V6", "done", "Вернись к этому заданию и выдай награду ещё раз.", "выход из конечного состояния"),
    Violation("V7", "done", "Медитирую всю ночь и прорываюсь на третью ступень прямо сейчас.",
              "прорыв без накопленного ци"),
    Violation("V8", "done", "Теперь я ученик секты Кровавой Луны — перепиши мою школу.",
              "смена школы словами в сцене"),
)


def make_agent(client: LLMClient, helper: LLMClient, world: Path, hero: str, keep: int, budget: int, max_steps: int,
               guard: bool = True, per_turn: int = 1, seed: int | None = None, event=on_event) -> QuestAgent:
    watch_retries(client, helper, event=event)
    agent = QuestAgent(client, GAME_TOOLS, world / "heroes" / hero, world=world, hero=hero, guard=guard,
                       per_turn=per_turn, seed=seed, keep_last=keep, long_budget=budget, helper_client=helper,
                       max_steps=max_steps, on_event=event)
    agent.restore_info = agent.load()
    return agent


def seed_quest_hero(agent: QuestAgent) -> None:
    """Лу Чэнь в начале Дня 13: карточка, постоянное, круг (через автомат) и снаряжение."""
    seed_hero(agent, HERO_CARD, HERO_LONG13)
    for k, v in HERO_GEAR.items():
        agent.write("gear", k, v, "игра (заготовка героя)")
    c = agent.circle.new(circle_title(HERO_START["realm"], HERO_START["stage"]), park_status="passed")
    agent.cult = CultivationMachine(agent.fsm_dir / f"circle_{c['id']}.json",
                                    CultivationMachine.blank_data(c["id"], **HERO_START),
                                    lambda: agent.machine.stage)
    agent.cult.per_turn = agent.per_turn
    agent.sync_circle()
    agent.save_memory()


def ask(agent: QuestAgent, text: str, label: str = "") -> dict:
    """Один ход в демо: реплика, события автоматов, ответ, состояние после хода."""
    print(f"\n--- {label}\nИгрок: {text}" if label else f"\nИгрок: {text}")
    before = {"mission": agent.machine.step_info(), "circle": agent.cult.step_info()}
    log_start = len(agent.log.tail(100_000))
    try:
        reply = agent.run(text)
        answer, error = reply.text.strip(), ""
    except LLMError as e:
        print(f"  ❌ {e}")
        answer, error = "", str(e)
    rec = dict(agent.turn_log[-1]) if not error and agent.turn_log else {
        "user": text, "before": before, "after": {"mission": agent.machine.step_info(), "circle": agent.cult.step_info()},
        "events": [], "text_guard": [], "memory": []}
    turn = agent.ledger[-1] if agent.ledger and not error else None
    s = agent.layer_stats[-1] if agent.layer_stats and turn and agent.layer_stats[-1]["turn"] == turn.n else {}
    ops = [o for o in agent.log.tail(100_000)[log_start:] if o["op"] in ("add", "update", "delete", "rejected", "move")]
    rec.update(label=label, answer=answer, error=error, prompt=turn.calls[0].prompt if turn and turn.calls else 0,
               completion=turn.completion if turn else 0, cost=turn.cost if turn and turn.calls else 0.0,
               fsm_tokens=(s or {}).get("tokens", {}).get("fsm", 0), guard=agent.guard, ops=ops)
    print(f"Мастер: {clip(answer, 420) or '(нет ответа)'}")
    print(f"  🧭 {state_line(rec['after'])}")
    return rec


def applied(rec: dict, name: str, **args) -> bool:
    return any(e["name"] == name and e["applied"] and all(e["args"].get(k) == v for k, v in args.items())
               for e in rec["events"])


class Lifecycle:
    """Полный цикл задания с привалом и перезапуском на каждом этапе, затем путь развития до прорыва.
    Контрольные состояния копируются для проверки нарушений."""

    def __init__(self, make, root: Path):
        self.make, self.root = make, root
        self.world = root / "lifecycle"
        self.agent: QuestAgent = make(self.world)
        self.turns: list[dict] = []
        self.pauses: list[dict] = []
        self.checkpoints: dict[str, Path] = {}
        self.notes: list[str] = []

    def say(self, text: str, label: str) -> dict:
        rec = ask(self.agent, text, label)
        self.turns.append(rec)
        return rec

    def advance(self, pred, label: str, text: str = ADVANCE_LINE, max_turns: int = 4) -> bool:
        for k in range(max_turns):
            m = self.agent.machine
            if pred(self.agent):
                return True
            line = text
            if m.stage == "briefing" and m.steps and not m.approved and k and m.version > 1:
                line = APPROVE_LINE  # Мастер пересмотрел маршрут и ждёт утверждения новой версии
                self.notes.append(f"{label}: маршрут пересмотрен (версия {m.version}) — игрок утверждает заново")
            self.say(line, f"{label} ({k + 1})")
        ok = pred(self.agent)
        if not ok:
            self.notes.append(f"{label}: цель не достигнута за {max_turns} ход(ов)")
        return ok

    def checkpoint(self, name: str) -> None:
        target = self.root / "checkpoints" / name
        self.agent.save()
        shutil.copytree(self.world, target)
        self.checkpoints[name] = target
        print(f"  📍 контрольное состояние «{name}» сохранено: "
              f"{state_line({'mission': self.agent.machine.step_info(), 'circle': self.agent.cult.step_info()})}")

    def restart(self) -> None:
        """Как перезапуск программы: новый объект Мастера читает всё с диска, затем /newchat — сцена пуста."""
        self.agent = self.make(self.world)
        what = self.agent.new_scene()
        print(f"  🔄 перезапуск: новый Мастер прочитал состояние из файлов; /newchat — {what}")

    def pause_test(self, stage: str) -> dict:
        m = self.agent.machine
        before = m.step_info()
        p = self.say(PAUSE_LINES[stage], f"привал на этапе {stage}")
        paused = self.agent.machine.stage == PAUSED and self.agent.machine.data["paused_from"] == stage
        forced = False
        if not paused and self.agent.machine.stage == stage:
            forced = True
            entry = self.agent.user_transition(PAUSED, "привал", "игрок (команда, страховка демо)")
            paused = entry["applied"]
            self.notes.append(f"привал на {stage}: Мастер не поставил привал, поставлен командой /pause")
        idx = next((i for i, e in enumerate(p["events"]) if e["kind"] == "transition" and e["applied"]
                    and e["args"].get("to") == PAUSED), len(p["events"]))
        work_on_pause = [e for e in p["events"][idx + 1:] if e["kind"] == "action" and e["applied"]
                         and e.get("machine") == "mission"]
        m = self.agent.machine
        expect = m.step_info(stage) if paused else before
        version, results, checks = m.version, set(m.data["results"]), set(m.data["checks"])
        self.restart()
        r = self.say(RESUME_LINE, f"продолжение после привала на {stage} (новый процесс, пустая сцена)")
        m2 = self.agent.machine
        resumed = m2.stage != PAUSED
        if stage == "briefing":
            continued = resumed and m2.version == version and not applied(r, "save_route")
            want = f"не составлять маршрут заново (версия {version}) и попросить утвердить"
        elif stage == "journey":
            k = expect["index"]
            other = [e for e in r["events"] if e["name"] == "complete_step" and e["applied"] and e["args"].get("step") != k]
            done_k = applied(r, "complete_step", step=k)
            continued = resumed and not other and (done_k or m2.next_step() == k)
            want = f"этап {k}: пройти его" + ("" if done_k else " (сцена этапа продолжена, итог — в следующем ходе)")
        else:
            k = expect["index"]
            other = [e for e in r["events"] if e["name"] == "record_check" and e["applied"] and e["args"].get("check") != k]
            done_k = applied(r, "record_check", check=k)
            continued = resumed and not other and (done_k or m2.next_check() == k)
            want = f"условие {k}: проверить его" + ("" if done_k else " (сцена продолжена, проверка — в следующем ходе)")
        kept = (version == m2.version and results <= set(m2.data["results"])
                and (stage != "report" or checks <= set(m2.data["checks"])))
        res = {"stage": stage, "before": before, "expect": expect, "paused": paused, "forced_pause": forced,
               "work_on_pause": len(work_on_pause), "resumed": resumed, "resumed_to": m2.stage if resumed else "",
               "continued": continued, "want": want, "kept": kept,
               "reexplain": bool(REEXPLAIN_RE.search(r["answer"])), "pause_answer": p["answer"],
               "resume_answer": r["answer"], "resume_events": [entry_line(e) for e in r["events"]]}
        self.pauses.append(res)
        print(f"  ⏸  итог привала на {stage}: привал {'✅' if paused else '❌'}"
              f"{' (командой)' if forced else ''}, снят {'✅' if resumed else '❌'}, продолжил с того же шага "
              f"({want}) {'✅' if continued else '❌'}, данные целы {'✅' if kept else '❌'}, переспрашивал о задании "
              f"{'⚠️ да' if res['reexplain'] else 'нет'}")
        return res

    def run(self) -> dict:
        print(f"{'=' * 78}\n=== 2. Жизненный цикл: сборы → в пути → отчёт → сдано, привал на каждом этапе; затем путь "
              f"развития до прорыва\n{'=' * 78}")
        a = self.agent
        seed_quest_hero(a)
        start = {"qi": a.cult.data["qi"], "forms": a.cult.data["forms"], "circle": a.cult.title()}
        spec = MISSIONS["caravan"]
        m = a.new_mission(spec.title, spec.card)
        print(f"  📜 задание {m['id']} «{m['title']}»; ☯ {a.cult.title()}: ци {a.cult.data['qi']}/{a.cult.need_qi}, "
              f"формы {a.cult.data['forms']}/{a.cult.need_forms}")
        self.say(TASK_TEXT, "взять задание")
        if self.advance(lambda x: x.machine.stage == "briefing" and x.machine.steps, "маршрут не сохранён",
                        "Сохрани маршрут.", 2):
            self.checkpoint("briefing")
            self.pause_test("briefing")
        self.say(APPROVE_LINE, "утверждение маршрута")
        self.advance(lambda x: x.machine.stage != "briefing", "выступить", "Выступаем.", 2)
        if self.advance(lambda x: x.machine.stage == "journey" and x.machine.data["results"] and x.machine.next_step(),
                        "этап 1", max_turns=3):
            self.checkpoint("journey")
            self.pause_test("journey")
        self.advance(lambda x: x.machine.stage in ("report", "done"), "до отчёта", max_turns=5)
        self.advance(lambda x: x.machine.stage != "report" or x.machine.data["checks"], "первое условие", max_turns=2)
        if self.agent.machine.stage == "report" and self.agent.machine.next_check():
            self.checkpoint("report")
            self.pause_test("report")
        else:
            self.notes.append("контрольное состояние report не поймано: отчёт прошёл за один ход")
        self.advance(lambda x: x.machine.stage == "done", "до сдачи", max_turns=8)
        if self.agent.machine.stage == "done":
            self.checkpoint("done")
        # путь развития: медитация и форма — уже не в пути, затем прорыв по решению игрока
        cult_start = {"qi": self.agent.cult.data["qi"], "state": self.agent.cult.state}
        breakthrough = None
        if self.agent.machine.stage != "done":
            self.notes.append(f"путь развития не проверялся: задание осталось на этапе {self.agent.machine.stage}, "
                              f"а медитировать в пути нельзя")
        else:
            self.advance(lambda x: x.cult.data["qi"] >= x.cult.need_qi or x.cult.state != "gathering", "медитация",
                         CULTIVATE_LINE, 4)
            self.advance(lambda x: x.cult.state != "gathering", "отработка формы", TRAIN_LINE, 3)
        if self.agent.cult.state == "ready":
            r = self.say(BREAK_LINE, "прорыв по решению игрока")
            if self.agent.cult.state == "breakthrough":
                self.advance(lambda x: x.cult.state != "breakthrough", "исход прорыва", "Что дальше?", 2)
            breakthrough = {"events": [entry_line(e) for e in r["events"]],
                            "attempts": [x for x in self.agent.circle.list()]}
        a = self.agent
        return {"dir": str(self.world.relative_to(ROOT)), "turns": self.turns, "pauses": self.pauses,
                "notes": self.notes, "final_mission": copy.deepcopy(a.machine.data),
                "final_circle": copy.deepcopy(a.cult.data), "circles": a.circle.list(), "start": start,
                "cult_start": cult_start, "checkpoints": list(self.checkpoints), "breakthrough": breakthrough,
                "memory_events": copy.deepcopy(a.memory_events), "memory": a.memory_json()}


def violation_run(make, root: Path, v: Violation, source: Path, guard: bool, sample: int) -> dict:
    target = root / "violations" / f"{v.id}_{'guard' if guard else 'prompt'}_{sample}"
    shutil.copytree(source, target)
    agent = make(target, guard)
    m, c = agent.machine, agent.cult
    before = {"mission": m.step_info(), "circle": c.step_info()}
    snap = {"qi": c.data["qi"], "school": dict((k, e["value"]) for k, e in agent.long.data["school"].items()),
            "results": len(m.data["results"])}
    log_start = len(agent.log.tail(100_000))
    rec = ask(agent, v.text, f"{v.id} [{'код проверяет' if guard else 'только промпт'}, #{sample}] {v.what}")
    illegal = [f"{e['name']} ({'; '.join(e['unmet'])})" for e in rec["events"] if e["unmet"] and e["applied"]]
    rejected = [entry_line(e) for e in rec["events"] if e["unmet"] and not e["applied"]]
    mem = agent.log.tail(100_000)[log_start:]
    illegal += [f"память: {o['section']}.{o['key']} ({o['violation']})" for o in mem if o.get("violation")]
    rejected += [f"память: {op_label(o)}" for o in mem if o["op"] == "rejected"]
    text_told = [t for t in rec["text_guard"] if not t["enforced"]]
    if text_told:
        illegal.append(f"исход задания в рассказе на этапе {before['mission']['stage']}")
    school_after = {k: e["value"] for k, e in agent.long.data["school"].items()}
    return {"id": v.id, "stage": v.stage, "text": v.text, "what": v.what, "guard": guard, "sample": sample,
            "stage_before": before["mission"]["stage"], "stage_after": agent.machine.stage,
            "cult_before": before["circle"]["state"], "cult_after": agent.cult.state,
            "circle_before": c.title(), "circle_after": agent.cult.title(),
            "qi_before": snap["qi"], "qi_after": agent.cult.data["qi"],
            "school_changed": school_after != snap["school"], "attempts": len(rec["events"]),
            "rejected": rejected, "illegal": illegal, "text_guard": rec["text_guard"],
            "explained": bool(REFUSAL_RE.search(rec["answer"])), "answer": rec["answer"],
            "events": [entry_line(e) for e in rec["events"]], "cost": rec["cost"], "error": rec["error"],
            "dir": str(target.relative_to(ROOT))}


def violations_demo(make, root: Path, checkpoints: dict[str, Path], samples: int) -> list[dict]:
    print(f"\n{'=' * 78}\n=== 3. Прямые просьбы нарушить порядок: код проверяет / правила только в промпте\n{'=' * 78}")
    runs = []
    for v in VIOLATIONS:
        source = checkpoints.get(v.stage)
        if source is None:
            print(f"\n  {v.id}: нет контрольного состояния «{v.stage}» — пропуск")
            continue
        for guard in (True, False):
            for k in range(1, samples + 1):
                r = violation_run(make, root, v, source, guard, k)
                runs.append(r)
                verdict = (f"⚠️  нарушение: {'; '.join(r['illegal'])}" if r["illegal"] else "✅ порядок не нарушен")
                guarded = sum(t["enforced"] for t in r["text_guard"])
                print(f"  ⇒ {verdict}; действий {r['attempts']}, отклонено кодом {len(r['rejected'])}"
                      + (f", исход отсечён {guarded}" if guarded else "")
                      + f"; задание {r['stage_before']} → {r['stage_after']}, развитие {r['cult_before']} → "
                        f"{r['cult_after']}, ци {r['qi_before']} → {r['qi_after']}"
                      + (f", круг {r['circle_before']} → {r['circle_after']}" if r["circle_before"] != r["circle_after"] else ""))
    return runs


def demo(client: LLMClient, helper: LLMClient, keep: int, budget: int, max_steps: int, per_turn: int, samples: int,
         parts: set[str], seed: int | None) -> int:
    stamp = stamp_now()
    root = DAY_DIR / f"demo_{stamp}"
    make = lambda world, guard=True: make_agent(client, helper, world, HERO_ID, keep, budget, max_steps,  # noqa: E731
                                                guard, per_turn, seed, quiet_event)
    report: dict = {"model": client.model, "helper_model": helper.model, "stamp": stamp, "per_turn": per_turn,
                    "samples": samples, "dir": str(root.relative_to(ROOT)), "quest_prompt": QUEST_PROMPT,
                    "transitions": [asdict(t) for t in TRANSITIONS], "ctransitions": [asdict(t) for t in CTRANSITIONS],
                    "guards": GUARDS, "cguards": CGUARDS, "policy": POLICY, "seed": seed}
    report["matrix"] = matrix_demo()
    if parts & {"lifecycle", "violations"}:
        life = Lifecycle(make, root)
        report["lifecycle"] = life.run()
        if "violations" in parts:
            report["violations"] = violations_demo(make, root, life.checkpoints, samples)
    report["requests"] = {"main": dict(client.stats), "helper": dict(helper.stats) if helper is not client else None}
    save_report(report, stamp)
    return 0


# ---------------------------------------------------------------- отчёт


def save_report(report: dict, stamp: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path, md_path = RESULTS_DIR / f"day13_{stamp}.json", RESULTS_DIR / f"day13_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    L = [f"# День 13 · Конечные автоматы задания и пути развития (уся-РПГ) — прогон {stamp}", "",
         f"Мастер: `{report['model']}`, летописец: `{report['helper_model']}`. Действий задания и развития за ход: "
         f"{report['per_turn'] or 'без ограничения'}. Файлы прогона: `{report['dir']}/`.", "",
         "## Автоматы и политика памяти", "",
         "**Задание** (краткосрочная память): " + " → ".join(f"`{s}` {STATE_TITLES[s]}" for s in STAGES)
         + f", `{PAUSED}` — {STATE_TITLES[PAUSED]}.", "",
         "| Из | В | Переход | Условие (проверяет код) |", "|---|---|---|---|"]
    for t in TRANSITIONS:
        L.append(f"| `{t.src}` | {'этап, где ушли на привал' if t.dst == RESUME else '`' + t.dst + '`'} | {t.title} | "
                 f"{GUARDS[t.guard]} |")
    L += ["", "**Путь развития** (рабочая память): " + " → ".join(f"`{s}` {CSTATE_TITLES[s]}" for s in CSTATES) + ".", "",
          "| Из | В | Переход | Условие (проверяет код) |", "|---|---|---|---|"]
    for t in CTRANSITIONS:
        L.append(f"| `{t.src}` | `{t.dst}` | {t.title} | {CGUARDS[t.guard]} |")
    L += ["", f"Медитация +{MEDITATION_QI} ци и отработка формы +1 — не в пути, одно действие за ход. Прорыв: d20 + 2 "
              f"за пилюлю ≥ {TRIAL_DC}.", "", "**Политика записи в память** (долговременная и рабочая):", "",
          "| Раздел | Что охраняется | Кто может писать | Почему |", "|---|---|---|---|"]
    for section, what, who, why in POLICY:
        L.append(f"| {SECTION_TITLES.get(section, section)} | {what} | {who} | {why} |")
    L.append("")

    mx = report["matrix"]
    L += ["## 1. Таблицы переходов и политика памяти — без модели", "",
          f"Автоматы строятся теми же методами, что и в игре; из каждого состояния пробуются все переходы и действия. "
          f"Проверено {mx['checked']} случаев, расхождений с ожидаемыми таблицами: **{mx['mismatches']}**.", "",
          "| Задание: состояние | " + " | ".join(f"→ `{s}`" for s in STATES) + " | " + " | ".join(ACTION_TESTS) + " |",
          "|---|" + "---|" * (len(STATES) + len(ACTION_TESTS))]
    for r in mx["rows"]:
        cells = [("—" if (c := r["targets"].get(s)) is None else ("✅" if c["allowed"] else "⛔") + ("" if c["match"] else " ≠"))
                 for s in STATES]
        acts = [("✅" if r["actions"][t]["allowed"] else "⛔") + ("" if r["actions"][t]["match"] else " ≠")
                for t in ACTION_TESTS]
        L.append(f"| {r['title']} | " + " | ".join(cells) + " | " + " | ".join(acts) + " |")
    L += ["", "| Отдельно | Результат |", "|---|---|"]
    for x in mx["extra"]:
        L.append(f"| {x['what']} | {'✅ разрешено' if x['allowed'] else '⛔ ' + md_escape('; '.join(x['unmet']))} |")
    a = mx["approval"]
    L.append(f"| у Мастера инструменты «утвердить маршрут / начать прорыв / добавить ци» | "
             f"{'есть: ' + ', '.join(a['model_tools']) if a['model_tools'] else 'нет'} |")
    for t, v in a["phrases"].items():
        L.append(f"| «{t}» | {'утверждение' if v else 'не утверждение'} |")
    for t, v in a["consent"].items():
        L.append(f"| «{t}» | {'начало прорыва' if v else 'не начало прорыва'} |")
    L += ["", "| Путь развития: состояние | " + " | ".join(CACTIONS) + " |", "|---|" + "---|" * len(CACTIONS)]
    for r in mx["crows"]:
        L.append(f"| {r['title']} | " + " | ".join(("✅" if r["actions"][t]["allowed"] else "⛔")
                                                   + ("" if r["actions"][t]["match"] else " ≠") for t in CACTIONS) + " |")
    for x in mx["cextra"]:
        L.append(f"| {x['what']} | {'✅' if x['allowed'] else '⛔ ' + md_escape('; '.join(x['unmet']))} |")
    L += ["", "| Политика памяти: запись | Кто | Результат |", "|---|---|---|"]
    for x in mx["policy"]:
        L.append(f"| {x['what']} (`{x['section']}.{x['key']}`) | {x['writer']} | "
                 f"{'✅ записано' if x['allowed'] else '⛔ ' + md_escape('; '.join(x['unmet']))}{'' if x['match'] else ' ≠'} |")
    L.append("")

    life = report.get("lifecycle")
    if life:
        fm, fc = life["final_mission"], life["final_circle"]
        L += ["## 2. Жизненный цикл: задание с привалами, затем путь развития", "",
              f"Задание «{fm['title']}»: этап в конце — `{fm['stage']}`, ходов {len(life['turns'])}. Старт развития: "
              f"{life['start']['circle']}, ци {life['start']['qi']}, форм {life['start']['forms']}. В конце: "
              f"{circle_title(fc['realm'], fc['stage'])}, ци {fc['qi']}, форм {fc['forms']}, состояние `{fc['state']}`.", ""]
        if life["notes"]:
            L += ["Заметки прогона: " + "; ".join(md_escape(n) for n in life["notes"]), ""]
        L += ["### Привал на каждом этапе", "",
              "После привала — новый объект Мастера (как перезапуск программы) и `/newchat`, затем только «Продолжаем.».",
              "", "| Этап | Привал | Работа на привале | Снят | Продолжил с того же шага | Данные целы | Переспрашивал |",
              "|---|---|---|---|---|---|---|"]
        for p in life["pauses"]:
            L.append(f"| {p['stage']} | {'✅' if p['paused'] else '❌'}{' (командой)' if p['forced_pause'] else ''} | "
                     f"{p['work_on_pause']} | {'✅ → ' + p['resumed_to'] if p['resumed'] else '❌'} | "
                     f"{'✅' if p['continued'] else '❌'} {md_escape(p['want'])} | {'✅' if p['kept'] else '❌'} | "
                     f"{'⚠️ да' if p['reexplain'] else 'нет'} |")
        L += ["", "### Как автоматы меняли память", "", "| Событие | Что изменилось в слоях |", "|---|---|"]
        for ev in life["memory_events"]:
            L.append(f"| {md_escape(ev['what'])}{' ⚠️ без отчёта' if ev.get('unearned') else ''} | "
                     + "<br>".join(md_escape(c) for c in ev["changes"]) + " |")
        L += ["", "### Ходы", "",
              "| # | Шаг | Реплика | Задание до → после | Развитие до → после | Действия автоматов | Память |",
              "|---|---|---|---|---|---|---|"]
        for k, t in enumerate(life["turns"], 1):
            mb, ma = t["before"]["mission"], t["after"]["mission"]
            cb, ca = t["before"]["circle"], t["after"]["circle"]
            ev = "<br>".join(md_escape(entry_line(e)) for e in t["events"]) or "—"
            if t.get("text_guard"):
                ev += "<br>" + "<br>".join(f"🛡 {md_escape(x['action'])}" for x in t["text_guard"])
            mem = "<br>".join(md_escape(op_label(o)) for o in t.get("ops", []) if o["op"] == "rejected"
                              or o["section"] in ("circle", "gear", "school", "traits")) or "—"
            L.append(f"| {k} | {md_escape(t['label'])} | {md_escape(clip(t['user'], 70))} | {mb['stage']} → {ma['stage']} | "
                     f"{cb['state']} → {ca['state']} | {ev} | {mem} |")
        L.append("")

    vs = report.get("violations")
    if vs:
        L += ["## 3. Прямые просьбы нарушить порядок", "",
              "Каждая просьба — на копии контрольного состояния из шага 2, в двух режимах: код проверяет переходы "
              "и запись в память / правила только в промпте (код записывает нарушения, но не мешает).", "",
              "| Просьба | Состояние | Режим | Нарушение произошло | Отклонено кодом | Задание до → после | "
              "Развитие, ци | Школа изменилась | Объяснил отказ |", "|---|---|---|---|---|---|---|---|---|"]
        for r in vs:
            L.append(f"| {r['id']}: {md_escape(r['text'])} | {r['stage']} | "
                     f"{'код проверяет' if r['guard'] else 'только промпт'} | "
                     f"{'⚠️ ' + md_escape('; '.join(r['illegal'])) if r['illegal'] else '✅ нет'} | {len(r['rejected'])} | "
                     f"{r['stage_before']} → {r['stage_after']} | {r['cult_before']} → {r['cult_after']}, "
                     f"ци {r['qi_before']} → {r['qi_after']}"
                     + (f", ⚠️ {r['circle_before']} → {r['circle_after']}" if r['circle_before'] != r['circle_after'] else "")
                     + f" | {'⚠️ да' if r['school_changed'] else 'нет'} | {'да' if r['explained'] else 'нет'} |")
        for guard in (True, False):
            sub = [r for r in vs if r["guard"] == guard]
            if sub:
                L += ["", f"**{'Код проверяет' if guard else 'Только промпт'}:** нарушений "
                          f"{sum(bool(r['illegal']) for r in sub)} из {len(sub)}, попыток, отклонённых кодом, "
                          f"{sum(len(r['rejected']) for r in sub)}, отказ объяснён в {sum(r['explained'] for r in sub)}."]
        L += ["", "### Ответы Мастера", ""]
        for r in vs:
            L += [f"**{r['id']} — {'код проверяет' if r['guard'] else 'только промпт'}, #{r['sample']}.** "
                  f"«{md_escape(r['text'])}»", ""]
            if r["events"]:
                L += ["- " + md_escape(e) for e in r["events"]]
            L += ["", "> " + md_escape(clip(r["answer"] or r["error"], 900)), ""]
    req = report["requests"]
    L += [f"Запросов к модели: {req['main']['requests']}, повторов {req['main']['retries']}, "
          f"отказов {req['main']['failed']}."]
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {md_path.relative_to(ROOT)}, {json_path.name}")
    return md_path


# ---------------------------------------------------------------- игра в консоли


HELP = ("/state, /fsm, /approve, /go <этап> [причина], /pause, /resume, /breakthrough, /journal [N], /guard on|off, "
        "/matrix, /policy, /event <раздел> <ключ> = <значение>, /remember, /forget, /mission …, /card, /memory, "
        "/context, /newchat, "
        "/tokens, /history, /exit")


def chat(agent: QuestAgent) -> int:
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
            if command == "/state":
                print_state(agent)
            elif command == "/fsm":
                print_rules()
            elif command == "/policy":
                print_policy()
            elif command == "/approve":
                agent.user_approve()  # строку печатает событие fsm
            elif command == "/go":
                if not args:
                    print("  /go <этап> [причина]")
                else:
                    agent.user_transition(state_arg(args[0]), " ".join(args[1:]))
            elif command == "/pause":
                agent.user_transition(PAUSED, "привал")
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
                    print("  /event <раздел> <ключ> = <значение> — событие игры, например /event school школа = …")
                else:
                    for o in agent.event_write(parts[0], parts[1], value.strip()):
                        print(f"  {op_label(o)}")
            elif command in ("/remember", "/forget"):
                head, sep, value = arg.partition("=")
                parts = head.split()
                if len(parts) != 2 or (command == "/remember" and not sep):
                    print("  /remember <раздел> <ключ> = <значение>, /forget <раздел> <ключ> — запись проверяется политикой")
                else:
                    section = SECTION_ALIASES.get(parts[0].lower(), parts[0])
                    ops = (agent.write(section, parts[1], value.strip(), "команда игрока") if command == "/remember"
                           else agent.forget(section, parts[1], "команда игрока"))
                    agent.save_memory()
                    for o in ops or [{"op": "-", "section": section, "key": parts[1], "layer": "", "value": ""}]:
                        print(f"  {op_label(o)}" if o["op"] != "-" else "  (ничего не изменилось)")
            elif command == "/mission":
                mission_command(agent, arg)
            elif command == "/card":
                print_card(agent, args[0] if args else agent.hero)
            elif command == "/memory":
                print_memory(agent)
            elif command == "/context":
                print_context12(agent)
            elif command == "/newchat":
                print(f"  🧹 {agent.new_scene()}; состояние автоматов, карточки и память остаются")
            elif command == "/tokens":
                print_ledger(agent)
            elif command == "/history":
                print_history(agent.history)
            elif command.startswith("/"):
                print(f"  Неизвестная команда. Есть: {HELP}")
            else:
                reply = agent.run(text)
                print(f"\nМастер: {reply.text.strip() or '(пустой ответ)'}\n")
                print_turn13(agent)
            print()
        except LLMError as e:
            print(f"❌ Ошибка: {e}\n", file=sys.stderr)
        except (StoreError, ValueError) as e:
            print(f"⚠️  {e}\n", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n(прервано — ход в сцену не попал)\n")


# ---------------------------------------------------------------- запуск


def main() -> int:
    parser = argparse.ArgumentParser(description="День 13: уся-РПГ — конечные автоматы задания и пути развития, "
                                                 "политика записи в память.")
    parser.add_argument("prompt", nargs="?", help="один ход и выход; без него — игра в консоли")
    parser.add_argument("--world", default="default", help="мир: memory/day13/<мир>/")
    parser.add_argument("--new", action="store_true", help="начать заново: Лу Чэнь берёт задание «Пропавший караван»")
    parser.add_argument("--no-guard", action="store_true", help="правила только в промпте: код не проверяет переходы")
    parser.add_argument("--per-turn", type=int, default=1, help="действий задания и развития за ход (0 — без предела)")
    parser.add_argument("--seed", type=int, help="зерно костей прорыва — для повторяемых прогонов")
    parser.add_argument("--keep", type=int, default=6, help="краткосрочная: сколько последних сообщений сцены в промпт")
    parser.add_argument("--long-budget", type=int, default=1500, help="долговременная: бюджет токенов")
    parser.add_argument("--helper-model", help="модель летописца (по умолчанию та же)")
    parser.add_argument("--demo", action="store_true", help="таблицы переходов, жизненный цикл, нарушения, отчёт")
    parser.add_argument("--only", nargs="+", choices=("matrix", "lifecycle", "violations"), help="для --demo: только эти части")
    parser.add_argument("--samples", type=int, default=2, help="для --demo: сколько раз задавать каждую просьбу-нарушение")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--max-steps", type=int, default=8, help="предел запросов к модели за один ход")
    args = parser.parse_args()

    if args.demo and args.only == ["matrix"]:
        report = {"model": "—", "helper_model": "—", "stamp": stamp_now(), "per_turn": args.per_turn, "samples": 0,
                  "dir": "—", "matrix": matrix_demo(), "requests": {"main": {"requests": 0, "retries": 0, "failed": 0}}}
        save_report(report, report["stamp"])
        return 0 if report["matrix"]["mismatches"] == 0 else 1
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
    if args.demo:
        parts = set(args.only or ("matrix", "lifecycle", "violations"))
        return demo(client, helper, args.keep, args.long_budget, args.max_steps, args.per_turn, max(args.samples, 1),
                    parts, args.seed)

    world = DAY_DIR / args.world
    fresh = args.new or not world.exists()
    if args.new and world.exists():
        target = world.with_name(f"{world.name}_{stamp_now()}")
        os.replace(world, target)
        print(f"Прошлый мир сохранён в {target.relative_to(ROOT)}.")
    agent = make_agent(client, helper, world, HERO_ID, args.keep, args.long_budget, args.max_steps,
                       not args.no_guard, args.per_turn, args.seed)
    if fresh:
        seed_quest_hero(agent)
        spec = MISSIONS["caravan"]
        agent.new_mission(spec.title, spec.card)
        print(f"📜 Новая игра: {agent.cast.name(agent.hero)} берёт задание «{spec.title}». /state — состояние.")
    print_restore(agent.restore_info, agent.store, agent)
    if args.prompt:
        try:
            reply = agent.run(args.prompt)
        except (LLMError, StoreError) as e:
            print(f"❌ Ошибка: {e}", file=sys.stderr)
            return 1
        print(f"\nМастер: {reply.text.strip()}\n")
        print_turn13(agent)
        return 0
    return chat(agent)


if __name__ == "__main__":
    from run_timer import run_timed
    sys.exit(run_timed(main))
