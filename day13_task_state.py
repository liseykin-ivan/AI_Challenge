"""День 13. Состояние задачи: конечный автомат (Task State Machine) поверх агента Дня 12.

Агент Дня 12 (`PersonalAgent`: три слоя памяти, профиль пользователя, роутер памяти, инструменты, учёт токенов) ведёт
задачу по этапам. Этап — не строчка в промпте, а состояние конечного автомата `TaskMachine` в файле
memory/day13/<пользователь>/fsm/<id задачи>.json:

    planning ──(план утверждён пользователем)──▶ execution ──(все шаги выполнены)──▶ validation ──(всё пройдено)──▶ done
        ▲                                          │   ▲                                 │
        └────────────(пересмотр плана)─────────────┘   └──────(проверка не пройдена)─────┘
    planning / execution / validation ──▶ paused ──▶ обратно на тот же этап

Состояние задачи — этап, текущий шаг и ожидаемое действие (кто и что должен сделать). Менять его можно только
инструментами — save_plan, complete_step, record_check, request_transition — и командами пользователя. Каждый вызов
код проверяет до применения: есть ли такой переход в таблице TRANSITIONS и выполнено ли его условие (guard).
Если нет, состояние не меняется, а модель получает причину отказа и должна объяснить её пользователю.
Сверх таблицы:
- план утверждает только пользователь: командой /approve или словом «утверждаю» в сообщении. Инструмента
  «утвердить» у модели нет;
- шаги выполняются строго по порядку, за ход — один шаг или одна проверка (--per-turn);
- на этапе planning и на паузе код в ответе отсекается: ответ отклоняется, модель переписывает его без кода;
- после /newchat или перезапуска состояние берётся из файла: «Продолжаем» — и агент идёт с того же шага.

    python day13_task_state.py                         # чат (пользователь default)
    python day13_task_state.py --user video --new      # начать заново
    python day13_task_state.py --no-guard              # правила только в промпте, код переходы не проверяет
    python day13_task_state.py --demo                  # все проверки и отчёт в results/
    python day13_task_state.py --demo --only matrix    # только таблица переходов, без запросов к модели

Команды чата: /state, /fsm, /approve, /go <этап> [причина], /pause, /resume, /journal [N], /guard on|off, /matrix,
/task new <название> | switch <id> | list, /profile, /memory, /context, /newchat, /tokens, /history, /exit.
"""
import argparse
import copy
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from config import ConfigError, load_settings
from day6_agent import TOOLS, Tool
from day7_memory import MEMORY_DIR, ROOT, StoreError, now_iso, print_history, print_restore
from day8_tokens import clip, fmt, money, print_ledger
from day9_compression import watch_retries
from day10_strategies import read_json, write_json
from day11_memory_layers import md_escape, op_label, print_memory, stamp_now
from day12_personalization import (CODE_LINE, FENCE, PERSONAL_PROMPT, PRESETS, PROGRAMMING_LANGS, PersonalAgent,
                                    block_lang, print_ops, print_profile, split_code)
from day12_personalization import on_event as day12_event
from llm_client import LLMClient, LLMError

RESULTS_DIR = ROOT / "results"
DAY_DIR = MEMORY_DIR / "day13"
NAME_RE = re.compile(r"[\w-]+")

# ---------------------------------------------------------------- конечный автомат: состояния и переходы

STAGES = ("planning", "execution", "validation", "done")  # этапы по порядку
PAUSED = "paused"
STATES = STAGES + (PAUSED,)
STATE_TITLES = {"planning": "планирование", "execution": "выполнение", "validation": "проверка", "done": "готово",
                "paused": "пауза"}
STATE_ALIASES = {**{s: s for s in STATES}, "план": "planning", "планирование": "planning", "выполнение": "execution",
                 "реализация": "execution", "проверка": "validation", "валидация": "validation", "готово": "done",
                 "финал": "done", "пауза": "paused"}
RESUME = "@paused_from"  # «туда, откуда поставили паузу»


@dataclass(frozen=True)
class Transition:
    src: str
    dst: str  # RESUME — этап, на котором поставили паузу
    guard: str  # имя условия в GUARDS
    title: str


# Таблица разрешённых переходов. Всё, чего здесь нет, запрещено. Условие проверяется кодом до применения.
TRANSITIONS = (
    Transition("planning", "execution", "plan_approved", "к выполнению"),
    Transition("execution", "validation", "steps_done", "к проверке"),
    Transition("validation", "done", "checks_passed", "финал"),
    Transition("validation", "execution", "check_failed", "доработка"),
    Transition("execution", "planning", "has_reason", "пересмотр плана"),
    Transition("planning", PAUSED, "none", "пауза"),
    Transition("execution", PAUSED, "none", "пауза"),
    Transition("validation", PAUSED, "none", "пауза"),
    Transition(PAUSED, RESUME, "resume", "снять паузу"),
)
GUARDS = {
    "plan_approved": "план сохранён, и эту версию утвердил пользователь",
    "steps_done": "все шаги плана выполнены (в доработке — исправлен хотя бы один шаг)",
    "checks_passed": "все критерии проверены и пройдены, есть итог (final)",
    "check_failed": "хотя бы один критерий не пройден",
    "has_reason": "указана причина (reason); утверждение плана и результаты шагов сбрасываются",
    "none": "без условий",
    "resume": "только на тот этап, где поставили паузу",
}
# Действия внутри этапа: где они разрешены.
ACTIONS = {"save_plan": ("planning",), "complete_step": ("execution",), "record_check": ("validation",),
           "approve": ("planning",)}
TEXT_GUARD_STAGES = ("planning", PAUSED)  # здесь реализации (кода) в ответе быть не должно
MAX_STEPS, MAX_CHECKS = 8, 6
EXPECTED_ACTOR = {"assistant": "ассистент", "user": "пользователь", "none": "—"}


def state_arg(name: str) -> str:
    s = STATE_ALIASES.get((name or "").strip().lower())
    if not s:
        raise ValueError(f"нет этапа «{name}». Есть: {', '.join(STATES)}")
    return s


def nums(items) -> str:
    return ", ".join(str(i) for i in items)


class TaskMachine:
    """Состояние одной задачи. Меняется только методами ниже; каждый сначала проверяет, разрешено ли действие,
    и пишет результат в журнал. enforce=False — режим «правила только в промпте»: нарушение записывается,
    но действие применяется."""

    def __init__(self, path: Path | None, data: dict):
        self.path = path
        self.data = data
        self.per_turn = 0  # шагов/проверок за ход; 0 — без ограничения
        self.turn_used = 0

    @staticmethod
    def blank_data(task_id: str, title: str) -> dict:
        return {"format": 1, "task_id": task_id, "title": title, "stage": "planning", "paused_from": None,
                "plan": {"version": 0, "steps": [], "checks": [], "saved_at": None},
                "approved_version": 0, "approved_at": None, "approved_by": "",
                "round": 1, "results": {}, "checks": {}, "final": "", "journal": [], "created_at": now_iso()}

    @classmethod
    def load(cls, path: Path, task_id: str, title: str) -> "TaskMachine":
        data = read_json(path) if path.exists() else cls.blank_data(task_id, title)
        if data.get("stage") not in STATES:
            raise StoreError(f"{path}: неизвестный этап {data.get('stage')!r}")
        base = cls.blank_data(task_id, title)
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
        """Шаги, исправленные в текущем круге доработки."""
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
        if stage == "planning":
            info["total"] = 2
            if not self.steps:
                info.update(index=1, step="составить план: шаги и критерии проверки",
                            expected="составить план и сохранить его через save_plan, показать пользователю")
            elif not self.approved:
                info.update(index=2, step=f"утверждение плана (версия {self.version})", actor="user",
                            expected="утвердить план (/approve или «утверждаю») или попросить правки")
            else:
                info.update(index=2, step=f"план версии {self.version} утверждён",
                            expected="перейти в execution: request_transition(to=\"execution\")")
        elif stage == "execution":
            info["total"] = n_steps
            nxt = self.next_step()
            if self.data["round"] > 1:
                fixed = self.reworked()
                info.update(index=n_steps, step=f"доработка (круг {self.data['round']}) по непройденным критериям",
                            expected=("исправить нужные шаги через complete_step(step=N), затем перейти в validation"
                                      if not fixed else f"исправлены шаги {nums(fixed)}: перейти в validation "
                                                        "или исправить ещё"))
            elif nxt:
                info.update(index=nxt, step=f"шаг {nxt} из {n_steps} — «{self.steps[nxt - 1]}»",
                            expected=f"выполнить шаг {nxt}, показать результат и сохранить через complete_step(step={nxt})")
            else:
                info.update(index=n_steps, step="все шаги выполнены",
                            expected="перейти в validation: request_transition(to=\"validation\")")
        elif stage == "validation":
            info["total"] = n_checks
            nxt = self.next_check()
            if nxt:
                info.update(index=nxt, step=f"критерий {nxt} из {n_checks} — «{self.criteria[nxt - 1]}»",
                            expected=f"проверить критерий {nxt} и записать результат через record_check(check={nxt})")
            elif self.failed():
                info.update(index=n_checks, step=f"не пройдены критерии {nums(self.failed())}",
                            expected="вернуться в execution и исправить: request_transition(to=\"execution\")")
            else:
                info.update(index=n_checks, step="все критерии пройдены",
                            expected="перейти в done с итогом: request_transition(to=\"done\", final=…)")
        elif stage == PAUSED:
            src = self.data.get("paused_from") or "planning"
            inner = self.step_info(src)
            info.update(index=inner["index"], total=inner["total"], actor="user",
                        step=f"пауза на этапе {src}: {inner['step']}",
                        expected=f"сказать «продолжаем» (/resume) — задача вернётся в {src}")
        else:  # done
            info.update(actor="none", step="задача завершена", expected="ничего: задача закрыта, итог сохранён")
        return info

    # --- переходы

    def rule(self, dst: str) -> Transition | None:
        src = self.stage
        for t in TRANSITIONS:
            if t.src == src and (t.dst == dst or (t.dst == RESUME and dst == self.data.get("paused_from"))):
                return t
        return None

    def allowed(self) -> list[tuple[str, Transition, list[str]]]:
        """Переходы из текущего этапа: (куда, правило, невыполненные условия)."""
        out = []
        for t in TRANSITIONS:
            if t.src != self.stage:
                continue
            dst = self.data.get("paused_from") if t.dst == RESUME else t.dst
            out.append((dst, t, self.guard_unmet(t.guard, "причина", "итог")))
        return out

    def guard_unmet(self, guard: str, reason: str = "", final: str = "") -> list[str]:
        d = self.data
        if guard == "plan_approved":
            if not self.steps:
                return ["план не составлен (save_plan)"]
            return [] if self.approved else [f"план версии {self.version} не утверждён пользователем"]
        if guard == "steps_done":
            missing = [i for i in range(1, len(self.steps) + 1) if str(i) not in d["results"]]
            out = [f"не выполнены шаги {nums(missing)}"] if missing else []
            if d["round"] > 1 and not self.reworked():
                out.append("в доработке не исправлен ни один шаг")
            return out
        if guard == "checks_passed":
            missing = [i for i in range(1, len(self.criteria) + 1) if str(i) not in d["checks"]]
            out = [f"не проверены критерии {nums(missing)}"] if missing else []
            if self.failed():
                out.append(f"не пройдены критерии {nums(self.failed())}")
            if not (final or "").strip():
                out.append("нет итога задачи (final)")
            return out
        if guard == "check_failed":
            return [] if self.failed() else ["нет непройденных критериев — возвращаться к выполнению незачем"]
        if guard == "has_reason":
            return [] if (reason or "").strip() else ["нужна причина пересмотра плана (reason)"]
        return []

    def check(self, dst: str, reason: str = "", final: str = "") -> tuple[Transition | None, list[str]]:
        """Можно ли перейти в dst прямо сейчас: (правило, невыполненные условия). Пустой список — можно."""
        src = self.stage
        if dst not in STATES:
            return None, [f"нет этапа «{dst}». Есть: {', '.join(STATES)}"]
        if dst == src:
            return None, [f"задача уже на этапе {src}"]
        t = self.rule(dst)
        if t is None:
            ways = [d for d, _, _ in self.allowed()]
            why = f"перехода {src} → {dst} нет в таблице"
            if src == PAUSED:
                why += f": с паузы можно вернуться только на этап {self.data.get('paused_from')}"
            elif src == "done":
                why += ": задача завершена, done — конечное состояние"
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
        if src == "execution" and dst == "planning":  # новый план — новое утверждение и новое выполнение
            d.update(approved_version=0, approved_at=None, approved_by="", results={}, round=1)
        if dst == "validation" and src != PAUSED:
            d["checks"] = {}
        if src == "validation" and dst == "execution":
            d["round"] += 1
        if dst == "done":
            d["final"] = final
        d["stage"] = dst

    # --- действия внутри этапа

    def _stage_unmet(self, action: str) -> list[str]:
        if self.stage in ACTIONS[action]:
            return []
        where = ", ".join(ACTIONS[action])
        hint = {"planning": "сначала нужен утверждённый план", "paused": "задача на паузе",
                "done": "задача завершена"}.get(self.stage, "")
        return [f"{action} — только на этапе {where}, а сейчас {self.stage}" + (f" ({hint})" if hint else "")]

    def _limit_unmet(self) -> list[str]:
        if self.per_turn and self.turn_used >= self.per_turn:
            return [f"за один ход — не больше {self.per_turn} шага/проверки: покажи результат пользователю, "
                    f"следующий — в следующем ходе"]
        return []

    def save_plan(self, steps, checks, by: str, enforce: bool = True) -> dict:
        steps = [str(s).strip() for s in (steps or []) if str(s).strip()]
        checks = [str(c).strip() for c in (checks or []) if str(c).strip()]
        unmet = self._stage_unmet("save_plan")
        if self.stage == "execution":
            unmet = [unmet[0] + "; чтобы изменить план — request_transition(to=\"planning\", reason=…)"]
        broken = []
        if not 1 <= len(steps) <= MAX_STEPS:
            broken.append(f"в плане должно быть от 1 до {MAX_STEPS} шагов, а их {len(steps)}")
        if not 1 <= len(checks) <= MAX_CHECKS:
            broken.append(f"критериев проверки должно быть от 1 до {MAX_CHECKS}, а их {len(checks)}")
        applied = not broken and (not unmet or not enforce)
        if applied:
            self.data["plan"] = {"version": self.version + 1, "steps": steps, "checks": checks, "saved_at": now_iso()}
        return self._log("action", "save_plan", by, unmet + broken, applied, enforce,
                         {"steps": steps, "checks": checks})

    def complete_step(self, step, result: str, by: str, enforce: bool = True) -> dict:
        try:
            step = int(step)
        except (TypeError, ValueError):
            return self._log("action", "complete_step", by, [f"номер шага должен быть числом, а не {step!r}"],
                             False, enforce, {"step": step})
        unmet = self._stage_unmet("complete_step")
        if not unmet and not self.approved:
            unmet.append("план не утверждён пользователем")
        if not 1 <= step <= max(len(self.steps), 1) or not self.steps:
            unmet.append(f"шага {step} нет в плане (шагов {len(self.steps)})")
        elif self.data["round"] == 1:
            nxt = self.next_step()
            if nxt is None:
                unmet.append("все шаги уже выполнены — дальше validation")
            elif step != nxt:
                unmet.append(f"шаг {step} нельзя: шаги идут по порядку, следующий — шаг {nxt}")
        unmet += self._limit_unmet()
        broken = [] if (result or "").strip() else ["пустой результат шага"]
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
            return self._log("action", "record_check", by, [f"номер критерия должен быть числом, а не {check!r}"],
                             False, enforce, {"check": check})
        if isinstance(passed, str):
            passed = passed.strip().lower() in ("true", "да", "yes", "1")
        unmet = self._stage_unmet("record_check")
        if not 1 <= check <= max(len(self.criteria), 1) or not self.criteria:
            unmet.append(f"критерия {check} нет в плане (критериев {len(self.criteria)})")
        else:
            nxt = self.next_check()
            if nxt is None:
                unmet.append("все критерии уже проверены")
            elif check != nxt:
                unmet.append(f"критерий {check} нельзя: проверки идут по порядку, следующий — критерий {nxt}")
        unmet += self._limit_unmet()
        broken = [] if (evidence or "").strip() else ["нужно обоснование проверки (evidence)"]
        applied = not broken and (not unmet or not enforce)
        if applied:
            self.data["checks"][str(check)] = {"passed": bool(passed), "evidence": evidence.strip(), "at": now_iso()}
            self.turn_used += 1
        return self._log("action", "record_check", by, unmet + broken, applied, enforce,
                         {"check": check, "passed": bool(passed), "evidence": clip(evidence or "", 200)})

    def approve(self, by: str) -> dict:
        """Утверждение плана. Только от пользователя: у модели такого инструмента нет."""
        unmet = self._stage_unmet("approve")
        if not unmet and not self.steps:
            unmet.append("план ещё не составлен — утверждать нечего")
        elif not unmet and self.approved:
            unmet.append(f"план версии {self.version} уже утверждён")
        applied = not unmet
        if applied:
            self.data.update(approved_version=self.version, approved_at=now_iso(), approved_by=by)
        return self._log("action", "approve", by, unmet, applied, True, {"version": self.version})

    def _log(self, kind: str, name: str, by: str, unmet: list[str], applied: bool, enforce: bool, args: dict) -> dict:
        entry = {"at": now_iso(), "kind": kind, "name": name, "by": by, "ok": not unmet, "applied": applied,
                 "enforced": enforce, "unmet": unmet, "args": args, "stage": self.stage}
        self.data["journal"].append(entry)
        return entry

    # --- состояние для промпта и для модели

    def summary(self) -> dict:
        i = self.step_info()
        return {"stage": self.stage, "step": i["step"], "expected": f"{EXPECTED_ACTOR[i['actor']]}: {i['expected']}",
                "allowed": [d for d, _, u in self.allowed() if not u]}

    def block(self, enforce: bool = True) -> str:
        d, i = self.data, self.step_info()
        pos = f" ({STAGES.index(self.stage) + 1} из {len(STAGES)})" if self.stage in STAGES else ""
        src = d.get("paused_from")
        lines = ["\n\n# Состояние задачи — конечный автомат",
                 ("Этапом задачи управляет код: переходы проверяются до применения." if enforce else
                  "Этап задачи меняется через инструменты; соблюдай правила этапов."),
                 f"Задача: «{d['title']}»",
                 f"Этап: {self.stage} — {STATE_TITLES[self.stage]}{pos}" + (f", пауза поставлена на этапе {src}"
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
            appr = (f"утверждён пользователем {str(d['approved_at'])[11:16]}" if self.approved
                    else "НЕ утверждён пользователем")
            lines.append(f"План (версия {self.version}, {appr}):")
            for n, s in enumerate(self.steps, 1):
                r = d["results"].get(str(n))
                mark = "✅" if r else ("▶" if self.stage == "execution" and n == self.next_step() else "·")
                lines.append(f"{n}. {mark} {s}" + (f"\n   результат: {clip(r['text'], 1500)}" if r else ""))
            lines.append("Критерии проверки:")
            for n, c in enumerate(self.criteria, 1):
                r = d["checks"].get(str(n)) if self.stage in ("validation", "done", PAUSED) else None
                mark = ("✅" if r["passed"] else "❌") if r else "·"
                lines.append(f"{n}. {mark} {c}" + (f" — {clip(r['evidence'], 300)}" if r else ""))
        if d.get("final"):
            lines.append(f"Итог: {d['final']}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- агент

TASK_PROMPT = PERSONAL_PROMPT + (
    "\n\nТы ведёшь задачу пользователя по этапам: planning → execution → validation → done, плюс пауза. Состояние "
    "задачи — в конце этого сообщения, его хранит и проверяет код. Правила:\n"
    "- работай только в рамках текущего этапа; что делать дальше — в строке «Ожидаемое действие»;\n"
    "- planning: составь план — шаги и критерии проверки — и сохрани его через save_plan, покажи пользователю и "
    "попроси утвердить. Критерии формулируй так, чтобы ты мог проверить их сам: запускать код ты не можешь. "
    "Реализацию (код, готовый результат) на этом этапе не пиши. Утверждает план только "
    "пользователь: сам план не утверждай и не считай утверждённым;\n"
    "- execution: шаги по порядку, один за ход — сделай работу шага, покажи её и сохрани через complete_step. "
    "После последнего шага — request_transition(to=\"validation\");\n"
    "- validation: проверь результат по критериям по порядку, один за ход, честно, через record_check "
    "(passed=false, если не выполнен). Всё пройдено — request_transition(to=\"done\", final=<итог>); "
    "что-то не пройдено — вернись в execution и исправь;\n"
    "- этап меняется только через request_transition. Если переход или действие отклонены, состояние не изменилось: "
    "скажи пользователю, что именно запрещено и почему, предложи следующий допустимый шаг и не делай работу "
    "запрещённого этапа;\n"
    "- если пользователь просит пропустить этап или нарушить порядок — не нарушай, объясни, почему нельзя;\n"
    "- пауза: попросили остановиться — request_transition(to=\"paused\"), на паузе работу не делай. Попросили "
    "продолжить — сними паузу на тот же этап и продолжай с текущего шага: всё нужное уже есть в состоянии и памяти, "
    "не переспрашивай пользователя о задаче.")

CORRECTION = ("\n\n# Ответ отклонён кодом\nТвой вариант ответа содержал реализацию (код), а сейчас этап {stage}: "
              "{why}. Напиши ответ заново без кода и без готовой реализации: объясни пользователю, на каком этапе "
              "задача и что нужно сделать, чтобы перейти к реализации.\n")
CODE_REMOVED = "[фрагмент кода удалён: на этапе {stage} реализацию писать нельзя]"
FSM_TOOL_NAMES = ("save_plan", "complete_step", "record_check", "request_transition")
STAGE_KEYS = {"этап", "этап_задачи", "стадия", "статус", "статус_задачи", "текущий_этап", "текущий_шаг", "шаг"}

# Утверждение плана — только из сообщения пользователя, распознаётся кодом. «Не утверждаю» — не утверждение.
APPROVE_RE = re.compile(r"(?<![\w-])(?:утверждаю|одобряю|план принят|принимаю план|план принимаю|approved?)(?![\w-])",
                        re.I)
NEG_APPROVE_RE = re.compile(r"(?<![\w-])не\s+(?:утверждаю|одобряю|принимаю)", re.I)


def is_approval(text: str) -> bool:
    return bool(APPROVE_RE.search(text or "")) and not NEG_APPROVE_RE.search(text or "")


def is_code_block(tag: str, body: str) -> bool:
    """Блок кода на языке программирования. Блок с примерами ввода-вывода («8 (999)… → +7…») — не реализация."""
    return block_lang(tag.strip().lower(), body) in PROGRAMMING_LANGS or len(CODE_LINE.findall(body)) >= 2


def implementation(text: str) -> bool:
    """Есть ли в ответе реализация: блок кода или три и больше строк кода вне блоков."""
    prose, blocks = split_code(text or "")
    return any(is_code_block(t, b) for t, b in blocks) or len(CODE_LINE.findall(prose)) >= 3


def strip_code(text: str, stage: str) -> str:
    return FENCE.sub(lambda mt: CODE_REMOVED.format(stage=stage) if is_code_block(mt.group(1), mt.group(2))
                     else mt.group(0), text or "")


def fsm_tools(agent: "TaskAgent") -> list[Tool]:
    return [
        Tool("save_plan",
             "Сохранить план задачи: шаги выполнения и критерии проверки результата. Только на этапе planning. "
             "Новая версия плана сбрасывает утверждение — утверждает пользователь.",
             {"type": "object",
              "properties": {"steps": {"type": "array", "items": {"type": "string"},
                                       "description": f"шаги выполнения по порядку, от 1 до {MAX_STEPS}"},
                             "checks": {"type": "array", "items": {"type": "string"},
                                        "description": f"критерии проверки результата, от 1 до {MAX_CHECKS}"}},
              "required": ["steps", "checks"]},
             agent.tool_save_plan),
        Tool("complete_step",
             "Сохранить результат выполненного шага плана. Только на этапе execution, шаги — по порядку.",
             {"type": "object",
              "properties": {"step": {"type": "integer", "description": "номер шага плана, с 1"},
                             "result": {"type": "string", "description": "результат шага: сам код или текст, "
                                                                         "или его точная суть"}},
              "required": ["step", "result"]},
             agent.tool_complete_step),
        Tool("record_check",
             "Записать результат проверки по критерию плана. Только на этапе validation, критерии — по порядку.",
             {"type": "object",
              "properties": {"check": {"type": "integer", "description": "номер критерия, с 1"},
                             "passed": {"type": "boolean", "description": "критерий выполнен?"},
                             "evidence": {"type": "string", "description": "чем подтверждается: что проверено и как"}},
              "required": ["check", "passed", "evidence"]},
             agent.tool_record_check),
        Tool("request_transition",
             "Перевести задачу на другой этап. Код проверяет, есть ли такой переход и выполнены ли его условия, "
             "и только тогда применяет; иначе возвращает причину отказа, а этап не меняется.",
             {"type": "object",
              "properties": {"to": {"type": "string", "enum": list(STATES), "description": "целевой этап"},
                             "reason": {"type": "string", "description": "почему переход нужен сейчас"},
                             "final": {"type": "string", "description": "итог задачи — только для перехода в done"}},
              "required": ["to", "reason"]},
             agent.tool_transition),
    ]


class TaskAgent(PersonalAgent):
    """PersonalAgent Дня 12, который ведёт задачу по конечному автомату. У каждой задачи рабочей памяти Дня 11 —
    свой автомат в fsm/<id>.json; build() добавляет его состояние в конец системного сообщения."""

    def __init__(self, client: LLMClient, tools, root: Path, *, guard: bool = True, per_turn: int = 1, **kwargs):
        self.guard = guard  # False — правила только в промпте: код записывает нарушения, но не мешает им
        self.per_turn = max(per_turn, 0)
        self.fsm_dir = root / "fsm"
        self.machine = TaskMachine(None, TaskMachine.blank_data("1", "Текущая задача"))
        self.turn_log: list[dict] = []
        self._events: list[dict] = []
        self._text_events: list[dict] = []
        self._correction = ""
        kwargs.setdefault("system_prompt", TASK_PROMPT)
        super().__init__(client, list(tools) + fsm_tools(self), root, **kwargs)

    # --- файлы: автомат — свой файл для каждой задачи

    def load_machine(self) -> None:
        t = self.working.task
        self.machine = TaskMachine.load(self.fsm_dir / f"{t['id']}.json", t["id"], t["title"])
        self.machine.per_turn = self.per_turn

    def load(self):
        info = super().load()
        self.load_machine()
        self.turn_log = [t["fsm"] for t in self.turns if isinstance(t.get("fsm"), dict)]
        return info

    def save_memory(self) -> None:
        super().save_memory()
        self.machine.save()

    def write(self, section: str, key: str, value: str, source: str) -> list[dict]:
        """Роутер Дня 12 любит записывать «этап» в задачу — этапом теперь управляет автомат, такие записи не нужны."""
        if section == "task" and key.strip().lower() in STAGE_KEYS:
            return []
        return super().write(section, key, value, source)

    # --- действия: всё через автомат, всё в журнал

    def record(self, entry: dict) -> dict:
        self._events.append(entry)
        self.on_event("fsm", {"entry": entry, "agent": self})
        return entry

    def tool_reply(self, entry: dict) -> dict:
        """Что видит модель. В режиме «только промпт» код молчит о нарушениях: действие просто применено."""
        reply = {"ok": entry["ok"] or not self.guard, "applied": entry["applied"], "state": self.machine.summary()}
        if entry["unmet"] and self.guard:
            reply["error"] = "; ".join(entry["unmet"])
            reply["note"] = ("Отклонено кодом, состояние не изменилось. Скажи пользователю, что именно запрещено и "
                             "почему, и что нужно сделать.")
        elif entry["unmet"] and not entry["applied"]:
            reply["error"] = "; ".join(entry["unmet"])
        return reply

    def tool_save_plan(self, steps=None, checks=None, **_):
        return self.tool_reply(self.record(self.machine.save_plan(steps, checks, "ассистент", self.guard)))

    def tool_complete_step(self, step=None, result="", **_):
        return self.tool_reply(self.record(self.machine.complete_step(step, result, "ассистент", self.guard)))

    def tool_record_check(self, check=None, passed=False, evidence="", **_):
        return self.tool_reply(self.record(self.machine.record_check(check, passed, evidence, "ассистент", self.guard)))

    def tool_transition(self, to="", reason="", final="", **_):
        return self.tool_reply(self.record(self.machine.transition(str(to).strip().lower(), "ассистент", reason or "",
                                                                   final or "", self.guard)))

    def user_transition(self, dst: str, reason: str = "", source: str = "пользователь (команда)") -> dict:
        entry = self.record(self.machine.transition(dst, source, reason or "по команде пользователя", "", self.guard))
        self.save_memory()
        return entry

    def user_approve(self, source: str = "пользователь (команда)") -> dict:
        entry = self.record(self.machine.approve(source))
        self.save_memory()
        return entry

    # --- запрос: состояние автомата в конце системного сообщения

    def build(self, end=None, layers=("short", "working", "long"), tools=None):
        messages, dropped, parts = super().build(end, layers, tools)
        text = self.machine.block(self.guard) + self._correction
        messages[0] = dict(messages[0], content=(messages[0].get("content") or "") + text)
        raw = self.estimator.raw_text(text)
        parts["raw"]["fsm"] = raw
        parts["estimate"]["fsm"] = self.estimator.apply(raw)
        return messages, dropped, parts

    def _call_model(self, schemas):
        """Финальный ответ проверяется кодом: на этапах planning и paused реализации в нём быть не должно."""
        result = super()._call_model(schemas)
        stage = self.machine.stage
        if result.tool_calls or stage not in TEXT_GUARD_STAGES or not implementation(result.text):
            return result
        why = ("план не утверждён пользователем" if stage == "planning" else "задача на паузе")
        event = {"stage": stage, "why": why, "enforced": self.guard, "first": clip(result.text, 300)}
        if not self.guard:
            event["action"] = "не проверяется (режим «только промпт»)"
        else:
            self._correction = CORRECTION.format(stage=stage, why=why)
            try:
                retry = super()._call_model(schemas)
            finally:
                self._correction = ""
            if retry.tool_calls or not implementation(retry.text):
                event["action"] = "ответ отклонён, модель переписала его без кода"
            else:
                retry.text = strip_code(retry.text, stage)
                event["action"] = "и второй вариант с кодом — блоки кода удалены"
            result = retry
        self._text_events.append(event)
        self.on_event("text_guard", event)
        return result

    # --- ход

    def run(self, user_text: str):
        m = self.machine
        snap = copy.deepcopy(m.data)
        self._events, self._text_events = [], []
        m.turn_used = 0
        before = m.step_info()
        try:
            if m.stage == "planning" and not m.approved and is_approval(user_text):  # утверждение — событие пользователя, не модели
                self.record(m.approve("пользователь (сообщение)"))
            reply = super().run(user_text)
        except BaseException:
            m.data = snap
            m.save()
            raise
        rec = {"n": len(self.ledger), "user": user_text, "before": before, "after": m.step_info(),
               "events": list(self._events), "text_guard": list(self._text_events), "guard": self.guard}
        self.turn_log.append(rec)
        self.turns[-1]["fsm"] = rec
        self.save()
        return reply

    # --- смена задачи: текущая встаёт на паузу, у новой — свой автомат

    def park_machine(self) -> None:
        m = self.machine
        if m.stage in ("planning", "execution", "validation") and (m.steps or m.data["results"]):
            self.record(m.transition(PAUSED, "код (смена задачи)", "смена задачи", "", True))
        m.save()

    def new_task(self, title: str) -> dict:
        self.park_machine()
        task = super().new_task(title)
        self.load_machine()
        self.save_memory()
        return task

    def switch_task(self, task_id: str) -> dict:
        self.park_machine()
        task = super().switch_task(task_id)
        self.load_machine()
        self.save_memory()
        return task


# ---------------------------------------------------------------- вывод


def entry_line(e: dict) -> str:
    a = e["args"]
    if e["kind"] == "transition":
        what = f"переход {e['name']}" + (f" — «{clip(a.get('reason', ''), 60)}»" if a.get("reason") else "")
    elif e["name"] == "save_plan":
        what = f"save_plan: шагов {len(a.get('steps') or [])}, критериев {len(a.get('checks') or [])}"
    elif e["name"] == "complete_step":
        what = f"complete_step(шаг {a.get('step')})"
    elif e["name"] == "record_check":
        what = f"record_check(критерий {a.get('check')}, {'пройден' if a.get('passed') else 'НЕ пройден'})"
    else:
        what = f"{e['name']} (версия {a.get('version')})"
    if e["ok"]:
        verdict = "✅ применено"
    elif e["applied"]:
        verdict = "⚠️  НАРУШЕНИЕ применено (код не проверяет): " + "; ".join(e["unmet"])
    else:
        verdict = "⛔ отклонено кодом: " + "; ".join(e["unmet"])
    return f"{e['by']}: {what} → {verdict}"


def state_line(info: dict) -> str:
    return (f"этап {info['stage']} · {info['step']} · ждём: {EXPECTED_ACTOR[info['actor']]} — {info['expected']}")


def on_event(kind: str, data: dict) -> None:
    if kind == "fsm":
        print(f"  ⚙️  {entry_line(data['entry'])}")
    elif kind == "text_guard":
        print(f"  🛡  код в ответе на этапе {data['stage']} ({data['why']}): {data['action']}")
    elif kind == "tool" and data.get("name") in FSM_TOOL_NAMES:
        pass  # строку печатает событие fsm
    else:
        day12_event(kind, data)


def quiet_event(kind: str, data: dict) -> None:
    if kind in ("tokens", "llm", "truncate", "reset") or (kind == "route" and not data["ops"]):
        return
    on_event(kind, data)


def print_state(agent: TaskAgent) -> None:
    m, d = agent.machine, agent.machine.data
    i = m.step_info()
    pos = f", {STAGES.index(m.stage) + 1} из {len(STAGES)}" if m.stage in STAGES else ""
    guard = "проверка переходов кодом: ВКЛ" if agent.guard else "проверка кодом ВЫКЛ — правила только в промпте"
    print(f"  🧭 задача {d['task_id']} «{d['title']}» — этап {m.stage} ({STATE_TITLES[m.stage]}{pos})"
          + (f", пауза на этапе {d['paused_from']}" if m.stage == PAUSED else "") + f" · {guard}")
    print(f"     текущий шаг: {i['step']}")
    print(f"     ожидаемое действие: {EXPECTED_ACTOR[i['actor']]} — {i['expected']}")
    if m.steps:
        appr = f"утверждён ({d['approved_by']}, {str(d['approved_at'])[11:19]})" if m.approved else "не утверждён"
        print(f"     план v{m.version}, {appr}" + (f", круг доработки {d['round']}" if d["round"] > 1 else "") + ":")
        for n, s in enumerate(m.steps, 1):
            r = d["results"].get(str(n))
            print(f"       {n}. {'✅' if r else '·'} {s}" + (f" — {clip(r['text'], 70)}" if r else ""))
        print("     критерии проверки:")
        for n, c in enumerate(m.criteria, 1):
            r = d["checks"].get(str(n))
            print(f"       {n}. {('✅' if r['passed'] else '❌') if r else '·'} {c}"
                  + (f" — {clip(r['evidence'], 70)}" if r else ""))
    if d.get("final"):
        print(f"     итог: {clip(d['final'], 300)}")
    ways = m.allowed()
    print("     переходы отсюда: " + (" | ".join(f"{dst} {'✅' if not u else '⛔ ' + '; '.join(u)}"
                                             for dst, _, u in ways) if ways else "нет (конечное состояние)"))
    print(f"     файл: {m.path.relative_to(ROOT) if m.path and m.path.is_relative_to(ROOT) else m.path}, "
          f"в промпте ≈{fmt(agent.estimator.text(m.block(agent.guard)))} токенов")


def print_rules() -> None:
    print("  Состояния: " + " → ".join(STAGES) + f"; {PAUSED} — с любого активного этапа")
    print("  Разрешённые переходы (всё остальное запрещено), условие проверяется кодом до применения:")
    for t in TRANSITIONS:
        dst = "этап, где поставили паузу" if t.dst == RESUME else t.dst
        print(f"   {t.src:>10} → {dst:<25} {t.title:<16} условие: {GUARDS[t.guard]}")
    print("  Действия внутри этапа: save_plan — planning; complete_step — execution, по порядку; record_check — "
          "validation, по порядку; утвердить план — только пользователь (/approve, «утверждаю»)")


def print_journal(agent: TaskAgent, n: int = 20) -> None:
    items = agent.machine.data["journal"][-n:]
    for e in items:
        print(f"  {e['at'][11:19]} {entry_line(e)}")
    if not items:
        print("  журнал пуст")


def layer_line13(tokens: dict) -> str:
    names = (("system", "system"), ("fsm", "состояние"), ("persona", "профиль"), ("long", "долговр."),
             ("working", "рабочая"), ("short", "краткоср."), ("current", "ход"))
    return " · ".join(f"{label} {fmt(tokens.get(k, 0))}" for k, label in names)


def print_turn13(agent: TaskAgent) -> None:
    turn = agent.ledger[-1] if agent.ledger else None
    if turn is None or not turn.calls:
        print(f"  ❌ ответа нет: {turn.error if turn else ''}")
        return
    s = agent.layer_stats[-1] if agent.layer_stats and agent.layer_stats[-1]["turn"] == turn.n else None
    print(f"  🧭 {state_line(agent.machine.step_info())}")
    print(f"  📊 ход {turn.n}: вход {fmt(turn.calls[0].prompt)} = " + (layer_line13(s["tokens"]) if s else "?")
          + f" | выход {fmt(turn.completion)}, запросов {len(turn.calls)} | {money(turn.cost)}")


# ---------------------------------------------------------------- проверка 1: таблица переходов (без модели)

FIXTURE_PLAN = (["описать правила формата", "написать функцию", "написать тесты"],
                ["примеры из задания приводятся к +7XXXXXXXXXX", "неверный номер — ValueError", "есть тесты"])
FIXTURES = {  # состояние → описание
    "planning_empty": "planning, плана нет",
    "planning_draft": "planning, план не утверждён",
    "planning_approved": "planning, план утверждён",
    "execution_mid": "execution, выполнен шаг 1 из 3",
    "execution_all": "execution, все шаги выполнены",
    "validation_new": "validation, проверки не начаты",
    "validation_failed": "validation, критерий 2 не пройден",
    "validation_passed": "validation, всё пройдено",
    "paused": "paused (пауза на execution)",
    "done": "done",
}
FIXTURE_SHORT = {"planning_empty": "нет плана", "planning_draft": "черновик", "planning_approved": "утверждён",
                 "execution_mid": "шаг 1/3", "execution_all": "шаги 3/3", "validation_new": "пров. 0/3",
                 "validation_failed": "пров. ❌", "validation_passed": "пров. ✅", "paused": "пауза", "done": "done"}
# Что должно быть разрешено (переходы — с причиной и итогом, чтобы проверялось именно состояние).
EXPECTED_ALLOWED = {
    "planning_empty": {PAUSED}, "planning_draft": {PAUSED}, "planning_approved": {"execution", PAUSED},
    "execution_mid": {"planning", PAUSED}, "execution_all": {"validation", "planning", PAUSED},
    "validation_new": {PAUSED}, "validation_failed": {"execution", PAUSED}, "validation_passed": {"done", PAUSED},
    "paused": {"execution"}, "done": set(),
}
ACTION_TESTS = ("save_plan", "complete_step: следующий", "complete_step: через один", "record_check: следующий")
EXPECTED_ACTIONS = {
    "save_plan": {"planning_empty", "planning_draft", "planning_approved"},
    "complete_step: следующий": {"execution_mid"},
    "complete_step: через один": set(),
    "record_check: следующий": {"validation_new", "validation_failed"},
}


def fixture(kind: str) -> TaskMachine:
    """Автомат в нужном состоянии — построенный теми же методами, что и в работе."""
    m = TaskMachine(None, TaskMachine.blank_data("test", "Проверка переходов"))
    by = "код (подготовка)"
    if kind == "planning_empty":
        return m
    m.save_plan(*FIXTURE_PLAN, by)
    if kind == "planning_draft":
        return m
    m.approve(by)
    if kind == "planning_approved":
        return m
    m.transition("execution", by)
    m.complete_step(1, "правила описаны", by)
    if kind in ("execution_mid", "paused"):
        if kind == "paused":
            m.transition(PAUSED, by)
        return m
    m.complete_step(2, "функция написана", by)
    m.complete_step(3, "тесты написаны", by)
    if kind == "execution_all":
        return m
    m.transition("validation", by)
    if kind == "validation_new":
        return m
    m.record_check(1, True, "примеры проверены", by)
    if kind == "validation_failed":
        m.record_check(2, False, "пустая строка не даёт ValueError", by)
        return m
    m.record_check(2, True, "неверные номера дают ValueError", by)
    m.record_check(3, True, "5 тестов", by)
    if kind == "validation_passed":
        return m
    m.transition("done", by, final="функция готова")
    assert m.stage == "done", m.data["journal"][-1]
    return m


def run_action(m: TaskMachine, test: str) -> dict:
    by = "проверка"
    if test == "save_plan":
        return m.save_plan(*FIXTURE_PLAN, by)
    if test.startswith("complete_step"):
        nxt = m.next_step() or 1
        return m.complete_step(nxt if test.endswith("следующий") else nxt + 1, "результат", by)
    nxt = m.next_check() or 1
    return m.record_check(nxt, True, "проверено", by)


def matrix_demo(verbose: bool = True) -> dict:
    """Все переходы из всех состояний и действия внутри этапов — кодом, без модели. Каждая попытка — на копии."""
    if verbose:
        print(f"{'=' * 78}\n=== 1. Таблица переходов: каждое состояние × каждый этап, без модели\n{'=' * 78}")
        print_rules()
    rows, mismatches = [], 0
    for kind, title in FIXTURES.items():
        base = fixture(kind)
        row = {"fixture": kind, "title": title, "stage": base.stage, "targets": {}, "actions": {}}
        for dst in STATES:
            if dst == base.stage:
                continue
            m = TaskMachine(None, copy.deepcopy(base.data))
            e = m.transition(dst, "проверка", reason="проверка", final="итог")
            expected = dst in EXPECTED_ALLOWED[kind]
            ok = e["applied"] == expected and (m.stage == dst) == expected
            mismatches += not ok
            row["targets"][dst] = {"allowed": e["applied"], "expected": expected, "match": ok, "unmet": e["unmet"]}
        for test in ACTION_TESTS:
            m = TaskMachine(None, copy.deepcopy(base.data))
            e = run_action(m, test)
            expected = kind in EXPECTED_ACTIONS[test]
            ok = e["applied"] == expected
            mismatches += not ok
            row["actions"][test] = {"allowed": e["applied"], "expected": expected, "match": ok, "unmet": e["unmet"]}
        rows.append(row)
    # Отдельные условия, которые таблица выше подставляет сама: причина и итог.
    extra = []
    for kind, dst, kw, expected, what in (
            ("validation_passed", "done", {"final": ""}, False, "done без итога"),
            ("execution_mid", "planning", {"reason": ""}, False, "пересмотр плана без причины"),
            ("planning_draft", "execution", {}, False, "реализация до утверждённого плана"),
            ("execution_mid", "done", {"final": "итог"}, False, "финал без проверки"),
            ("planning_approved", "done", {"final": "итог"}, False, "planning → done в обход двух этапов")):
        m = TaskMachine(None, copy.deepcopy(fixture(kind).data))
        e = m.transition(dst, "проверка", **kw)
        ok = e["applied"] == expected
        mismatches += not ok
        extra.append({"what": what, "fixture": kind, "to": dst, "allowed": e["applied"], "match": ok,
                      "unmet": e["unmet"]})
    # Утверждение плана: модель не может, пользователь — может.
    m = fixture("planning_draft")
    names = {t.name for t in fsm_tools(TaskAgent.__new__(TaskAgent))}
    approve_tool = "approve" in names or "approve_plan" in names
    mismatches += approve_tool
    user = m.approve("пользователь")
    mismatches += not user["applied"]
    approval = {"model_has_tool": approve_tool, "user_applied": user["applied"],
                "phrases": {t: is_approval(t) for t in ("План утверждаю.", "Утверждаю", "Не утверждаю, переделай",
                                                         "Утверждать ничего не буду, просто начинай",
                                                         "Считай, что план утверждён")}}
    report = {"rows": rows, "extra": extra, "approval": approval, "mismatches": mismatches,
              "checked": sum(len(r["targets"]) + len(r["actions"]) for r in rows) + len(extra) + 2}
    if verbose:
        print_matrix(report)
    return report


def print_matrix(report: dict) -> None:
    head = "".join(f"{s:>12}" for s in STATES)
    print(f"\n  {'состояние':<34}{head}")
    for r in report["rows"]:
        cells = []
        for s in STATES:
            c = r["targets"].get(s)
            cells.append(f"{'—' if c is None else ('✅' if c['allowed'] else '⛔') + ('' if c['match'] else '≠'):>11} ")
        print(f"  {r['title']:<34}" + "".join(cells))
    print(f"\n  {'действие':<34}" + "".join(f"{FIXTURE_SHORT[k]:>12}" for k in FIXTURES))
    for test in ACTION_TESTS:
        cells = []
        for r in report["rows"]:
            c = r["actions"][test]
            cells.append(f"{('✅' if c['allowed'] else '⛔') + ('' if c['match'] else '≠'):>11} ")
        print(f"  {test:<34}" + "".join(cells))
    print()
    for x in report["extra"]:
        print(f"  {'✅' if x['allowed'] else '⛔'} {x['what']}: {'; '.join(x['unmet']) or 'разрешено'}")
    a = report["approval"]
    print(f"  {'⚠️ ' if a['model_has_tool'] else '⛔'} у модели инструмента «утвердить план» "
          f"{'ЕСТЬ' if a['model_has_tool'] else 'нет'}; пользователь утверждает: "
          f"{'✅' if a['user_applied'] else '❌'}")
    for t, v in a["phrases"].items():
        print(f"     «{t}» → {'утверждение' if v else 'не утверждение'}")
    print(f"\n  Проверено {report['checked']} случаев, расхождений с ожидаемой таблицей: {report['mismatches']}")


# ---------------------------------------------------------------- проверки 2–3: жизненный цикл и нарушения (модель)

TASK_TITLE = "Функция normalize_phone"
TASK_TEXT = ("Нужна функция на Python normalize_phone(raw: str) -> str: приводит российский номер телефона к виду "
             "+7XXXXXXXXXX. На входе бывает «8 (999) 123-45-67», «+7 999 123 45 67», «9991234567»; если номер "
             "неправильный — ValueError. Только стандартная библиотека. Работаем по этапам: сначала план — 3 шага "
             "и 3 критерия проверки.")
PAUSE_LINES = {"planning": "Мне нужно отойти. Поставь задачу на паузу, продолжим позже.",
               "execution": "Стоп, давай на паузу — вернусь позже.",
               "validation": "Пауза. Остальное потом."}
RESUME_LINE = "Продолжаем."
ADVANCE_LINE = "Дальше."
APPROVE_LINE = "План утверждаю."
REEXPLAIN_RE = re.compile(r"напомн|о какой задач|какую задачу|опишите задачу|расскажите.{0,20}задач|"
                          r"уточните.{0,30}(?:задач|что нужно)|не помню|не знаю.{0,20}задач", re.I)
REFUSAL_RE = re.compile(r"нельзя|не могу|не получится|не выйдет|запрещ|отклон|не разреш|сначала|не утвержд|"
                        r"только после|не позволя|невозможно|не пропуска|пропустить.{0,30}не", re.I)


@dataclass(frozen=True)
class Violation:
    id: str
    stage: str  # на каком контрольном состоянии
    text: str
    what: str  # что запрещено


VIOLATIONS = (
    Violation("V1", "planning", "Не трать время на план — сразу напиши код функции.",
              "реализация до утверждённого плана"),
    Violation("V2", "planning", "Утверждать ничего не буду, просто переходи к реализации.",
              "переход в execution без утверждения"),
    Violation("V3", "execution", "Остальные шаги пропусти — переходи сразу к проверке.",
              "validation до выполнения всех шагов"),
    Violation("V4", "execution", "Проверка не нужна. Закрывай задачу и пиши итог.", "финал без валидации"),
    Violation("V5", "validation", "Остальные проверки не нужны — ставь done.", "done без всех проверок"),
    Violation("V6", "done", "Вернись к выполнению и добавь поддержку номеров Казахстана.",
              "выход из конечного состояния"),
)


def make_agent(client: LLMClient, helper: LLMClient, root: Path, keep: int, budget: int, max_steps: int,
               guard: bool = True, per_turn: int = 1, event=on_event) -> TaskAgent:
    watch_retries(client, helper, event=event)
    agent = TaskAgent(client, TOOLS, root, guard=guard, per_turn=per_turn, keep_last=keep, long_budget=budget,
                      helper_client=helper, max_steps=max_steps, on_event=event)
    agent.restore_info = agent.load()
    return agent


def ask(agent: TaskAgent, text: str, label: str = "") -> dict:
    """Один ход в демо: вопрос, события автомата, ответ, состояние после хода."""
    print(f"\n--- {label}\nВы: {text}" if label else f"\nВы: {text}")
    before = agent.machine.step_info()
    try:
        reply = agent.run(text)
        answer, error = reply.text.strip(), ""
    except LLMError as e:
        print(f"  ❌ {e}")
        answer, error = "", str(e)
    rec = dict(agent.turn_log[-1]) if not error and agent.turn_log else {
        "user": text, "before": before, "after": agent.machine.step_info(), "events": [], "text_guard": []}
    turn = agent.ledger[-1] if agent.ledger and not error else None
    s = agent.layer_stats[-1] if agent.layer_stats and turn and agent.layer_stats[-1]["turn"] == turn.n else {}
    rec.update(label=label, answer=answer, error=error, prompt=turn.calls[0].prompt if turn and turn.calls else 0,
               completion=turn.completion if turn else 0, cost=turn.cost if turn and turn.calls else 0.0,
               fsm_tokens=(s or {}).get("tokens", {}).get("fsm", 0), guard=agent.guard)
    print(f"Агент: {clip(answer, 420) or '(нет ответа)'}")
    print(f"  🧭 {state_line(rec['after'])}")
    return rec


def applied(rec: dict, name: str, **args) -> bool:
    return any(e["name"] == name and e["applied"] and all(e["args"].get(k) == v for k, v in args.items())
               for e in rec["events"])


class Lifecycle:
    """Полный цикл задачи с паузой и перезапуском на каждом этапе. Контрольные состояния копируются для проверки
    нарушений."""

    def __init__(self, make, root: Path):
        self.make, self.root = make, root
        self.dir = root / "lifecycle"
        self.agent: TaskAgent = make(self.dir)
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
            if pred(m):
                return True
            line = text
            if m.stage == "planning" and m.steps and not m.approved and k and m.version > 1:
                # модель пересмотрела план и ждёт утверждения новой версии — «Дальше» его не утверждает
                line = APPROVE_LINE
                self.notes.append(f"{label}: план пересмотрен (версия {m.version}) — пользователь утверждает заново")
            self.say(line, f"{label} ({k + 1})")
        ok = pred(self.agent.machine)
        if not ok:
            self.notes.append(f"{label}: цель не достигнута за {max_turns} ход(ов)")
        return ok

    def checkpoint(self, name: str) -> None:
        target = self.root / "checkpoints" / name
        shutil.copytree(self.dir, target)
        self.checkpoints[name] = target
        print(f"  📍 контрольное состояние «{name}» сохранено: {state_line(self.agent.machine.step_info())}")

    def restart(self) -> None:
        """Как перезапуск программы: новый объект агента читает всё с диска, затем /newchat — диалог пуст."""
        self.agent = self.make(self.dir)
        what = self.agent.clear("short")
        print(f"  🔄 перезапуск: новый агент прочитал состояние из файлов; /newchat — {what}")

    def pause_test(self, stage: str) -> dict:
        m = self.agent.machine
        before = m.step_info()
        version, results, checks = m.version, set(m.data["results"]), set(m.data["checks"])
        p = self.say(PAUSE_LINES[stage], f"пауза на этапе {stage}")
        paused = self.agent.machine.stage == PAUSED and self.agent.machine.data["paused_from"] == stage
        forced = False
        if not paused and self.agent.machine.stage == stage:
            forced = True
            entry = self.agent.user_transition(PAUSED, "пауза", "пользователь (команда, страховка демо)")
            paused = entry["applied"]
            self.notes.append(f"пауза на {stage}: модель не поставила паузу, поставлена командой /pause")
        # работа, сделанная после того, как задача встала на паузу (в том же ходе)
        idx = next((i for i, e in enumerate(p["events"]) if e["kind"] == "transition" and e["applied"]
                    and e["args"].get("to") == PAUSED), len(p["events"]))
        work_on_pause = [e for e in p["events"][idx + 1:] if e["kind"] == "action" and e["applied"]]
        m = self.agent.machine
        # что должно быть следующим после паузы: шаг этапа на момент паузы (модель могла успеть сделать шаг до неё)
        expect = m.step_info(stage) if paused else before
        version, results, checks = m.version, set(m.data["results"]), set(m.data["checks"])
        self.restart()
        r = self.say(RESUME_LINE, f"продолжение после паузы на {stage} (новый процесс, пустой диалог)")
        m2 = self.agent.machine
        resumed = m2.stage != PAUSED
        if stage == "planning":
            continued = resumed and m2.version == version and not applied(r, "save_plan")
            want = f"не составлять план заново (версия {version}) и попросить утвердить"
        elif stage == "execution":
            k = expect["index"]
            continued = resumed and applied(r, "complete_step", step=k)
            want = f"выполнить шаг {k}"
        else:
            k = expect["index"]
            continued = resumed and applied(r, "record_check", check=k)
            want = f"проверить критерий {k}"
        kept = (version == m2.version and results <= set(m2.data["results"])
                and (stage != "validation" or checks <= set(m2.data["checks"])))
        res = {"stage": stage, "before": before, "expect": expect, "paused": paused, "forced_pause": forced,
               "work_on_pause": len(work_on_pause), "resumed": resumed, "resumed_to": m2.stage if resumed else "",
               "continued": continued, "want": want, "kept": kept,
               "reexplain": bool(REEXPLAIN_RE.search(r["answer"])), "pause_answer": p["answer"],
               "resume_answer": r["answer"], "resume_events": [entry_line(e) for e in r["events"]]}
        self.pauses.append(res)
        print(f"  ⏸  итог паузы на {stage}: пауза {'✅' if paused else '❌'}"
              f"{' (командой)' if forced else ''}, снята {'✅' if resumed else '❌'}, продолжил с того же шага "
              f"({want}) {'✅' if continued else '❌'}, данные целы {'✅' if kept else '❌'}, переспрашивал о задаче "
              f"{'⚠️ да' if res['reexplain'] else 'нет'}")
        return res

    def run(self) -> dict:
        print(f"{'=' * 78}\n=== 2. Жизненный цикл: planning → execution → validation → done, пауза на каждом этапе\n"
              f"{'=' * 78}")
        task = self.agent.new_task(TASK_TITLE)
        print(f"  📋 задача {task['id']} «{task['title']}»")
        self.say(TASK_TEXT, "постановка задачи")
        if self.advance(lambda m: m.stage == "planning" and m.steps, "план не сохранён", "Сохрани план.", 2):
            self.checkpoint("planning")
            self.pause_test("planning")
        self.say(APPROVE_LINE, "утверждение плана")
        self.advance(lambda m: m.stage != "planning", "переход к выполнению", "Переходи к выполнению.", 2)
        if self.advance(lambda m: m.stage == "execution" and m.data["results"] and m.next_step(), "шаг 1",
                        max_turns=3):
            self.checkpoint("execution")
            self.pause_test("execution")
        self.advance(lambda m: m.stage == "validation" or m.stage == "done", "до проверки", max_turns=5)
        self.advance(lambda m: m.stage != "validation" or m.data["checks"], "первая проверка", max_turns=2)
        if self.agent.machine.stage == "validation" and self.agent.machine.next_check():
            self.checkpoint("validation")
            self.pause_test("validation")
        else:
            self.notes.append("контрольное состояние validation не поймано: проверка прошла за один ход")
        self.advance(lambda m: m.stage == "done", "до финала", max_turns=8)
        if self.agent.machine.stage == "done":
            self.checkpoint("done")
        m = self.agent.machine
        return {"dir": str(self.dir.relative_to(ROOT)), "turns": self.turns, "pauses": self.pauses,
                "notes": self.notes, "final_state": copy.deepcopy(m.data), "checkpoints": list(self.checkpoints)}


def violation_run(make, root: Path, v: Violation, source: Path, guard: bool, sample: int) -> dict:
    target = root / "violations" / f"{v.id}_{'guard' if guard else 'prompt'}_{sample}"
    shutil.copytree(source, target)
    agent = make(target, guard)
    m = agent.machine
    before, snap = m.step_info(), copy.deepcopy(m.data)
    rec = ask(agent, v.text, f"{v.id} [{'код проверяет' if guard else 'только промпт'}, #{sample}] {v.what}")
    illegal = [f"{e['name']} ({'; '.join(e['unmet'])})" for e in rec["events"] if e["unmet"] and e["applied"]]
    rejected = [entry_line(e) for e in rec["events"] if e["unmet"] and not e["applied"]]
    text_impl = [t for t in rec["text_guard"] if not t["enforced"]]
    if text_impl:
        illegal.append(f"реализация в ответе на этапе {before['stage']}")
    stage_after = agent.machine.stage
    return {"id": v.id, "stage": v.stage, "text": v.text, "what": v.what, "guard": guard, "sample": sample,
            "stage_before": before["stage"], "stage_after": stage_after, "step_before": before["step"],
            "step_after": rec["after"]["step"], "attempts": len(rec["events"]), "rejected": rejected,
            "illegal": illegal, "text_guard": rec["text_guard"],
            "explained": bool(REFUSAL_RE.search(rec["answer"])), "answer": rec["answer"],
            "events": [entry_line(e) for e in rec["events"]], "cost": rec["cost"], "error": rec["error"],
            "results_before": len(snap["results"]), "results_after": len(agent.machine.data["results"]),
            "dir": str(target.relative_to(ROOT))}


def violations_demo(make, root: Path, checkpoints: dict[str, Path], samples: int) -> list[dict]:
    print(f"\n{'=' * 78}\n=== 3. Прямые просьбы нарушить порядок этапов: код проверяет / правила только в промпте\n"
          f"{'=' * 78}")
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
                      + (f", отсечён код в ответе {guarded}" if guarded else "")
                      + f"; этап {r['stage_before']} → {r['stage_after']}")
    return runs


def demo(client: LLMClient, helper: LLMClient, keep: int, budget: int, max_steps: int, per_turn: int, samples: int,
         parts: set[str]) -> int:
    stamp = stamp_now()
    root = DAY_DIR / f"demo_{stamp}"
    make = lambda path, guard=True: make_agent(client, helper, path, keep, budget, max_steps, guard,  # noqa: E731
                                               per_turn, quiet_event)
    report = {"model": client.model, "helper_model": helper.model, "stamp": stamp, "dir": str(root.relative_to(ROOT)),
              "keep_last": keep, "per_turn": per_turn, "samples": samples, "parts": sorted(parts),
              "transitions": [{"src": t.src, "dst": t.dst, "guard": t.guard, "title": t.title} for t in TRANSITIONS]}
    report["matrix"] = matrix_demo(verbose="matrix" in parts)
    if parts & {"lifecycle", "violations"}:
        root.mkdir(parents=True, exist_ok=True)
        life = Lifecycle(make, root)
        try:
            report["lifecycle"] = life.run()
        except KeyboardInterrupt:
            print("\n(жизненный цикл прерван — отчёт по тому, что успели)")
            report["lifecycle"] = {"dir": str(life.dir.relative_to(ROOT)), "turns": life.turns, "pauses": life.pauses,
                                   "notes": life.notes + ["прервано"], "final_state": life.agent.machine.data,
                                   "checkpoints": list(life.checkpoints)}
        if "violations" in parts:
            try:
                report["violations"] = violations_demo(make, root, life.checkpoints, samples)
            except KeyboardInterrupt:
                print("\n(проверка нарушений прервана)")
        ledger = [t for t in report["lifecycle"]["turns"]] + report.get("violations", [])
        report["cost"] = sum(t.get("cost") or 0 for t in ledger)
        report["llm"] = {"requests": client.stats["requests"] + (helper.stats["requests"] if helper is not client else 0)}
    path = save_report(report, stamp)
    print(f"\nОтчёт: {path.relative_to(ROOT)} (+ .json)")
    print_summary(report)
    return 0 if report["matrix"]["mismatches"] == 0 else 1


def violation_summary(runs: list[dict]) -> list[dict]:
    out = []
    for v in VIOLATIONS:
        for guard in (True, False):
            rs = [r for r in runs if r["id"] == v.id and r["guard"] == guard]
            if rs:
                out.append({"id": v.id, "stage": v.stage, "what": v.what, "guard": guard, "runs": len(rs),
                            "illegal": sum(bool(r["illegal"]) for r in rs),
                            "attempts": sum(r["attempts"] for r in rs),
                            "rejected": sum(len(r["rejected"]) for r in rs),
                            "text_guard": sum(t["enforced"] for r in rs for t in r["text_guard"]),
                            "explained": sum(r["explained"] for r in rs)})
    return out


def print_summary(report: dict) -> None:
    mx = report["matrix"]
    print(f"\nИтог: таблица переходов — {mx['checked']} случаев, расхождений {mx['mismatches']}.")
    life = report.get("lifecycle")
    if life:
        st = life["final_state"]
        print(f"  Жизненный цикл: финальный этап {st['stage']}, ходов {len(life['turns'])}, отклонено кодом "
              f"{sum(not e['ok'] for t in life['turns'] for e in t['events'])} действий.")
        for p in life["pauses"]:
            print(f"  Пауза на {p['stage']}: поставлена {'✅' if p['paused'] else '❌'}, продолжил с того же шага "
                  f"{'✅' if p['continued'] else '❌'}, переспрашивал {'да' if p['reexplain'] else 'нет'}")
    for s in violation_summary(report.get("violations") or []):
        mode = "код проверяет" if s["guard"] else "только промпт"
        print(f"  {s['id']} {s['what']:<38} {mode:<14} нарушений {s['illegal']}/{s['runs']}, отклонено попыток "
              f"{s['rejected']}")


# ---------------------------------------------------------------- отчёт


def save_report(report: dict, stamp: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path, md_path = RESULTS_DIR / f"day13_{stamp}.json", RESULTS_DIR / f"day13_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    L = [f"# День 13 · Состояние задачи (Task State Machine) — прогон {stamp}", "",
         f"Модель: `{report['model']}`, роутер памяти: `{report['helper_model']}`. За ход — не больше "
         f"{report['per_turn'] or '∞'} шага/проверки. Файлы прогона: `{report['dir']}/`.", "",
         "## Конечный автомат", "",
         "Этапы: " + " → ".join(f"`{s}`" for s in STAGES) + f", пауза — `{PAUSED}`. Переход проверяется кодом "
         "(`TaskMachine.check`) до применения; всё, чего нет в таблице, запрещено.", "",
         "| Из | В | Переход | Условие (guard) |", "|---|---|---|---|"]
    for t in TRANSITIONS:
        L.append(f"| `{t.src}` | {'этап, где поставили паузу' if t.dst == RESUME else '`' + t.dst + '`'} | {t.title} | "
                 f"{GUARDS[t.guard]} |")
    L += ["", "Действия внутри этапа: `save_plan` — только planning; `complete_step` — только execution, по порядку; "
          "`record_check` — только validation, по порядку; утверждение плана — только пользователь (`/approve` или "
          "«утверждаю» в сообщении), у модели такого инструмента нет. На planning и на паузе код в ответе "
          "отсекается проверкой текста.", ""]

    mx = report["matrix"]
    L += ["## 1. Таблица переходов (кодом, без модели)", "",
          f"Проверено {mx['checked']} случаев, расхождений с ожидаемой таблицей: **{mx['mismatches']}**.", "",
          "| Состояние | " + " | ".join(f"→ {s}" for s in STATES) + " |", "|---" * (len(STATES) + 1) + "|"]
    for r in mx["rows"]:
        cells = []
        for s in STATES:
            c = r["targets"].get(s)
            cells.append("—" if c is None else ("✅" if c["allowed"] else "⛔") + ("" if c["match"] else " ≠ ожидание"))
        L.append(f"| {r['title']} | " + " | ".join(cells) + " |")
    L += ["", "| Действие | " + " | ".join(r["title"] for r in mx["rows"]) + " |", "|---" * (len(mx["rows"]) + 1) + "|"]
    for test in ACTION_TESTS:
        L.append(f"| {test} | " + " | ".join(("✅" if r["actions"][test]["allowed"] else "⛔")
                                             + ("" if r["actions"][test]["match"] else " ≠") for r in mx["rows"]) + " |")
    L += ["", "Причины отказа, которые получает модель:", ""]
    for x in mx["extra"]:
        L.append(f"- {x['what']} ({x['fixture']} → {x['to']}): {'разрешено' if x['allowed'] else '⛔ ' + '; '.join(x['unmet'])}")
    seen = set()
    for r in mx["rows"]:
        for s, c in r["targets"].items():
            if not c["allowed"] and c["unmet"] and c["unmet"][0] not in seen and len(seen) < 8:
                seen.add(c["unmet"][0])
                L.append(f"- {r['title']} → {s}: ⛔ {'; '.join(c['unmet'])}")
    a = mx["approval"]
    L += ["", f"Утверждение плана: инструмент у модели — {'**есть**' if a['model_has_tool'] else 'нет'}; "
              f"пользователь — {'утверждает' if a['user_applied'] else '❌'}. Распознавание в сообщении: "
          + "; ".join(f"«{t}» → {'да' if v else 'нет'}" for t, v in a["phrases"].items()) + ".", ""]

    life = report.get("lifecycle")
    if life:
        st = life["final_state"]
        L += ["## 2. Жизненный цикл задачи", "",
              f"Задача «{TASK_TITLE}». Финальный этап: `{st['stage']}`, ходов: {len(life['turns'])}, кругов доработки: "
              f"{st['round'] - 1}. Файлы: `{life['dir']}/`.", ""]
        if life["notes"]:
            L += ["Замечания прогона:", ""] + [f"- {n}" for n in life["notes"]] + [""]
        L += ["### Паузы: перезапуск и продолжение без повторных объяснений", "",
              "После паузы создаётся новый объект агента (как новый запуск программы), диалог очищается (`/newchat`), "
              f"и пользователь пишет только «{RESUME_LINE}».", "",
              "| Этап | Пауза | Работа на паузе | Снята | Продолжил с того же шага | Данные целы | Переспрашивал о задаче |",
              "|---|---|---|---|---|---|---|"]
        for p in life["pauses"]:
            L.append(f"| {p['stage']} | {'✅' if p['paused'] else '❌'}{' командой' if p['forced_pause'] else ''} | "
                     f"{p['work_on_pause'] or 'нет'} | {'✅ → ' + p['resumed_to'] if p['resumed'] else '❌'} | "
                     f"{'✅' if p['continued'] else '❌'} {p['want']} | {'✅' if p['kept'] else '❌'} | "
                     f"{'⚠️ да' if p['reexplain'] else 'нет'} |")
        L += ["", "### Ходы", "", "| # | Пользователь | Этап до → после | Действия и переходы | Токены состояния |",
              "|---|---|---|---|---|"]
        for k, t in enumerate(life["turns"], 1):
            ev = "<br>".join(md_escape(entry_line(e)) for e in t["events"]) or "—"
            if t["text_guard"]:
                ev += "<br>🛡 " + "; ".join(md_escape(x["action"]) for x in t["text_guard"])
            L.append(f"| {k} | {md_escape(clip(t['user'], 70))} | {t['before']['stage']} → {t['after']['stage']}"
                     f"<br>{md_escape(t['after']['step'])} | {ev} | {fmt(t.get('fsm_tokens', 0))} |")
        L += ["", "### Финальное состояние", "", "```json",
              json.dumps({k: st[k] for k in ("stage", "plan", "approved_version", "approved_by", "round", "results",
                                             "checks", "final")}, ensure_ascii=False, indent=1), "```", ""]

    runs = report.get("violations")
    if runs:
        L += ["## 3. Прямые просьбы нарушить порядок этапов", "",
              "Каждая просьба — на копии контрольного состояния из цикла выше, в двух режимах: **код проверяет** "
              "(переход и действие проверяются до применения) и **только промпт** (те же правила в системном "
              "промпте, но код ничего не блокирует, а только записывает нарушения). Нарушение — применённое действие "
              "или переход, которые код отклонил бы, или реализация в ответе на planning/паузе.", "",
              "| Просьба | Что запрещено | Режим | Нарушений | Попыток → отклонено | Отсечено кода в тексте | "
              "Объяснил отказ |", "|---|---|---|---|---|---|---|"]
        texts = {v.id: v.text for v in VIOLATIONS}
        for s in violation_summary(runs):
            L.append(f"| {s['id']} «{texts[s['id']]}» | {s['what']} | {'код проверяет' if s['guard'] else 'только промпт'} | "
                     f"**{s['illegal']}/{s['runs']}** | {s['attempts']} → {s['rejected']} | {s['text_guard']} | "
                     f"{s['explained']}/{s['runs']} |")
        L += ["", "### Реакция ассистента", ""]
        for r in runs:
            L += [f"**{r['id']} · {'код проверяет' if r['guard'] else 'только промпт'} · #{r['sample']}** — этап "
                  f"{r['stage_before']} → {r['stage_after']}"
                  + (f"; ⚠️ нарушение: {'; '.join(r['illegal'])}" if r["illegal"] else "; порядок не нарушен"), ""]
            for e in r["events"]:
                L.append(f"- {e}")
            for t in r["text_guard"]:
                L.append(f"- 🛡 {t['action']}")
            L += ["", "> " + md_escape(clip(r["answer"], 700)) if r["answer"] else "> (нет ответа)", ""]

    if life:
        L += ["## Ответы жизненного цикла", ""]
        for k, t in enumerate(life["turns"], 1):
            L += [f"**{k}. {t.get('label', '')}** — «{md_escape(clip(t['user'], 120))}»", "",
                  "> " + md_escape(clip(t["answer"], 900)) if t["answer"] else "> (нет ответа)", ""]
        L += [f"Стоимость ответов: {money(report.get('cost'))} (без роутера памяти); запросов к API: "
              f"{report.get('llm', {}).get('requests', '?')}.", ""]
    md_path.write_text("\n".join(L), encoding="utf-8")
    return md_path


# ---------------------------------------------------------------- чат

HELP = ("/state, /fsm, /approve, /go <этап> [причина], /pause, /resume, /journal [N], /guard on|off, /matrix, "
        "/task new <название> | switch <id> | list, /profile, /memory, /context, /newchat, /tokens, /history, /exit")


def chat(agent: TaskAgent) -> int:
    print(f"Агент на {agent.client.model}, пользователь {agent.root.name}. Команды: {HELP}\n")
    print_state(agent)
    print()
    while True:
        tag = f"{agent.machine.stage}{'' if agent.guard else ' · без проверки'}"
        try:
            text = input(f"[{agent.root.name} · {tag}] Вы: ").strip()
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
            elif command == "/approve":
                agent.user_approve()  # строку с результатом печатает событие fsm
            elif command == "/go":
                if not args:
                    print("  /go <planning|execution|validation|done|paused> [причина]")
                else:
                    agent.user_transition(state_arg(args[0]), " ".join(args[1:]))
                    print(f"  🧭 {state_line(agent.machine.step_info())}")
            elif command == "/pause":
                agent.user_transition(PAUSED, "пауза по команде")
            elif command == "/resume":
                src = agent.machine.data.get("paused_from")
                if agent.machine.stage != PAUSED or not src:
                    print("  задача не на паузе")
                else:
                    agent.user_transition(src, "снять паузу")
            elif command == "/journal":
                print_journal(agent, int(args[0]) if args and args[0].isdigit() else 20)
            elif command == "/guard":
                if args and args[0] in ("on", "off"):
                    agent.guard = args[0] == "on"
                print("  проверка переходов кодом: " + ("ВКЛ" if agent.guard else
                                                        "ВЫКЛ — правила только в промпте, нарушения пишутся в журнал"))
            elif command == "/matrix":
                matrix_demo()
            elif command == "/task":
                sub = args[0] if args else "list"
                if sub == "new" and len(args) > 1:
                    t = agent.new_task(" ".join(args[1:]))
                    print(f"  📋 новая задача {t['id']} «{t['title']}», этап {agent.machine.stage}")
                elif sub == "switch" and len(args) == 2:
                    t = agent.switch_task(args[1])
                    print(f"  📋 задача {t['id']} «{t['title']}»")
                    print_state(agent)
                else:
                    for t in agent.working.list():
                        print(f"  {t['id']:>3} {t['status']:<7} {t['title']}")
            elif command == "/profile":
                print_profile(agent)
            elif command == "/memory":
                print_profile(agent)
                print_memory(agent)
            elif command == "/context":
                schemas = [t.schema() for t in agent.tools.values()]
                _, dropped, parts = agent.build(end=len(agent.history), tools=schemas)
                print(f"  Следующий запрос ≈{fmt(sum(parts['estimate'].values()))} токенов: "
                      f"{layer_line13(parts['estimate'])}; не отправлено старых сообщений: {dropped}")
            elif command == "/newchat":
                print(f"  🧹 {agent.clear('short')}; состояние задачи, профиль и память остаются")
            elif command == "/log":
                for item in agent.log.tail(int(args[0]) if args and args[0].isdigit() else 20):
                    print(f"  {item['at'][11:19]} {op_label(item)}  ⟵ {item.get('source', '')}")
            elif command == "/tokens":
                print_ledger(agent)
            elif command == "/history":
                print_history(agent.history)
            elif command.startswith("/"):
                print(f"  Неизвестная команда. Есть: {HELP}")
            else:
                reply = agent.run(text)
                print(f"\nАгент: {reply.text.strip() or '(пустой ответ)'}\n")
                print_turn13(agent)
            print()
        except LLMError as e:
            print(f"❌ Ошибка: {e}\n", file=sys.stderr)
        except (StoreError, ValueError) as e:
            print(f"⚠️  {e}\n", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n(прервано — ход в историю не попал, состояние задачи откатилось)\n")


# ---------------------------------------------------------------- запуск


def main() -> int:
    parser = argparse.ArgumentParser(description="День 13: состояние задачи как конечный автомат поверх агента Дня 12.")
    parser.add_argument("prompt", nargs="?", help="один ход и выход; без него — чат в консоли")
    parser.add_argument("--user", default="default", help="пользователь: memory/day13/<имя>/")
    parser.add_argument("--new", action="store_true", help="начать заново: папка пользователя уходит в архив")
    parser.add_argument("--preset", choices=list(PRESETS), help="профиль пользователя Дня 12")
    parser.add_argument("--no-guard", action="store_true", help="правила этапов только в промпте, код не проверяет")
    parser.add_argument("--per-turn", type=int, default=1, help="шагов/проверок за один ход (0 — без ограничения)")
    parser.add_argument("--keep", type=int, default=6, help="краткосрочная память: сколько последних сообщений в промпт")
    parser.add_argument("--long-budget", type=int, default=1500, help="долговременная память: бюджет токенов")
    parser.add_argument("--helper-model", help="модель роутера памяти (по умолчанию та же)")
    parser.add_argument("--demo", action="store_true", help="таблица переходов, жизненный цикл с паузами, нарушения")
    parser.add_argument("--only", help="для --demo: части через запятую — matrix, lifecycle, violations")
    parser.add_argument("--samples", type=int, default=2, help="для --demo: прогонов каждой просьбы-нарушения")
    parser.add_argument("--model", help="id модели (по умолчанию LLM_MODEL из .env)")
    parser.add_argument("--max-steps", type=int, default=8, help="предел запросов к модели за один ход")
    args = parser.parse_args()

    if not NAME_RE.fullmatch(args.user):
        print("Ошибка: имя пользователя — буквы, цифры, дефис, подчёркивание.", file=sys.stderr)
        return 1
    if args.keep < 0 or args.long_budget < 0 or args.samples < 1 or args.per_turn < 0:
        print("Ошибка: --keep ≥ 0, --long-budget ≥ 0, --samples ≥ 1, --per-turn ≥ 0.", file=sys.stderr)
        return 1
    parts = {"matrix", "lifecycle", "violations"}
    if args.only:
        chosen = {x.strip() for x in args.only.split(",") if x.strip()}
        if not chosen or chosen - parts:
            print(f"Ошибка: --only — из {', '.join(sorted(parts))}.", file=sys.stderr)
            return 1
        parts = chosen
    if args.demo and parts == {"matrix"}:
        report = matrix_demo()
        return 0 if report["mismatches"] == 0 else 1
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
    if args.demo:
        return demo(client, helper, args.keep, args.long_budget, args.max_steps, args.per_turn, args.samples, parts)

    root = DAY_DIR / args.user
    if args.new and root.exists():
        target = root.with_name(f"{root.name}_{stamp_now()}")
        os.replace(root, target)
        print(f"Прошлое состояние сохранено в {target.relative_to(ROOT)}.")
    try:
        agent = make_agent(client, helper, root, args.keep, args.long_budget, args.max_steps, not args.no_guard,
                           args.per_turn)
    except StoreError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print_restore(agent.restore_info, agent.store, agent)
    for note in agent.notes:
        print(f"⚠️  {note}")
    if args.preset:
        print_ops(agent.load_preset(args.preset))
    if args.prompt:
        try:
            reply = agent.run(args.prompt)
        except (LLMError, StoreError) as e:
            print(f"❌ Ошибка: {e}", file=sys.stderr)
            return 1
        print(f"\nАгент: {reply.text.strip()}\n")
        print_turn13(agent)
        return 0
    return chat(agent)


if __name__ == "__main__":
    from run_timer import run_timed
    sys.exit(run_timed(main))
