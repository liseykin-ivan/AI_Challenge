"""Публикует результаты дня: коммит в main + ветка week_<N>/day_<M> на том же коммите, push обеих.

    python tasks\\publish_day.py                       # последний день из tasks\\week_*_day_*.md
    python tasks\\publish_day.py --week 1 --day 4
    python tasks\\publish_day.py -m "День 4: <что сделано>"
    python tasks\\publish_day.py --message-file C:\\temp\\msg.txt --yes
    python tasks\\publish_day.py --dry-run             # только проверки, ничего не меняет
    python tasks\\publish_day.py --no-push             # коммит и ветка локально, без push
    python tasks\\publish_day.py --push-only           # отправить уже сделанные коммит и ветку

Что проверяется до любых изменений:
  - текущая ветка main, нет незавершённого merge/rebase и .lock-файлов в .git;
  - main не разошёлся с origin/main. Если локальный main просто отстал (например, прошлый день
    отправили с другого места), а изменённые файлы совпадают с версией на GitHub, — main без потерь
    переносится на origin/main;
  - ветки week_N/day_M ещё нет ни локально, ни на GitHub;
  - в коммит не попадают .env, ключ API из .env и файлы больше 20 МБ.
Push атомарный: main и ветка дня уходят вместе или не уходят вовсе. В конце сверяется,
что на GitHub обе ветки указывают на новый коммит.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"
MAIN = "main"
REMOTE = "origin"
MAX_FILE_MB = 20
FORBIDDEN = [re.compile(p) for p in (r"(^|/)\.env$", r"(^|/)\.venv/", r"\.bundle$", r"(^|/)__pycache__/")]


class Stop(Exception):
    """Проверка не пройдена: ничего не делаем, объясняем почему."""


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and result.returncode != 0:
        raise Stop(f"git {' '.join(args)}:\n{(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True).returncode == 0


def say(text: str = "") -> None:
    print(text, flush=True)

# ---------------------------------------------------------------- день и сообщение

def latest_day() -> tuple[int, int]:
    days = []
    for f in TASKS_DIR.glob("week_*_day_*.md"):
        m = re.fullmatch(r"week_(\d+)_day_(\d+)\.md", f.name)
        if m:
            days.append((int(m.group(1)), int(m.group(2))))
    if not days:
        raise Stop("в tasks/ нет файлов week_*_day_*.md — укажите --week и --day.")
    return max(days)


def task_title(week: int, day: int) -> str:
    """Название из первой строки tasks/week_N_day_M.md: «# Неделя 1 · День 3 — Разные способы рассуждения»."""
    f = TASKS_DIR / f"week_{week}_day_{day}.md"
    if not f.exists():
        return ""
    first = f.read_text(encoding="utf-8-sig").splitlines()[0] if f.stat().st_size else ""
    return first.split("—", 1)[1].strip() if "—" in first else ""


def build_message(args, week: int, day: int) -> str:
    if args.message_file:
        text = Path(args.message_file).read_text(encoding="utf-8-sig").strip()
        if not text:
            raise Stop(f"файл сообщения пуст: {args.message_file}")
        return text + "\n"
    if args.message:
        return args.message.strip() + "\n"
    title = task_title(week, day)
    if not title:
        raise Stop("не удалось взять название из файла задания — передайте -m или --message-file.")
    return f"День {day}: {title[0].lower() + title[1:]}\n"

# ---------------------------------------------------------------- проверки

def check_repo_state() -> None:
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != MAIN:
        raise Stop(f"сейчас ветка «{branch}», а публиковать нужно из {MAIN}: git switch {MAIN}")
    git_dir = Path(git("rev-parse", "--absolute-git-dir"))
    for marker in ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD"):
        if (git_dir / marker).exists():
            raise Stop(f"не завершена операция git ({marker}) — закончите или отмените её.")
    locks = [p.name for p in git_dir.glob("*.lock")]
    if locks:
        raise Stop(f"в .git остались lock-файлы {locks}: другой git ещё работает или упал. "
                   "Если git не запущен — удалите их вручную.")


def sync_with_remote(dry_run: bool) -> str:
    """Локальный main должен содержать origin/main. Если просто отстал — переносим без потерь или останавливаемся.
    Возвращает коммит, от которого считать будущий коммит (для показа в --dry-run)."""
    say(f"Получаю {REMOTE}…")
    git("fetch", REMOTE)
    remote = f"{REMOTE}/{MAIN}"
    if not git_ok("rev-parse", "--verify", "-q", remote):
        return "HEAD"  # пустой репозиторий на GitHub
    if git_ok("merge-base", "--is-ancestor", remote, "HEAD"):
        return "HEAD"  # origin/main уже в истории — всё хорошо
    if not git_ok("merge-base", "--is-ancestor", "HEAD", remote):
        raise Stop(f"локальный {MAIN} и {remote} разошлись (у каждого свои коммиты). "
                   "Это нужно решить вручную — скрипт ничего не трогал.")
    # Локальный main просто отстал. Безопасно, только если всё, что изменилось на GitHub,
    # в рабочей папке либо не тронуто, либо уже лежит ровно в той же версии.
    changed = git("diff", "--name-only", "HEAD", remote).splitlines()
    conflicts = []
    for path in changed:
        local = ROOT / path
        try:
            remote_bytes = subprocess.run(["git", "show", f"{remote}:{path}"], cwd=ROOT,
                                          capture_output=True, check=True).stdout
        except subprocess.CalledProcessError:
            remote_bytes = None  # файл на GitHub удалён
        head_has = git_ok("cat-file", "-e", f"HEAD:{path}")
        if remote_bytes is None:
            if local.exists() and head_has and git_ok("diff", "--quiet", "HEAD", "--", path):
                continue  # удалён на GitHub, локально не трогали — reset просто покажет его удалённым
            if local.exists():
                conflicts.append(path)
            continue
        if not local.exists():
            if head_has:
                conflicts.append(path)  # удалён локально, а на GitHub изменился
            continue
        if local.read_bytes() == remote_bytes:
            continue
        if head_has and git_ok("diff", "--quiet", "HEAD", "--", path):
            continue  # локально не трогали — просто получит версию с GitHub
        conflicts.append(path)
    behind = git("rev-list", "--count", f"HEAD..{remote}")
    if conflicts:
        raise Stop(f"{MAIN} отстаёт от {remote} на {behind} коммит(а), и эти файлы изменены и там, и здесь:\n  "
                   + "\n  ".join(conflicts) + "\nСведите их вручную — скрипт ничего не трогал.")
    say(f"Локальный {MAIN} отставал от {remote} на {behind} коммит(а) — переношу без потерь.")
    if dry_run:
        return remote
    untouched = [p for p in changed if git_ok("cat-file", "-e", f"HEAD:{p}")
                 and git_ok("diff", "--quiet", "HEAD", "--", p)]
    git("reset", "-q", remote)  # HEAD и индекс = GitHub, рабочие файлы не трогаем
    # Файлы, которые локально не меняли, приводим к версии GitHub — иначе `git add -A` откатил бы чужие правки.
    for p in untouched:
        if git_ok("cat-file", "-e", f"{remote}:{p}"):
            git("checkout", remote, "--", p)
        else:
            (ROOT / p).unlink(missing_ok=True)
    return "HEAD"


def check_branch_free(branch: str) -> None:
    if git_ok("rev-parse", "--verify", "-q", f"refs/heads/{branch}"):
        raise Stop(f"локальная ветка {branch} уже есть. Если день уже опубликован — ничего делать не нужно; "
                   "если надо отправить его заново — это решение вручную.")
    if git(*["ls-remote", "--heads", REMOTE, branch]):
        raise Stop(f"ветка {branch} уже есть на GitHub.")


def api_keys() -> list[str]:
    env = ROOT / ".env"
    if not env.exists():
        return []
    keys = []
    for line in env.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        m = re.match(r"\s*[A-Z0-9_]*(KEY|TOKEN|SECRET)[A-Z0-9_]*\s*=\s*(.+)", line)
        if m:
            value = m.group(2).strip().strip("'\"")
            if len(value) >= 12 and value != "your-key-here":
                keys.append(value)
    return keys


def check_staged() -> list[str]:
    files = git("diff", "--cached", "--name-only").splitlines()
    if not files:
        raise Stop("нечего коммитить: изменений нет.")
    bad = [f for f in files if any(p.search(f) for p in FORBIDDEN)]
    if bad:
        raise Stop(f"в коммит попали файлы, которым там не место: {bad}. Добавьте их в .gitignore.")
    big = [f for f in files if (ROOT / f).exists() and (ROOT / f).stat().st_size > MAX_FILE_MB * 2**20]
    if big:
        raise Stop(f"файлы больше {MAX_FILE_MB} МБ: {big}.")
    diff = git("diff", "--cached", "--no-color", "-U0")
    leaked = [k[:4] + "…" for k in api_keys() if k in diff]
    if leaked:
        raise Stop(f"в изменениях найден ключ из .env ({', '.join(leaked)}). Уберите его из файлов.")
    return files


def ensure_identity() -> list[str]:
    """Если имя/почта git не настроены, берём их из последнего коммита (как у прошлых дней)."""
    if git("config", "user.name", check=False) and git("config", "user.email", check=False):
        return []
    name = git("log", "-1", "--format=%an", check=False)
    email = git("log", "-1", "--format=%ae", check=False)
    if not (name and email):
        raise Stop("не настроены git user.name и user.email.")
    return ["-c", f"user.name={name}", "-c", f"user.email={email}"]

# ---------------------------------------------------------------- push

def push(branch: str) -> None:
    head = git("rev-parse", "HEAD")
    if git("rev-parse", branch) != head:
        raise Stop(f"ветка {branch} указывает не на текущий коммит {MAIN} — проверьте вручную.")
    say(f"Отправляю {MAIN} и {branch} одной операцией…")
    git("push", "--atomic", "-u", REMOTE, MAIN, branch)
    git("branch", "-q", "-u", f"{REMOTE}/{MAIN}", MAIN)
    listing = git("ls-remote", REMOTE, f"refs/heads/{MAIN}", f"refs/heads/{branch}").splitlines()
    remote_refs = {ref: sha for sha, ref in (line.split("\t") for line in listing)}
    for ref in (MAIN, branch):
        if remote_refs.get(f"refs/heads/{ref}") != head:
            raise Stop(f"после push ветка {ref} на GitHub указывает не на {head[:7]} — проверьте вручную.")
    say(f"Готово: на GitHub {MAIN} и {branch} → {head[:7]}.")


def confirm(question: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes", "д", "да"):
        raise Stop("отменено.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Коммит дня в main + ветка week_N/day_M + push.")
    p.add_argument("--week", type=int)
    p.add_argument("--day", type=int)
    p.add_argument("-m", "--message", help="сообщение коммита (по умолчанию «День M: <название задания>»)")
    p.add_argument("--message-file", help="файл с полным сообщением коммита (лучше вне папки проекта)")
    p.add_argument("--dry-run", action="store_true", help="только проверки и список файлов")
    p.add_argument("--no-push", action="store_true", help="коммит и ветка без отправки на GitHub")
    p.add_argument("--push-only", action="store_true", help="отправить уже сделанные коммит и ветку дня")
    p.add_argument("--yes", action="store_true", help="не спрашивать подтверждение")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if (args.week is None) != (args.day is None):
            raise Stop("--week и --day указываются вместе.")
        week, day = (args.week, args.day) if args.week else latest_day()
        branch = f"week_{week}/day_{day}"
        say(f"День: неделя {week}, день {day} → ветка {branch}")
        check_repo_state()

        if args.push_only:
            if not git_ok("rev-parse", "--verify", "-q", f"refs/heads/{branch}"):
                raise Stop(f"локальной ветки {branch} нет — сначала запустите без --push-only.")
            git("fetch", REMOTE)
            if not git_ok("merge-base", "--is-ancestor", f"{REMOTE}/{MAIN}", "HEAD"):
                raise Stop(f"{REMOTE}/{MAIN} ушёл вперёд после коммита — сведите вручную.")
            push(branch)
            return 0

        message = build_message(args, week, day)
        base = sync_with_remote(args.dry_run)
        check_branch_free(branch)

        git("add", "-A")  # уважает .gitignore
        try:  # до коммита любая остановка возвращает индекс как было
            check_staged()
            say("\nВ коммит войдут:")
            for line in git("diff", "--cached", "--stat", base).splitlines():
                say("  " + line)
            say("\nСообщение коммита:\n  " + message.strip().replace("\n", "\n  "))
            if args.dry_run:
                git("reset", "-q")
                say("\n--dry-run: проверки пройдены, ничего не изменено.")
                return 0
            confirm(f"\nСоздать коммит, ветку {branch}{'' if args.no_push else ' и отправить на GitHub'}?", args.yes)
        except Stop:
            git("reset", "-q")
            raise
        identity = ensure_identity()
        subprocess.run(["git", *identity, "commit", "-q", "-F", "-"], cwd=ROOT, input=message,
                       text=True, encoding="utf-8", check=True)
        git("branch", branch)
        say(f"Коммит {git('rev-parse', '--short', 'HEAD')} и ветка {branch} созданы.")
        if args.no_push:
            say("--no-push: на GitHub не отправлено. Отправить: python tasks\\publish_day.py --push-only")
            return 0
        try:
            push(branch)
        except Stop as e:
            say(f"\nКоммит и ветка остались локально, но push не прошёл:\n{e}\n"
                "Повторить отправку: python tasks\\publish_day.py --push-only")
            return 1
        return 0
    except Stop as e:
        say(f"\nОстановлено: {e}")
        return 1
    except subprocess.CalledProcessError as e:
        say(f"\nОшибка git (код {e.returncode}). Состояние: git status")
        return 1


if __name__ == "__main__":
    if os.name != "nt":
        # Вне Windows (например, в сессии Claude) нет менеджера паролей: пусть push сразу падает, а не ждёт ввода.
        os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    sys.exit(main())
