"""Запуск main() с замером времени: по окончании печатает, сколько шёл скрипт.

    if __name__ == "__main__":
        from run_timer import run_timed
        sys.exit(run_timed(main))

Время печатается всегда: при обычном выходе, при sys.exit(), при ошибке и при Ctrl+C.
"""
import sys
import time
from datetime import datetime


def format_duration(seconds: float) -> str:
    """41.3 с | 23 мин 14 с | 1 ч 05 мин 14 с"""
    if seconds < 60:
        return f"{seconds:.1f} с"
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} ч {minutes:02d} мин {secs:02d} с"
    return f"{minutes} мин {secs:02d} с"


def run_timed(main):
    """Вызывает main(), возвращает её код выхода и в конце печатает время выполнения."""
    started_at = datetime.now()
    start = time.perf_counter()
    try:
        return main()
    finally:
        elapsed = time.perf_counter() - start
        finished_at = datetime.now()
        sys.stdout.flush()
        print(f"\nВремя выполнения: {format_duration(elapsed)} "
              f"(старт {started_at:%H:%M:%S}, финиш {finished_at:%H:%M:%S})", flush=True)
