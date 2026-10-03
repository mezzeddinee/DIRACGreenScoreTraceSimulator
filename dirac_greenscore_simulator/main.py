from pathlib import Path

try:
    from .app import run
except ImportError:  # direct script-style execution fallback
    from app import run


def main() -> None:
    run(base=Path(__file__).resolve().parent, tick_minutes=1)


if __name__ == "__main__":
    main()
