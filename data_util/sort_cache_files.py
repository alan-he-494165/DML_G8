from __future__ import annotations

import shutil
from pathlib import Path

SOURCE_DIR = Path(r"D:\imperial_homework\third_year\i-explore\DML\DML_G8\data_for_process\cache_output")
CHINA_DIR = SOURCE_DIR / "china_stock"
AMERICA_DIR = SOURCE_DIR / "america_stock"


def is_numeric_stem(path: Path) -> bool:
    return path.stem.isdigit()


def is_alpha_stem(path: Path) -> bool:
    stem = path.stem
    return stem.isalpha() and stem.isascii()


def main() -> None:
    CHINA_DIR.mkdir(parents=True, exist_ok=True)
    AMERICA_DIR.mkdir(parents=True, exist_ok=True)

    moved_china = 0
    moved_america = 0
    skipped = 0

    for item in SOURCE_DIR.iterdir():
        if item.is_dir():
            continue

        if is_numeric_stem(item):
            target = CHINA_DIR / item.name
            shutil.move(str(item), str(target))
            moved_china += 1
        elif is_alpha_stem(item):
            target = AMERICA_DIR / item.name
            shutil.move(str(item), str(target))
            moved_america += 1
        else:
            skipped += 1

    print(f"Moved to china_stock: {moved_china}")
    print(f"Moved to america_stock: {moved_america}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
