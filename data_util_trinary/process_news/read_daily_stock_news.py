import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NewsData:
    symbol: str
    datetime: str
    sentiment_score: float


def main() -> None:
    pkl_path = Path(r"D:\imperial_homework\third_year\i-explore\DML\DML_G8\data_for_process\news_daily_stock\2020\01\2020-01-03.pkl")

    if not pkl_path.exists():
        print(f"File not found: {pkl_path}")
        return

    with pkl_path.open('rb') as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} records")
    for item in data[:10]:
        print(item)


if __name__ == '__main__':
    main()
