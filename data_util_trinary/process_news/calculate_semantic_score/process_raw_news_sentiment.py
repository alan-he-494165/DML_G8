import os
import pickle
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ.setdefault('TRANSFORMERS_ALLOW_TORCH_LOAD_UNSAFE', '1')


@dataclass
class NewsItem:
    datetime: str
    content: str
    channels: str


@dataclass
class StockNameInfo:
    symbol: str
    ts_code: str
    name: str
    fullname: str
    former_names: list[str]


@dataclass
class NewsData:
    symbol: str
    datetime: str
    sentiment_score: float


def load_mapping(mapping_path: Path) -> dict[str, list[str]]:
    with mapping_path.open('rb') as f:
        symbol_to_info = pickle.load(f)

    symbol_to_names: dict[str, list[str]] = {}
    for symbol, info in symbol_to_info.items():
        names = []
        for name in [info.name, info.fullname]:
            if name and name not in names:
                names.append(name)
        for former in info.former_names or []:
            if former and former not in names:
                names.append(former)
        symbol_to_names[symbol] = names

    return symbol_to_names


def build_name_lookup(symbol_to_names: dict[str, list[str]]):
    name_to_symbol: dict[str, str] = {}
    for symbol, names in symbol_to_names.items():
        for name in names:
            if name and name not in name_to_symbol:
                name_to_symbol[name] = symbol

    sorted_names = sorted(name_to_symbol.keys(), key=len, reverse=True)
    return name_to_symbol, sorted_names


def match_symbols(text: str, name_to_symbol: dict[str, str], sorted_names: list[str]) -> list[str]:
    matched = []
    seen = set()
    for name in sorted_names:
        if name in text:
            symbol = name_to_symbol[name]
            if symbol not in seen:
                matched.append(symbol)
                seen.add(symbol)
    return matched


def finbert_sentiment(text: str, tokenizer, model, device) -> float:
    if tokenizer is None or model is None:
        return 0.0

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()

    neg, _, pos = probs.tolist()
    return pos - neg


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    mapping_path = base_dir / 'process_news' / 'match_stock_name' / 'stock_name_mapping.pkl'
    raw_news_dir = base_dir.parent / 'data_for_process' / 'raw_news' / '2025'
    output_base = base_dir.parent / 'data_for_process' / 'news_daily_stock' / '2025'

    symbol_to_names = load_mapping(mapping_path)
    name_to_symbol, sorted_names = build_name_lookup(symbol_to_names)

    print('Loading FinBERT model...')
    model_name = 'ProsusAI/finbert'
    tokenizer = None
    model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=True,
        )
        model.to(device)
        model.eval()
        print(f'Model loaded on {device}')
    except Exception as exc:
        print(f'FinBERT load failed, fallback to neutral scores: {str(exc)[:120]}')

    news_files = sorted(raw_news_dir.rglob('*.pkl'))
    print(f'Found {len(news_files)} news files')

    for news_file in news_files:
        results: list[NewsData] = []
        try:
            with news_file.open('rb') as f:
                items = pickle.load(f)
        except Exception as exc:
            print(f'Failed to load {news_file}: {exc}')
            continue

        for item in items:
            text = getattr(item, 'content', '') or ''
            if not text:
                continue

            matched_symbols = match_symbols(text, name_to_symbol, sorted_names)
            if not matched_symbols:
                continue

            for symbol in matched_symbols:
                try:
                    score = finbert_sentiment(text, tokenizer, model, device)
                except Exception as exc:
                    print(f'Sentiment error in {news_file.name}: {str(exc)[:80]}')
                    continue

                results.append(NewsData(
                    symbol=symbol,
                    datetime=getattr(item, 'datetime', ''),
                    sentiment_score=round(float(score), 4),
                ))

        rel_path = news_file.relative_to(raw_news_dir)
        output_path = output_base / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('wb') as f:
            pickle.dump(results, f)

        print(f'Processed file: {news_file} (saved {len(results)} records)')

    print(f'Final save complete in {output_base}')


if __name__ == '__main__':
    main()
