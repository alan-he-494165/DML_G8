import os
import pickle
import queue
import threading
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


BATCH_SIZE = 32
QUEUE_MAXSIZE = 16  # max batches buffered in queue


def finbert_sentiment_batch(texts: list[str], tokenizer, model, device) -> list[float]:
    """Run FinBERT inference on a batch of texts. Returns list of (pos - neg) scores."""
    if tokenizer is None or model is None:
        return [0.0] * len(texts)

    inputs = tokenizer(texts, return_tensors='pt', truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    neg = probs[:, 0]
    pos = probs[:, 2]
    return (pos - neg).tolist()


def producer(
    news_files: list[Path],
    raw_news_dir: Path,
    output_base: Path,
    name_to_symbol: dict[str, str],
    sorted_names: list[str],
    batch_queue: queue.Queue,
) -> None:
    """
    CPU thread: loads files, matches symbols, and feeds batches into the queue.
    Each item put on the queue is either:
      - (batch, output_path, rel_results_so_far) tuple for GPU inference
      - None sentinel to signal completion
    """
    for news_file in news_files:
        try:
            with news_file.open('rb') as f:
                items = pickle.load(f)
        except Exception as exc:
            print(f'Failed to load {news_file}: {exc}')
            continue

        rel_path = news_file.relative_to(raw_news_dir)
        output_path = output_base / rel_path

        pending: list[tuple[str, list[str], str]] = []
        for item in items:
            text = getattr(item, 'content', '') or ''
            if not text:
                continue
            matched_symbols = match_symbols(text, name_to_symbol, sorted_names)
            if not matched_symbols:
                continue
            pending.append((text, matched_symbols, getattr(item, 'datetime', '')))

        # Split into batches and enqueue
        batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
        batch_queue.put(('file_start', output_path, news_file.name, len(batches)))
        for batch in batches:
            batch_queue.put(('batch', batch))

    batch_queue.put(('done', None))


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
    print(f'Using device: {device}')

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

    batch_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    # Start producer thread
    t = threading.Thread(
        target=producer,
        args=(news_files, raw_news_dir, output_base, name_to_symbol, sorted_names, batch_queue),
        daemon=True,
    )
    t.start()

    # GPU consumer loop
    current_output_path = None
    current_file_name = None
    current_results: list[NewsData] = []
    remaining_batches = 0

    while True:
        msg = batch_queue.get()

        if msg[0] == 'done':
            break

        if msg[0] == 'file_start':
            _, current_output_path, current_file_name, remaining_batches = msg
            current_results = []
            continue

        # msg[0] == 'batch'
        batch = msg[1]
        texts = [t for t, _, _ in batch]
        try:
            scores = finbert_sentiment_batch(texts, tokenizer, model, device)
        except Exception as exc:
            print(f'Batch sentiment error in {current_file_name}: {str(exc)[:80]}')
            scores = [0.0] * len(batch)

        for score, (_, symbols, dt) in zip(scores, batch):
            for symbol in symbols:
                current_results.append(NewsData(
                    symbol=symbol,
                    datetime=dt,
                    sentiment_score=round(float(score), 4),
                ))

        remaining_batches -= 1
        if remaining_batches == 0:
            current_output_path.parent.mkdir(parents=True, exist_ok=True)
            with current_output_path.open('wb') as f:
                pickle.dump(current_results, f)
            print(f'Processed file: {current_file_name} (saved {len(current_results)} records)')

    t.join()
    print(f'Final save complete in {output_base}')


if __name__ == '__main__':
    main()
