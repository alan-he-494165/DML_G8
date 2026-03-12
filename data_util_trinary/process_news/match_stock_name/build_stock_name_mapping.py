import os
import time
import pickle
from dataclasses import dataclass
from pathlib import Path
import tushare as ts


@dataclass
class StockNameInfo:
    symbol: str
    ts_code: str
    name: str
    fullname: str
    former_names: list[str]


def load_symbols(path: Path) -> list[str]:
    symbols = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            symbol = line.strip()
            if symbol:
                symbols.append(symbol)
    return symbols


def main() -> None:
    pro = ts.pro_api('31c93f930abfd98726b28a7d984e5d56163f69c1f040ba97d6d4436a')

    base_dir = Path(__file__).resolve().parent
    symbols_path = base_dir.parent / 'zhongzheng500.txt'
    output_path = base_dir / 'stock_name_mapping.pkl'

    symbols = load_symbols(symbols_path)
    symbol_set = set(symbols)
    print(f'Loaded {len(symbols)} symbols')

    stock_basic = pro.stock_basic(list_status='L', fields='ts_code,symbol,name,fullname')
    stock_basic = stock_basic[stock_basic['symbol'].isin(symbol_set)]

    symbol_to_info: dict[str, StockNameInfo] = {}
    not_found = symbol_set - set(stock_basic['symbol'].tolist())
    if not_found:
        print(f'Missing symbols in stock_basic: {len(not_found)}')

    for _, row in stock_basic.iterrows():
        symbol = row['symbol']
        ts_code = row['ts_code']
        name = row['name'] or symbol
        fullname = row['fullname'] or name

        former_names: list[str] = []
        try:
            df_namechange = pro.namechange(
                ts_code=ts_code,
                start_date='',
                end_date='',
                limit='',
                offset='',
                fields=['ts_code', 'name', 'start_date', 'end_date', 'ann_date', 'change_reason'],
            )

            if df_namechange is not None and not df_namechange.empty:
                current_name = df_namechange.iloc[0]['name']
                for i in range(1, len(df_namechange)):
                    former_name = df_namechange.iloc[i]['name']
                    if former_name and former_name != current_name and former_name not in former_names:
                        former_names.append(former_name)
        except Exception as exc:
            print(f'Namechange error for {ts_code}: {str(exc)[:80]}')

        symbol_to_info[symbol] = StockNameInfo(
            symbol=symbol,
            ts_code=ts_code,
            name=name,
            fullname=fullname,
            former_names=former_names,
        )
        print(f"Fetched name change for {ts_code}")

    with output_path.open('wb') as f:
        pickle.dump(symbol_to_info, f)

    print(f'Saved mapping to {output_path}')


if __name__ == '__main__':
    main()
