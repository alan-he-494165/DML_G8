from fetcher import Ticker_Day, YFFetcher

test = Ticker_Day.from_yf('AAPL', '2021-01-01', '2025-12-31')
print(test.moving_average())