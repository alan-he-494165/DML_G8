from fetcher_yf import Ticker_Day, YFFetcher

ticker_tech = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'ADBE', 'ORCL', 'IBM', 'CRM', 'SAP', 'TXN', 'QCOM', 'AVGO', 'AMD', 'NOW', 'ZM', 'SNOW', 'UBER', 'LYFT', 'TWTR', 'SQ', 'PYPL', 'SHOP', 'SPOT', 'DOCU', 'DDOG', 'FSLY']
ticker_finance = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW', 'PNC', 'USB', 'TFC', 'COF', 'BK', 'STT', 'AIG', 'MMC', 'ALL', 'CME', 'ICE', 'SPGI', 'V', 'MA', 'PYPL', 'ADP', 'FIS', 'FISV', 'INTU', 'PAYC', 'VRSN', 'DFS']
ticker_health = ['JNJ', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'BMY', 'LLY', 'GILD', 'AMGN', 'MDT', 'DHR', 'ZTS', 'SNY', 'REGN', 'BIIB', 'VRTX', 'ISRG', 'EW', 'HUM', 'CI', 'CNC', 'ANTM', 'UNH', 'CVS', 'WBA', 'LH', 'BAX', 'BDX', 'ALGN', 'IDXX']
ticker_energy = ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'PSX', 'VLO', 'MPC', 'HES', 'KMI', 'WMB', 'FTI', 'PXD', 'DVN', 'CVE', 'ENB', 'SU', 'CNQ', 'TRP', 'IMO', 'BTE', 'AR', 'FET', 'NBL', 'OKE', 'RRC', 'APA', 'CLR', 'MRO', 'CXO']

tech_data = []
finance_data = []
health_data = []
energy_data = []

for ticker in ticker_tech:
    data = Ticker_Day.from_yf(ticker, '2020-01-01', '2023-12-31')
    tech_data.append((ticker, data))

for ticker in ticker_finance:
    data = Ticker_Day.from_yf(ticker, '2020-01-01', '2023-12-31')
    finance_data.append((ticker, data))

for ticker in ticker_health:
    data = Ticker_Day.from_yf(ticker, '2020-01-01', '2023-12-31')
    health_data.append((ticker, data))

for ticker in ticker_energy:
    data = Ticker_Day.from_yf(ticker, '2020-01-01', '2023-12-31')
    energy_data.append((ticker, data))
