import pickle
import sys
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# Add project root to path for pickle to find the class
sys.path.insert(0, str(Path(__file__).parent.parent))
from fetcher.fetcher_yf import Ticker_Day

cache_dir = Path(__file__).parent.parent / 'cache'


def load_all_stocks(limit=50):
    """Load stock data from cache directory."""
    stocks = []
    for pkl_file in list(cache_dir.glob('*.pkl'))[:limit]:
        with open(pkl_file, 'rb') as f:
            stocks.append(pickle.load(f))
    return stocks


def create_features(stock):
    """Create normalized features from stock data.

    All features are lagged (from previous day) to avoid data leakage.
    Target is today's close-to-close return.
    """
    open_prices = np.array(stock.open)
    high = np.array(stock.high)
    low = np.array(stock.low)
    close = np.array(stock.close)
    volume = np.array(stock.volume)

    # Previous day's data (lagged by 1 day to avoid leakage)
    prev_open = np.roll(open_prices, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_volume = np.roll(volume, 1)

    # Fill first values
    prev_open[0] = open_prices[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    prev_volume[0] = volume[0]

    # Normalize previous day's prices relative to previous day's open
    prev_high_ratio = prev_high / prev_open
    prev_low_ratio = prev_low / prev_open
    prev_close_ratio = prev_close / prev_open

    X = np.column_stack([prev_high_ratio, prev_low_ratio, prev_close_ratio, prev_volume])
    # Target is log return: log(close_today / close_yesterday)
    y = np.log(close / prev_close)
    return X, y, prev_close


def pearson_similarity(stock1, stock2):
    """Calculate Pearson correlation between two stocks' close prices."""
    close1 = np.array(stock1.close)
    close2 = np.array(stock2.close)
    # Align lengths by using the shorter one
    min_len = min(len(close1), len(close2))
    corr, _ = pearsonr(close1[-min_len:], close2[-min_len:])
    return corr


def compute_similarity_matrix(stocks):
    """Compute pairwise Pearson similarity matrix for all stocks."""
    n = len(stocks)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                corr = pearson_similarity(stocks[i], stocks[j])
                sim_matrix[i, j] = corr
                sim_matrix[j, i] = corr
    return sim_matrix


def create_augmented_features(stocks, sim_matrix):
    """Create features augmented with similarity-weighted features from other stocks.

    For each stock i at time t, append weighted average of other stocks' features,
    where weights are based on similarity matrix.
    """
    n_stocks = len(stocks)

    # First, align all stocks to common time index
    # Find common date range
    all_dates = set(stocks[0].date)
    for stock in stocks[1:]:
        all_dates &= set(stock.date)
    common_dates = sorted(all_dates)

    if len(common_dates) == 0:
        raise ValueError("No common dates found across stocks")

    # Build feature matrices for each stock on common dates
    # All features are lagged (from previous day) to avoid data leakage
    stock_features = []  # List of (X, y, prev_close) per stock
    for stock in stocks:
        date_to_idx = {d: i for i, d in enumerate(stock.date)}
        indices = [date_to_idx[d] for d in common_dates]

        open_prices = np.array([stock.open[i] for i in indices])
        high = np.array([stock.high[i] for i in indices])
        low = np.array([stock.low[i] for i in indices])
        close = np.array([stock.close[i] for i in indices])
        volume = np.array([stock.volume[i] for i in indices])

        # Previous day's data (lagged by 1 day to avoid leakage)
        prev_open = np.roll(open_prices, 1)
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_volume = np.roll(volume, 1)

        # Fill first values
        prev_open[0] = open_prices[0]
        prev_high[0] = high[0]
        prev_low[0] = low[0]
        prev_close[0] = close[0]
        prev_volume[0] = volume[0]

        # Normalize previous day's prices relative to previous day's open
        prev_high_ratio = prev_high / prev_open
        prev_low_ratio = prev_low / prev_open
        prev_close_ratio = prev_close / prev_open

        X = np.column_stack([prev_high_ratio, prev_low_ratio, prev_close_ratio, prev_volume])
        # Target is log return: log(close_today / close_yesterday)
        y = np.log(close / prev_close)
        stock_features.append((X, y, prev_close))

    # Now augment each stock's features with all other stocks' features scaled by similarity
    X_all, y_all, stock_indices = [], [], []
    n_base_features = stock_features[0][0].shape[1]

    for i in range(n_stocks):
        X_i, y_i, _ = stock_features[i]
        n_samples = X_i.shape[0]

        # Collect all other stocks' features, each scaled by similarity to stock i
        all_neighbor_features = []
        for j in range(n_stocks):
            if j != i:
                X_j, _, _ = stock_features[j]
                sim_weight = max(0, sim_matrix[i, j])  # Clip negative similarities
                all_neighbor_features.append(X_j * sim_weight)

        # Concatenate: [own features, stock0_features*sim, stock1_features*sim, ...]
        X_augmented = np.column_stack([X_i] + all_neighbor_features)
        X_all.append(X_augmented)
        y_all.append(y_i)
        stock_indices.extend([i] * n_samples)

    return np.vstack(X_all), np.concatenate(y_all), np.array(stock_indices), common_dates


def compute_rank_ic(y_true, y_pred):
    """Compute Rank IC (Spearman correlation) between predictions and actuals."""
    corr, _ = spearmanr(y_true, y_pred)
    return corr


def train_general_model(stocks, max_depth=10, sim_matrix=None):
    """Train a general model on all stocks with similarity-augmented features.

    Parameters
    ----------
    stocks : list
        List of stock data objects.
    max_depth : int
        Maximum depth for XGBoost.
    sim_matrix : np.ndarray, optional
        Similarity matrix for feature augmentation.
    """
    if sim_matrix is not None:
        X_all, y_all, stock_indices, common_dates = create_augmented_features(stocks, sim_matrix)
        print(f"Using augmented features: {X_all.shape[1]} features (4 base + 4*{len(stocks)-1} neighbor features)")
    else:
        # Fallback to basic features
        X_all, y_all = [], []
        for i, stock in enumerate(stocks):
            X, y, _ = create_features(stock)
            X_all.append(X)
            y_all.append(y)
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)
        print(f"Using basic features: {X_all.shape[1]} features")

    print(f"Total samples: {X_all.shape[0]}")

    # Split
    train_size = int(len(y_all) * 0.7)
    X_train, X_test = X_all[:train_size], X_all[train_size:]
    y_train, y_test = y_all[:train_size], y_all[train_size:]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=max_depth, learning_rate=0.1)
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    train_ic = compute_rank_ic(y_train, model.predict(X_train))
    test_ic = compute_rank_ic(y_test, model.predict(X_test))
    print(f"General Model (max_depth={max_depth}) - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Train IC: {train_ic:.4f}, Test IC: {test_ic:.4f}")
    return model, train_r2, test_r2, train_ic, test_ic


def screen_max_depth(stocks, sim_matrix, depths=range(2, 10, 1)):
    """Screen max_depth parameter for best test Rank IC."""
    if depths is None:
        depths = [2, 3, 4, 5, 6, 8, 10, 15, 20]

    results = []
    for depth in depths:
        model, train_r2, test_r2, train_ic, test_ic = train_general_model(
            stocks, max_depth=depth, sim_matrix=sim_matrix
        )
        results.append({'max_depth': depth, 'train_r2': train_r2, 'test_r2': test_r2,
                        'train_ic': train_ic, 'test_ic': test_ic, 'model': model})
        print(f"max_depth={depth}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, Train IC={train_ic:.4f}, Test IC={test_ic:.4f}")

    # Find best by test Rank IC
    best = max(results, key=lambda x: x['test_ic'])
    print(f"\nBest max_depth: {best['max_depth']} (Test IC: {best['test_ic']:.4f})")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot R²
    axes[0].plot([r['max_depth'] for r in results], [r['train_r2'] for r in results], 'o-', label='Train R²')
    axes[0].plot([r['max_depth'] for r in results], [r['test_r2'] for r in results], 'o-', label='Test R²')
    axes[0].set_xlabel('max_depth')
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² vs max_depth')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Rank IC
    axes[1].plot([r['max_depth'] for r in results], [r['train_ic'] for r in results], 'o-', label='Train IC')
    axes[1].plot([r['max_depth'] for r in results], [r['test_ic'] for r in results], 'o-', label='Test IC')
    axes[1].set_xlabel('max_depth')
    axes[1].set_ylabel('Rank IC')
    axes[1].set_title('Rank IC vs max_depth')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    return results, best['model'], best['max_depth']


def create_augmented_features_single(stock, stocks, sim_matrix, stock_idx):
    """Create augmented features for a single target stock."""
    n_stocks = len(stocks)

    # Find common dates
    all_dates = set(stocks[0].date)
    for s in stocks[1:]:
        all_dates &= set(s.date)
    common_dates = sorted(all_dates)

    # Build feature matrices for all stocks on common dates
    # All features are lagged (from previous day) to avoid data leakage
    stock_features = []
    for s in stocks:
        date_to_idx = {d: i for i, d in enumerate(s.date)}
        indices = [date_to_idx[d] for d in common_dates]

        open_prices = np.array([s.open[i] for i in indices])
        high = np.array([s.high[i] for i in indices])
        low = np.array([s.low[i] for i in indices])
        close = np.array([s.close[i] for i in indices])
        volume = np.array([s.volume[i] for i in indices])

        # Previous day's data (lagged by 1 day to avoid leakage)
        prev_open = np.roll(open_prices, 1)
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_volume = np.roll(volume, 1)

        # Fill first values
        prev_open[0] = open_prices[0]
        prev_high[0] = high[0]
        prev_low[0] = low[0]
        prev_close[0] = close[0]
        prev_volume[0] = volume[0]

        # Normalize previous day's prices relative to previous day's open
        prev_high_ratio = prev_high / prev_open
        prev_low_ratio = prev_low / prev_open
        prev_close_ratio = prev_close / prev_open

        X = np.column_stack([prev_high_ratio, prev_low_ratio, prev_close_ratio, prev_volume])
        # Target is log return: log(close_today / close_yesterday)
        y = np.log(close / prev_close)
        stock_features.append((X, y, prev_close))

    # Get target stock features
    X_i, y_i, prev_close = stock_features[stock_idx]

    # Collect all other stocks' features, each scaled by similarity
    all_neighbor_features = []
    for j in range(n_stocks):
        if j != stock_idx:
            X_j, _, _ = stock_features[j]
            sim_weight = max(0, sim_matrix[stock_idx, j])
            all_neighbor_features.append(X_j * sim_weight)

    X_augmented = np.column_stack([X_i] + all_neighbor_features)
    return X_augmented, y_i, prev_close, common_dates


def finetune_for_stock(base_model, stock, stocks, sim_matrix, stock_idx, n_estimators=50, max_depth=3):
    """Finetune the general model for a specific stock using augmented features.

    Original approach: continues training all features (neighbor weights can change).
    """
    X, y, open_prices, common_dates = create_augmented_features_single(
        stock, stocks, sim_matrix, stock_idx
    )
    print(f"Finetuning with augmented features: {X.shape[1]} features, {X.shape[0]} samples")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # Continue training from base model
    finetuned = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05)
    finetuned.fit(X_train, y_train, xgb_model=base_model.get_booster())

    train_r2 = finetuned.score(X_train, y_train)
    test_r2 = finetuned.score(X_test, y_test)
    train_ic = compute_rank_ic(y_train, finetuned.predict(X_train))
    test_ic = compute_rank_ic(y_test, finetuned.predict(X_test))
    print(f"Finetuned ({stock.symbol}) - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Train IC: {train_ic:.4f}, Test IC: {test_ic:.4f}")
    return finetuned, train_r2, test_r2, train_ic, test_ic


def finetune_option1(base_model, stock, stocks, sim_matrix, stock_idx, n_estimators=50, max_depth=3):
    """Option 1: Two-stage prediction.

    Use general model's prediction as a fixed "neighbor_score" feature,
    then train a small model on [own_features, fixed_neighbor_score].
    """
    X_full, y, open_prices, common_dates = create_augmented_features_single(
        stock, stocks, sim_matrix, stock_idx
    )

    # Get fixed predictions from general model (this captures neighbor contribution)
    neighbor_score = base_model.predict(X_full).reshape(-1, 1)

    # Extract only own features (first 4 columns)
    X_own = X_full[:, :4]

    # New features: [own_features, fixed_neighbor_score]
    X_new = np.column_stack([X_own, neighbor_score])

    print(f"Option 1: {X_new.shape[1]} features (4 own + 1 fixed neighbor score)")
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, shuffle=False)

    # Train new model on this reduced feature set
    finetuned = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05)
    finetuned.fit(X_train, y_train)

    train_r2 = finetuned.score(X_train, y_train)
    test_r2 = finetuned.score(X_test, y_test)
    train_ic = compute_rank_ic(y_train, finetuned.predict(X_train))
    test_ic = compute_rank_ic(y_test, finetuned.predict(X_test))
    print(f"Option 1 ({stock.symbol}) - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Train IC: {train_ic:.4f}, Test IC: {test_ic:.4f}")
    return finetuned, train_r2, test_r2, train_ic, test_ic, neighbor_score


def finetune_option2(base_model, stock, stocks, sim_matrix, stock_idx, n_estimators=50, max_depth=3):
    """Option 2: Pre-compute aggregated neighbor feature.

    Compute a single weighted-average neighbor feature (fixed from similarity matrix),
    then finetune only on [own_features, aggregated_neighbor_features].
    """
    X_full, y, open_prices, common_dates = create_augmented_features_single(
        stock, stocks, sim_matrix, stock_idx
    )

    # Extract own features (first 4 columns)
    X_own = X_full[:, :4]

    # Extract neighbor features (remaining columns) and aggregate them
    # Each neighbor has 4 features, already scaled by similarity
    n_stocks = len(stocks)
    neighbor_features = X_full[:, 4:]  # Shape: (n_samples, 4*(n_stocks-1))

    # Reshape to (n_samples, n_stocks-1, 4) and sum across neighbors
    n_neighbors = n_stocks - 1
    neighbor_features_reshaped = neighbor_features.reshape(-1, n_neighbors, 4)
    aggregated_neighbors = neighbor_features_reshaped.sum(axis=1)  # Shape: (n_samples, 4)

    # New features: [own_features, aggregated_neighbor_features]
    X_new = np.column_stack([X_own, aggregated_neighbors])

    print(f"Option 2: {X_new.shape[1]} features (4 own + 4 aggregated neighbor)")
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, shuffle=False)

    # Train new model (not continuing from base, since feature space is different)
    finetuned = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05)
    finetuned.fit(X_train, y_train)

    train_r2 = finetuned.score(X_train, y_train)
    test_r2 = finetuned.score(X_test, y_test)
    train_ic = compute_rank_ic(y_train, finetuned.predict(X_train))
    test_ic = compute_rank_ic(y_test, finetuned.predict(X_test))
    print(f"Option 2 ({stock.symbol}) - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Train IC: {train_ic:.4f}, Test IC: {test_ic:.4f}")
    return finetuned, train_r2, test_r2, train_ic, test_ic


def predict_and_plot(model, stock, stocks, sim_matrix, stock_idx, plot=True):
    """Make predictions and plot results using augmented features."""
    X, y, prev_close, common_dates = create_augmented_features_single(
        stock, stocks, sim_matrix, stock_idx
    )
    train_size = int(len(y) * 0.7)  # Match 0.3 test split

    # y is log return: log(close / prev_close)
    # To get close price: close = prev_close * exp(log_return)
    y_pred_return = model.predict(X)
    y_pred = np.exp(y_pred_return) * prev_close
    y_actual = np.exp(y) * prev_close

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Close prices
        axes[0].scatter(common_dates[train_size:], y_actual[train_size:], label='Actual', alpha=0.7, s=10)
        axes[0].scatter(common_dates[train_size:], y_pred[train_size:], label='Predicted', alpha=0.7, s=10)
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Close Price')
        axes[0].set_title(f'Close Price - Test Data ({stock.symbol})')
        axes[0].legend()
        axes[0].grid(True)

        # Plot 2: Log Returns
        axes[1].scatter(common_dates[train_size:], y[train_size:], label='Actual Log Return', alpha=0.7, s=10)
        axes[1].scatter(common_dates[train_size:], y_pred_return[train_size:], label='Predicted Log Return', alpha=0.7, s=10)
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Log Return')
        axes[1].set_title(f'Daily Return - Test Data ({stock.symbol})')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Load all stocks and train general model
    print("Loading all stocks...")
    stocks = load_all_stocks()
    print(f"Loaded {len(stocks)} stocks")

    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    sim_matrix = compute_similarity_matrix(stocks)
    print(f"Similarity matrix shape: {sim_matrix.shape}")

    # Screen max_depth for best Rank IC
    print("\nScreening max_depth...")
    results, general_model, optimal_depth = screen_max_depth(stocks, sim_matrix)
    print(f"\nUsing optimal max_depth={optimal_depth} for finetuning")

    # Benchmark finetuning on first 10 stocks
    n_benchmark = min(10, len(stocks))
    all_results = []

    for target_idx in range(n_benchmark):
        target_stock = stocks[target_idx]
        print(f"\n{'='*60}")
        print(f"Evaluating on {target_stock.symbol} ({target_idx + 1}/{n_benchmark})")
        print(f"{'='*60}")

        # Get augmented features for this stock
        X, y, open_prices, common_dates = create_augmented_features_single(
            target_stock, stocks, sim_matrix, target_idx
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        # General model (before finetuning)
        general_train_r2 = general_model.score(X_train, y_train)
        general_test_r2 = general_model.score(X_test, y_test)
        general_train_ic = compute_rank_ic(y_train, general_model.predict(X_train))
        general_test_ic = compute_rank_ic(y_test, general_model.predict(X_test))
        print(f"\n--- General model (before finetuning) ---")
        print(f"Train R²: {general_train_r2:.4f}, Test R²: {general_test_r2:.4f}, Train IC: {general_train_ic:.4f}, Test IC: {general_test_ic:.4f}")

        # Original finetuning (all features trainable)
        print(f"\n--- Original finetuning (all features trainable) ---")
        _, orig_train_r2, orig_test_r2, orig_train_ic, orig_test_ic = finetune_for_stock(
            general_model, target_stock, stocks, sim_matrix, target_idx,
            max_depth=optimal_depth
        )

        # Option 1: Two-stage with fixed neighbor score
        print(f"\n--- Option 1: Fixed neighbor score from general model ---")
        _, opt1_train_r2, opt1_test_r2, opt1_train_ic, opt1_test_ic, _ = finetune_option1(
            general_model, target_stock, stocks, sim_matrix, target_idx,
            max_depth=optimal_depth
        )

        # Option 2: Aggregated neighbor features
        print(f"\n--- Option 2: Aggregated neighbor features ---")
        _, opt2_train_r2, opt2_test_r2, opt2_train_ic, opt2_test_ic = finetune_option2(
            general_model, target_stock, stocks, sim_matrix, target_idx,
            max_depth=optimal_depth
        )

        # Store results
        all_results.append({
            'symbol': target_stock.symbol,
            'general_train_r2': general_train_r2,
            'general_test_r2': general_test_r2,
            'general_train_ic': general_train_ic,
            'general_test_ic': general_test_ic,
            'orig_train_r2': orig_train_r2,
            'orig_test_r2': orig_test_r2,
            'orig_train_ic': orig_train_ic,
            'orig_test_ic': orig_test_ic,
            'opt1_train_r2': opt1_train_r2,
            'opt1_test_r2': opt1_test_r2,
            'opt1_train_ic': opt1_train_ic,
            'opt1_test_ic': opt1_test_ic,
            'opt2_train_r2': opt2_train_r2,
            'opt2_test_r2': opt2_test_r2,
            'opt2_train_ic': opt2_train_ic,
            'opt2_test_ic': opt2_test_ic,
        })

    # Summary table for all stocks - Rank IC
    print(f"\n{'='*120}")
    print(f"BENCHMARK SUMMARY (Rank IC) - First {n_benchmark} Stocks")
    print(f"{'='*120}")
    print(f"{'Stock':<10} {'General':^25} {'Orig Finetune':^25} {'Option 1':^25} {'Option 2':^25}")
    print(f"{'':10} {'Train IC':>12} {'Test IC':>12} {'Train IC':>12} {'Test IC':>12} {'Train IC':>12} {'Test IC':>12} {'Train IC':>12} {'Test IC':>12}")
    print(f"{'-'*120}")

    for r in all_results:
        print(f"{r['symbol']:<10} {r['general_train_ic']:>12.4f} {r['general_test_ic']:>12.4f} "
              f"{r['orig_train_ic']:>12.4f} {r['orig_test_ic']:>12.4f} "
              f"{r['opt1_train_ic']:>12.4f} {r['opt1_test_ic']:>12.4f} "
              f"{r['opt2_train_ic']:>12.4f} {r['opt2_test_ic']:>12.4f}")

    # Compute averages for IC
    avg_general_train_ic = np.mean([r['general_train_ic'] for r in all_results])
    avg_general_test_ic = np.mean([r['general_test_ic'] for r in all_results])
    avg_orig_train_ic = np.mean([r['orig_train_ic'] for r in all_results])
    avg_orig_test_ic = np.mean([r['orig_test_ic'] for r in all_results])
    avg_opt1_train_ic = np.mean([r['opt1_train_ic'] for r in all_results])
    avg_opt1_test_ic = np.mean([r['opt1_test_ic'] for r in all_results])
    avg_opt2_train_ic = np.mean([r['opt2_train_ic'] for r in all_results])
    avg_opt2_test_ic = np.mean([r['opt2_test_ic'] for r in all_results])

    print(f"{'-'*120}")
    print(f"{'AVERAGE':<10} {avg_general_train_ic:>12.4f} {avg_general_test_ic:>12.4f} "
          f"{avg_orig_train_ic:>12.4f} {avg_orig_test_ic:>12.4f} "
          f"{avg_opt1_train_ic:>12.4f} {avg_opt1_test_ic:>12.4f} "
          f"{avg_opt2_train_ic:>12.4f} {avg_opt2_test_ic:>12.4f}")
    print(f"{'='*120}")

    # Plot representative outcome for first stock
    print(f"\nPlotting prediction for {stocks[0].symbol}...")
    predict_and_plot(general_model, stocks[0], stocks, sim_matrix, 0)
