import os
import importlib
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = 'data_for_process/cache_raw_stock'

def load_all_tickers():
    """Load all cached ticker data"""
    tickers = {}
    cache_files = Path(CACHE_DIR).glob('*.pkl')

    module_map = {
        'fetcher': 'data_util.fetcher',
        'fetcher.fetcher_yf': 'data_util.fetcher.fetcher_yf',
        'fetcher.fetcher_ts': 'data_util.fetcher.fetcher_ts'
    }

    class RemapUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            mapped = module_map.get(module, module)
            if mapped != module:
                importlib.import_module(mapped)
            return super().find_class(mapped, name)

    def load_pickle(path):
        with open(path, 'rb') as f:
            return RemapUnpickler(f).load()
    
    for pkl_file in cache_files:
        if not pkl_file.stem.isdigit():
            continue
        try:
            ticker_data = load_pickle(pkl_file)
            symbol = ticker_data.symbol
            tickers[symbol] = ticker_data
            print(f"Loaded: {symbol}")
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
    
    return tickers

def calculate_volatility_stats(tickers):
    """Calculate volatility statistics for all stocks"""
    stats = []
    
    for symbol, ticker in tickers.items():
        if not ticker.high or not ticker.low:
            continue
            
        high_array = np.array(ticker.high)
        low_array = np.array(ticker.low)
        
        # Calculate amplitude (high-low)
        amplitude = high_array - low_array
        
        # Calculate averages
        avg_high = np.mean(high_array)
        avg_low = np.mean(low_array)
        avg_amplitude = np.mean(amplitude)
        
        # Calculate max values
        max_high = np.max(high_array)
        max_amplitude = np.max(amplitude)
        
        # Calculate standard deviation (volatility)
        std_amplitude = np.std(amplitude)
        
        # Calculate range
        range_high = np.max(high_array) - np.min(high_array)
        range_low = np.max(low_array) - np.min(low_array)
        
        stats.append({
            'symbol': symbol,
            'avg_high': avg_high,
            'avg_low': avg_low,
            'avg_amplitude': avg_amplitude,
            'max_high': max_high,
            'max_amplitude': max_amplitude,
            'std_amplitude': std_amplitude,
            'range_high': range_high,
            'range_low': range_low,
            'num_records': len(ticker.high)
        })
    
    if not stats:
        return pd.DataFrame(
            columns=[
                'symbol',
                'avg_high',
                'avg_low',
                'avg_amplitude',
                'max_high',
                'max_amplitude',
                'std_amplitude',
                'range_high',
                'range_low',
                'num_records'
            ]
        )

    return pd.DataFrame(stats)

def plot_volatility_analysis(stats_df):
    """Plot volatility analysis charts"""
    if stats_df.empty:
        print("No data available for volatility plots.")
        return
    
    # Sort by average amplitude and view top 30 stocks with largest amplitude
    top_volatile = stats_df.nlargest(30, 'avg_amplitude')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Stock Volatility Analysis', fontsize=16, fontweight='bold')
    
    # 1. Average amplitude (High-Low)
    ax1 = axes[0, 0]
    ax1.barh(top_volatile['symbol'], top_volatile['avg_amplitude'], color='steelblue')
    ax1.set_xlabel('Average Amplitude (High-Low)', fontsize=12)
    ax1.set_title('Top 30 Stocks with Largest Average Amplitude', fontsize=12)
    ax1.invert_yaxis()
    
    # 2. Standard deviation of amplitude (volatility rate)
    top_volatility_std = stats_df.nlargest(30, 'std_amplitude')
    ax2 = axes[0, 1]
    ax2.barh(top_volatility_std['symbol'], top_volatility_std['std_amplitude'], color='orangered')
    ax2.set_xlabel('Amplitude Standard Deviation (Volatility)', fontsize=12)
    ax2.set_title('Top 30 Stocks with Highest Volatility Rate', fontsize=12)
    ax2.invert_yaxis()
    
    # 3. Maximum amplitude
    top_max_amplitude = stats_df.nlargest(30, 'max_amplitude')
    ax3 = axes[1, 0]
    ax3.barh(top_max_amplitude['symbol'], top_max_amplitude['max_amplitude'], color='seagreen')
    ax3.set_xlabel('Maximum Amplitude', fontsize=12)
    ax3.set_title('Top 30 Stocks with Largest Maximum Amplitude', fontsize=12)
    ax3.invert_yaxis()
    
    # 4. Distribution of amplitude (histogram)
    ax4 = axes[1, 1]
    ax4.hist(stats_df['avg_amplitude'], bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Average Amplitude', fontsize=12)
    ax4.set_ylabel('Number of Stocks', fontsize=12)
    ax4.set_title('Distribution of Average Amplitude Across All Stocks', fontsize=12)
    ax4.axvline(stats_df['avg_amplitude'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_df["avg_amplitude"].mean():.2f}')
    ax4.axvline(stats_df['avg_amplitude'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {stats_df["avg_amplitude"].median():.2f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('volatility_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: volatility_analysis.png")
    plt.show()

def plot_additional_analysis(stats_df):
    """Plot additional analysis charts"""
    if stats_df.empty:
        print("No data available for detailed plots.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Stock Volatility Detailed Analysis', fontsize=16, fontweight='bold')
    
    # 1. Average high price distribution
    ax1 = axes[0, 0]
    ax1.hist(stats_df['avg_high'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Average High Price', fontsize=12)
    ax1.set_ylabel('Number of Stocks', fontsize=12)
    ax1.set_title('Distribution of Average High Price Across All Stocks', fontsize=12)
    
    # 2. Average low price distribution
    ax2 = axes[0, 1]
    ax2.hist(stats_df['avg_low'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Average Low Price', fontsize=12)
    ax2.set_ylabel('Number of Stocks', fontsize=12)
    ax2.set_title('Distribution of Average Low Price Across All Stocks', fontsize=12)
    
    # 3. Amplitude standard deviation distribution
    ax3 = axes[1, 0]
    ax3.hist(stats_df['std_amplitude'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Amplitude Standard Deviation (Volatility)', fontsize=12)
    ax3.set_ylabel('Number of Stocks', fontsize=12)
    ax3.set_title('Distribution of Stock Volatility', fontsize=12)
    ax3.axvline(stats_df['std_amplitude'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_df["std_amplitude"].mean():.2f}')
    ax3.legend()
    
    # 4. Scatter plot: average high price vs amplitude
    ax4 = axes[1, 1]
    scatter = ax4.scatter(stats_df['avg_high'], stats_df['avg_amplitude'], 
                         c=stats_df['std_amplitude'], cmap='viridis', 
                         s=100, alpha=0.6, edgecolors='black')
    ax4.set_xlabel('Average High Price', fontsize=12)
    ax4.set_ylabel('Average Amplitude', fontsize=12)
    ax4.set_title('Average High Price vs Average Amplitude (Color = Volatility)', fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Volatility', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('volatility_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: volatility_detailed_analysis.png")
    plt.show()

def print_statistics_summary(stats_df):
    """Print statistics summary"""
    if stats_df.empty:
        print("\n" + "="*60)
        print("Stock Volatility Statistics Summary")
        print("="*60)
        print("Total number of stocks: 0")
        print("No data available for volatility statistics.")
        print("="*60)
        return
    print("\n" + "="*60)
    print("Stock Volatility Statistics Summary")
    print("="*60)
    print(f"Total number of stocks: {len(stats_df)}")
    print(f"\nAverage Amplitude (High-Low) Statistics:")
    print(f"  Mean: {stats_df['avg_amplitude'].mean():.4f}")
    print(f"  Median: {stats_df['avg_amplitude'].median():.4f}")
    print(f"  Standard Deviation: {stats_df['avg_amplitude'].std():.4f}")
    print(f"  Min: {stats_df['avg_amplitude'].min():.4f}")
    print(f"  Max: {stats_df['avg_amplitude'].max():.4f}")
    
    print(f"\nAmplitude Standard Deviation (Volatility Rate) Statistics:")
    print(f"  Mean: {stats_df['std_amplitude'].mean():.4f}")
    print(f"  Median: {stats_df['std_amplitude'].median():.4f}")
    print(f"  Standard Deviation: {stats_df['std_amplitude'].std():.4f}")
    print(f"  Min: {stats_df['std_amplitude'].min():.4f}")
    print(f"  Max: {stats_df['std_amplitude'].max():.4f}")
    
    print(f"\nTop 10 stocks with highest volatility (by average amplitude):")
    top_10 = stats_df.nlargest(10, 'avg_amplitude')[['symbol', 'avg_amplitude', 'std_amplitude', 'num_records']]
    for idx, row in top_10.iterrows():
        print(f"  {row['symbol']:8s}: avg_amplitude={row['avg_amplitude']:.4f}, volatility={row['std_amplitude']:.4f}, data_points={row['num_records']}")
    
    print(f"\nBottom 10 stocks with lowest volatility (by average amplitude):")
    bottom_10 = stats_df.nsmallest(10, 'avg_amplitude')[['symbol', 'avg_amplitude', 'std_amplitude', 'num_records']]
    for idx, row in bottom_10.iterrows():
        print(f"  {row['symbol']:8s}: avg_amplitude={row['avg_amplitude']:.4f}, volatility={row['std_amplitude']:.4f}, data_points={row['num_records']}")
    
    print("\n" + "="*60)
    
    # Save detailed statistics to CSV
    stats_df_sorted = stats_df.sort_values('avg_amplitude', ascending=False)
    stats_df_sorted.to_csv('volatility_statistics.csv', index=False)
    print("Detailed statistics saved to: volatility_statistics.csv")

if __name__ == '__main__':
    print("Starting to load all stock data...")
    tickers = load_all_tickers()
    print(f"\nLoaded {len(tickers)} stocks")
    
    print("\nCalculating volatility statistics...")
    stats_df = calculate_volatility_stats(tickers)
    
    print("\nPrinting statistics summary...")
    print_statistics_summary(stats_df)
    
    print("\nGenerating visualization charts...")
    plot_volatility_analysis(stats_df)
    plot_additional_analysis(stats_df)
    
    print("\nAnalysis complete!")
