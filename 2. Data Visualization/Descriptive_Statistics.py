#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ethereum DEX Trading Behavior Simplified Analysis
Generate only: Table 1 + Figure 1 + Figure 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set English font and style
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """Main analysis function"""
    
    print("=" * 100)
    print("📊 Ethereum DEX Trading Behavior Simplified Analysis")
    print("=" * 100)
    
    # Load and process data
    df = load_and_preprocess_data()
    
    if df is None:
        print("❌ Failed to load data. Exiting.")
        return None
    
    # Run analysis
    print("\n🔹 Starting simplified analysis...")
    
    # 1. Generate Table 1: Descriptive Statistics of Transaction Amounts
    generate_descriptive_statistics_table(df)
    
    # 2. Generate Figure 1: 24-Hour Distribution Pattern
    create_24hour_pattern(df)
    
    # 3. Generate Figure 2: Scatter Plot
    create_frequency_amount_scatter(df)
    
    print("\n🎉 Analysis completed successfully!")
    print("📁 Output files:")
    print("   - Table 1: Descriptive Statistics (console output)")
    print("   - Figure 1: 24_hour_trading_pattern.png")
    print("   - Figure 2: frequency_vs_amount_scatter.png")
    
    return df

def load_and_preprocess_data():
    """Load and preprocess trading data"""
    
    print("📊 Loading and preprocessing data...")
    
    # Try to load real data files
    df = None
    
    # Try to load main trading data
    for filename in ['filtered_trades.csv', 'ethereum_trades.csv', 'trading_data.csv']:
        try:
            df = pd.read_csv(filename)
            print(f"✅ Successfully loaded: {filename} ({df.shape[0]:,} rows × {df.shape[1]} columns)")
            break
        except FileNotFoundError:
            continue
    
    # If no real data found, generate demo data
    if df is None:
        print("⚠️  No data files found. Generating demo data for analysis...")
        df = generate_demo_data()
        print(f"✅ Generated demo data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Preprocess the data
    df = preprocess_data(df)
    
    return df

def generate_demo_data():
    """Generate realistic demo trading data"""
    
    np.random.seed(42)
    n_trades = 50000
    n_wallets = 1500
    
    # Generate wallet addresses
    wallets = [f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}" 
               for _ in range(n_wallets)]
    
    # Generate timestamps over 90 days
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2024-03-31')
    timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')[:n_trades]
    
    # Generate trading data with realistic patterns
    demo_df = pd.DataFrame({
        'wallet_address': np.random.choice(wallets, n_trades),
        'block_time': timestamps,
        'amount_usd': np.random.lognormal(mean=6, sigma=2, size=n_trades),
        'token_bought_symbol': np.random.choice([
            'USDC', 'WETH', 'USDT', 'DAI', 'WBTC', 'UNI', 'LINK', 'AAVE', 'MATIC', 'CRV'
        ], n_trades),
        'dex_name': np.random.choice([
            'Uniswap V3', 'Uniswap V2', 'SushiSwap', '1inch', 'Curve'
        ], n_trades, p=[0.4, 0.25, 0.15, 0.1, 0.1])
    })
    
    return demo_df

def preprocess_data(df):
    """Preprocess the trading data"""
    
    print("🔧 Preprocessing data...")
    
    # Identify key columns
    columns = {
        'amount': find_column(df, ['amount_usd', 'amount', 'value_usd', 'usd']),
        'time': find_column(df, ['block_time', 'timestamp', 'time', 'date']),
        'wallet': find_column(df, ['wallet_address', 'taker', 'user_address', 'address']),
        'token': find_column(df, ['token_bought_symbol', 'token_symbol', 'symbol']),
        'dex': find_column(df, ['dex_name', 'project', 'exchange', 'protocol'])
    }
    
    print("📋 Identified columns:")
    for key, col in columns.items():
        print(f"   {key}: {col}")
    
    # Process time column
    if columns['time']:
        df[columns['time']] = pd.to_datetime(df[columns['time']])
        df['date'] = df[columns['time']].dt.date
        df['hour'] = df[columns['time']].dt.hour
        df['day_of_week'] = df[columns['time']].dt.day_name()
        df['is_weekend'] = df[columns['time']].dt.weekday.isin([5, 6])
    
    # Clean amount data
    if columns['amount'] and df[columns['amount']].dtype in ['float64', 'int64']:
        # Remove extreme outliers (keep 99% of data)
        q01 = df[columns['amount']].quantile(0.005)
        q99 = df[columns['amount']].quantile(0.995)
        before_len = len(df)
        df = df[(df[columns['amount']] >= q01) & (df[columns['amount']] <= q99)]
        print(f"   Removed {before_len - len(df):,} extreme outliers")
    
    print(f"✅ Final dataset: {len(df):,} transactions")
    
    return df

def find_column(df, keywords):
    """Find column containing any of the keywords"""
    for keyword in keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                return col
    return None

def generate_descriptive_statistics_table(df):
    """Generate Table 1: Descriptive Statistics of Transaction Amounts"""
    
    print("\n" + "="*80)
    print("📋 TABLE 1: DESCRIPTIVE STATISTICS OF TRANSACTION AMOUNTS")
    print("="*80)
    
    amount_col = find_column(df, ['amount_usd', 'amount', 'value_usd'])
    
    if not amount_col:
        print("❌ No amount column found")
        return
    
    amounts = df[amount_col].dropna()
    
    # Calculate statistics
    stats_desc = amounts.describe()
    
    print("\n% LaTeX Table for Academic Paper")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Descriptive Statistics of Transaction Amounts}")
    print("\\label{tab:descriptive_stats}")
    print("\\begin{tabular}{lr}")
    print("\\toprule")
    print("Statistic & Value (USD) \\\\")
    print("\\midrule")
    print(f"Count & {len(amounts):,} \\\\")
    print(f"Mean & {stats_desc['mean']:.2f} \\\\")
    print(f"Median & {stats_desc['50%']:.2f} \\\\")
    print(f"Standard Deviation & {stats_desc['std']:.2f} \\\\")
    print(f"Minimum & {stats_desc['min']:.2f} \\\\")
    print(f"25th Percentile & {stats_desc['25%']:.2f} \\\\")
    print(f"75th Percentile & {stats_desc['75%']:.2f} \\\\")
    print(f"Maximum & {stats_desc['max']:.2f} \\\\")
    print(f"Skewness & {stats.skew(amounts):.3f} \\\\")
    print(f"Kurtosis & {stats.kurtosis(amounts):.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print(f"\n📊 Summary Statistics:")
    print(f"Sample Size: {len(amounts):,}")
    print(f"Total Volume: ${amounts.sum():,.0f}")
    print(f"Mean: ${stats_desc['mean']:,.2f}")
    print(f"Median: ${stats_desc['50%']:,.2f}")
    print(f"Std Dev: ${stats_desc['std']:,.2f}")
    print(f"Skewness: {stats.skew(amounts):.3f}")
    print(f"Kurtosis: {stats.kurtosis(amounts):.3f}")

def create_24hour_pattern(df):
    """Create Figure 1: 24-Hour Distribution Pattern of Ethereum DEX Trading"""
    
    print("\n" + "="*80)
    print("📊 FIGURE 1: 24-HOUR DISTRIBUTION PATTERN")
    print("="*80)
    
    if 'hour' not in df.columns:
        print("❌ No time data available")
        return
    
    # Set clean style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    
    plt.figure(figsize=(12, 6))
    
    hourly_counts = df.groupby('hour').size()
    plt.plot(hourly_counts.index, hourly_counts.values, 'o-', 
            linewidth=3, markersize=8, color='#2E86AB', markerfacecolor='#A23B72')
    
    # Highlight peak and low hours
    peak_hour = hourly_counts.idxmax()
    low_hour = hourly_counts.idxmin()
    plt.scatter(peak_hour, hourly_counts[peak_hour], color='red', s=150, 
               label=f'Peak: {peak_hour}:00', zorder=5)
    plt.scatter(low_hour, hourly_counts[low_hour], color='blue', s=150, 
               label=f'Low: {low_hour}:00', zorder=5)
    
    plt.title('24-Hour Distribution Pattern of Ethereum DEX Trading', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day (UTC)', fontsize=12)
    plt.ylabel('Transaction Count', fontsize=12)
    plt.xticks(range(0, 24, 2))
    plt.legend()
    
    # Clean styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig('24_hour_trading_pattern.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.show()
    print(f"✅ Figure 1: 24-hour trading pattern saved as '24_hour_trading_pattern.png'")

def create_frequency_amount_scatter(df):
    """Create Figure 2: Scatter Plot of Wallet Transaction Frequency versus Average Transaction Amount"""
    
    print("\n" + "="*80)
    print("📊 FIGURE 2: FREQUENCY VS AMOUNT SCATTER PLOT")
    print("="*80)
    
    amount_col = find_column(df, ['amount_usd', 'amount', 'value_usd'])
    wallet_col = find_column(df, ['wallet_address', 'taker', 'user_address'])
    
    if not amount_col or not wallet_col:
        print("❌ Required columns not found")
        return
    
    # Set clean style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    
    plt.figure(figsize=(10, 8))
    
    wallet_stats = df.groupby(wallet_col).agg({
        amount_col: ['count', 'mean']
    })
    wallet_stats.columns = ['transaction_count', 'avg_amount']
    
    # Sample for better visualization if too many points
    if len(wallet_stats) > 2000:
        sample_data = wallet_stats.sample(2000, random_state=42)
    else:
        sample_data = wallet_stats
    
    plt.scatter(sample_data['transaction_count'], sample_data['avg_amount'], 
               alpha=0.6, s=30, c='darkblue', edgecolors='white', linewidth=0.5)
    plt.title('Scatter Plot of Wallet Transaction Frequency versus Average Transaction Amount', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Transaction Count per Wallet', fontsize=12)
    plt.ylabel('Average Transaction Amount (USD)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    
    # Add correlation info
    correlation = sample_data['transaction_count'].corr(sample_data['avg_amount'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=plt.gca().transAxes, fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Clean styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig('frequency_vs_amount_scatter.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.show()
    print(f"✅ Figure 2: Frequency vs amount scatter plot saved as 'frequency_vs_amount_scatter.png'")

if __name__ == "__main__":
    print("🚀 Starting Simplified Ethereum DEX Trading Analysis...")
    result = main()
    if result is not None:
        print("🎉 Analysis completed successfully!")
        print("\n📋 Summary of outputs:")
        print("   1. Table 1: Descriptive Statistics (LaTeX format in console)")
        print("   2. Figure 1: 24_hour_trading_pattern.png")
        print("   3. Figure 2: frequency_vs_amount_scatter.png")
    else:
        print("❌ Analysis failed.")
