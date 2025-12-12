import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

"""
Factor Strategy Guardian - Regime Classification and Strategy Implementation
VIX Term-Structure Based Regime Switching Framework
FIXED VERSION - Using Real DAILY Factor Data (correct extraction from dataOnFactors.xlsx)
"""

print("="*80)
print("FACTOR STRATEGY GUARDIAN - REGIME CLASSIFICATION & BACKTESTING")
print("="*80)

# ==================== 1. LOAD DATA ====================
print("\n" + "="*80)
print("STEP 1: Loading Data")
print("="*80)

# -------- 1a. Load VIX futures data --------
df_vix = pd.read_excel('data.xlsx', header=1)
df_vix.columns = df_vix.columns.str.strip()
df_vix['Date'] = pd.to_datetime(df_vix['Date'])
df_vix = df_vix.sort_values('Date').reset_index(drop=True)

# Extract VX1 and VX2 prices
df_vix['VX1_Price'] = pd.to_numeric(df_vix['VX1'], errors='coerce')
df_vix['VX2_Price'] = pd.to_numeric(df_vix['VX2'], errors='coerce')

# -------- 1b. Load DAILY factor data (fixed) --------
print("\nLoading factor data (DAILY only)...")
df_factors_raw = pd.read_excel('dataOnFactors.xlsx', header=None)

rows, cols = df_factors_raw.shape

# 1) Find all blocks of the form [Dates | PX_LAST] (each block = one factor)
factor_blocks = []
for col in range(cols - 1):
    for row in range(rows):
        cell = df_factors_raw.iat[row, col]
        if isinstance(cell, str) and cell.strip().lower() == "dates":
            next_cell = df_factors_raw.iat[row, col + 1]
            if isinstance(next_cell, str) and "px_last" in next_cell.lower():
                factor_blocks.append({
                    "date_col": col,
                    "data_col": col + 1,
                    "dates_row": row,
                })
            # Once we find "Dates" in this column, stop scanning this column
            break

# 2) Detect WEEKLY section; treat any block whose date_col >= weekly_start_col as WEEKLY and drop
weekly_start_col = None
for r in range(min(10, rows)):  # only need to scan first few rows for the label
    for c in range(cols):
        val = df_factors_raw.iat[r, c]
        if isinstance(val, str) and "weekly" in val.lower():
            weekly_start_col = c
            break
    if weekly_start_col is not None:
        break

daily_blocks = []
for blk in factor_blocks:
    if weekly_start_col is not None and blk["date_col"] >= weekly_start_col:
        # skip WEEKLY factors
        continue
    daily_blocks.append(blk)

print(f"Identified {len(daily_blocks)} DAILY factor blocks")

# 3) Helper: infer the factor name by scanning upward above PX_LAST column
def infer_factor_name(df, data_col, dates_row):
    blacklist_words = ["start date", "end date", "last price", "dates", "daily", "weekly"]
    for r in range(dates_row - 1, -1, -1):
        val = df.iat[r, data_col]
        if pd.isna(val):
            continue
        s = str(val).strip()
        if not s:
            continue
        low = s.lower()

        # Skip metadata-like rows
        if any(b in low for b in blacklist_words):
            continue
        if "index" in low:
            continue

        # Skip cells that look like dates
        try:
            dt = pd.to_datetime(s, errors="raise")
            if 1900 <= dt.year <= 2100:
                continue
        except Exception:
            pass

        # Remaining strings are likely factor names
        return s

    # Fallback name if nothing reasonable is found
    return f"Factor_{data_col}"

# 4) Extract each DAILY factor block into its own time series
all_factors = {}
for blk in daily_blocks:
    date_col = blk["date_col"]
    data_col = blk["data_col"]
    dates_row = blk["dates_row"]

    factor_name = infer_factor_name(df_factors_raw, data_col, dates_row)

    # Optional manual cleanup for common typos
    rename_map = {
        "Qualiy": "Quality",
        "Qualitiy": "Quality",
    }
    factor_name = rename_map.get(factor_name, factor_name)

    start_row = dates_row + 1  # data starts below the "Dates" row

    dates = df_factors_raw.iloc[start_row:, date_col]
    values = df_factors_raw.iloc[start_row:, data_col]

    factor_df = pd.DataFrame({
        "Date": dates,
        factor_name: values,
    })

    # Convert types
    factor_df["Date"] = pd.to_datetime(factor_df["Date"], errors="coerce")
    factor_df[factor_name] = pd.to_numeric(factor_df[factor_name], errors="coerce")

    factor_df = factor_df.dropna()

    if len(factor_df) > 0:
        all_factors[factor_name] = factor_df
        print(f"  {factor_name}: {len(factor_df)} days")

# 5) Merge all DAILY factors into a single dataframe
if len(all_factors) == 0:
    print("\n❌ ERROR: No DAILY factors found in dataOnFactors.xlsx")
    raise SystemExit(1)

df_factors = None
for factor_name, factor_df in all_factors.items():
    if df_factors is None:
        df_factors = factor_df
    else:
        df_factors = df_factors.merge(factor_df, on="Date", how="outer")

df_factors = df_factors.sort_values("Date").reset_index(drop=True)
factor_columns = [col for col in df_factors.columns if col != "Date"]

print(f"\n✓ DAILY factor data merged: {len(df_factors)} days")
print(f"✓ Factors available: {factor_columns}")

# -------- 1c. Load risk-free rate --------
df_rf_raw = pd.read_excel('rf_3m_tbill_daily.xlsx')
df_rf_split = df_rf_raw.iloc[:, 0].str.split(',', expand=True)
df_rf_split.columns = ['Date', 'RF_3M', 'rf_daily']
df_rf_split['Date'] = pd.to_datetime(df_rf_split['Date'])
df_rf_split['rf_daily'] = pd.to_numeric(df_rf_split['rf_daily'])
df_rf_split = df_rf_split.sort_values('Date').reset_index(drop=True)

print(f"\nData Summary:")
print(f"  VIX data: {len(df_vix)} days")
print(f"  Factor data (DAILY): {len(df_factors)} days")
print(f"  Risk-free rate: {len(df_rf_split)} days")

# ==================== 2. CALCULATE VIX TERM STRUCTURE ====================
print("\n" + "="*80)
print("STEP 2: Calculate VIX Term Structure Metrics")
print("="*80)

window = 252  # 1-year rolling window
df_vix['VIX_Slope'] = (df_vix['VX2_Price'] / df_vix['VX1_Price']) - 1
df_vix['VIX_Slope_pct'] = df_vix['VIX_Slope'] * 100

df_vix['VIX_Slope_Mean'] = df_vix['VIX_Slope'].rolling(window=window, min_periods=60).mean()
df_vix['VIX_Slope_Std'] = df_vix['VIX_Slope'].rolling(window=window, min_periods=60).std()
df_vix['VIX_Zscore'] = (df_vix['VIX_Slope'] - df_vix['VIX_Slope_Mean']) / df_vix['VIX_Slope_Std']

# Remove NaN values for Z-score
df_vix = df_vix.dropna(subset=['VIX_Zscore']).reset_index(drop=True)

print(f"\nVIX Slope Statistics:")
print(f"  Mean: {df_vix['VIX_Slope_pct'].mean():.3f}%")
print(f"  Std Dev: {df_vix['VIX_Slope_pct'].std():.3f}%")
print(f"  Min: {df_vix['VIX_Slope_pct'].min():.3f}%")
print(f"  Max: {df_vix['VIX_Slope_pct'].max():.3f}%")

print(f"\nVIX Z-Score Statistics:")
print(f"  Mean: {df_vix['VIX_Zscore'].mean():.3f}")
print(f"  Std Dev: {df_vix['VIX_Zscore'].std():.3f}")
print(f"  Min: {df_vix['VIX_Zscore'].min():.3f}")
print(f"  Max: {df_vix['VIX_Zscore'].max():.3f}")

# ==================== 3. DEFINE 3 RISK REGIMES ====================
print("\n" + "="*80)
print("STEP 3: Define 3 Risk Regimes")
print("="*80)

threshold_low = -0.5   # Risk-Off threshold (high stress)
threshold_high = 0.5   # Risk-On threshold (low stress)

def classify_regime(z_score):
    if z_score < threshold_low:
        return 'Risk-Off'
    elif z_score > threshold_high:
        return 'Risk-On'
    else:
        return 'Neutral'

df_vix['Regime'] = df_vix['VIX_Zscore'].apply(classify_regime)

print(f"\nRegime Distribution:")
regime_counts = df_vix['Regime'].value_counts()
for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
    if regime in regime_counts.index:
        count = regime_counts[regime]
        pct = count / len(df_vix) * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")

# ==================== 4. MERGE DATA ====================
print("\n" + "="*80)
print("STEP 4: Merge All Data")
print("="*80)

df_merged = df_vix[['Date', 'VX1_Price', 'VX2_Price', 'VIX_Slope', 'VIX_Slope_pct',
                    'VIX_Zscore', 'Regime']].copy()
df_merged = df_merged.merge(df_rf_split[['Date', 'rf_daily']], on='Date', how='left')

df_merged = df_merged.merge(df_factors, on='Date', how='inner')

print(f"\nMerged dataset shape: {df_merged.shape}")
print(f"Date range: {df_merged['Date'].min()} to {df_merged['Date'].max()}")
print(f"Columns: {df_merged.columns.tolist()}")

# ==================== 5. CALCULATE FACTOR RETURNS ====================
print("\n" + "="*80)
print("STEP 5: Calculate Factor Returns")
print("="*80)

factor_return_cols = []
for factor in factor_columns:
    return_col = f'{factor}_Return'
    df_merged[return_col] = df_merged[factor].pct_change()
    factor_return_cols.append(return_col)

# Drop the first row (NaN returns)
df_merged = df_merged.iloc[1:].reset_index(drop=True)

print(f"\nFactor returns calculated for {len(factor_columns)} factors")

print("\nCreating equal-weighted composite factor portfolio...")
df_merged['Factor_Returns'] = df_merged[factor_return_cols].mean(axis=1)

if df_merged['Factor_Returns'].isna().any():
    print(f"⚠️  Warning: {df_merged['Factor_Returns'].isna().sum()} NaN values in factor returns")
    df_merged['Factor_Returns'] = df_merged['Factor_Returns'].fillna(method='ffill')
    df_merged['Factor_Returns'] = df_merged['Factor_Returns'].fillna(0)

print(f"\nComposite Factor Returns Statistics:")
print(f"  Mean (daily): {df_merged['Factor_Returns'].mean()*100:.4f}%")
print(f"  Std (daily): {df_merged['Factor_Returns'].std()*100:.4f}%")
print(f"  Min: {df_merged['Factor_Returns'].min()*100:.2f}%")
print(f"  Max: {df_merged['Factor_Returns'].max()*100:.2f}%")
print(f"  Total days: {len(df_merged)}")

# ==================== 6. STRATEGY IMPLEMENTATION ====================
print("\n" + "="*80)
print("STEP 6: Implement Strategy")
print("="*80)

STRATEGY_CONFIG = {
    'Risk-Off': {
        'factor_weight': 0.0,      # Exit factor exposure
        'hedge_ratio': 1.0,        # Full hedge
        'description': 'High stress: Exit factors, full hedge'
    },
    'Neutral': {
        'factor_weight': 0.5,      # 50% factor exposure
        'hedge_ratio': 0.5,        # 50% hedge
        'description': 'Moderate: Partial exposure'
    },
    'Risk-On': {
        'factor_weight': 1.0,      # Full factor exposure
        'hedge_ratio': 0.0,        # No hedge
        'description': 'Low stress: Full factor exposure'
    }
}

print("\nStrategy Configuration:")
for regime, config in STRATEGY_CONFIG.items():
    print(f"\n{regime}:")
    print(f"  Factor Weight: {config['factor_weight']*100:.0f}%")
    print(f"  Hedge Ratio: {config['hedge_ratio']*100:.0f}%")
    print(f"  Description: {config['description']}")

df_merged['Factor_Weight'] = df_merged['Regime'].map(
    lambda x: STRATEGY_CONFIG[x]['factor_weight'])
df_merged['Hedge_Ratio'] = df_merged['Regime'].map(
    lambda x: STRATEGY_CONFIG[x]['hedge_ratio'])

# Compute individual futures returns
df_merged['VX1_Return'] = df_merged['VX1_Price'].pct_change()
df_merged['VX2_Return'] = df_merged['VX2_Price'].pct_change()
'''
# Dollar-neutral long VX1 / short VX2 spread return
df_merged['Hedge_Returns'] = (
    df_merged['VX1_Return'] * df_merged['VX2_Price'].shift(1)
    - df_merged['VX2_Return'] * df_merged['VX1_Price'].shift(1)
) / (df_merged['VX1_Price'].shift(1) + df_merged['VX2_Price'].shift(1))
'''

window = 20
df_merged['VX1_Vol'] = df_merged['VX1_Return'].rolling(window).std()
df_merged['VX2_Vol'] = df_merged['VX2_Return'].rolling(window).std()

# Inverse-volatility weighting (risk parity)
df_merged['VX1_Weight'] = 1 / df_merged['VX1_Vol']
df_merged['VX2_Weight'] = 1 / df_merged['VX2_Vol']

# Normalize the weights
total_weight = df_merged['VX1_Weight'] + df_merged['VX2_Weight']
df_merged['VX1_Weight'] = df_merged['VX1_Weight'] / total_weight
df_merged['VX2_Weight'] = df_merged['VX2_Weight'] / total_weight

# Calendar spread returns
df_merged['Hedge_Returns'] = (
    df_merged['VX1_Return'] * df_merged['VX1_Weight'] -
    df_merged['VX2_Return'] * df_merged['VX2_Weight']
)
df_merged['Hedge_Returns'] = df_merged['Hedge_Returns'].fillna(0)

df_merged['Strategy_Returns'] = (
    df_merged['Factor_Returns'] * df_merged['Factor_Weight'] +
    df_merged['Hedge_Returns'] * df_merged['Hedge_Ratio']
)

df_merged['Benchmark_Returns'] = df_merged['Factor_Returns']

df_merged['Strategy_Cumulative'] = (1 + df_merged['Strategy_Returns']).cumprod()
df_merged['Benchmark_Cumulative'] = (1 + df_merged['Benchmark_Returns']).cumprod()

print(f"\nReturn Statistics (daily):")
print(f"\nStrategy:")
print(f"  Mean Return: {df_merged['Strategy_Returns'].mean()*100:.4f}%")
print(f"  Std Dev: {df_merged['Strategy_Returns'].std()*100:.4f}%")
print(f"  Total Return: {(df_merged['Strategy_Cumulative'].iloc[-1] - 1)*100:.2f}%")

print(f"\nBenchmark (Buy & Hold):")
print(f"  Mean Return: {df_merged['Benchmark_Returns'].mean()*100:.4f}%")
print(f"  Std Dev: {df_merged['Benchmark_Returns'].std()*100:.4f}%")
print(f"  Total Return: {(df_merged['Benchmark_Cumulative'].iloc[-1] - 1)*100:.2f}%")

# ==================== 7. CALCULATE PERFORMANCE METRICS ====================
print("\n" + "="*80)
print("STEP 7: Calculate Performance Metrics")
print("="*80)

def calculate_metrics(returns, rf_rate=None):
    if rf_rate is None:
        rf_daily = 0.0
    else:
        rf_daily = rf_rate.mean() if hasattr(rf_rate, 'mean') else rf_rate

    annual_factor = 252

    mean_return = returns.mean() * annual_factor
    std_return = returns.std() * np.sqrt(annual_factor)

    excess_returns = returns - rf_daily
    if excess_returns.std() > 0:
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(annual_factor)
    else:
        sharpe = 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0

    win_rate = (returns > 0).sum() / len(returns)

    downside_returns = returns[returns < rf_daily]
    downside_std = downside_returns.std() * np.sqrt(annual_factor)
    sortino = (mean_return - rf_daily * annual_factor) / downside_std if downside_std > 0 else 0

    return {
        'Mean Annual Return': mean_return,
        'Annual Volatility': std_return,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Maximum Drawdown': max_drawdown,
        'Calmar Ratio': calmar,
        'Win Rate': win_rate
    }

rf_daily = df_merged['rf_daily'].fillna(0)
strategy_metrics = calculate_metrics(df_merged['Strategy_Returns'], rf_daily)
benchmark_metrics = calculate_metrics(df_merged['Benchmark_Returns'], rf_daily)

print("\n" + "="*80)
print("PERFORMANCE METRICS COMPARISON")
print("="*80)

metrics_df = pd.DataFrame({
    'Strategy': strategy_metrics,
    'Benchmark': benchmark_metrics
})

print(f"\n{metrics_df.to_string()}")

print("\n" + "="*80)
print("STRATEGY IMPROVEMENTS")
print("="*80)
print(f"\nSharpe Ratio Improvement: {(strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']):.4f}")
print(f"Volatility Reduction: {(benchmark_metrics['Annual Volatility'] - strategy_metrics['Annual Volatility'])*100:.2f}%")
print(f"Max Drawdown Reduction: {(benchmark_metrics['Maximum Drawdown'] - strategy_metrics['Maximum Drawdown'])*100:.2f}%")

# ==================== 8. REGIME ANALYSIS ====================
print("\n" + "="*80)
print("STEP 8: Analyze Returns by Regime")
print("="*80)

regime_analysis = df_merged.groupby('Regime').agg({
    'Strategy_Returns': ['mean', 'std', 'count'],
    'Benchmark_Returns': ['mean', 'std'],
    'VIX_Zscore': ['mean', 'std']
}).round(6)

print("\nReturns by Regime (daily):")
print(regime_analysis)

print("\n" + "-"*80)
print("Annualized Returns by Regime:")
for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
    regime_data = df_merged[df_merged['Regime'] == regime]
    if len(regime_data) > 0:
        strat_annual = regime_data['Strategy_Returns'].mean() * 252 * 100
        bench_annual = regime_data['Benchmark_Returns'].mean() * 252 * 100
        print(f"\n{regime}:")
        print(f"  Strategy: {strat_annual:.2f}%")
        print(f"  Benchmark: {bench_annual:.2f}%")
        print(f"  Difference: {strat_annual - bench_annual:.2f}%")

# ==================== 9. KEY MARKET EVENTS ====================
print("\n" + "="*80)
print("STEP 9: Analyze Key Market Events")
print("="*80)

key_events = {
    'VIX Spike 2018': ('2018-02-01', '2018-02-28'),
    '2018 Q4 Selloff': ('2018-10-01', '2018-12-31'),
    'COVID-19 Crash': ('2020-02-15', '2020-04-15'),
    'COVID Recovery': ('2020-04-16', '2020-06-30'),
    '2022 Inflation': ('2022-01-01', '2022-03-31'),
}

for event_name, (start, end) in key_events.items():
    mask = (df_merged['Date'] >= start) & (df_merged['Date'] <= end)
    event_data = df_merged[mask]

    if len(event_data) > 0:
        regime_dist = event_data['Regime'].value_counts(normalize=True) * 100
        strat_ret = (1 + event_data['Strategy_Returns']).prod() - 1
        bench_ret = (1 + event_data['Benchmark_Returns']).prod() - 1

        print(f"\n{event_name} ({start} to {end}):")
        print(f"  Days: {len(event_data)}")
        print(f"  Strategy Return: {strat_ret*100:.2f}%")
        print(f"  Benchmark Return: {bench_ret*100:.2f}%")
        print(f"  Outperformance: {(strat_ret - bench_ret)*100:.2f}%")
        print(f"  Regime Distribution:")
        for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
            if regime in regime_dist.index:
                print(f"    {regime}: {regime_dist[regime]:.1f}%")

# ==================== 10. SAVE RESULTS ====================
print("\n" + "="*80)
print("STEP 10: Save Results")
print("="*80)

output_file = 'strategy_backtest_results.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Daily data
    df_merged.to_excel(writer, sheet_name='Daily_Data', index=False)
    # Performance metrics
    metrics_df.to_excel(writer, sheet_name='Performance_Metrics')
    # Regime analysis
    regime_analysis.to_excel(writer, sheet_name='Regime_Analysis')

    # Strategy configuration
    config_df = pd.DataFrame([
        {'Regime': k, 'Factor_Weight': v['factor_weight'],
         'Hedge_Ratio': v['hedge_ratio'], 'Description': v['description']}
        for k, v in STRATEGY_CONFIG.items()
    ])
    config_df.to_excel(writer, sheet_name='Strategy_Config', index=False)

    # Individual factor performance
    factor_perf = pd.DataFrame()
    for factor in factor_columns:
        return_col = f'{factor}_Return'
        ann_ret = df_merged[return_col].mean() * 252 * 100
        ann_vol = df_merged[return_col].std() * np.sqrt(252) * 100
        sharpe = (
            df_merged[return_col].mean() / df_merged[return_col].std() * np.sqrt(252)
            if df_merged[return_col].std() > 0 else 0
        )
        factor_perf[factor] = [ann_ret, ann_vol, sharpe]

    factor_perf.index = ['Annual Return (%)', 'Annual Volatility (%)', 'Sharpe Ratio']
    factor_perf.to_excel(writer, sheet_name='Factor_Performance')

print(f"\nResults saved to: {output_file}")

# ==================== 11. VISUALIZATIONS ====================
print("\n" + "="*80)
print("STEP 11: Create Visualizations")
print("="*80)

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Plot 1: Cumulative Returns
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_merged['Date'], (df_merged['Strategy_Cumulative'] - 1) * 100,
         label='Strategy', linewidth=2, color='blue')
ax1.plot(df_merged['Date'], (df_merged['Benchmark_Cumulative'] - 1) * 100,
         label='Benchmark (Buy & Hold)', linewidth=2, color='gray', alpha=0.7)
ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Drawdown
ax2 = fig.add_subplot(gs[1, :])
strategy_dd = (df_merged['Strategy_Cumulative'] /
               df_merged['Strategy_Cumulative'].expanding().max() - 1) * 100
benchmark_dd = (df_merged['Benchmark_Cumulative'] /
                df_merged['Benchmark_Cumulative'].expanding().max() - 1) * 100
ax2.fill_between(df_merged['Date'], strategy_dd, 0, alpha=0.3, color='blue', label='Strategy')
ax2.fill_between(df_merged['Date'], benchmark_dd, 0, alpha=0.3, color='gray', label='Benchmark')
ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Rolling Sharpe Ratio
ax3 = fig.add_subplot(gs[2, 0])
rolling_window = 252
strategy_rolling_sharpe = (
    df_merged['Strategy_Returns'].rolling(rolling_window).mean() /
    df_merged['Strategy_Returns'].rolling(rolling_window).std() * np.sqrt(252)
)
benchmark_rolling_sharpe = (
    df_merged['Benchmark_Returns'].rolling(rolling_window).mean() /
    df_merged['Benchmark_Returns'].rolling(rolling_window).std() * np.sqrt(252)
)
ax3.plot(df_merged['Date'], strategy_rolling_sharpe, label='Strategy', linewidth=1.5, color='blue')
ax3.plot(df_merged['Date'], benchmark_rolling_sharpe, label='Benchmark',
         linewidth=1.5, color='gray', alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
ax3.set_title('Rolling Sharpe Ratio (252-day)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Sharpe Ratio', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Factor Weight Time Series by Regime
ax4 = fig.add_subplot(gs[2, 1])
colors_map = {'Risk-Off': 'red', 'Neutral': 'yellow', 'Risk-On': 'green'}
for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
    mask = df_merged['Regime'] == regime
    ax4.scatter(df_merged.loc[mask, 'Date'], df_merged.loc[mask, 'Factor_Weight'],
                c=colors_map[regime], label=regime, s=2, alpha=0.5)
ax4.set_title('Factor Weight by Regime', fontsize=12, fontweight='bold')
ax4.set_ylabel('Factor Weight', fontsize=10)
ax4.set_ylim(-0.1, 1.1)
ax4.legend(fontsize=8, markerscale=3)
ax4.grid(True, alpha=0.3)

# Plot 5: Monthly Returns Distribution
ax5 = fig.add_subplot(gs[3, 0])
strategy_monthly = df_merged.set_index('Date')['Strategy_Returns'].resample('M').sum() * 100
benchmark_monthly = df_merged.set_index('Date')['Benchmark_Returns'].resample('M').sum() * 100
ax5.hist(strategy_monthly, bins=30, alpha=0.5, label='Strategy', color='blue')
ax5.hist(benchmark_monthly, bins=30, alpha=0.5, label='Benchmark', color='gray')
ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax5.set_title('Monthly Returns Distribution', fontsize=12, fontweight='bold')
ax5.set_xlabel('Monthly Return (%)', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: Regime Distribution
ax6 = fig.add_subplot(gs[3, 1])
regime_counts = df_merged['Regime'].value_counts()
colors = [colors_map[r] for r in regime_counts.index]
bars = ax6.bar(regime_counts.index, regime_counts.values,
               color=colors, alpha=0.7, edgecolor='black')
ax6.set_title('Regime Distribution', fontsize=12, fontweight='bold')
ax6.set_ylabel('Days', fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, regime_counts.values):
    height = bar.get_height()
    pct = count / len(df_merged) * 100
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(count)}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.savefig('strategy_backtest_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: strategy_backtest_analysis.png")

# ==================== 12. SUMMARY ====================
print("\n" + "="*80)
print("EXECUTION SUMMARY")
print("="*80)

print(f"""
Analysis Period: {df_merged['Date'].min().strftime('%Y-%m-%d')} to {df_merged['Date'].max().strftime('%Y-%m-%d')}
Total Days: {len(df_merged)}

FACTORS USED: {', '.join(factor_columns)}

REGIME DISTRIBUTION:  Risk-Off:  {(df_merged['Regime']=='Risk-Off').sum()} days ({(df_merged['Regime']=='Risk-Off').sum()/len(df_merged)*100:.1f}%)
  Neutral:   {(df_merged['Regime']=='Neutral').sum()} days ({(df_merged['Regime']=='Neutral').sum()/len(df_merged)*100:.1f}%)
  Risk-On:   {(df_merged['Regime']=='Risk-On').sum()} days ({(df_merged['Regime']=='Risk-On').sum()/len(df_merged)*100:.1f}%)

STRATEGY PERFORMANCE:
  Sharpe Ratio:      {strategy_metrics['Sharpe Ratio']:.4f}
  Annual Return:     {strategy_metrics['Mean Annual Return']*100:.2f}%
  Annual Volatility: {strategy_metrics['Annual Volatility']*100:.2f}%
  Max Drawdown:      {strategy_metrics['Maximum Drawdown']*100:.2f}%
  Calmar Ratio:      {strategy_metrics['Calmar Ratio']:.4f}
  Sortino Ratio:     {strategy_metrics['Sortino Ratio']:.4f}
  Win Rate:          {strategy_metrics['Win Rate']*100:.1f}%

BENCHMARK PERFORMANCE:
  Sharpe Ratio:      {benchmark_metrics['Sharpe Ratio']:.4f}
  Annual Return:     {benchmark_metrics['Mean Annual Return']*100:.2f}%
  Annual Volatility: {benchmark_metrics['Annual Volatility']*100:.2f}%
  Max Drawdown:      {benchmark_metrics['Maximum Drawdown']*100:.2f}%
  Sortino Ratio:     {benchmark_metrics['Sortino Ratio']:.4f}
  Win Rate:          {benchmark_metrics['Win Rate']*100:.1f}%

IMPROVEMENTS:
  Sharpe Ratio:     {(strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']):+.4f}
  Annual Return:    {(strategy_metrics['Mean Annual Return'] - benchmark_metrics['Mean Annual Return'])*100:+.2f}%
  Volatility:       {(strategy_metrics['Annual Volatility'] - benchmark_metrics['Annual Volatility'])*100:+.2f}%
  Max Drawdown:     {(strategy_metrics['Maximum Drawdown'] - benchmark_metrics['Maximum Drawdown'])*100:+.2f}%
  Sortino Ratio:    {(strategy_metrics['Sortino Ratio'] - benchmark_metrics['Sortino Ratio']):+.4f}

KEY INSIGHTS:
  1. Risk-Adjusted Returns: {'IMPROVED' if strategy_metrics['Sharpe Ratio'] > benchmark_metrics['Sharpe Ratio'] else 'DECLINED'} by {abs(strategy_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']):.2f}
  2. Drawdown Protection: {'SUCCESSFUL' if strategy_metrics['Maximum Drawdown'] > benchmark_metrics['Maximum Drawdown'] else 'FAILED'} - reduced by {abs((strategy_metrics['Maximum Drawdown'] - benchmark_metrics['Maximum Drawdown'])*100):.1f}%
  3. Volatility Management: {'EFFECTIVE' if strategy_metrics['Annual Volatility'] < benchmark_metrics['Annual Volatility'] else 'INEFFECTIVE'} - {'reduced' if strategy_metrics['Annual Volatility'] < benchmark_metrics['Annual Volatility'] else 'increased'} by {abs((strategy_metrics['Annual Volatility'] - benchmark_metrics['Annual Volatility'])*100):.1f}%

REGIME EFFECTIVENESS:
  Risk-Off Protection: {(df_merged[df_merged['Regime']=='Risk-Off']['Strategy_Returns'].mean() - df_merged[df_merged['Regime']=='Risk-Off']['Benchmark_Returns'].mean())*252*100:+.2f}% annual outperformance
  Neutral Performance: {(df_merged[df_merged['Regime']=='Neutral']['Strategy_Returns'].mean() - df_merged[df_merged['Regime']=='Neutral']['Benchmark_Returns'].mean())*252*100:+.2f}% annual difference
  Risk-On Capture:     {(df_merged[df_merged['Regime']=='Risk-On']['Strategy_Returns'].mean() / df_merged[df_merged['Regime']=='Risk-On']['Benchmark_Returns'].mean())*100:.1f}% of benchmark returns

FACTOR PORTFOLIO COMPOSITION:
  Number of Factors: {len(factor_columns)}
  Weighting Scheme:  Equal-weighted composite
  Rebalancing:       Daily

STRATEGY PARAMETERS:
  VIX Z-Score Window:      {window} days (1 year)
  Risk-Off Threshold:      Z < {threshold_low}
  Risk-On Threshold:       Z > {threshold_high}
  Risk-Off Allocation:     {STRATEGY_CONFIG['Risk-Off']['factor_weight']*100:.0f}% factors, {STRATEGY_CONFIG['Risk-Off']['hedge_ratio']*100:.0f}% hedge
  Neutral Allocation:      {STRATEGY_CONFIG['Neutral']['factor_weight']*100:.0f}% factors, {STRATEGY_CONFIG['Neutral']['hedge_ratio']*100:.0f}% hedge
  Risk-On Allocation:      {STRATEGY_CONFIG['Risk-On']['factor_weight']*100:.0f}% factors, {STRATEGY_CONFIG['Risk-On']['hedge_ratio']*100:.0f}% hedge

FILES GENERATED:
  1. strategy_backtest_results.xlsx  - Complete backtest data and analysis
  2. strategy_backtest_analysis.png  - Performance visualizations
""")
