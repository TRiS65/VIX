import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ==================== LOAD DATA ====================
# Load VIX regime data
df_regime = pd.read_excel('data.xlsx', header=1)
df_regime['Date'] = pd.to_datetime(df_regime['Date'])
df_regime = df_regime.sort_values('Date').reset_index(drop=True)

# Calculate VIX metrics
df_regime['VX1_Price'] = pd.to_numeric(df_regime['VX1'], errors='coerce')
df_regime['VX2_Price'] = pd.to_numeric(df_regime['VX2'], errors='coerce')
df_regime['VIX_Slope'] = (df_regime['VX2_Price'] / df_regime['VX1_Price']) - 1

# Calculate Z-score
window = 252
df_regime['VIX_Slope_Mean'] = df_regime['VIX_Slope'].rolling(window=window, min_periods=60).mean()
df_regime['VIX_Slope_Std'] = df_regime['VIX_Slope'].rolling(window=window, min_periods=60).std()
df_regime['VIX_Zscore'] = (df_regime['VIX_Slope'] - df_regime['VIX_Slope_Mean']) / df_regime['VIX_Slope_Std']

# Classify regime
threshold_low = -0.5
threshold_high = 0.5

def classify_regime(z_score):
    if pd.isna(z_score):
        return 'Unknown'
    if z_score < threshold_low:
        return 'Risk-Off'
    elif z_score > threshold_high:
        return 'Risk-On'
    else:
        return 'Neutral'

df_regime['Regime'] = df_regime['VIX_Zscore'].apply(classify_regime)
df_regime = df_regime[df_regime['Regime'] != 'Unknown'].reset_index(drop=True)

print(f"VIX Data: {len(df_regime)} days from {df_regime['Date'].min().date()} to {df_regime['Date'].max().date()}")

# Load factor data with proper structure parsing
df_factors_raw = pd.read_excel('dataOnFactors.xlsx', header=None)

# Identify DAILY factor columns only
daily_factor_groups = []

for col_idx in range(df_factors_raw.shape[1]):
    # Check if this column has frequency indicator in row 0
    freq_indicator = df_factors_raw.iloc[0, col_idx]
    
    # Look for "Dates" in row 7 and factor name in row 2
    if col_idx < df_factors_raw.shape[1] - 1:
        dates_marker = df_factors_raw.iloc[7, col_idx]
        factor_name = df_factors_raw.iloc[2, col_idx + 1]  # Factor name is in next column
        
        # Only process DAILY factors
        if pd.notna(dates_marker) and str(dates_marker).lower() == 'dates':
            # Check if this is a DAILY factor group
            is_daily = False
            
            # Look backwards in the row to find DAILY marker
            for check_col in range(max(0, col_idx - 5), col_idx + 1):
                if df_factors_raw.iloc[0, check_col] == 'DAILY':
                    is_daily = True
                    break
            
            # If no DAILY found before, check if WEEKLY is after (then this is DAILY)
            if not is_daily:
                for check_col in range(col_idx, min(df_factors_raw.shape[1], col_idx + 20)):
                    if df_factors_raw.iloc[0, check_col] == 'WEEKLY':
                        is_daily = True
                        break
            
            if is_daily and pd.notna(factor_name):
                daily_factor_groups.append({
                    'name': str(factor_name).strip(),
                    'date_col': col_idx,
                    'data_col': col_idx + 1
                })

print(f"\nIdentified {len(daily_factor_groups)} DAILY factors:")
for group in daily_factor_groups:
    print(f"  - {group['name']}")

# Extract DAILY factor data
all_factors = {}

for group in daily_factor_groups:
    factor_name = group['name']
    date_col = group['date_col']
    data_col = group['data_col']
    
    # Data starts from row 8
    dates = df_factors_raw.iloc[8:, date_col]
    values = df_factors_raw.iloc[8:, data_col]
    
    factor_df = pd.DataFrame({
        'Date': dates,
        factor_name: values
    })
    
    factor_df['Date'] = pd.to_datetime(factor_df['Date'], errors='coerce')
    factor_df[factor_name] = pd.to_numeric(factor_df[factor_name], errors='coerce')
    factor_df = factor_df.dropna()
    
    if len(factor_df) > 0:
        all_factors[factor_name] = factor_df
        print(f"    {factor_name}: {len(factor_df)} days")

# Merge all factors
df_factors = None
for factor_name, factor_df in all_factors.items():
    if df_factors is None:
        df_factors = factor_df
    else:
        df_factors = df_factors.merge(factor_df, on='Date', how='outer')

df_factors = df_factors.sort_values('Date').reset_index(drop=True)
factor_columns = [col for col in df_factors.columns if col != 'Date']

print(f"\nFactor Data: {len(df_factors)} days, {len(factor_columns)} factors")
print(f"Factors: {', '.join(factor_columns)}")

# ==================== MERGE AND CALCULATE RETURNS ====================
df_merged = df_regime[['Date', 'VIX_Zscore', 'Regime']].merge(
    df_factors[['Date'] + factor_columns], 
    on='Date', 
    how='inner'
)

# Calculate daily returns
for factor in factor_columns:
    df_merged[f'{factor}_Return'] = df_merged[factor].pct_change()

df_merged = df_merged.iloc[1:].reset_index(drop=True)

print(f"Analysis Period: {len(df_merged)} days from {df_merged['Date'].min().date()} to {df_merged['Date'].max().date()}\n")

# ==================== ANALYZE PERFORMANCE BY REGIME ====================
results = {}

for factor in factor_columns:
    return_col = f'{factor}_Return'
    
    if df_merged[return_col].isna().all():
        continue
    
    results[factor] = {}
    
    for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
        regime_data = df_merged[df_merged['Regime'] == regime][return_col].dropna()
        
        if len(regime_data) == 0:
            continue
        
        stats_dict = {
            'count': len(regime_data),
            'mean_daily': regime_data.mean(),
            'std_daily': regime_data.std(),
            'mean_annual': regime_data.mean() * 252,
            'std_annual': regime_data.std() * np.sqrt(252),
            'sharpe': (regime_data.mean() / regime_data.std() * np.sqrt(252)) if regime_data.std() > 0 else 0,
            'min': regime_data.min(),
            'max': regime_data.max(),
            'median': regime_data.median(),
            'skew': regime_data.skew(),
            'win_rate': (regime_data > 0).sum() / len(regime_data)
        }
        
        results[factor][regime] = stats_dict

# Print annualized returns
print("ANNUALIZED RETURNS BY REGIME (%)\n")
summary_data = []
for factor in results.keys():
    row = {'Factor': factor}
    for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
        if regime in results[factor]:
            row[regime] = results[factor][regime]['mean_annual'] * 100
        else:
            row[regime] = np.nan
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

print(f"\nAverage Across All Factors:")
for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
    avg = df_summary[regime].mean()
    print(f"  {regime}: {avg:.2f}%")

# Print detailed statistics
for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
    print(f"\n\nDETAILED STATISTICS - {regime}\n")
    
    detailed_data = []
    for factor in results.keys():
        if regime in results[factor]:
            stats = results[factor][regime]
            detailed_data.append({
                'Factor': factor,
                'Days': stats['count'],
                'Annual Return (%)': f"{stats['mean_annual']*100:.2f}",
                'Annual Vol (%)': f"{stats['std_annual']*100:.2f}",
                'Sharpe': f"{stats['sharpe']:.3f}",
                'Win Rate (%)': f"{stats['win_rate']*100:.1f}",
                'Min Daily (%)': f"{stats['min']*100:.2f}",
                'Max Daily (%)': f"{stats['max']*100:.2f}"
            })
    
    if detailed_data:
        df_detailed = pd.DataFrame(detailed_data)
        print(df_detailed.to_string(index=False))

# Print best/worst performers
print("\n\nBEST AND WORST PERFORMERS BY REGIME\n")

for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
    regime_returns = []
    for factor in results.keys():
        if regime in results[factor]:
            regime_returns.append({
                'Factor': factor,
                'Annual_Return': results[factor][regime]['mean_annual']
            })
    
    if regime_returns:
        df_regime_returns = pd.DataFrame(regime_returns).sort_values('Annual_Return', ascending=False)
        
        print(f"{regime}:")
        print(f"  Best Performer:")
        for idx, row in df_regime_returns.head(1).iterrows():
            print(f"    {row['Factor']}: {row['Annual_Return']*100:.2f}%")
        
        print(f"  Worst Performer:")
        for idx, row in df_regime_returns.tail(1).iterrows():
            print(f"    {row['Factor']}: {row['Annual_Return']*100:.2f}%")
        print()

# ==================== CORRELATION ANALYSIS ====================
print("\nFACTOR CORRELATION WITH VIX Z-SCORE\n")

correlations = []
for factor in factor_columns:
    return_col = f'{factor}_Return'
    data = df_merged[[return_col, 'VIX_Zscore']].dropna()
    
    if len(data) > 30:
        corr, p_value = pearsonr(data[return_col], data['VIX_Zscore'])
        correlations.append({
            'Factor': factor,
            'Correlation': corr,
            'P-Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

df_corr = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
print(df_corr.to_string(index=False))

print("\nInterpretation:")
print("  Positive correlation: Factor performs better in Risk-On conditions")
print("  Negative correlation: Factor suffers during market stress (needs hedging)")

# ==================== VISUALIZATIONS ====================
fig = plt.figure(figsize=(20, 16))

# Plot 1: Heatmap of Annual Returns
ax1 = plt.subplot(3, 2, 1)
heatmap_data = df_summary.set_index('Factor')[['Risk-Off', 'Neutral', 'Risk-On']]
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Annual Return (%)'}, ax=ax1,
            vmin=-50, vmax=50)
ax1.set_title('Factor Returns by Regime (Annual %)', fontsize=14, fontweight='bold')
ax1.set_xlabel('')
ax1.set_ylabel('')

# Plot 2: Bar chart comparison
ax2 = plt.subplot(3, 2, 2)
x = np.arange(len(df_summary))
width = 0.25

for i, regime in enumerate(['Risk-Off', 'Neutral', 'Risk-On']):
    values = df_summary[regime]
    ax2.bar(x + i*width, values, width, label=regime, alpha=0.8)

ax2.set_xlabel('Factor', fontsize=12)
ax2.set_ylabel('Annual Return (%)', fontsize=12)
ax2.set_title('Factor Returns Comparison Across Regimes', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels(df_summary['Factor'], rotation=45, ha='right')
ax2.legend()
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Sharpe Ratio comparison
ax3 = plt.subplot(3, 2, 3)
sharpe_data = []
for factor in results.keys():
    row = {'Factor': factor}
    for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
        if regime in results[factor]:
            row[regime] = results[factor][regime]['sharpe']
        else:
            row[regime] = np.nan
    sharpe_data.append(row)

df_sharpe = pd.DataFrame(sharpe_data)
heatmap_sharpe = df_sharpe.set_index('Factor')[['Risk-Off', 'Neutral', 'Risk-On']]
sns.heatmap(heatmap_sharpe, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Sharpe Ratio'}, ax=ax3,
            vmin=-2, vmax=2)
ax3.set_title('Factor Sharpe Ratios by Regime', fontsize=14, fontweight='bold')
ax3.set_xlabel('')
ax3.set_ylabel('')

# Plot 4: Win Rate comparison
ax4 = plt.subplot(3, 2, 4)
winrate_data = []
for factor in results.keys():
    row = {'Factor': factor}
    for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
        if regime in results[factor]:
            row[regime] = results[factor][regime]['win_rate'] * 100
        else:
            row[regime] = np.nan
    winrate_data.append(row)

df_winrate = pd.DataFrame(winrate_data)
heatmap_winrate = df_winrate.set_index('Factor')[['Risk-Off', 'Neutral', 'Risk-On']]
sns.heatmap(heatmap_winrate, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
            cbar_kws={'label': 'Win Rate (%)'}, ax=ax4,
            vmin=30, vmax=70)
ax4.set_title('Factor Win Rates by Regime (%)', fontsize=14, fontweight='bold')
ax4.set_xlabel('')
ax4.set_ylabel('')

# Plot 5: Volatility comparison
ax5 = plt.subplot(3, 2, 5)
vol_data = []
for factor in results.keys():
    row = {'Factor': factor}
    for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
        if regime in results[factor]:
            row[regime] = results[factor][regime]['std_annual'] * 100
        else:
            row[regime] = np.nan
    vol_data.append(row)

df_vol = pd.DataFrame(vol_data)
x = np.arange(len(df_vol))
width = 0.25

for i, regime in enumerate(['Risk-Off', 'Neutral', 'Risk-On']):
    values = df_vol[regime]
    ax5.bar(x + i*width, values, width, label=regime, alpha=0.8)

ax5.set_xlabel('Factor', fontsize=12)
ax5.set_ylabel('Annual Volatility (%)', fontsize=12)
ax5.set_title('Factor Volatility Across Regimes', fontsize=14, fontweight='bold')
ax5.set_xticks(x + width)
ax5.set_xticklabels(df_vol['Factor'], rotation=45, ha='right')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Correlation with VIX Z-score
ax6 = plt.subplot(3, 2, 6)
if not df_corr.empty:
    colors = ['red' if x < 0 else 'green' for x in df_corr['Correlation']]
    bars = ax6.barh(df_corr['Factor'], df_corr['Correlation'], color=colors, alpha=0.6)
    
    for i, (idx, row) in enumerate(df_corr.iterrows()):
        if row['Significant'] == 'Yes':
            ax6.text(row['Correlation'], i, ' *', ha='left' if row['Correlation'] > 0 else 'right',
                    va='center', fontsize=16, fontweight='bold')
    
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax6.set_xlabel('Correlation with VIX Z-Score', fontsize=12)
    ax6.set_title('Factor-VIX Correlation (* = significant)', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.text(0.02, 0.98, 'Negative = Suffers in stress\nPositive = Benefits in stress',
            transform=ax6.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('factor_regime_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved: factor_regime_analysis.png")

# ==================== SAVE RESULTS ====================
output_file = 'factor_regime_analysis.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='Annual_Returns', index=False)
    
    for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
        detailed_data = []
        for factor in results.keys():
            if regime in results[factor]:
                stats = results[factor][regime]
                detailed_data.append({
                    'Factor': factor,
                    'Days': stats['count'],
                    'Mean_Daily': stats['mean_daily'],
                    'Std_Daily': stats['std_daily'],
                    'Mean_Annual': stats['mean_annual'],
                    'Std_Annual': stats['std_annual'],
                    'Sharpe': stats['sharpe'],
                    'Win_Rate': stats['win_rate'],
                    'Min_Daily': stats['min'],
                    'Max_Daily': stats['max'],
                    'Median': stats['median'],
                    'Skew': stats['skew']
                })
        
        if detailed_data:
            df_regime_detail = pd.DataFrame(detailed_data)
            df_regime_detail.to_excel(writer, sheet_name=f'{regime}_Stats', index=False)
    
    df_corr.to_excel(writer, sheet_name='VIX_Correlations', index=False)
    df_sharpe.to_excel(writer, sheet_name='Sharpe_Ratios', index=False)
    df_winrate.to_excel(writer, sheet_name='Win_Rates', index=False)
    df_vol.to_excel(writer, sheet_name='Volatility', index=False)

print(f"Results saved: {output_file}")

# ==================== KEY INSIGHTS ====================
print("\n\nKEY INSIGHTS\n")

# Factors most hurt by Risk-Off
riskoff_data = []
for factor in results.keys():
    if 'Risk-Off' in results[factor] and 'Risk-On' in results[factor]:
        riskoff_return = results[factor]['Risk-Off']['mean_annual']
        riskon_return = results[factor]['Risk-On']['mean_annual']
        spread = riskon_return - riskoff_return
        riskoff_data.append({
            'Factor': factor,
            'Risk-Off_Return': riskoff_return * 100,
            'Risk-On_Return': riskon_return * 100,
            'Spread': spread * 100
        })

df_riskoff = pd.DataFrame(riskoff_data).sort_values('Spread', ascending=False)

print("Factors with Biggest Performance Difference (Risk-On vs Risk-Off):")
for idx, row in df_riskoff.iterrows():
    print(f"  {row['Factor']}: Spread = {row['Spread']:.1f}% "
          f"(Risk-Off: {row['Risk-Off_Return']:.1f}%, Risk-On: {row['Risk-On_Return']:.1f}%)")

# Most consistent factors
vol_riskoff = []
for factor in results.keys():
    if 'Risk-Off' in results[factor]:
        vol_riskoff.append({
            'Factor': factor,
            'Vol': results[factor]['Risk-Off']['std_annual'] * 100
        })

df_vol_riskoff = pd.DataFrame(vol_riskoff).sort_values('Vol')

print("\nMost Stable Factors in Risk-Off (by volatility):")
for idx, row in df_vol_riskoff.iterrows():
    print(f"  {row['Factor']}: {row['Vol']:.1f}% annual volatility")

# Hedging recommendations
print("\n\nHEDGING RECOMMENDATIONS\n")

print("Must Hedge (negative returns in Risk-Off):")
must_hedge = df_riskoff[df_riskoff['Risk-Off_Return'] < 0]
if len(must_hedge) > 0:
    for idx, row in must_hedge.iterrows():
        print(f"  {row['Factor']}: {row['Risk-Off_Return']:.1f}% in Risk-Off")
else:
    print("  None - all factors positive in Risk-Off")

print("\nNo Hedging Needed (positive returns in Risk-Off):")
optional_hedge = df_riskoff[df_riskoff['Risk-Off_Return'] >= 0]
if len(optional_hedge) > 0:
    for idx, row in optional_hedge.iterrows():
        print(f"  {row['Factor']}: {row['Risk-Off_Return']:.1f}% in Risk-Off (defensive)")
else:
    print("  None - all factors negative in Risk-Off")

print(f"\n\nAnalysis complete - {len(results)} DAILY factors analyzed over {len(df_merged)} days")
