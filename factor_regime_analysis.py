import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

"""
Factor Performance by Regime Analysis
Analyzes how each factor performs under different VIX regimes
"""

print("=" * 80)
print("FACTOR PERFORMANCE BY REGIME ANALYSIS")
print("=" * 80)

# ==================== 1. LOAD DATA ====================
print("\n" + "=" * 80)
print("STEP 1: Loading Data")
print("=" * 80)

# Load regime classification results
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

print(f"VIX regime data loaded: {len(df_regime)} days")
print(f"Date range: {df_regime['Date'].min()} to {df_regime['Date'].max()}")

# Load factor data
df_factors_raw = pd.read_excel('dataOnFactors.xlsx', header=None)

# Identify factor groups (each group has its own date and data columns)
# Look for rows with "Dates" keyword
factor_groups = []
for col_idx in range(df_factors_raw.shape[1]):
    col_data = df_factors_raw.iloc[:, col_idx]
    
    # Find "Dates" row
    dates_row = None
    for row_idx in range(min(20, len(col_data))):
        if pd.notna(col_data.iloc[row_idx]) and 'date' in str(col_data.iloc[row_idx]).lower():
            dates_row = row_idx
            break
    
    if dates_row is not None:
        # Find factor name (usually 2-3 rows above Dates)
        factor_name = None
        for lookback in range(2, 6):
            if row_idx - lookback >= 0:
                candidate = df_factors_raw.iloc[dates_row - lookback, col_idx]
                if pd.notna(candidate) and isinstance(candidate, str) and len(candidate) > 2:
                    factor_name = candidate
                    break
        
        if factor_name is None:
            factor_name = f"Factor_{col_idx}"
        
        factor_groups.append({
            'name': factor_name,
            'date_col': col_idx,
            'data_col': col_idx + 1 if col_idx + 1 < df_factors_raw.shape[1] else col_idx,
            'dates_row': dates_row
        })

print(f"\nIdentified {len(factor_groups)} factor groups")

# Extract each factor group
all_factors = {}

for group in factor_groups:
    factor_name = group['name']
    date_col = group['date_col']
    data_col = group['data_col']
    start_row = group['dates_row'] + 1
    
    # Extract dates and values
    dates = df_factors_raw.iloc[start_row:, date_col]
    values = df_factors_raw.iloc[start_row:, data_col]
    
    # Create dataframe
    factor_df = pd.DataFrame({
        'Date': dates,
        factor_name: values
    })
    
    # Convert to datetime and numeric
    factor_df['Date'] = pd.to_datetime(factor_df['Date'], errors='coerce')
    factor_df[factor_name] = pd.to_numeric(factor_df[factor_name], errors='coerce')
    
    # Drop NaN
    factor_df = factor_df.dropna()
    
    if len(factor_df) > 0:
        all_factors[factor_name] = factor_df
        print(f"  {factor_name}: {len(factor_df)} days")

# Merge all factors on Date
if len(all_factors) == 0:
    print("\nError: No factors loaded")
    exit(1)

df_factors = None
for factor_name, factor_df in all_factors.items():
    if df_factors is None:
        df_factors = factor_df
    else:
        df_factors = df_factors.merge(factor_df, on='Date', how='outer')

df_factors = df_factors.sort_values('Date').reset_index(drop=True)
factor_columns = [col for col in df_factors.columns if col != 'Date']

print(f"\nFactor data merged: {len(df_factors)} days")
print(f"Factors: {factor_columns}")

# ==================== 2. MERGE DATA ====================
print("\n" + "=" * 80)
print("STEP 2: Merge Regime and Factor Data")
print("=" * 80)

df_merged = df_regime[['Date', 'VIX_Zscore', 'Regime']].merge(
    df_factors[['Date'] + factor_columns], 
    on='Date', 
    how='inner'
)

print(f"Merged dataset: {len(df_merged)} days")
print(f"Date range: {df_merged['Date'].min()} to {df_merged['Date'].max()}")

# ==================== 3. CALCULATE FACTOR RETURNS ====================
print("\n" + "=" * 80)
print("STEP 3: Calculate Factor Returns")
print("=" * 80)

# Calculate daily returns for each factor
for factor in factor_columns:
    df_merged[f'{factor}_Return'] = df_merged[factor].pct_change()

# Remove first row with NaN returns
df_merged = df_merged.iloc[1:].reset_index(drop=True)

print(f"Factor returns calculated for {len(factor_columns)} factors")
print(f"Analysis period: {len(df_merged)} days")

# ==================== 4. ANALYZE PERFORMANCE BY REGIME ====================
print("\n" + "=" * 80)
print("STEP 4: Analyze Factor Performance by Regime")
print("=" * 80)

# Create results dictionary
results = {}

for factor in factor_columns:
    return_col = f'{factor}_Return'
    
    # Skip if all NaN
    if df_merged[return_col].isna().all():
        continue
    
    results[factor] = {}
    
    for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
        regime_data = df_merged[df_merged['Regime'] == regime][return_col].dropna()
        
        if len(regime_data) == 0:
            continue
        
        # Calculate statistics
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

# Print summary table
print("\n" + "=" * 80)
print("ANNUALIZED RETURNS BY REGIME")
print("=" * 80)

summary_data = []
for factor in results.keys():
    row = {'Factor': factor}
    for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
        if regime in results[factor]:
            row[regime] = results[factor][regime]['mean_annual']
        else:
            row[regime] = np.nan
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)
print("\n" + df_summary.to_string(index=False))

# Calculate average across all factors
print("\n" + "-" * 80)
print("AVERAGE ACROSS ALL FACTORS:")
for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
    avg = df_summary[regime].mean()
    print(f"  {regime}: {avg*100:.2f}%")

# ==================== 5. DETAILED STATISTICS TABLE ====================
print("\n" + "=" * 80)
print("DETAILED STATISTICS")
print("=" * 80)

for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
    print(f"\n{'=' * 80}")
    print(f"REGIME: {regime}")
    print(f"{'=' * 80}")
    
    detailed_data = []
    for factor in results.keys():
        if regime in results[factor]:
            stats = results[factor][regime]
            detailed_data.append({
                'Factor': factor,
                'Days': stats['count'],
                'Annual Return': f"{stats['mean_annual']*100:.2f}%",
                'Annual Vol': f"{stats['std_annual']*100:.2f}%",
                'Sharpe': f"{stats['sharpe']:.3f}",
                'Win Rate': f"{stats['win_rate']*100:.1f}%",
                'Min Daily': f"{stats['min']*100:.2f}%",
                'Max Daily': f"{stats['max']*100:.2f}%"
            })
    
    if detailed_data:
        df_detailed = pd.DataFrame(detailed_data)
        print("\n" + df_detailed.to_string(index=False))

# ==================== 6. IDENTIFY BEST/WORST PERFORMERS ====================
print("\n" + "=" * 80)
print("BEST AND WORST PERFORMERS BY REGIME")
print("=" * 80)

for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
    print(f"\n{regime}:")
    
    regime_returns = []
    for factor in results.keys():
        if regime in results[factor]:
            regime_returns.append({
                'Factor': factor,
                'Annual_Return': results[factor][regime]['mean_annual']
            })
    
    if regime_returns:
        df_regime_returns = pd.DataFrame(regime_returns).sort_values('Annual_Return', ascending=False)
        
        print(f"\n  Top 3 Performers:")
        for idx, row in df_regime_returns.head(3).iterrows():
            print(f"    {row['Factor']}: {row['Annual_Return']*100:.2f}%")
        
        print(f"\n  Bottom 3 Performers:")
        for idx, row in df_regime_returns.tail(3).iterrows():
            print(f"    {row['Factor']}: {row['Annual_Return']*100:.2f}%")

# ==================== 7. CORRELATION ANALYSIS ====================
print("\n" + "=" * 80)
print("STEP 5: Factor Correlation with VIX Z-Score")
print("=" * 80)

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
print("\n" + df_corr.to_string(index=False))

print("\nInterpretation:")
print("  Positive correlation: Factor performs better when Z-score is higher (Risk-On)")
print("  Negative correlation: Factor performs worse when Z-score is higher (suffers in stress)")

# ==================== 8. VISUALIZATIONS ====================
print("\n" + "=" * 80)
print("STEP 6: Create Visualizations")
print("=" * 80)

# Set up the plot
fig = plt.figure(figsize=(20, 16))

# Plot 1: Heatmap of Annual Returns
ax1 = plt.subplot(3, 2, 1)
heatmap_data = df_summary.set_index('Factor')[['Risk-Off', 'Neutral', 'Risk-On']] * 100
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
    values = df_summary[regime] * 100
    ax2.bar(x + i*width, values, width, label=regime, 
            alpha=0.8)

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
    
    # Add significance markers
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
print("\n✓ Visualization saved: factor_regime_analysis.png")

# ==================== 9. SAVE DETAILED RESULTS ====================
print("\n" + "=" * 80)
print("STEP 7: Save Detailed Results")
print("=" * 80)

output_file = 'factor_regime_analysis.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Summary table
    df_summary.to_excel(writer, sheet_name='Annual_Returns', index=False)
    
    # Detailed statistics for each regime
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
    
    # Correlations
    df_corr.to_excel(writer, sheet_name='VIX_Correlations', index=False)
    
    # Sharpe ratios
    df_sharpe.to_excel(writer, sheet_name='Sharpe_Ratios', index=False)
    
    # Win rates
    df_winrate.to_excel(writer, sheet_name='Win_Rates', index=False)
    
    # Volatility
    df_vol.to_excel(writer, sheet_name='Volatility', index=False)

print(f"\n✓ Detailed results saved: {output_file}")

# ==================== 10. KEY INSIGHTS ====================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

# Find factors most hurt by Risk-Off
print("\nFactors Most Hurt by Risk-Off:")
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
print(f"\nTop 3 factors with biggest Risk-On vs Risk-Off spread:")
for idx, row in df_riskoff.head(3).iterrows():
    print(f"  {row['Factor']}: Spread = {row['Spread']:.1f}% "
          f"(Risk-Off: {row['Risk-Off_Return']:.1f}%, Risk-On: {row['Risk-On_Return']:.1f}%)")

# Find most consistent factors
print("\nMost Consistent Factors (lowest volatility in Risk-Off):")
vol_riskoff = []
for factor in results.keys():
    if 'Risk-Off' in results[factor]:
        vol_riskoff.append({
            'Factor': factor,
            'Vol': results[factor]['Risk-Off']['std_annual'] * 100
        })

df_vol_riskoff = pd.DataFrame(vol_riskoff).sort_values('Vol')
print(f"\nTop 3 most stable factors in Risk-Off:")
for idx, row in df_vol_riskoff.head(3).iterrows():
    print(f"  {row['Factor']}: {row['Vol']:.1f}% annual volatility")

# Hedging recommendations
print("\n" + "=" * 80)
print("HEDGING RECOMMENDATIONS")
print("=" * 80)

print("\nBased on the analysis:")

# Factors to definitely hedge
print("\n1. MUST HEDGE (negative returns in Risk-Off):")
must_hedge = df_riskoff[df_riskoff['Risk-Off_Return'] < 0].head(5)
for idx, row in must_hedge.iterrows():
    print(f"   - {row['Factor']}: {row['Risk-Off_Return']:.1f}% in Risk-Off")

# Factors that might not need hedging
print("\n2. OPTIONAL HEDGE (positive or neutral in Risk-Off):")
optional_hedge = df_riskoff[df_riskoff['Risk-Off_Return'] >= 0]
for idx, row in optional_hedge.iterrows():
    print(f"   - {row['Factor']}: {row['Risk-Off_Return']:.1f}% in Risk-Off (defensive)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"""
Files Generated:
  1. factor_regime_analysis.png - Comprehensive visualizations
  2. factor_regime_analysis.xlsx - Detailed statistics

Summary:
  - {len(results)} factors analyzed
  - {len(df_merged)} days of data
  - 3 regimes: Risk-Off, Neutral, Risk-On
  
Next Steps:
  1. Review which factors need hedging most
  2. Consider selective hedging strategy
  3. Optimize factor weights by regime
""")
