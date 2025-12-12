import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

"""
Factor Strategy Guardian - Regime Classification Analysis
VIX Term-Structure Based Market State Classification
"""

# ==================== 1. DATA LOADING ====================
print("="*80)
print("REGIME CLASSIFICATION ANALYSIS")
print("="*80)

# Load VIX futures data
df_vix = pd.read_excel('data.xlsx', header=1)
df_vix.columns = df_vix.columns.str.strip()
df_vix['Date'] = pd.to_datetime(df_vix['Date'])
df_vix = df_vix.sort_values('Date').reset_index(drop=True)
df_vix['VX1_Price'] = pd.to_numeric(df_vix['VX1'], errors='coerce')
df_vix['VX2_Price'] = pd.to_numeric(df_vix['VX2'], errors='coerce')

# Load factor data
df_factors_raw = pd.read_excel('dataOnFactors.xlsx', header=None)
df_factors = None
for i in range(10):
    if df_factors_raw.iloc[i].astype(str).str.contains('Dates|Date').any():
        try:
            df_factors = df_factors_raw.iloc[i+1:].copy()
            df_factors.columns = df_factors_raw.iloc[i]
            df_factors = df_factors.reset_index(drop=True)
            date_cols = [col for col in df_factors.columns if 'Date' in str(col)]
            if date_cols:
                df_factors['Date'] = pd.to_datetime(df_factors[date_cols[0]], errors='coerce')
                df_factors = df_factors.dropna(subset=['Date'])
                df_factors = df_factors.sort_values('Date').reset_index(drop=True)
                break
        except:
            continue

if df_factors is None:
    df_factors = pd.DataFrame()

# Load risk-free rate
df_rf_raw = pd.read_excel('rf_3m_tbill_daily.xlsx')
df_rf_split = df_rf_raw.iloc[:, 0].str.split(',', expand=True)
df_rf_split.columns = ['Date', 'RF_3M', 'rf_daily']
df_rf_split['Date'] = pd.to_datetime(df_rf_split['Date'])
df_rf_split['rf_daily'] = pd.to_numeric(df_rf_split['rf_daily'])
df_rf_split = df_rf_split.sort_values('Date').reset_index(drop=True)

print(f"\nData loaded: {len(df_vix)} days")

# ==================== 2. CALCULATE VIX TERM STRUCTURE ====================
# Calculate slope and Z-score
window = 252
df_vix['VIX_Slope'] = (df_vix['VX2_Price'] / df_vix['VX1_Price']) - 1
df_vix['VIX_Slope_pct'] = df_vix['VIX_Slope'] * 100
df_vix['VIX_Slope_Mean'] = df_vix['VIX_Slope'].rolling(window=window, min_periods=60).mean()
df_vix['VIX_Slope_Std'] = df_vix['VIX_Slope'].rolling(window=window, min_periods=60).std()
df_vix['VIX_Zscore'] = (df_vix['VIX_Slope'] - df_vix['VIX_Slope_Mean']) / df_vix['VIX_Slope_Std']
df_vix = df_vix.dropna(subset=['VIX_Zscore']).reset_index(drop=True)

print(f"\nVIX Slope Statistics:")
print(f"  Mean: {df_vix['VIX_Slope_pct'].mean():.3f}%")
print(f"  Std:  {df_vix['VIX_Slope_pct'].std():.3f}%")
print(f"\nVIX Z-Score Statistics:")
print(f"  Mean: {df_vix['VIX_Zscore'].mean():.3f}")
print(f"  Std:  {df_vix['VIX_Zscore'].std():.3f}")

# ==================== 3. DEFINE MARKET REGIMES ====================
# Method 1: Fixed Z-score thresholds (RECOMMENDED)
threshold_low = -0.5
threshold_high = 0.5

def classify_regime_fixed(z_score):
    if z_score < threshold_low:
        return 'Risk-Off'
    elif z_score > threshold_high:
        return 'Risk-On'
    else:
        return 'Neutral'

df_vix['Regime_Fixed'] = df_vix['VIX_Zscore'].apply(classify_regime_fixed)

# Method 2: Percentile-based thresholds
percentile_33 = df_vix['VIX_Zscore'].quantile(0.33)
percentile_67 = df_vix['VIX_Zscore'].quantile(0.67)

def classify_regime_percentile(z_score):
    if z_score < percentile_33:
        return 'Risk-Off'
    elif z_score > percentile_67:
        return 'Risk-On'
    else:
        return 'Neutral'

df_vix['Regime_Percentile'] = df_vix['VIX_Zscore'].apply(classify_regime_percentile)

# Method 3: Absolute slope thresholds
slope_low = -0.02
slope_high = 0.02

def classify_regime_slope(slope):
    if slope < slope_low:
        return 'Risk-Off'
    elif slope > slope_high:
        return 'Risk-On'
    else:
        return 'Neutral'

df_vix['Regime_Slope'] = df_vix['VIX_Slope'].apply(classify_regime_slope)

# Print classification results
print("\n" + "="*80)
print("REGIME CLASSIFICATION METHODS")
print("="*80)

print("\nMethod 1 - Fixed Z-score Thresholds (Low=-0.5, High=0.5):")
print(df_vix['Regime_Fixed'].value_counts().sort_index())
print(f"Percentages:\n{(df_vix['Regime_Fixed'].value_counts(normalize=True).sort_index() * 100).round(1)}")

print(f"\nMethod 2 - Percentile Thresholds (33%, 67%):")
print(f"  33rd percentile: {percentile_33:.3f}")
print(f"  67th percentile: {percentile_67:.3f}")
print(df_vix['Regime_Percentile'].value_counts().sort_index())

print(f"\nMethod 3 - Absolute Slope Thresholds (-2%, +2%):")
print(df_vix['Regime_Slope'].value_counts().sort_index())

# ==================== 4. MERGE ALL DATA ====================
df_merged = df_vix[['Date', 'VX1_Price', 'VX2_Price', 'VIX_Slope', 'VIX_Slope_pct', 
                     'VIX_Zscore', 'Regime_Fixed', 'Regime_Percentile', 'Regime_Slope']].copy()
df_merged = df_merged.merge(df_rf_split[['Date', 'rf_daily']], on='Date', how='left')

# Extract factor columns
factor_cols = []
if not df_factors.empty and 'Date' in df_factors.columns:
    for col in df_factors.columns:
        if col not in ['Date', 'Dates'] and col is not None:
            try:
                test_series = pd.to_numeric(df_factors[col], errors='coerce')
                if not test_series.isna().all():
                    factor_cols.append(col)
            except:
                pass
    
    if factor_cols:
        df_factors_clean = df_factors[['Date'] + factor_cols].copy()
        for col in factor_cols:
            df_factors_clean[col] = pd.to_numeric(df_factors_clean[col], errors='coerce')
        df_merged = df_merged.merge(df_factors_clean, on='Date', how='left')

print(f"\nFinal dataset: {df_merged.shape[0]} days, {df_merged.shape[1]} columns")
print(f"Date range: {df_merged['Date'].min().strftime('%Y-%m-%d')} to {df_merged['Date'].max().strftime('%Y-%m-%d')}")

# ==================== 5. REGIME CHARACTERISTICS ====================
print("\n" + "="*80)
print("REGIME CHARACTERISTICS")
print("="*80)

regime_stats = df_merged.groupby('Regime_Fixed').agg({
    'VIX_Zscore': ['count', 'mean', 'std', 'min', 'max'],
    'VIX_Slope_pct': ['mean', 'std', 'min', 'max'],
    'VX1_Price': ['mean', 'std']
}).round(3)

print("\nRegime Statistics:")
print(regime_stats)

# Calculate regime transitions
df_merged['Regime_Change'] = df_merged['Regime_Fixed'] != df_merged['Regime_Fixed'].shift(1)
transitions = df_merged[df_merged['Regime_Change']].copy()
print(f"\nTotal regime transitions: {len(transitions)}")

# Calculate average regime duration
regime_durations = []
current_regime = df_merged['Regime_Fixed'].iloc[0]
duration = 1

for i in range(1, len(df_merged)):
    if df_merged['Regime_Fixed'].iloc[i] == current_regime:
        duration += 1
    else:
        regime_durations.append({'Regime': current_regime, 'Duration': duration})
        current_regime = df_merged['Regime_Fixed'].iloc[i]
        duration = 1
regime_durations.append({'Regime': current_regime, 'Duration': duration})

df_durations = pd.DataFrame(regime_durations)
print("\nAverage Regime Duration (days):")
print(df_durations.groupby('Regime')['Duration'].agg(['mean', 'median', 'max']).round(1))

# ==================== 6. SAVE RESULTS ====================
output_file = 'regime_classification_results.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_merged.to_excel(writer, sheet_name='Full_Data', index=False)
    regime_stats.to_excel(writer, sheet_name='Regime_Stats')
    df_durations.to_excel(writer, sheet_name='Regime_Durations', index=False)
    
    params = pd.DataFrame({
        'Parameter': ['Window_Days', 'Z_Threshold_Low', 'Z_Threshold_High', 
                      'Percentile_33', 'Percentile_67', 'Slope_Low', 'Slope_High'],
        'Value': [window, threshold_low, threshold_high, 
                  percentile_33, percentile_67, slope_low, slope_high]
    })
    params.to_excel(writer, sheet_name='Parameters', index=False)

print(f"\nResults saved to: {output_file}")

# ==================== 7. VISUALIZATION ====================
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# Plot 1: VIX Slope time series
ax1 = axes[0]
ax1.plot(df_merged['Date'], df_merged['VIX_Slope_pct'], linewidth=0.8, color='black', alpha=0.7)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Risk-On threshold (+2%)')
ax1.axhline(y=-2, color='red', linestyle='--', alpha=0.5, label='Risk-Off threshold (-2%)')
ax1.set_title('VIX Term Structure Slope (VX2/VX1 - 1)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Slope (%)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: VIX Z-Score time series
ax2 = axes[1]
ax2.plot(df_merged['Date'], df_merged['VIX_Zscore'], linewidth=0.8, color='blue')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Risk-On threshold (0.5)')
ax2.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Risk-Off threshold (-0.5)')
ax2.fill_between(df_merged['Date'], -0.5, 0.5, alpha=0.1, color='gray', label='Neutral Zone')
ax2.set_title('VIX Slope Z-Score (252-day rolling)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Z-Score', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Regime classification (color-coded)
ax3 = axes[2]
regime_colors = {'Risk-Off': '#D32F2F', 'Neutral': '#FFA726', 'Risk-On': '#388E3C'}
for regime in ['Risk-Off', 'Neutral', 'Risk-On']:
    mask = df_merged['Regime_Fixed'] == regime
    ax3.scatter(df_merged.loc[mask, 'Date'], df_merged.loc[mask, 'VIX_Zscore'], 
               c=regime_colors[regime], label=regime, s=2, alpha=0.6)
ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
ax3.axhline(y=-0.5, color='black', linestyle='--', alpha=0.3)
ax3.set_title('Market Regimes (Fixed Threshold Method)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Z-Score', fontsize=12)
ax3.legend(markerscale=5)
ax3.grid(True, alpha=0.3)

# Plot 4: Regime distribution
ax4 = axes[3]
regime_counts = df_merged['Regime_Fixed'].value_counts().sort_index()
colors_bar = [regime_colors[regime] for regime in regime_counts.index]
bars = ax4.bar(regime_counts.index, regime_counts.values, color=colors_bar, alpha=0.7, edgecolor='black')
ax4.set_title('Regime Distribution', fontsize=14, fontweight='bold')
ax4.set_ylabel('Days', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, regime_counts.values):
    height = bar.get_height()
    pct = count / len(df_merged) * 100
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('regime_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: regime_analysis.png")


print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
