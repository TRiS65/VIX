import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载和清洗 ====================
print("=" * 60)
print("步骤 1: 加载数据")
print("=" * 60)

# 读取VIX期货数据
df_vix = pd.read_excel('data.xlsx', header=1)
df_vix.columns = df_vix.columns.str.strip()
print(f"\n原始VIX数据形状: {df_vix.shape}")
print(f"列名: {df_vix.columns.tolist()[:10]}")

# 清洗VIX数据
df_vix['Date'] = pd.to_datetime(df_vix['Date'])
df_vix = df_vix.sort_values('Date').reset_index(drop=True)

# 使用已有的VX1和VX2数据
df_vix['VX1_Price'] = pd.to_numeric(df_vix['VX1'], errors='coerce')
df_vix['VX2_Price'] = pd.to_numeric(df_vix['VX2'], errors='coerce')

# 检查是否已经有计算好的slope和z-score
if 'Slope' in df_vix.columns and 'Slope_z' in df_vix.columns:
    df_vix['VIX_Slope_existing'] = pd.to_numeric(df_vix['Slope'], errors='coerce')
    df_vix['VIX_Zscore_existing'] = pd.to_numeric(df_vix['Slope_z'], errors='coerce')
    print("\n✓ 数据中已包含Slope和Slope_z，我们会重新计算以验证")

# 读取因子数据
df_factors_raw = pd.read_excel('dataOnFactors.xlsx', header=None)
print(f"\n原始因子数据形状: {df_factors_raw.shape}")

# 简化因子数据处理 - 找到Dates列
df_factors = None
for i in range(10):  # 检查前10行
    if df_factors_raw.iloc[i].astype(str).str.contains('Dates|Date').any():
        try:
            df_factors = df_factors_raw.iloc[i+1:].copy()
            df_factors.columns = df_factors_raw.iloc[i]
            df_factors = df_factors.reset_index(drop=True)
            
            # 找到日期列
            date_cols = [col for col in df_factors.columns if 'Date' in str(col)]
            if date_cols:
                df_factors['Date'] = pd.to_datetime(df_factors[date_cols[0]], errors='coerce')
                df_factors = df_factors.dropna(subset=['Date'])
                df_factors = df_factors.sort_values('Date').reset_index(drop=True)
                break
        except:
            continue

if df_factors is None:
    print("警告: 无法正确解析因子数据，将继续进行VIX分析")
    df_factors = pd.DataFrame()  # 空DataFrame

# 读取无风险利率数据
df_rf_raw = pd.read_excel('rf_3m_tbill_daily.xlsx')
df_rf_split = df_rf_raw.iloc[:, 0].str.split(',', expand=True)
df_rf_split.columns = ['Date', 'RF_3M', 'rf_daily']
df_rf_split['Date'] = pd.to_datetime(df_rf_split['Date'])
df_rf_split['rf_daily'] = pd.to_numeric(df_rf_split['rf_daily'])
df_rf_split = df_rf_split.sort_values('Date').reset_index(drop=True)

print(f"\n✓ VIX数据: {len(df_vix)} 天")
print(f"✓ 因子数据: {len(df_factors)} 天")
print(f"✓ 无风险利率数据: {len(df_rf_split)} 天")

# ==================== 2. 计算VIX Term Structure ====================
print("\n" + "=" * 60)
print("步骤 2: 计算VIX期限结构")
print("=" * 60)

# 计算slope和Z-score
df_vix['VIX_Slope'] = (df_vix['VX2_Price'] / df_vix['VX1_Price']) - 1
df_vix['VIX_Slope_pct'] = df_vix['VIX_Slope'] * 100

# 计算滚动Z-score (252天窗口)
window = 252
df_vix['VIX_Slope_Mean'] = df_vix['VIX_Slope'].rolling(window=window, min_periods=60).mean()
df_vix['VIX_Slope_Std'] = df_vix['VIX_Slope'].rolling(window=window, min_periods=60).std()
df_vix['VIX_Zscore'] = (df_vix['VIX_Slope'] - df_vix['VIX_Slope_Mean']) / df_vix['VIX_Slope_Std']

# 删除缺失值
df_vix = df_vix.dropna(subset=['VIX_Zscore']).reset_index(drop=True)

print(f"\nVIX Slope 统计:")
print(f"  均值: {df_vix['VIX_Slope_pct'].mean():.3f}%")
print(f"  标准差: {df_vix['VIX_Slope_pct'].std():.3f}%")
print(f"  最小值: {df_vix['VIX_Slope_pct'].min():.3f}%")
print(f"  最大值: {df_vix['VIX_Slope_pct'].max():.3f}%")

print(f"\nVIX Z-Score 统计:")
print(f"  均值: {df_vix['VIX_Zscore'].mean():.3f}")
print(f"  标准差: {df_vix['VIX_Zscore'].std():.3f}")
print(f"  最小值: {df_vix['VIX_Zscore'].min():.3f}")
print(f"  最大值: {df_vix['VIX_Zscore'].max():.3f}")

# ==================== 3. 定义3个风险Regime ====================
print("\n" + "=" * 60)
print("步骤 3: 定义3个风险Regime")
print("=" * 60)

# 方法1: 基于Z-score的固定阈值
threshold_low = -0.5   # 低于此值为Risk-Off
threshold_high = 0.5   # 高于此值为Risk-On

def classify_regime_fixed(z_score):
    """基于固定阈值分类"""
    if z_score < threshold_low:
        return 'Risk-Off'  # 高压力
    elif z_score > threshold_high:
        return 'Risk-On'   # 低压力
    else:
        return 'Neutral'   # 中等

df_vix['Regime_Fixed'] = df_vix['VIX_Zscore'].apply(classify_regime_fixed)

# 方法2: 基于分位数的动态分类（更平衡）
percentile_33 = df_vix['VIX_Zscore'].quantile(0.33)
percentile_67 = df_vix['VIX_Zscore'].quantile(0.67)

def classify_regime_percentile(z_score):
    """基于分位数分类"""
    if z_score < percentile_33:
        return 'Risk-Off'
    elif z_score > percentile_67:
        return 'Risk-On'
    else:
        return 'Neutral'

df_vix['Regime_Percentile'] = df_vix['VIX_Zscore'].apply(classify_regime_percentile)

# 方法3: 基于VIX Slope的绝对值
slope_low = -0.02   # -2%
slope_high = 0.02   # +2%

def classify_regime_slope(slope):
    """基于VIX Slope绝对值分类"""
    if slope < slope_low:
        return 'Risk-Off'  # Backwardation
    elif slope > slope_high:
        return 'Risk-On'   # Strong Contango
    else:
        return 'Neutral'

df_vix['Regime_Slope'] = df_vix['VIX_Slope'].apply(classify_regime_slope)

# 打印分类结果
print("\n方法1 - 固定Z-score阈值 (Low=-0.5, High=0.5):")
print(df_vix['Regime_Fixed'].value_counts().sort_index())
print(f"比例: \n{df_vix['Regime_Fixed'].value_counts(normalize=True).sort_index() * 100}")

print(f"\n方法2 - 分位数阈值 (33%, 67%):")
print(f"  33rd percentile: {percentile_33:.3f}")
print(f"  67th percentile: {percentile_67:.3f}")
print(df_vix['Regime_Percentile'].value_counts().sort_index())
print(f"比例: \n{df_vix['Regime_Percentile'].value_counts(normalize=True).sort_index() * 100}")

print(f"\n方法3 - VIX Slope阈值 (-2%, +2%):")
print(df_vix['Regime_Slope'].value_counts().sort_index())
print(f"比例: \n{df_vix['Regime_Slope'].value_counts(normalize=True).sort_index() * 100}")

# ==================== 4. 合并所有数据 ====================
print("\n" + "=" * 60)
print("步骤 4: 合并所有数据")
print("=" * 60)

# 合并数据
df_merged = df_vix[['Date', 'VX1_Price', 'VX2_Price', 'VIX_Slope', 'VIX_Slope_pct', 
                     'VIX_Zscore', 'Regime_Fixed', 'Regime_Percentile', 'Regime_Slope']].copy()

df_merged = df_merged.merge(df_rf_split[['Date', 'rf_daily']], on='Date', how='left')

# 提取因子数据（需要识别具体的因子列）
factor_cols = []
if not df_factors.empty and 'Date' in df_factors.columns:
    for col in df_factors.columns:
        if col not in ['Date', 'Dates'] and col is not None:
            try:
                # 尝试转换为数值
                test_series = pd.to_numeric(df_factors[col], errors='coerce')
                if not test_series.isna().all():  # 如果不是全部NaN
                    factor_cols.append(col)
            except:
                pass

    if factor_cols:
        df_factors_clean = df_factors[['Date'] + factor_cols].copy()
        for col in factor_cols:
            df_factors_clean[col] = pd.to_numeric(df_factors_clean[col], errors='coerce')
        
        df_merged = df_merged.merge(df_factors_clean, on='Date', how='left')
        print(f"\n✓ 合并了 {len(factor_cols)} 个因子列")
    else:
        print("\n! 未找到有效的因子数据列")
else:
    print("\n! 因子数据为空或缺少日期列")

print(f"\n最终数据集形状: {df_merged.shape}")
print(f"日期范围: {df_merged['Date'].min()} 到 {df_merged['Date'].max()}")

# ==================== 5. 分析每个Regime的特征 ====================
print("\n" + "=" * 60)
print("步骤 5: 分析Regime特征")
print("=" * 60)

# 使用固定阈值方法作为主要分类
regime_stats = df_merged.groupby('Regime_Fixed').agg({
    'VIX_Zscore': ['count', 'mean', 'std', 'min', 'max'],
    'VIX_Slope_pct': ['mean', 'std', 'min', 'max'],
    'VX1_Price': ['mean', 'std']
}).round(3)

print("\n各Regime统计特征:")
print(regime_stats)

# 计算Regime转换
df_merged['Regime_Change'] = df_merged['Regime_Fixed'] != df_merged['Regime_Fixed'].shift(1)
transitions = df_merged[df_merged['Regime_Change']].copy()
print(f"\n总共发生 {len(transitions)} 次Regime转换")

# 计算每个Regime的平均持续时间
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
print("\n各Regime平均持续时间（天数）:")
print(df_durations.groupby('Regime')['Duration'].agg(['mean', 'median', 'max']))

# ==================== 6. 保存结果 ====================
print("\n" + "=" * 60)
print("步骤 6: 保存结果")
print("=" * 60)

# 保存到Excel
output_file = 'regime_classification_results.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_merged.to_excel(writer, sheet_name='Full_Data', index=False)
    regime_stats.to_excel(writer, sheet_name='Regime_Stats')
    df_durations.to_excel(writer, sheet_name='Regime_Durations', index=False)
    
    # 保存参数设置
    params = pd.DataFrame({
        'Parameter': ['Window_Days', 'Z_Threshold_Low', 'Z_Threshold_High', 
                      'Percentile_33', 'Percentile_67', 'Slope_Low', 'Slope_High'],
        'Value': [window, threshold_low, threshold_high, 
                  percentile_33, percentile_67, slope_low, slope_high]
    })
    params.to_excel(writer, sheet_name='Parameters', index=False)

print(f"\n✓ 结果已保存到: {output_file}")

# ==================== 7. 可视化 ====================
print("\n" + "=" * 60)
print("步骤 7: 创建可视化")
print("=" * 60)

fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# 图1: VIX Slope时间序列
ax1 = axes[0]
ax1.plot(df_merged['Date'], df_merged['VIX_Slope_pct'], linewidth=0.8, color='black', alpha=0.7)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Risk-On threshold (+2%)')
ax1.axhline(y=-2, color='red', linestyle='--', alpha=0.5, label='Risk-Off threshold (-2%)')
ax1.set_title('VIX Term Structure Slope (VX2/VX1 - 1)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Slope (%)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: VIX Z-Score时间序列
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

# 图3: Regime分类（颜色编码）
ax3 = axes[2]
regime_colors = {'Risk-Off': 'red', 'Neutral': 'yellow', 'Risk-On': 'green'}
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

# 图4: Regime分布柱状图
ax4 = axes[3]
regime_counts = df_merged['Regime_Fixed'].value_counts().sort_index()
colors_bar = [regime_colors[regime] for regime in regime_counts.index]
bars = ax4.bar(regime_counts.index, regime_counts.values, color=colors_bar, alpha=0.7, edgecolor='black')
ax4.set_title('Regime Distribution', fontsize=14, fontweight='bold')
ax4.set_ylabel('Days', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值和百分比
for bar, count in zip(bars, regime_counts.values):
    height = bar.get_height()
    pct = count / len(df_merged) * 100
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('regime_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ 可视化已保存到: regime_analysis.png")

# ==================== 8. 总结报告 ====================
print("\n" + "=" * 60)
print("分析总结")
print("=" * 60)

print(f"""
数据期间: {df_merged['Date'].min().strftime('%Y-%m-%d')} 至 {df_merged['Date'].max().strftime('%Y-%m-%d')}
总天数: {len(df_merged)} 天

【推荐方法: 固定Z-score阈值】
- Risk-Off (Z < -0.5):  {(df_merged['Regime_Fixed']=='Risk-Off').sum()} 天 ({(df_merged['Regime_Fixed']=='Risk-Off').sum()/len(df_merged)*100:.1f}%)
- Neutral (-0.5 ≤ Z ≤ 0.5): {(df_merged['Regime_Fixed']=='Neutral').sum()} 天 ({(df_merged['Regime_Fixed']=='Neutral').sum()/len(df_merged)*100:.1f}%)
- Risk-On (Z > 0.5):    {(df_merged['Regime_Fixed']=='Risk-On').sum()} 天 ({(df_merged['Regime_Fixed']=='Risk-On').sum()/len(df_merged)*100:.1f}%)

各Regime特征:
- Risk-Off:  强Backwardation，VIX曲线倒挂，市场高度紧张
- Neutral:   过渡状态，市场相对平衡
- Risk-On:   Contango结构，VIX曲线正常，市场平静

关键市场事件:
- 2018年2月: VIX飙升事件
- 2018年Q4: 股市大幅下跌
- 2020年3月: COVID-19市场崩盘
- 2022年: 俄乌冲突和加息周期

下一步:
1. 在每个Regime下测试因子表现
2. 构建Regime切换策略
3. 计算风险调整后收益
""")

print("\n✓ 分析完成！")
