import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, skew
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    feat_matrix_path = PROJECT_ROOT / "data" / "aggregated" / "feature_matrix.csv"
    if not feat_matrix_path.exists():
        print(f"Error: Could not find {feat_matrix_path}")
        return
        
    df = pd.read_csv(feat_matrix_path)
    
    print("\n── Feature Matrix Audit Report ──")
    
    # 1. Print basics
    n_final = len(df)
    print(f"Final Recordings (n): {n_final}")
    
    hqi_mean = df['hqi'].mean()
    hqi_std = df['hqi'].std()
    hqi_min = df['hqi'].min()
    hqi_max = df['hqi'].max()
    print(f"HQI: Mean={hqi_mean:.3f}, Std={hqi_std:.3f}, Min={hqi_min:.3f}, Max={hqi_max:.3f}")
    
    print("\nPer-country counts:")
    country_counts = df['country'].value_counts()
    print(country_counts.to_string())
    
    # 2. Plot: HQI distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['hqi'].dropna(), bins=30, alpha=0.7, color='indigo', edgecolor='black')
    plt.title('HQI Distribution')
    plt.xlabel('Habitat Quality Index (HQI)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    
    hqi_hist_path = PROJECT_ROOT / "checkpoints" / "cp4_hqi_distribution.png"
    plt.savefig(hqi_hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    hqi_skew = skew(df['hqi'].dropna())
    print(f"\nHQI Skewness: {hqi_skew:.3f}")
    if abs(hqi_skew) > 1.5:
        print("[!] WARNING: HQI is strongly skewed. Consider transformation.")
        
    # 3. Plot: Correlation heatmap of mean ecoacoustic features vs HQI
    mean_cols = [c for c in df.columns if c.startswith('mean_')]
    cols_to_correlate = ['hqi'] + mean_cols
    corr_matrix = df[cols_to_correlate].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix[['hqi']].sort_values(by='hqi', ascending=False),
                annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation: Mean Ecoacoustic Features vs HQI')
    
    heatmap_path = PROJECT_ROOT / "checkpoints" / "cp4_feature_hqi_correlations.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Plot: Boxplot of HQI per country
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='country', y='hqi', data=df, palette='Set2')
    plt.title('HQI Distribution per Country')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    country_box_path = PROJECT_ROOT / "checkpoints" / "cp4_hqi_per_country.png"
    plt.savefig(country_box_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Plot: HQI vs bout_count scatter
    df_valid_bouts = df.dropna(subset=['hqi', 'bout_count'])
    plt.figure(figsize=(8, 6))
    plt.scatter(df_valid_bouts['bout_count'], df_valid_bouts['hqi'], alpha=0.5, color='darkgreen')
    plt.title('HQI vs Bout Count')
    plt.xlabel('Bout Count')
    plt.ylabel('HQI')
    plt.grid(alpha=0.3)
    
    bout_scatter_path = PROJECT_ROOT / "checkpoints" / "cp4_hqi_vs_bout_count.png"
    plt.savefig(bout_scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Pearson r between bout_count and HQI
    r_bouts, p_bouts = pearsonr(df_valid_bouts['bout_count'], df_valid_bouts['hqi'])
    print(f"Pearson r (bout_count vs HQI): {r_bouts:.3f} (p={p_bouts:.3e})")
    
    # 7. Detect outliers (> 4 std from mean)
    eco_cols = [c for c in df.columns if c.startswith('mean_') or c.startswith('std_') or c.startswith('min_') or c.startswith('max_')]
    outlier_ids = set()
    
    for col in eco_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        # Find indices where > 4 std
        outliers = df[np.abs(df[col] - mean_val) > 4 * std_val]['recording_id'].tolist()
        outlier_ids.update(outliers)
        
    print(f"Detected {len(outlier_ids)} outlier recordings (>4 std on any feature).")
    df_outliers = pd.DataFrame({'recording_id': list(outlier_ids)})
    outliers_path = PROJECT_ROOT / "checkpoints" / "cp4_outliers.csv"
    df_outliers.to_csv(outliers_path, index=False)
    print(f"[✓] Saved outliers to {outliers_path.name}")
    
    # 8. Data Quality Verdict
    print("\n── Data Quality Verdict ──")
    flags = 0
    if n_final < 300:
        print("[!] FLAG: Dataset size < 300. May be too small for spatial CV.")
        flags += 1
    if abs(r_bouts) > 0.3:
        print(f"[!] FLAG: Strong correlation between bout_count and HQI (r={r_bouts:.3f}). Confound detected.")
        flags += 1
    small_countries = country_counts[country_counts < 20]
    if len(small_countries) > 0:
        print(f"[!] FLAG: {len(small_countries)} countries have < 20 recordings. Consider dropping or pooling:")
        for c, count in small_countries.items():
            print(f"    - {c}: {count}")
        flags += 1
        
    if flags == 0:
        print("[✓] ALL CLEAR! Feature matrix looks robust.")
        
    # Save summary
    summary_data = {
        'n_final': n_final,
        'hqi_mean': hqi_mean,
        'hqi_std': hqi_std,
        'hqi_skewness': hqi_skew,
        'pearson_bout_count_hqi': r_bouts,
        'outlier_count': len(outlier_ids),
        'quality_flags': flags
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_path = PROJECT_ROOT / "checkpoints" / "cp4_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[✓] Saved summary to {summary_path.name}")
    print("[✓] Generated all plots in checkpoints/")

if __name__ == "__main__":
    main()
