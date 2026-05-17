import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_RAW

def main():
    log_path = PROJECT_ROOT / DATA_RAW / "audio_attrition_log.csv"
    if not log_path.exists():
        print(f"Error: Could not find attrition log at {log_path}")
        return

    # Load data
    df = pd.read_csv(log_path)
    
    # 1. Print metrics
    total_accepted = len(df[df['status'] == 'accepted'])
    print("\n── Quality Audit Report ──")
    print(f"Total accepted recordings: {total_accepted} out of {len(df)}")
    
    print("\nPer-status breakdown:")
    print(df['status'].value_counts().to_string())
    
    print("\nRejection rates by quality tier:")
    tiers = df['quality_tier'].dropna().unique()
    summary_data = []
    
    for t in sorted(tiers):
        df_t = df[df['quality_tier'] == t]
        total = len(df_t)
        if total == 0:
            continue
        rejected = len(df_t[df_t['status'] != 'accepted'])
        rate = rejected / total
        print(f"  Tier {t}: {rate*100:.1f}% ({rejected}/{total})")
        summary_data.append({'tier': t, 'total': total, 'rejected': rejected, 'rejection_rate': rate})
    
    # 2. Plot: SNR distribution histogram (accepted vs rejected)
    plt.figure(figsize=(10, 6))
    snr_accepted = df[(df['status'] == 'accepted') & df['snr_db'].notna()]['snr_db']
    snr_rejected = df[(df['status'] != 'accepted') & df['snr_db'].notna()]['snr_db']
    
    plt.hist(snr_accepted, bins=30, alpha=0.6, label='Accepted', color='green', edgecolor='black')
    plt.hist(snr_rejected, bins=30, alpha=0.6, label='Rejected', color='red', edgecolor='black')
    plt.title('SNR Distribution: Accepted vs Rejected')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    snr_hist_path = PROJECT_ROOT / "checkpoints" / "cp2_snr_distribution.png"
    plt.savefig(snr_hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[✓] Saved SNR distribution plot to {snr_hist_path.name}")
    
    # 3. Plot: HQI vs SNR scatter plot
    plt.figure(figsize=(10, 6))
    df_valid = df.dropna(subset=['snr_db', 'hqi'])
    
    plt.scatter(
        df_valid[df_valid['status'] == 'accepted']['hqi'],
        df_valid[df_valid['status'] == 'accepted']['snr_db'],
        alpha=0.5, label='Accepted', color='green', s=20
    )
    plt.scatter(
        df_valid[df_valid['status'] != 'accepted']['hqi'],
        df_valid[df_valid['status'] != 'accepted']['snr_db'],
        alpha=0.5, label='Rejected', color='red', s=20
    )
    
    plt.title('HQI vs SNR (Bias Check)')
    plt.xlabel('Habitat Quality Index (HQI)')
    plt.ylabel('SNR (dB)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    hqi_snr_path = PROJECT_ROOT / "checkpoints" / "cp2_hqi_vs_snr.png"
    plt.savefig(hqi_snr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved HQI vs SNR scatter plot to {hqi_snr_path.name}")
    
    # 4. Print: Spearman correlation between HQI and SNR
    corr, p_value = spearmanr(df_valid['hqi'], df_valid['snr_db'])
    print(f"\nSpearman correlation between HQI and SNR: {corr:.3f} (p-value: {p_value:.3e})")
    
    # 5. Print warning if urban bias detected
    if corr > 0.3:
        print("\n[!] WARNING: Urban bias detected! High HQI correlates strongly with better SNR.")
        print("    Low-HQI (urban) recordings are disproportionately noisier and may be systematically rejected.")
        print("    Consider lowering the SNR threshold or using adaptive filtering to preserve urban samples.")
    else:
        print("\n[✓] No significant urban bias detected (correlation <= 0.3).")
        
    # 6. Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    # Add general stats row
    summary_df = pd.concat([summary_df, pd.DataFrame([{
        'tier': 'ALL',
        'total': len(df),
        'rejected': len(df[df['status'] != 'accepted']),
        'rejection_rate': len(df[df['status'] != 'accepted']) / len(df)
    }])], ignore_index=True)
    
    summary_csv_path = PROJECT_ROOT / "checkpoints" / "cp2_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[✓] Saved tabular summary to {summary_csv_path.name}")

if __name__ == "__main__":
    main()
