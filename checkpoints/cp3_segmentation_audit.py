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

from config import DATA_SEGMENTS, DATA_HQI

def main():
    det_log_path = PROJECT_ROOT / DATA_SEGMENTS / "detection_log.csv"
    hqi_path = PROJECT_ROOT / DATA_HQI / "hqi_scores.csv"
    
    if not det_log_path.exists() or not hqi_path.exists():
        print("Error: Could not find required CSV files.")
        return

    # Load data
    df_det = pd.read_csv(det_log_path)
    df_hqi = pd.read_csv(hqi_path)
    
    df_det['recording_id'] = df_det['recording_id'].astype(str)
    df_hqi['recording_id'] = df_hqi['recording_id'].astype(str)
    
    # Check if there are any detections
    if len(df_det) == 0:
        print("No detections found in detection_log.csv")
        return
        
    print("\n── Segmentation Audit Report ──")
    
    # Group detections by recording_id
    rec_stats = df_det.groupby('recording_id').agg(
        bout_count=('segment_id', 'count'),
        mean_confidence=('confidence', 'mean')
    ).reset_index()
    
    # Merge with HQI scores
    df_merged = rec_stats.merge(df_hqi[['recording_id', 'hqi']], on='recording_id', how='left')
    
    # 1. Print metrics
    total_segments = len(df_det)
    mean_segs = df_merged['bout_count'].mean()
    median_segs = df_merged['bout_count'].median()
    min_segs = df_merged['bout_count'].min()
    max_segs = df_merged['bout_count'].max()
    
    print(f"Total segments extracted: {total_segments}")
    print(f"Segments per recording: Mean={mean_segs:.2f}, Median={median_segs:.1f}, Min={min_segs}, Max={max_segs}")
    
    # 2. Count of recordings with < 3 bouts
    dropped_recs = len(df_merged[df_merged['bout_count'] < 3])
    print(f"Recordings with < 3 bouts (to be dropped later): {dropped_recs}")
    
    # 3. Plot: Histogram of confidence scores
    plt.figure(figsize=(10, 6))
    plt.hist(df_det['confidence'], bins=20, alpha=0.7, color='teal', edgecolor='black')
    plt.title('Distribution of Confidence Scores')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    
    conf_hist_path = PROJECT_ROOT / "checkpoints" / "cp3_confidence_hist.png"
    plt.savefig(conf_hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[✓] Saved confidence histogram to {conf_hist_path.name}")
    
    # Drop rows without HQI for correlations
    df_valid = df_merged.dropna(subset=['hqi', 'mean_confidence', 'bout_count'])
    
    if len(df_valid) == 0:
        print("No valid recordings with HQI scores to plot.")
        return

    # 4. Plot: Scatter of mean confidence vs HQI
    plt.figure(figsize=(10, 6))
    plt.scatter(df_valid['hqi'], df_valid['mean_confidence'], alpha=0.5, color='purple', s=20)
    plt.title('Mean Confidence vs HQI (Bias Check)')
    plt.xlabel('Habitat Quality Index (HQI)')
    plt.ylabel('Mean Confidence per Recording')
    plt.grid(alpha=0.3)
    
    conf_hqi_path = PROJECT_ROOT / "checkpoints" / "cp3_confidence_vs_hqi.png"
    plt.savefig(conf_hqi_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Mean Confidence vs HQI scatter plot to {conf_hqi_path.name}")

    # 5. Plot: Segments per recording vs HQI
    plt.figure(figsize=(10, 6))
    plt.scatter(df_valid['hqi'], df_valid['bout_count'], alpha=0.5, color='orange', s=20)
    plt.title('Segments per Recording vs HQI')
    plt.xlabel('Habitat Quality Index (HQI)')
    plt.ylabel('Number of Segments (Bouts)')
    plt.grid(alpha=0.3)
    
    bouts_hqi_path = PROJECT_ROOT / "checkpoints" / "cp3_bouts_vs_hqi.png"
    plt.savefig(bouts_hqi_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Segments vs HQI scatter plot to {bouts_hqi_path.name}")
    
    # 6. Print Spearman correlations
    corr_conf, p_conf = spearmanr(df_valid['hqi'], df_valid['mean_confidence'])
    corr_bouts, p_bouts = spearmanr(df_valid['hqi'], df_valid['bout_count'])
    
    print("\n── Bias Analysis ──")
    print(f"Correlation (HQI & Mean Confidence): {corr_conf:.3f} (p-value: {p_conf:.3e})")
    print(f"Correlation (HQI & Bout Count)     : {corr_bouts:.3f} (p-value: {p_bouts:.3e})")
    
    if corr_conf > 0.3:
        print("\n[!] WARNING: Confidence drops with HQI. Urban bias is confirmed!")
        print("    Urban recordings (low HQI) have systematically lower confidence scores.")
    else:
        print("\n[✓] No significant urban bias detected in confidence scores.")

    if corr_bouts > 0.3:
        print("[!] WARNING: Urban recordings yield significantly fewer segments (bouts).")
        
    # 7. Save summary CSV
    summary_data = [{
        'total_segments': total_segments,
        'mean_segments_per_rec': mean_segs,
        'median_segments_per_rec': median_segs,
        'recs_with_lt_3_bouts': dropped_recs,
        'spearman_hqi_vs_confidence': corr_conf,
        'spearman_hqi_vs_bouts': corr_bouts
    }]
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = PROJECT_ROOT / "checkpoints" / "cp3_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[✓] Saved summary CSV to {summary_csv_path.name}")

if __name__ == "__main__":
    main()
