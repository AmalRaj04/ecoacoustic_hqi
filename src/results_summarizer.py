import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def get_interpretation_hint(feature_name):
    if 'min_frequency_hz' in feature_name:
        return "Lombard effect / urban noise avoidance"
    elif 'peak_frequency_hz' in feature_name:
        return "Vocal shift in noise"
    elif 'spectral_entropy' in feature_name:
        return "Acoustic diversity / competition"
    elif 'temporal_entropy' in feature_name:
        return "Rhythmic structure"
    elif 'harmonicity_hnr_db' in feature_name:
        return "Vocal clarity / stress indicator"
    elif 'bout_duration_s' in feature_name:
        return "Energy expenditure / territory defense"
    elif 'syllable_rate_per_s' in feature_name:
        return "Vocal effort / male quality"
    elif 'modulation_index' in feature_name:
        return "Signal distinctiveness"
    return "Aggregate metric"

def main():
    print("Generating Final Research Summary...")
    
    out_dir = PROJECT_ROOT / "outputs"
    cp_dir = PROJECT_ROOT / "checkpoints"
    
    # 1. Dataset pipeline
    try:
        cp1 = pd.read_csv(cp_dir / "cp1_summary.csv").set_index('Metric')
        n_raw = int(cp1.loc['Total Recordings', 'Value'])
    except:
        n_raw = "N/A"
        
    try:
        cp2 = pd.read_csv(cp_dir / "cp2_summary.csv").set_index('tier')
        n_accepted = int(cp2.loc['ALL', 'total'] - cp2.loc['ALL', 'rejected'])
    except:
        n_accepted = "N/A"
        
    try:
        cp3 = pd.read_csv(cp_dir / "cp3_summary.csv")
        n_segments = int(cp3['total_segments'].iloc[0])
    except:
        n_segments = "N/A"
        
    try:
        cp4 = pd.read_csv(cp_dir / "cp4_summary.csv")
        n_final = int(cp4['n_final'].iloc[0])
    except:
        n_final = "N/A"
        
    pipeline_text = f"""### 1. Dataset Pipeline
- **Initial Recordings Downloaded:** {n_raw}
- **Accepted after Audio Quality Audit (SNR/Clipping):** {n_accepted}
- **Total BirdNET Segments Extracted:** {n_segments}
- **Final Recordings for ML (>= 3 bouts):** {n_final}
"""

    # 2. Model performance
    oof_path = out_dir / "oof_predictions.csv"
    if oof_path.exists():
        oof_df = pd.read_csv(oof_path)
        rmse = np.sqrt(mean_squared_error(oof_df['hqi'], oof_df['predicted_hqi']))
        mae = mean_absolute_error(oof_df['hqi'], oof_df['predicted_hqi'])
        r2 = r2_score(oof_df['hqi'], oof_df['predicted_hqi'])
        
        try:
            with open(out_dir / "best_feature_set.txt", "r") as f:
                best_fs = f.read().strip()
        except:
            best_fs = "Unknown"
            
        perf_text = f"""### 2. Model Performance (Out-of-Fold Spatial CV)
- **Best Feature Set:** {best_fs}
- **RMSE:** {rmse:.4f}
- **MAE:** {mae:.4f}
- **R² Score:** {r2:.4f}
"""
    else:
        perf_text = "### 2. Model Performance\n*Out-of-fold predictions not found.*\n"

    # 3. Ablation result
    ablation_text = f"""### 3. Ablation Result
The best feature set was determined to be **{best_fs}**. 
(For full ablation comparisons between ecoacoustic, embeddings, and fused sets, please refer to the ml_trainer.py console logs).
"""

    # 4. Top 5 acoustic features
    shap_path = out_dir / "shap" / "feature_importance.csv"
    feat_matrix_path = PROJECT_ROOT / "data" / "aggregated" / "feature_matrix.csv"
    
    if shap_path.exists() and feat_matrix_path.exists():
        shap_df = pd.read_csv(shap_path)
        fm_df = pd.read_csv(feat_matrix_path)
        
        # Filter to ecoacoustic features only for the report if there are PCA features
        eco_shap = shap_df[shap_df['feature_name'].str.startswith(('mean_', 'std_', 'min_', 'max_'))].copy()
        
        if len(eco_shap) > 0:
            top5 = eco_shap.head(5)
            
            top5_text = "### 4. Top 5 Ecoacoustic Features by SHAP Importance\n\n"
            top5_text += "| Rank | Feature | Mean |SHAP| | Direction (r vs HQI) | Ecological Hint |\n"
            top5_text += "|---|---|---|---|---|\n"
            
            for i, row in top5.iterrows():
                feat = row['feature_name']
                val = row['mean_abs_shap']
                
                # Correlation with HQI
                if feat in fm_df.columns:
                    r, _ = pearsonr(fm_df[feat], fm_df['hqi'])
                    direction = f"{r:+.2f}"
                else:
                    direction = "N/A"
                    
                hint = get_interpretation_hint(feat)
                top5_text += f"| {row['rank']} | `{feat}` | {val:.4f} | {direction} | {hint} |\n"
        else:
            top5_text = "### 4. Top Features\nTop features are embedding PCA components (no direct ecological interpretation).\n"
    else:
        top5_text = "### 4. Top 5 Ecoacoustic Features\n*SHAP feature importance not found.*\n"

    # 5. Spatial error analysis
    if oof_path.exists():
        oof_df['error_sq'] = (oof_df['hqi'] - oof_df['predicted_hqi']) ** 2
        block_stats = oof_df.groupby('spatial_block').agg(
            rmse=('error_sq', lambda x: np.sqrt(np.mean(x))),
            n_recordings=('error_sq', 'count')
        )
        
        # Filter for blocks with at least 5 recordings to avoid noise
        valid_blocks = block_stats[block_stats['n_recordings'] >= 5].copy()
        
        if len(valid_blocks) > 0:
            worst_blocks = valid_blocks.sort_values('rmse', ascending=False).head(3)
            
            spatial_text = "### 5. Spatial Error Analysis\n"
            spatial_text += "Geographic blocks with the highest prediction error (minimum 5 recordings):\n\n"
            spatial_text += "| Spatial Block ID | RMSE | N Recordings |\n"
            spatial_text += "|---|---|---|\n"
            
            for block_id, row in worst_blocks.iterrows():
                spatial_text += f"| `{block_id}` | {row['rmse']:.4f} | {row['n_recordings']} |\n"
        else:
            spatial_text = "### 5. Spatial Error Analysis\n*Not enough data per block to compute robust spatial errors.*\n"
    else:
        spatial_text = "### 5. Spatial Error Analysis\n*Out-of-fold predictions not found.*\n"

    # Compile Final Report
    report = f"""# Ecoacoustic Habitat Quality Index (HQI) - Final Research Report

{pipeline_text}
{perf_text}
{ablation_text}
{top5_text}
{spatial_text}
"""

    print("\n" + "="*50)
    print(report)
    print("="*50 + "\n")
    
    report_path = out_dir / "final_report.md"
    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"[✓] Saved final research summary to {report_path}")

if __name__ == "__main__":
    main()
