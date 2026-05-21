import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_AGGREGATED, SPATIAL_BLOCK_SIZE_KM, XGBOOST_PARAMS

def create_spatial_blocks(df, block_size_km):
    # 1 degree lat is ~ 111.32 km
    # 1 degree lon is ~ 111.32 * cos(lat) km
    deg_lat = block_size_km / 111.32
    mean_lat = df['lat'].mean()
    deg_lng = block_size_km / (111.32 * np.cos(np.radians(mean_lat)))
    
    lat_block = np.floor(df['lat'] / deg_lat)
    lng_block = np.floor(df['lng'] / deg_lng)
    
    return (lat_block.astype(str) + '_' + lng_block.astype(str)).values

def main():
    print("Starting ML Training Pipeline")
    agg_dir = PROJECT_ROOT / DATA_AGGREGATED
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    feat_matrix_path = agg_dir / "feature_matrix.csv"
    emb_matrix_path = agg_dir / "embedding_matrix.npy"
    
    if not feat_matrix_path.exists():
        print(f"Error: Could not find {feat_matrix_path}")
        return
        
    df = pd.read_csv(feat_matrix_path)
    embeddings = np.load(emb_matrix_path)
    
    print(f"Loaded {len(df)} recordings.")
    
    # Check for NaN in targets or features
    df = df.fillna(0) # Simple imputation for missing features
    
    # 2. Build feature sets
    eco_cols = [c for c in df.columns if c.startswith('mean_') or c.startswith('std_') or c.startswith('min_') or c.startswith('max_')]
    
    X_eco = df[eco_cols].values
    X_emb = embeddings
    X_fused = np.hstack([X_eco, X_emb])
    
    y = df['hqi'].values
    
    feature_sets = {
        'ecoacoustic_only': X_eco,
        'embeddings_only': X_emb,
        'fused': X_fused
    }
    
    # 3. Spatial block cross-validation
    groups = create_spatial_blocks(df, SPATIAL_BLOCK_SIZE_KM)
    unique_groups = np.unique(groups)
    print(f"Created {len(unique_groups)} spatial blocks ({SPATIAL_BLOCK_SIZE_KM} km grid).")
    
    print("\nBlock-level HQI distribution (first 10 blocks):")
    block_hqi = df.groupby(groups)['hqi'].agg(['count', 'mean', 'std'])
    print(block_hqi.head(10).to_string())
    
    # 4. Train and evaluate
    models = {
        'Ridge': Ridge(alpha=1.0),
        'XGBoost': xgb.XGBRegressor(**XGBOOST_PARAMS)
    }
    
    results = []
    oof_preds_dict = {}
    
    logo = LeaveOneGroupOut()
    
    print("\nStarting Spatial Block CV...")
    for fs_name, X in feature_sets.items():
        for model_name, model in models.items():
            print(f"Evaluating {model_name} on {fs_name}...")
            oof_preds = np.zeros_like(y)
            
            for train_idx, test_idx in logo.split(X, y, groups):
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                
                model.fit(X_train, y_train)
                oof_preds[test_idx] = model.predict(X_test)
                
            rmse = np.sqrt(mean_squared_error(y, oof_preds))
            mae = mean_absolute_error(y, oof_preds)
            r2 = r2_score(y, oof_preds)
            
            combo_name = f"{model_name} | {fs_name}"
            results.append({
                'Configuration': combo_name,
                'Model': model_name,
                'Feature Set': fs_name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
            oof_preds_dict[combo_name] = oof_preds
            
    # 5. Results table
    res_df = pd.DataFrame(results).sort_values(by='RMSE')
    print("\n── Model Evaluation Results (Sorted by RMSE) ──")
    print(res_df[['Configuration', 'RMSE', 'MAE', 'R2']].to_string(index=False))
    
    # Flag ablation
    xgb_eco = res_df[(res_df['Model'] == 'XGBoost') & (res_df['Feature Set'] == 'ecoacoustic_only')]['RMSE'].values[0]
    xgb_fused = res_df[(res_df['Model'] == 'XGBoost') & (res_df['Feature Set'] == 'fused')]['RMSE'].values[0]
    
    print("\n── Ablation Analysis ──")
    print(f"XGBoost (Eco-only) RMSE: {xgb_eco:.4f}")
    print(f"XGBoost (Fused)    RMSE: {xgb_fused:.4f}")
    if xgb_fused < xgb_eco:
        print("[!] YES, adding embeddings improves over ecoacoustic-only features!")
    else:
        print("[!] NO, adding embeddings does not improve over ecoacoustic-only features.")
        
    # 6. Best model
    best_config = res_df.iloc[0]
    best_model_name = best_config['Model']
    best_fs_name = best_config['Feature Set']
    best_combo = best_config['Configuration']
    print(f"\nBest Model Configuration: {best_combo}")
    
    best_X = feature_sets[best_fs_name]
    best_model = models[best_model_name]
    
    # Train on full dataset
    print("Training best model on full dataset...")
    best_model.fit(best_X, y)
    
    best_model_path = out_dir / "best_model.pkl"
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
        
    with open(out_dir / "best_feature_set.txt", 'w') as f:
        f.write(best_fs_name)
        
    df_oof = df[['recording_id', 'hqi', 'lat', 'lng', 'country']].copy()
    df_oof['predicted_hqi'] = oof_preds_dict[best_combo]
    df_oof['spatial_block'] = groups
    
    oof_path = out_dir / "oof_predictions.csv"
    df_oof.to_csv(oof_path, index=False)
    
    # 7. Plots
    plt.figure(figsize=(8, 8))
    plt.scatter(df_oof['hqi'], df_oof['predicted_hqi'], alpha=0.6, color='teal')
    
    # Draw diagonal line for reference
    min_val = min(df_oof['hqi'].min(), df_oof['predicted_hqi'].min())
    max_val = max(df_oof['hqi'].max(), df_oof['predicted_hqi'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Actual vs Predicted HQI\n(OOF Predictions: {best_combo})')
    plt.xlabel('Actual HQI')
    plt.ylabel('Predicted HQI')
    plt.grid(alpha=0.3)
    plt.savefig(out_dir / "actual_vs_predicted_hqi.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # RMSE per spatial block
    df_oof['error_sq'] = (df_oof['hqi'] - df_oof['predicted_hqi']) ** 2
    block_rmse = df_oof.groupby('spatial_block')['error_sq'].mean().apply(np.sqrt)
    block_counts = df_oof.groupby('spatial_block').size()
    
    # Only plot blocks with at least 5 recordings to avoid noise
    valid_blocks = block_counts[block_counts >= 5].index
    block_rmse_valid = block_rmse[valid_blocks].sort_values()
    
    plt.figure(figsize=(12, 6))
    if len(block_rmse_valid) > 0:
        block_rmse_valid.plot(kind='bar', color='coral')
    plt.title('RMSE per Spatial Block (blocks with n>=5)')
    plt.xlabel('Spatial Block ID')
    plt.ylabel('RMSE')
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(out_dir / "rmse_per_spatial_block.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[✓] Saved best model to {best_model_path.name}")
    print("[✓] Saved OOF predictions and plots to outputs/")

if __name__ == "__main__":
    main()
