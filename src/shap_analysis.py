import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from pathlib import Path
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def get_interpretation_hint(feature_name):
    if 'min_frequency_hz' in feature_name:
        return "Minimum frequency (Lombard effect / urban noise avoidance)"
    elif 'peak_frequency_hz' in feature_name:
        return "Peak energy frequency (vocal shift in noise)"
    elif 'spectral_entropy' in feature_name:
        return "Spectral complexity (acoustic diversity / competition)"
    elif 'temporal_entropy' in feature_name:
        return "Temporal energy distribution (rhythmic structure)"
    elif 'harmonicity_hnr_db' in feature_name:
        return "Harmonic-to-noise ratio (vocal clarity / stress indicator)"
    elif 'bout_duration_s' in feature_name:
        return "Bout duration (energy expenditure / territory defense)"
    elif 'syllable_rate_per_s' in feature_name:
        return "Syllable rate (vocal effort / male quality)"
    elif 'modulation_index' in feature_name:
        return "Amplitude modulation (signal distinctiveness)"
    return "Aggregate acoustic metric"

def main():
    print("Starting Explainability Analysis (SHAP)")
    
    out_dir = PROJECT_ROOT / "outputs"
    shap_dir = out_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data and best configuration
    with open(out_dir / "best_feature_set.txt", "r") as f:
        best_fs = f.read().strip()
        
    feat_matrix_path = PROJECT_ROOT / "data" / "aggregated" / "feature_matrix.csv"
    emb_matrix_path = PROJECT_ROOT / "data" / "aggregated" / "embedding_matrix.npy"
    
    df = pd.read_csv(feat_matrix_path).fillna(0)
    embeddings = np.load(emb_matrix_path)
    y = df['hqi'].values
    
    eco_cols = [c for c in df.columns if c.startswith('mean_') or c.startswith('std_') or c.startswith('min_') or c.startswith('max_')]
    X_eco = df[eco_cols].values
    
    if best_fs == "ecoacoustic_only":
        X = X_eco
        feature_names = eco_cols
        
    elif best_fs in ["embeddings_only", "fused"]:
        print("\nApplying PCA to embeddings as requested...")
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(embeddings)
        
        n_components = X_pca.shape[1]
        var_explained = np.sum(pca.explained_variance_ratio_)
        print(f"Retained {n_components} PCA components explaining {var_explained:.2%} of variance.")
        
        with open(out_dir / "pca_model.pkl", "wb") as f:
            pickle.dump(pca, f)
            
        pca_cols = [f"PCA_{i+1}" for i in range(n_components)]
        
        if best_fs == "embeddings_only":
            X = X_pca
            feature_names = pca_cols
        else: # fused
            X = np.hstack([X_eco, X_pca])
            feature_names = eco_cols + pca_cols
            
        print("Refitting best model on PCA-transformed features for SHAP explainability...")
        with open(out_dir / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
            
        # Refit to match PCA feature shape
        model.fit(X, y)
    else:
        print(f"Unknown feature set: {best_fs}")
        return
        
    # 3. Compute SHAP values
    print("Computing SHAP values...")
    if isinstance(model, XGBRegressor) or hasattr(model, 'get_booster'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        # For Ridge or Linear
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)
        
    # 4a) SHAP summary plot — ALL features
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.savefig(shap_dir / "shap_summary_all.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4b) SHAP summary plot — ECOACOUSTIC features only
    if best_fs in ["ecoacoustic_only", "fused"]:
        X_eco_only = X[:, :len(eco_cols)]
        shap_eco_only = shap_values[:, :len(eco_cols)]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_eco_only, X_eco_only, feature_names=eco_cols, show=False)
        plt.savefig(shap_dir / "shap_summary_ecoacoustic.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Save feature importance
        mean_abs_shap = np.abs(shap_eco_only).mean(axis=0)
        df_imp = pd.DataFrame({
            'feature_name': eco_cols,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        df_imp['rank'] = df_imp.index + 1
        df_imp.to_csv(shap_dir / "feature_importance.csv", index=False)
        
        print("\n── Top 10 Ecoacoustic Features by Mean |SHAP| ──")
        print(df_imp.head(10).to_string(index=False))
        
        top_5 = df_imp.head(5)['feature_name'].tolist()
        
        # 4c) SHAP dependence plots
        for feat in top_5:
            feat_idx = eco_cols.index(feat)
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(X_eco_only[:, feat_idx], shap_eco_only[:, feat_idx], 
                             c=y, cmap='viridis', alpha=0.7)
            plt.colorbar(sc, label='HQI')
            plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
            plt.title(f'SHAP Dependence: {feat}')
            plt.xlabel(feat)
            plt.ylabel('SHAP Value')
            plt.savefig(shap_dir / f"shap_dependence_{feat}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        # 4d) SHAP vs HQI scatter for TOP feature
        top_feat_idx = eco_cols.index(top_5[0])
        plt.figure(figsize=(8, 6))
        plt.scatter(y, shap_eco_only[:, top_feat_idx], alpha=0.6, color='coral')
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.title(f'SHAP Value vs HQI Gradient for {top_5[0]}')
        plt.xlabel('Habitat Quality Index (HQI)')
        plt.ylabel(f'SHAP Value ({top_5[0]})')
        plt.grid(alpha=0.3)
        plt.savefig(shap_dir / "shap_hqi_gradient.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Ecological Interpretation Summary
        print("\n── Ecological Interpretation Summary ──")
        for feat in top_5:
            feat_idx = eco_cols.index(feat)
            # Check direction: correlate feature value with SHAP value
            
            std_feat = np.std(X_eco_only[:, feat_idx])
            if std_feat > 1e-6:
                corr = np.corrcoef(X_eco_only[:, feat_idx], shap_eco_only[:, feat_idx])[0, 1]
            else:
                corr = 0
            
            direction = "INCREASE" if corr > 0 else "DECREASE"
            
            val = df_imp[df_imp['feature_name'] == feat]['mean_abs_shap'].values[0]
            hint = get_interpretation_hint(feat)
            
            print(f"Feature   : {feat}")
            print(f"Mean|SHAP|: {val:.4f}")
            print(f"Direction : Higher values {direction} predicted HQI (r={corr:.2f})")
            print(f"Ecol. Hint: {hint}\n")
            
    else:
        # Just embeddings
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        df_imp = pd.DataFrame({
            'feature_name': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        df_imp.to_csv(shap_dir / "feature_importance.csv", index=False)
        print("\n── Top 10 Features by Mean |SHAP| ──")
        print(df_imp.head(10).to_string(index=False))

    print(f"\n[✓] Saved SHAP analysis results and plots to {shap_dir}")

if __name__ == "__main__":
    main()
