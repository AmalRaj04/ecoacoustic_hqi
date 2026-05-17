import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_FEATURES, DATA_HQI, DATA_RAW, DATA_SEGMENTS, MIN_BOUTS_PER_RECORDING

def parse_length_to_minutes(length_str):
    try:
        parts = str(length_str).strip().split(':')
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1])/60.0
        elif len(parts) == 3:
            return int(parts[0])*60 + int(parts[1]) + int(parts[2])/60.0
    except:
        pass
    return np.nan

def main():
    print("Starting Recording Aggregator Pipeline")
    
    features_dir = PROJECT_ROOT / DATA_FEATURES
    agg_dir = PROJECT_ROOT / "data" / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load datasets
    print("Loading datasets...")
    eco_df = pd.read_csv(features_dir / "ecoacoustic_features.csv")
    emb_ids = pd.read_csv(features_dir / "embedding_ids.csv")
    embeddings = np.load(features_dir / "embeddings.npy")
    hqi_df = pd.read_csv(PROJECT_ROOT / DATA_HQI / "hqi_scores.csv")
    det_log = pd.read_csv(PROJECT_ROOT / DATA_SEGMENTS / "detection_log.csv")
    xc_meta = pd.read_csv(PROJECT_ROOT / DATA_RAW / "xc_metadata.csv")
    
    # Standardize ID types
    eco_df['segment_id'] = eco_df['segment_id'].astype(str)
    eco_df['recording_id'] = eco_df['recording_id'].astype(str)
    emb_ids['segment_id'] = emb_ids['segment_id'].astype(str)
    emb_ids['recording_id'] = emb_ids['recording_id'].astype(str)
    hqi_df['recording_id'] = hqi_df['recording_id'].astype(str)
    det_log['recording_id'] = det_log['recording_id'].astype(str)
    xc_meta['id'] = xc_meta['id'].astype(str)
    
    # 2. Validations
    print("Validating dimensions and alignments...")
    if embeddings.shape[1] != 1024:
        raise ValueError(f"CRITICAL: Embedding dimension is {embeddings.shape[1]}, expected 1024.")
    if len(embeddings) != len(emb_ids):
        raise ValueError(f"CRITICAL: Embeddings length ({len(embeddings)}) mismatch with IDs length ({len(emb_ids)}).")
        
    eco_segs = set(eco_df['segment_id'])
    emb_segs = set(emb_ids['segment_id'])
    missing_in_emb = eco_segs - emb_segs
    missing_in_eco = emb_segs - eco_segs
    
    if missing_in_emb:
        print(f"WARNING: {len(missing_in_emb)} segments in eco features missing from embeddings.")
    if missing_in_eco:
        print(f"WARNING: {len(missing_in_eco)} segments in embeddings missing from eco features.")
        
    common_segs = eco_segs.intersection(emb_segs)
    print(f"Total valid aligned segments: {len(common_segs)}")
    
    if len(common_segs) == 0:
        print("CRITICAL: No overlapping segments found. Exiting.")
        return
        
    eco_df = eco_df[eco_df['segment_id'].isin(common_segs)]
    emb_ids = emb_ids[emb_ids['segment_id'].isin(common_segs)]
    
    # Ensure aligned order for embeddings
    emb_df = pd.DataFrame(embeddings)
    emb_df['segment_id'] = emb_ids['segment_id'].values
    emb_df['recording_id'] = emb_ids['recording_id'].values
    
    # 3. Filtering recordings with >= MIN_BOUTS_PER_RECORDING
    bouts_per_rec = eco_df.groupby('recording_id')['segment_id'].count()
    valid_recs = bouts_per_rec[bouts_per_rec >= MIN_BOUTS_PER_RECORDING].index
    
    TEST_MODE = False
    if TEST_MODE:
        valid_recs = valid_recs[:10]
        print(f"TEST_MODE ON: Restricting to first 10 qualifying recordings.")
    
    dropped_count = len(bouts_per_rec) - len(valid_recs)
    print(f"Dropped {dropped_count} recordings with < {MIN_BOUTS_PER_RECORDING} bouts.")
    print(f"Qualifying recordings for aggregation: {len(valid_recs)}")
    
    eco_valid = eco_df[eco_df['recording_id'].isin(valid_recs)]
    emb_valid = emb_df[emb_df['recording_id'].isin(valid_recs)]
    
    # Base features
    feature_cols = [
        'min_frequency_hz', 'peak_frequency_hz', 'spectral_entropy', 'temporal_entropy',
        'harmonicity_hnr_db', 'bout_duration_s', 'syllable_rate_per_s', 'modulation_index'
    ]
    
    # Join with xc_metadata to get recording length
    # Prepare metadata map
    meta_map = xc_meta.set_index('id')
    
    # 4. Aggregation
    print("Aggregating ecoacoustic features and embeddings per recording...")
    final_records = []
    final_embs = []
    final_emb_ids = []
    
    # Supress warnings for all-nan slices during aggregation
    warnings.filterwarnings('ignore', r'All-NaN slice encountered')
    warnings.filterwarnings('ignore', r'Mean of empty slice')
    
    for rec_id in tqdm(valid_recs, desc="Aggregating"):
        # Ecoacoustic agg
        rec_eco = eco_valid[eco_valid['recording_id'] == rec_id]
        
        # Calculate stats, ignoring NaNs
        agg_feats = {'recording_id': rec_id}
        for col in feature_cols:
            vals = rec_eco[col].values
            agg_feats[f"mean_{col}"] = np.nanmean(vals)
            agg_feats[f"std_{col}"] = np.nanstd(vals)
            agg_feats[f"min_{col}"] = np.nanmin(vals)
            agg_feats[f"max_{col}"] = np.nanmax(vals)
            
        # Bout count and bpm
        bout_count = len(rec_eco)
        agg_feats['bout_count'] = bout_count
        
        length_str = meta_map.loc[rec_id, 'length'] if rec_id in meta_map.index else None
        length_min = parse_length_to_minutes(length_str)
        agg_feats['bouts_per_minute'] = bout_count / length_min if length_min and length_min > 0 else np.nan
        
        final_records.append(agg_feats)
        
        # Embedding agg
        rec_emb = emb_valid[emb_valid['recording_id'] == rec_id].drop(columns=['segment_id', 'recording_id']).values
        emb_mean = np.nanmean(rec_emb, axis=0)
        emb_std = np.nanstd(rec_emb, axis=0)
        
        # Concatenate mean and std
        emb_concat = np.concatenate([emb_mean, emb_std])
        final_embs.append(emb_concat)
        final_emb_ids.append({'recording_id': rec_id})
        
    agg_df = pd.DataFrame(final_records)
    
    # Merge targets and metadata
    print("Merging metadata and HQI scores...")
    hqi_map = hqi_df.set_index('recording_id')
    
    agg_df['hqi'] = agg_df['recording_id'].map(hqi_map['hqi'])
    agg_df['lat'] = agg_df['recording_id'].map(meta_map['lat'])
    agg_df['lng'] = agg_df['recording_id'].map(meta_map['lon'])
    agg_df['country'] = agg_df['recording_id'].map(meta_map['country'])
    agg_df['quality'] = agg_df['recording_id'].map(meta_map['quality'])
    agg_df['month'] = agg_df['recording_id'].map(meta_map['month'])
    
    # Rearrange columns
    metadata_cols = ['recording_id', 'lat', 'lng', 'country', 'quality', 'month', 'hqi', 'bout_count', 'bouts_per_minute']
    eco_cols = [c for c in agg_df.columns if c not in metadata_cols]
    final_cols = metadata_cols + eco_cols
    agg_df = agg_df[final_cols]
    
    # Save files
    out_feat_matrix = agg_dir / "feature_matrix.csv"
    out_emb_matrix = agg_dir / "embedding_matrix.npy"
    out_emb_ids = agg_dir / "embedding_recording_ids.csv"
    
    agg_df.to_csv(out_feat_matrix, index=False)
    
    final_emb_matrix = np.vstack(final_embs)
    np.save(out_emb_matrix, final_emb_matrix)
    pd.DataFrame(final_emb_ids).to_csv(out_emb_ids, index=False)
    
    # Print summary
    print("\n── Aggregation Summary ──")
    print(f"Final Recordings (n) : {len(agg_df)}")
    print(f"Feature Matrix Shape : {agg_df.shape}")
    print(f"Embedding Matrix     : {final_emb_matrix.shape}")
    
    valid_hqi = agg_df['hqi'].dropna()
    print(f"HQI Mean ± Std       : {valid_hqi.mean():.3f} ± {valid_hqi.std():.3f}")
    
    print("\nRecordings per Country:")
    country_counts = agg_df['country'].value_counts()
    for country, count in country_counts.items():
        print(f"  {country:<15}: {count:>4}")

    print(f"\n[✓] Saved feature_matrix.csv to {out_feat_matrix}")
    print(f"[✓] Saved embedding_matrix.npy to {out_emb_matrix}")

if __name__ == "__main__":
    main()
