import os
import sys
import time
import pandas as pd
import numpy as np
import librosa
import parselmouth
import soundfile as sf
import tensorflow.lite as tflite
from scipy.stats import entropy
from scipy.signal import find_peaks, hilbert
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_SEGMENTS, DATA_FEATURES

def extract_ecoacoustic_features(y, sr, start_s, end_s):
    features = {}
    features['bout_duration_s'] = end_s - start_s
    
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    
    # 1. min_frequency_hz
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mean_energy = np.mean(S)
    std_energy = np.std(S)
    threshold = mean_energy + std_energy
    
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=sr/2)
    active_bins = np.any(S > threshold, axis=1)
    if np.any(active_bins):
        min_bin_idx = np.where(active_bins)[0][0]
        features['min_frequency_hz'] = mel_freqs[min_bin_idx]
    else:
        features['min_frequency_hz'] = np.nan
        
    # 2. peak_frequency_hz
    D = np.abs(librosa.stft(y))
    mean_mag = np.mean(D, axis=1)
    fft_freqs = librosa.fft_frequencies(sr=sr)
    peak_bin = np.argmax(mean_mag)
    features['peak_frequency_hz'] = fft_freqs[peak_bin]
    
    # 3. spectral_entropy
    power_spectrum = mean_mag ** 2
    if np.sum(power_spectrum) > 0:
        norm_power = power_spectrum / np.sum(power_spectrum)
        features['spectral_entropy'] = entropy(norm_power)
    else:
        features['spectral_entropy'] = np.nan
        
    # 4. temporal_entropy
    if np.sum(rms) > 0:
        norm_rms = rms / np.sum(rms)
        features['temporal_entropy'] = entropy(norm_rms)
    else:
        features['temporal_entropy'] = np.nan
        
    # 7. syllable_rate_per_s
    norm_rms_peak = rms / (np.max(rms) + 1e-10)
    peaks, _ = find_peaks(norm_rms_peak, distance=10, prominence=0.1)
    features['syllable_rate_per_s'] = len(peaks) / features['bout_duration_s'] if features['bout_duration_s'] > 0 else 0
    
    # 8. modulation_index
    analytic_signal = hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    mean_env = np.mean(amplitude_envelope)
    if mean_env > 0:
        features['modulation_index'] = np.std(amplitude_envelope) / mean_env
    else:
        features['modulation_index'] = np.nan

    return features

def extract_harmonicity(wav_path):
    try:
        snd = parselmouth.Sound(str(wav_path))
        harmonicity = snd.to_harmonicity()
        hnr_values = harmonicity.values[0]
        valid_hnr = hnr_values[(hnr_values > 0) & (~np.isnan(hnr_values))]
        if len(valid_hnr) > 0:
            return np.mean(valid_hnr)
    except Exception as e:
        # Expected if signal doesn't have clear periodicity
        pass
    return np.nan

class BirdNETEmbeddingExtractor:
    def __init__(self):
        model_path = PROJECT_ROOT / 'venv/lib/python3.11/site-packages/birdnetlib/models/analyzer/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite'
        self.interpreter = tflite.Interpreter(
            model_path=str(model_path),
            experimental_preserve_all_tensors=True
        )
        self.interpreter.allocate_tensors()
        self.input_idx = self.interpreter.get_input_details()[0]['index']

    def extract(self, wav_path):
        y, sr = librosa.load(wav_path, sr=48000, mono=True)
        target_samples = 144000
        
        if len(y) > target_samples:
            y = y[:target_samples]
        elif len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
            
        self.interpreter.set_tensor(self.input_idx, np.float32(y).reshape(1, target_samples))
        self.interpreter.invoke()
        
        emb = self.interpreter.get_tensor(545)
        return emb[0]

def main():
    print("Starting Production Ecoacoustic Feature Extraction")
    
    det_log_path = PROJECT_ROOT / DATA_SEGMENTS / "detection_log.csv"
    if not det_log_path.exists():
        print(f"Error: Could not find {det_log_path}")
        return
        
    df_det = pd.read_csv(det_log_path)
    print(f"Total segments found in log: {len(df_det)}")
    
    features_dir = PROJECT_ROOT / DATA_FEATURES
    features_dir.mkdir(parents=True, exist_ok=True)
    
    segments_dir = PROJECT_ROOT / DATA_SEGMENTS
    
    out_csv = features_dir / "ecoacoustic_features.csv"
    emb_out_npy = features_dir / "embeddings.npy"
    ids_out_csv = features_dir / "embedding_ids.csv"
    
    # Load existing to support resume
    processed_segs = set()
    extracted_data = []
    embedding_ids = []
    embeddings_list = []
    
    if out_csv.exists():
        df_existing = pd.read_csv(out_csv)
        processed_segs = set(df_existing['segment_id'].astype(str))
        extracted_data = df_existing.to_dict('records')
        print(f"Resuming: Loaded {len(processed_segs)} previously processed ecoacoustic features.")
        
    if emb_out_npy.exists() and ids_out_csv.exists():
        embeddings_list = list(np.load(emb_out_npy))
        df_ids = pd.read_csv(ids_out_csv)
        embedding_ids = df_ids.to_dict('records')
        print(f"Resuming: Loaded {len(embeddings_list)} previously extracted embeddings.")
        
        # Ensure parity between CSV and NPY
        if len(embedding_ids) != len(embeddings_list):
            print("WARNING: embedding_ids length mismatch with embeddings.npy. Truncating to shorter.")
            min_len = min(len(embedding_ids), len(embeddings_list))
            embedding_ids = embedding_ids[:min_len]
            embeddings_list = embeddings_list[:min_len]

    # Filter df_det to unprocessed
    df_det['segment_id'] = df_det['segment_id'].astype(str)
    df_todo = df_det[~df_det['segment_id'].isin(processed_segs)]
    print(f"Segments left to process: {len(df_todo)}")
    
    if len(df_todo) == 0:
        print("All segments already processed.")
        return
        
    extractor = BirdNETEmbeddingExtractor()
    
    start_time = time.time()
    processed_count = 0
    failed_count = 0
    
    for _, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Extracting Features"):
        seg_id = row['segment_id']
        rec_id = row['recording_id']
        start_s = row['start_s']
        end_s = row['end_s']
        
        wav_path = segments_dir / f"{seg_id}.wav"
        
        if not wav_path.exists():
            print(f"\n[!] Missing WAV for {seg_id}")
            failed_count += 1
            continue
            
        try:
            # Load for ecoacoustic
            y, sr = librosa.load(wav_path, sr=None)
            
            # PART A - Ecoacoustic
            feats = extract_ecoacoustic_features(y, sr, start_s, end_s)
            feats['harmonicity_hnr_db'] = extract_harmonicity(wav_path)
            
            # PART B - Embedding
            emb = extractor.extract(wav_path)
            
            # Validate embedding
            if emb.shape[0] != 1024 or np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                raise ValueError(f"Invalid embedding generated (shape={emb.shape}, has_nan={np.any(np.isnan(emb))})")
                
            # Append success
            extracted_data.append({
                'segment_id': seg_id,
                'recording_id': rec_id,
                'min_frequency_hz': feats['min_frequency_hz'],
                'peak_frequency_hz': feats['peak_frequency_hz'],
                'spectral_entropy': feats['spectral_entropy'],
                'temporal_entropy': feats['temporal_entropy'],
                'harmonicity_hnr_db': feats['harmonicity_hnr_db'],
                'bout_duration_s': feats['bout_duration_s'],
                'syllable_rate_per_s': feats['syllable_rate_per_s'],
                'modulation_index': feats['modulation_index']
            })
            
            embeddings_list.append(emb)
            embedding_ids.append({'segment_id': seg_id, 'recording_id': rec_id})
            
            processed_count += 1
            
            # Incremental save every 100
            if processed_count % 100 == 0:
                pd.DataFrame(extracted_data).to_csv(out_csv, index=False)
                np.save(emb_out_npy, np.vstack(embeddings_list))
                pd.DataFrame(embedding_ids).to_csv(ids_out_csv, index=False)
                
        except Exception as e:
            failed_count += 1
            # We don't crash, just log and skip
            pass
            
    # Final save
    if processed_count > 0:
        pd.DataFrame(extracted_data).to_csv(out_csv, index=False)
        np.save(emb_out_npy, np.vstack(embeddings_list))
        pd.DataFrame(embedding_ids).to_csv(ids_out_csv, index=False)
        
    elapsed = time.time() - start_time
    
    print("\n── Feature Extraction Summary ──")
    print(f"Runtime Elapsed     : {elapsed:.2f} s")
    print(f"Newly Processed     : {processed_count}")
    print(f"Failed Segments     : {failed_count}")
    print(f"Total Eco Features  : {len(extracted_data)}")
    
    if len(embeddings_list) > 0:
        final_emb_shape = np.vstack(embeddings_list).shape
        print(f"Embedding Matrix    : {final_emb_shape}")
    
    df_out = pd.DataFrame(extracted_data)
    if not df_out.empty:
        print("\n── NaN Counts per Feature ──")
        print(df_out.isna().sum().to_string())

if __name__ == "__main__":
    main()
