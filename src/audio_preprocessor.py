import os
import sys
import time
import logging
import requests
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to sys.path so config can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_RAW, DATA_HQI, DATA_CLEAN, TARGET_SR,
    BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, MIN_SNR_DB
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def download_audio(url: str, output_path: Path) -> bool:
    """
    Downloads an audio file via streaming.
    Returns: bool indicating success or existing.
    """
    if output_path.exists():
        return True

    if pd.isna(url) or not url:
        return False

    if url.startswith('//'):
        url = 'https:' + url

    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()  # Check for 404, 403, etc.
        
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return True
        
    except Exception as e:
        log.warning(f"Download failed for {url}: {e}")
        if output_path.exists():
            output_path.unlink()  # Remove partial downloads
        return False

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter using filtfilt for zero phase shift."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def estimate_snr(y, fs):
    """Estimate SNR using RMS energy of signal vs noise proxy bands."""
    # Signal band: 2000 - 9000 Hz
    y_sig = butter_bandpass_filter(y, 2000, 9000, fs, order=4)
    rms_sig = np.sqrt(np.mean(y_sig**2))
    
    # Noise proxy: 200 - 800 Hz
    y_noise = butter_bandpass_filter(y, 200, 800, fs, order=4)
    rms_noise = np.sqrt(np.mean(y_noise**2))
    
    # Prevent divide by zero
    if rms_noise < 1e-10:
        if rms_sig > 1e-10:
            return 100.0  # Perfect SNR
        else:
            return 0.0    # Absolute silence
            
    snr = 20 * np.log10(rms_sig / rms_noise)
    return snr

def detect_clipping(y, threshold=0.98, max_fraction=0.001):
    """Detect if audio is overly clipped."""
    fraction_clipped = np.mean(np.abs(y) > threshold)
    return fraction_clipped > max_fraction

def main():
    log.info("Starting Production-Grade Resumable Audio Pipeline")
    
    # STEP 1 — Read CSVs
    xc_meta_path = PROJECT_ROOT / DATA_RAW / "xc_metadata.csv"
    hqi_path = PROJECT_ROOT / DATA_HQI / "hqi_scores.csv"
    log_path = PROJECT_ROOT / DATA_RAW / "audio_attrition_log.csv"
    
    if not xc_meta_path.exists() or not hqi_path.exists():
        log.error(f"Missing required CSV files. Please check paths.")
        return

    df_meta = pd.read_csv(xc_meta_path)
    df_hqi = pd.read_csv(hqi_path)
    
    df_meta['id'] = df_meta['id'].astype(str)
    df_hqi['recording_id'] = df_hqi['recording_id'].astype(str)
    
    # Inner join
    df_merged = df_meta.merge(df_hqi, left_on='id', right_on='recording_id', how='inner')
    log.info(f"Total merged recordings: {len(df_merged)}")
    
    audio_dir = PROJECT_ROOT / DATA_RAW / "audio"
    clean_dir = PROJECT_ROOT / DATA_CLEAN
    audio_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing attrition log to resume
    processed_ids = set()
    attrition_log = []
    if log_path.exists():
        df_existing = pd.read_csv(log_path)
        df_existing['recording_id'] = df_existing['recording_id'].astype(str)
        processed_ids = set(df_existing['recording_id'].unique())
        attrition_log = df_existing.to_dict('records')
        log.info(f"Resuming from existing log: {len(processed_ids)} recordings already processed.")
        
    # Filter df_merged to only unprocessed
    df_todo = df_merged[~df_merged['recording_id'].isin(processed_ids)]
    log.info(f"Recordings left to process: {len(df_todo)}")
    
    meter = pyln.Meter(TARGET_SR)
    
    # STEP 2 & 3 — For each recording: process
    new_processed_count = 0
    
    for _, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Processing Audio"):
        rec_id = str(row['id'])
        url = row['file_url']
        mp3_path = audio_dir / f"{rec_id}.mp3"
        wav_path = clean_dir / f"{rec_id}.wav"
        
        status = 'accepted'
        snr_db = np.nan
        
        # Check if clean file already exists, if so skip processing (assume accepted)
        # But this case shouldn't happen if they are not in attrition_log, unless log was deleted.
        if wav_path.exists():
            log.info(f"[{rec_id}] Skipping - Processed WAV already exists.")
            status = 'accepted'
            
        else:
            # a) Download audio file
            success = download_audio(url, mp3_path)
            
            if not success:
                log.error(f"[{rec_id}] Download failed.")
                status = 'download_failed'
            else:
                try:
                    # b) Load, resample to 48000 Hz, convert to mono
                    try:
                        y, sr = librosa.load(mp3_path, sr=None, mono=True)
                        if sr != TARGET_SR:
                            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                        sr = TARGET_SR
                    except Exception as e:
                        log.error(f"[{rec_id}] Decoding failure: {e}")
                        status = 'decoding_failed'
                        y = None
                        
                    if status == 'accepted' and y is not None:
                        if len(y) == 0:
                            log.error(f"[{rec_id}] Empty audio file")
                            status = 'decoding_failed'
                        else:
                            # e) Detect clipping before applying gains or filters
                            if detect_clipping(y):
                                log.warning(f"[{rec_id}] Rejected: Clipping detected.")
                                status = 'rejected_clipping'
                            else:
                                # d) Estimate SNR
                                snr_db = estimate_snr(y, sr)
                                
                                if snr_db < MIN_SNR_DB:
                                    log.warning(f"[{rec_id}] Rejected: SNR ({snr_db:.2f} dB) < {MIN_SNR_DB} dB.")
                                    status = 'rejected_snr'
                                else:
                                    # c) Apply gentle bandpass filter
                                    y_filtered = butter_bandpass_filter(y, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, sr, order=4)
                                    
                                    # f) Loudness normalisation to -23 LUFS
                                    loudness = meter.integrated_loudness(y_filtered)
                                    if np.isinf(loudness):
                                        log.error(f"[{rec_id}] Rejected: Infinite loudness (silent).")
                                        status = 'decoding_failed'
                                    else:
                                        y_norm = pyln.normalize.loudness(y_filtered, loudness, -23.0)
                                        
                                        # Soft clip to ensure no overflow after normalisation
                                        y_norm = np.clip(y_norm, -1.0, 1.0)
                                        
                                        # g) Export to clean/ as 16-bit PCM WAV
                                        try:
                                            sf.write(wav_path, y_norm, TARGET_SR, subtype='PCM_16')
                                        except Exception as e:
                                            log.error(f"[{rec_id}] Export failure: {e}")
                                            status = 'export_failed'
                                            
                except Exception as e:
                    log.error(f"[{rec_id}] Unexpected failure: {e}")
                    status = 'unexpected_failed'
                    
        # Track rejection
        attrition_log.append({
            'recording_id': rec_id,
            'status': status,
            'snr_db': round(snr_db, 2) if not np.isnan(snr_db) else np.nan,
            'quality_tier': row['quality'],
            'hqi': row['hqi']
        })
        
        new_processed_count += 1
        
        # Save attrition logs incrementally every 25 recordings
        if new_processed_count % 25 == 0:
            pd.DataFrame(attrition_log).to_csv(log_path, index=False)
            
        # Be polite to the API if we downloaded something
        if not wav_path.exists():
            time.sleep(0.5)

    # Final save
    if new_processed_count > 0:
        pd.DataFrame(attrition_log).to_csv(log_path, index=False)
        log.info(f"Saved final attrition log to {log_path}")
    
    # STEP 4 — Print summary table
    df_log = pd.DataFrame(attrition_log)
    if len(df_log) > 0:
        print("\n── Audio Preprocessing Summary ──")
        status_counts = df_log['status'].value_counts()
        for st, count in status_counts.items():
            print(f"  {st:<20}: {count:>4}")
            
        print("\n── Rejection Rates by Quality Tier ──")
        tiers = df_log['quality_tier'].dropna().unique()
        for t in sorted(tiers):
            df_t = df_log[df_log['quality_tier'] == t]
            total_t = len(df_t)
            if total_t == 0:
                continue
            rejected_t = len(df_t[df_t['status'] != 'accepted'])
            rate = 100 * rejected_t / total_t
            print(f"  Tier {t}: {rate:5.1f}% ({rejected_t}/{total_t})")

if __name__ == "__main__":
    main()
