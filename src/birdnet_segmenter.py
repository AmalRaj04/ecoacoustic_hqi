import os
import sys
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BIRDNET_CONFIDENCE_THRESHOLD, 
    MIN_SEGMENT_DURATION_S, 
    DATA_CLEAN, 
    DATA_SEGMENTS, 
    DATA_RAW
)

def extract_segment(y, sr, start_s, end_s, out_path):
    start_idx = int(start_s * sr)
    end_idx = int(end_s * sr)
    segment = y[start_idx:end_idx]
    sf.write(out_path, segment, sr, subtype='PCM_16')

def main():
    print("Initializing BirdNET Analyzer...")
    analyzer = Analyzer()
    
    attrition_path = PROJECT_ROOT / DATA_RAW / "audio_attrition_log.csv"
    if not attrition_path.exists():
        print(f"Error: Could not find {attrition_path}")
        return
        
    df_log = pd.read_csv(attrition_path)
    df_accepted = df_log[df_log['status'] == 'accepted'].copy()
    
    clean_dir = PROJECT_ROOT / DATA_CLEAN
    segments_dir = PROJECT_ROOT / DATA_SEGMENTS
    segments_dir.mkdir(parents=True, exist_ok=True)
    
    detections_log = []
    
    processed_count = 0
    zero_detections_count = 0
    total_segments = 0
    
    for _, row in tqdm(df_accepted.iterrows(), total=len(df_accepted), desc="Segmenting Audio"):
        rec_id = str(row['recording_id'])
        hqi = row['hqi']
        wav_path = clean_dir / f"{rec_id}.wav"
        
        if not wav_path.exists():
            continue
            
        try:
            # Analyze using birdnetlib
            # Note: default chunk_size is 3.0s in birdnetlib
            recording = Recording(
                analyzer,
                str(wav_path),
                min_conf=BIRDNET_CONFIDENCE_THRESHOLD,
                overlap=0.0
            )
            recording.analyze()
            
            # Load audio for slicing if there are any detections
            if recording.detections:
                y, sr = librosa.load(wav_path, sr=None, mono=True)
            else:
                y, sr = None, None
                
            found_detections = False
            for det in recording.detections:
                species = det['common_name']
                sci_name = det['scientific_name']
                
                # Filter for target species
                if species != "Great Tit" and sci_name != "Parus major":
                    continue
                    
                conf = det['confidence']
                if conf < BIRDNET_CONFIDENCE_THRESHOLD:
                    continue
                    
                start_s = det['start_time']
                end_s = det['end_time']
                duration = end_s - start_s
                
                if duration < MIN_SEGMENT_DURATION_S:
                    continue
                    
                found_detections = True
                start_ms = int(start_s * 1000)
                seg_id = f"{rec_id}_seg_{start_ms}"
                seg_path = segments_dir / f"{seg_id}.wav"
                
                extract_segment(y, sr, start_s, end_s, seg_path)
                
                detections_log.append({
                    'recording_id': rec_id,
                    'segment_id': seg_id,
                    'start_s': start_s,
                    'end_s': end_s,
                    'duration_s': duration,
                    'confidence': conf,
                    'hqi': hqi
                })
                total_segments += 1
                
            processed_count += 1
            if not found_detections:
                zero_detections_count += 1
                
        except Exception as e:
            print(f"Error processing {rec_id}: {e}")
            
    # Save detection log
    df_detections = pd.DataFrame(detections_log)
    log_out = segments_dir / "detection_log.csv"
    if not df_detections.empty:
        df_detections.to_csv(log_out, index=False)
    else:
        # Create empty CSV with columns
        pd.DataFrame(columns=[
            'recording_id', 'segment_id', 'start_s', 'end_s', 
            'duration_s', 'confidence', 'hqi'
        ]).to_csv(log_out, index=False)
        
    print(f"\n[✓] Detection log saved to {log_out}")
    
    # Summary
    mean_segs = total_segments / processed_count if processed_count > 0 else 0
    print("\n── BirdNET Segmentation Summary ──")
    print(f"Total recordings processed: {processed_count}")
    print(f"Total segments extracted: {total_segments}")
    print(f"Mean segments per recording: {mean_segs:.2f}")
    print(f"Recordings with zero detections: {zero_detections_count}")

if __name__ == "__main__":
    main()
