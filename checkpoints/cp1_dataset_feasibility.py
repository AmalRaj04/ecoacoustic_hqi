import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Make project-root importable ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATA_RAW, CHECKPOINTS_DIR, SPATIAL_BLOCK_SIZE_KM

def estimate_spatial_blocks(df, block_size_km):
    """Rough estimate of the number of spatial blocks."""
    if df.empty:
        return 0
    
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    min_lon, max_lon = df['lon'].min(), df['lon'].max()
    
    # 1 degree of latitude is approx 111 km
    dist_y = (max_lat - min_lat) * 111
    
    # 1 degree of longitude is approx 111 km * cos(latitude)
    avg_lat = np.radians(df['lat'].mean())
    dist_x = (max_lon - min_lon) * 111 * np.cos(avg_lat)
    
    blocks_x = max(1, int(np.ceil(dist_x / block_size_km)))
    blocks_y = max(1, int(np.ceil(dist_y / block_size_km)))
    
    return blocks_x * blocks_y

def main():
    metadata_path = os.path.join(PROJECT_ROOT, DATA_RAW, "xc_metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found. Run xc_collector.py first.")
        sys.exit(1)
        
    df = pd.read_csv(metadata_path)
    
    os.makedirs(os.path.join(PROJECT_ROOT, CHECKPOINTS_DIR), exist_ok=True)
    
    total_count = len(df)
    print("=" * 50)
    print("DATASET FEASIBILITY REPORT")
    print("=" * 50)
    print(f"Total recordings: {total_count}")
    
    if total_count < 500:
        print("\nWARNING: Dataset has fewer than 500 recordings. This may be too small for robust spatial cross-validation.")
    
    print("\n--- Per-Country Breakdown ---")
    country_counts = df['country'].value_counts()
    print(country_counts.to_string())
    
    print("\n--- Per-Quality Breakdown ---")
    quality_counts = df['quality'].value_counts()
    print(quality_counts.to_string())
    
    # Bounding box
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    min_lon, max_lon = df['lon'].min(), df['lon'].max()
    print("\n--- Geographic Bounding Box ---")
    print(f"Latitude:  {min_lat:.4f} to {max_lat:.4f}")
    print(f"Longitude: {min_lon:.4f} to {max_lon:.4f}")
    
    # Spatial blocks estimate
    num_blocks = estimate_spatial_blocks(df, SPATIAL_BLOCK_SIZE_KM)
    print(f"\nExpected number of {SPATIAL_BLOCK_SIZE_KM} km spatial blocks (rough estimate): ~{num_blocks}")
    
    # Save summary CSV
    summary_path = os.path.join(PROJECT_ROOT, CHECKPOINTS_DIR, "cp1_summary.csv")
    summary_df = pd.DataFrame({
        'Metric': [
            'Total Recordings', 
            'Total Countries', 
            'Min Latitude', 
            'Max Latitude', 
            'Min Longitude', 
            'Max Longitude', 
            f'Estimated {SPATIAL_BLOCK_SIZE_KM}km Blocks'
        ],
        'Value': [
            total_count,
            len(country_counts),
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            num_blocks
        ]
    })
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[✓] Summary saved to {summary_path}")
    
    # Plotting
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 5))
    
    # 1. Histogram of recording months
    ax1 = plt.subplot(1, 2, 1)
    df['month'].value_counts().sort_index().plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Recordings by Month')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    
    # 2. Map of recording locations
    ax2 = plt.subplot(1, 2, 2)
    countries = df['country'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(countries)))
    
    for country, color in zip(countries, colors):
        subset = df[df['country'] == country]
        ax2.scatter(subset['lon'], subset['lat'], label=country, color=color, alpha=0.7, s=20, edgecolors='none')
        
    ax2.set_title('Recording Locations')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    map_path = os.path.join(PROJECT_ROOT, CHECKPOINTS_DIR, "cp1_location_map.png")
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Plots saved to {map_path}")

if __name__ == "__main__":
    main()
