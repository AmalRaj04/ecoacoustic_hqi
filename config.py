# config.py — central configuration, edit these before running any stage

# ── Geographic filter ──────────────────────────────────────────────────
COUNTRIES = ["Germany", "Netherlands", "Belgium", "France",
             "Switzerland", "Austria", "Czech Republic"]

# ── Season filter (breeding season only) ──────────────────────────────
SEASON_START_MONTH = 3   # March
SEASON_END_MONTH   = 6   # June inclusive

# ── Xeno-Canto filters ────────────────────────────────────────────────
SPECIES         = "Parus major"
QUALITY_TIERS   = ["A", "B", "C"]
REQUIRE_GPS     = True

# ── Audio preprocessing ───────────────────────────────────────────────
TARGET_SR       = 48000
TARGET_BITS     = 16
BANDPASS_LOW_HZ = 1500
BANDPASS_HIGH_HZ = 10000
MIN_SNR_DB      = 15.0   # recordings below this are rejected

# ── BirdNET segmentation ──────────────────────────────────────────────
BIRDNET_CONFIDENCE_THRESHOLD = 0.5   # FIXED from 0.8 (avoid urban bias)
MIN_SEGMENT_DURATION_S       = 1.0
MIN_BOUTS_PER_RECORDING      = 3    # recordings with fewer bouts are dropped

# ── HQI formula ───────────────────────────────────────────────────────
# Weights must sum to a defined range; second min-max applied after sum
HQI_WEIGHTS = {
    "ndvi":          +0.30,   # Pettorelli et al. 2005 — NDVI as habitat quality proxy
    "naturalness":   +0.25,   # ESA WorldCover naturalness score
    "nightlights":   -0.20,   # Cinzano et al. 2001 — light pollution as disturbance
    "human_footprint": -0.15, # Venter et al. 2016 — cumulative human impact
    "road_impact":   -0.10,   # van der Ree et al. 2015 — road network as fragmentation
}
HQI_BUFFER_RADIUS_M = 1000   # 1 km around each recording location

# ── ML model ─────────────────────────────────────────────────────────
SPATIAL_BLOCK_SIZE_KM = 50   # spatial cross-validation block size
XGBOOST_PARAMS = {
    "n_estimators": 500, "max_depth": 5,
    "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1,
    "reg_lambda": 1.0, "random_state": 42
}

# ── Paths ─────────────────────────────────────────────────────────────
DATA_RAW        = "data/raw/"
DATA_CLEAN      = "data/clean/"
DATA_SEGMENTS   = "data/segments/"
DATA_FEATURES   = "data/features/"
DATA_AGGREGATED = "data/aggregated/"
DATA_HQI        = "data/hqi/"
CHECKPOINTS_DIR = "checkpoints/"
OUTPUTS_DIR     = "outputs/"