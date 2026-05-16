"""
hqi_extractor.py
────────────────
Compute a Habitat Quality Index (HQI) for every recording in
data/raw/xc_metadata.csv using Google Earth Engine and (optionally)
OSMnx / overpy for road density.

Run from the project root:
    python src/hqi_extractor.py

All fixes applied
-----------------
FIX-1  Bad dates (e.g. '1986-06-00'): sanitised in Python before GEE call.
FIX-2  VIIRS empty collections: ee.Algorithms.If guards + DMSP fallback.
FIX-3  Per-record fallback: a failed batch retries each record individually.
FIX-4  Dictionary.getNumber() only takes 1 arg (no default param in GEE API).
       Pattern: ee.Number(ee.Algorithms.If(col.size().gt(0), col.mean()
                .reduceRegion(...).get("KEY"), 0))
       Never:   dict.getNumber("KEY", 0)   ← that is the error you just saw
"""

import logging
import math
import os
import sys
import time
import warnings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import ee

from config import (
    HQI_WEIGHTS,
    HQI_BUFFER_RADIUS_M,
    DATA_RAW,
    DATA_HQI,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Road library — prefer OSMnx, fall back to overpy ─────────────────────────
ROAD_BACKEND = None

try:
    import osmnx as ox
    ROAD_BACKEND = "osmnx"
    log.info("Road backend: OSMnx")
except ImportError:
    try:
        import overpy as _overpy_mod
        if not hasattr(_overpy_mod, "API"):
            raise ImportError("overpy has no 'API' attribute — reinstall it.")
        overpy = _overpy_mod
        ROAD_BACKEND = "overpy"
        log.info("Road backend: overpy")
    except ImportError as _e:
        warnings.warn(
            f"No road backend ({_e}). road_impact will be 0.\n"
            "Install:  pip install osmnx   OR   pip install overpy",
            stacklevel=2,
        )

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
GEE_PROJECT = os.getenv("GEE_PROJECT", "")

BATCH_SIZE         = 50
OVERPASS_DELAY_S   = 1.2
MODIS_SCALE_FACTOR = 0.0001
PARTIAL_CSV        = os.path.join(PROJECT_ROOT, DATA_HQI, "hqi_partial.csv")
VIIRS_START_DATE   = "2012-04-01"
DMSP_START_DATE    = "1992-01-01"

LC_FROM   = [10,  20,  30,  40,  50,  60,  70,  80,  90,  95,  100]
LC_TO_INT = [100, 80,  70,  30,   0,  40,  50,  50,  85,  90,   60]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — GEE init
# ═══════════════════════════════════════════════════════════════════════════════

def init_gee() -> None:
    if not GEE_PROJECT:
        raise EnvironmentError("GEE_PROJECT not set in .env")
    try:
        ee.Initialize(project=GEE_PROJECT)
        log.info("GEE initialised  (project=%s)", GEE_PROJECT)
    except Exception as exc:
        raise RuntimeError(f"ee.Initialize() failed: {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════════
# FIX-1 — Date cleaning
# ═══════════════════════════════════════════════════════════════════════════════

def clean_date(raw) -> str:
    """Clamp day/month=00 to 01 so ee.Date() can always parse the string."""
    if not raw or pd.isna(raw):
        return "2000-01-01"
    parts = str(raw).strip().split("-")
    try:
        year  = int(parts[0]) if len(parts) > 0 else 2000
        month = int(parts[1]) if len(parts) > 1 else 1
        day   = int(parts[2]) if len(parts) > 2 else 1
    except (ValueError, IndexError):
        return "2000-01-01"
    year  = max(1970, min(year,  2030))
    month = max(1,    min(month, 12))
    day   = max(1,    min(day,   28))
    return f"{year:04d}-{month:02d}-{day:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1a–d — GEE extraction
# ═══════════════════════════════════════════════════════════════════════════════

_STATIC_IMAGES: dict = {}


def _get_static_images() -> dict:
    if _STATIC_IMAGES:
        return _STATIC_IMAGES

    _STATIC_IMAGES["naturalness"] = (
        ee.Image("ESA/WorldCover/v200/2021")
        .select("Map")
        .remap(LC_FROM, LC_TO_INT)
        .divide(100.0)
        .rename("naturalness")
    )
    _STATIC_IMAGES["hfi"] = (
        ee.ImageCollection("CSP/HM/GlobalHumanModification")
        .select("gHM")
        .first()
    )
    _STATIC_IMAGES["viirs_start"] = ee.Date(VIIRS_START_DATE)
    _STATIC_IMAGES["dmsp_start"]  = ee.Date(DMSP_START_DATE)
    return _STATIC_IMAGES


def _safe_reduce(image, band: str, geometry, scale: int) -> "ee.Number":
    """
    Reduce an image over a geometry and return the result as ee.Number.
    Returns 0 if the band is missing from the result dict (null-safe).

    FIX-4: Dictionary.getNumber() accepts only 1 argument in the GEE
    Python API — there is no default parameter.  Use .get() inside
    ee.Algorithms.If to supply the fallback safely.
    """
    d = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=scale,
        maxPixels=int(1e9),
    )
    # ee.Dictionary.contains() checks key existence server-side
    return ee.Number(
        ee.Algorithms.If(
            d.contains(band),
            d.get(band),   # .get() returns ee.ComputedObject; cast via ee.Number() above
            0,
        )
    )


def _safe_collection_reduce(
    collection: "ee.ImageCollection",
    band: str,
    geometry,
    scale: int,
) -> "ee.Number":
    """
    Mean-reduce a collection only if it is non-empty; return 0 otherwise.
    FIX-2 + FIX-4 combined: guards empty collections AND avoids the
    two-argument getNumber() call.
    """
    return ee.Number(
        ee.Algorithms.If(
            collection.size().gt(0),
            _safe_reduce(collection.mean(), band, geometry, scale),
            0,
        )
    )


def extract_gee_batch(records: list[dict]) -> list[dict]:
    static          = _get_static_images()
    naturalness_img = static["naturalness"]
    hfi_img         = static["hfi"]
    viirs_start     = static["viirs_start"]
    dmsp_start      = static["dmsp_start"]

    features = [
        ee.Feature(
            ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
            {"recording_id": str(r["id"]), "date": str(r["date"])},
        )
        for r in records
    ]
    fc = ee.FeatureCollection(features)

    def _extract(feature):
        date    = ee.Date(feature.get("date"))
        buf     = feature.geometry().buffer(HQI_BUFFER_RADIUS_M)
        d_start = date.advance(-30, "day")
        d_end   = date.advance( 30, "day")

        # a) NDVI — MODIS 061, 250 m, 16-day composite
        ndvi_val = _safe_collection_reduce(
            ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterDate(d_start, d_end)
                .select("NDVI"),
            "NDVI", buf, 250,
        )

        # b) Naturalness — ESA WorldCover (STATIC)
        nat_val = _safe_reduce(naturalness_img, "naturalness", buf, 10)

        # c) Nighttime lights — VIIRS → DMSP → 0 (FIX-2)
        viirs_val = _safe_collection_reduce(
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
                .filterDate(d_start, d_end)
                .select("avg_rad"),
            "avg_rad", buf, 500,
        )

        # DMSP stable_lights rescaled: DN/63 * 50 ≈ nW/cm²/sr equivalent
        dmsp_raw = _safe_collection_reduce(
            ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS")
                .filterDate(d_start, d_end)
                .select("stable_lights"),
            "stable_lights", buf, 1000,
        )
        dmsp_val = dmsp_raw.divide(63).multiply(50)

        nl_val = ee.Number(
            ee.Algorithms.If(
                date.millis().gte(viirs_start.millis()),
                viirs_val,
                ee.Algorithms.If(
                    date.millis().gte(dmsp_start.millis()),
                    dmsp_val,
                    0,
                ),
            )
        )

        # d) Human Footprint Index — STATIC
        hfi_val = _safe_reduce(hfi_img, "gHM", buf, 1000)

        return feature.set({
            "ndvi":        ndvi_val,
            "naturalness": nat_val,
            "nightlights": nl_val,
            "hfi":         hfi_val,
        })

    result_info = fc.map(_extract).getInfo()

    rows = []
    for feat in result_info["features"]:
        p = feat["properties"]
        rows.append({
            "recording_id":    p.get("recording_id"),
            "ndvi_raw":        p.get("ndvi"),
            "naturalness_raw": p.get("naturalness"),
            "nightlights_raw": p.get("nightlights"),
            "hfi_raw":         p.get("hfi"),
        })
    return rows


def _nan_row(record_id: str) -> dict:
    return {
        "recording_id":    str(record_id),
        "ndvi_raw":        np.nan,
        "naturalness_raw": np.nan,
        "nightlights_raw": np.nan,
        "hfi_raw":         np.nan,
    }


def extract_all_gee(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process all records.  FIX-3: if a full batch fails, retry each of its
    records individually so only the broken one gets NaN.
    """
    records = df[["id", "lat", "lon", "date"]].to_dict("records")
    n       = len(records)

    done_ids: set       = set()
    existing_rows: list = []
    if os.path.exists(PARTIAL_CSV):
        prev      = pd.read_csv(PARTIAL_CSV)
        done_ids  = set(prev["recording_id"].astype(str))
        existing_rows = prev.to_dict("records")
        log.info("Resuming: %d done, %d remaining.", len(done_ids), n - len(done_ids))

    pending = [r for r in records if str(r["id"]) not in done_ids]
    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    all_rows = list(existing_rows)

    for b_idx, batch in enumerate(tqdm(batches, desc="GEE batches", unit="batch")):
        success = False
        for attempt in range(1, 4):
            try:
                rows = extract_gee_batch(batch)
                all_rows.extend(rows)
                success = True
                break
            except Exception as exc:
                wait = 5 * attempt
                log.warning(
                    "Batch %d/%d failed (attempt %d/3): %s — retrying in %ds",
                    b_idx + 1, len(batches), attempt, exc, wait,
                )
                time.sleep(wait)

        if not success:
            log.warning(
                "Batch %d/%d: falling back to per-record mode (%d records).",
                b_idx + 1, len(batches), len(batch),
            )
            rescued = 0
            for record in batch:
                try:
                    row = extract_gee_batch([record])
                    all_rows.extend(row)
                    rescued += 1
                except Exception as exc:
                    log.warning("Record %s failed: %s — NaN.", record["id"], exc)
                    all_rows.append(_nan_row(record["id"]))
            log.info(
                "Batch %d/%d: rescued %d / %d via per-record fallback.",
                b_idx + 1, len(batches), rescued, len(batch),
            )

        processed = min((b_idx + 1) * BATCH_SIZE, len(pending))
        log.info("Progress: %d / %d (batch %d / %d)",
                 len(done_ids) + processed, n, b_idx + 1, len(batches))
        os.makedirs(os.path.dirname(PARTIAL_CSV), exist_ok=True)
        pd.DataFrame(all_rows).to_csv(PARTIAL_CSV, index=False)

    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1e — Road impact
# ═══════════════════════════════════════════════════════════════════════════════

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R    = 6_371_000.0
    phi1 = math.radians(lat1);  phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1);  dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(min(a, 1.0)))


def _road_osmnx(lat, lon, radius_m) -> float:
    try:
        import osmnx as ox
        G = ox.graph_from_point((lat, lon), dist=radius_m, network_type="all")
        return float(ox.graph_to_gdfs(G, nodes=False)["length"].sum())
    except Exception as exc:
        log.warning("OSMnx (%.4f,%.4f): %s", lat, lon, exc)
        return np.nan


def _road_overpy(lat, lon, radius_m, api) -> float:
    d   = radius_m / 111_000.0
    bbox = (lat-d, lon-d, lat+d, lon+d)
    q   = (f"[out:json][timeout:30];"
           f"(way[\"highway\"]({bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}););"
           f"out geom;")
    try:
        result = api.query(q)
        total  = 0.0
        for way in result.ways:
            ns = way.nodes
            for i in range(len(ns)-1):
                total += haversine_m(ns[i].lat, ns[i].lon, ns[i+1].lat, ns[i+1].lon)
        return total
    except Exception as exc:
        log.warning("Overpass (%.4f,%.4f): %s", lat, lon, exc)
        return np.nan


def extract_road_impact(df: pd.DataFrame) -> np.ndarray:
    if ROAD_BACKEND is None:
        log.warning("No road backend — road_impact_raw = 0.")
        return np.zeros(len(df), dtype=float)

    overpy_api = None
    if ROAD_BACKEND == "overpy":
        try:
            import overpy as _op
            overpy_api = _op.API(
                url="https://overpass-api.de/api/interpreter",
                max_retry_count=3, retry_timeout=10,
            )
        except Exception as exc:
            log.error("overpy.API() failed: %s — road_impact = 0.", exc)
            return np.zeros(len(df), dtype=float)

    values = np.full(len(df), np.nan)
    for i, row in enumerate(tqdm(df.itertuples(), total=len(df),
                                  desc=f"Road ({ROAD_BACKEND})", unit="rec")):
        if ROAD_BACKEND == "osmnx":
            values[i] = _road_osmnx(row.lat, row.lon, HQI_BUFFER_RADIUS_M)
        else:
            values[i] = _road_overpy(row.lat, row.lon, HQI_BUFFER_RADIUS_M, overpy_api)
            time.sleep(OVERPASS_DELAY_S)
    return values


# ═══════════════════════════════════════════════════════════════════════════════
# STEPS 2–4 — Normalise + HQI
# ═══════════════════════════════════════════════════════════════════════════════

def minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.where(series.isna(), np.nan, 0.5), index=series.index)
    return (series - lo) / (hi - lo)


def compute_hqi(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["ndvi_raw", "naturalness_raw", "nightlights_raw",
                "hfi_raw", "road_impact_raw"]:
        n = df[col].isna().sum()
        if n:
            med = df[col].median()
            log.warning("Imputing %d NaN in %s with median %.4f", n, col, med)
            df[col] = df[col].fillna(med)

    df["ndvi_raw"]         = df["ndvi_raw"] * MODIS_SCALE_FACTOR
    df["ndvi_norm"]        = minmax_norm(df["ndvi_raw"])
    df["naturalness_norm"] = minmax_norm(df["naturalness_raw"])
    df["nightlights_norm"] = minmax_norm(df["nightlights_raw"])
    df["hfi_norm"]         = minmax_norm(df["hfi_raw"])
    df["road_impact_norm"] = minmax_norm(df["road_impact_raw"])

    w = HQI_WEIGHTS
    df["raw_hqi"] = (
          w["ndvi"]            * df["ndvi_norm"]
        + w["naturalness"]     * df["naturalness_norm"]
        + w["nightlights"]     * df["nightlights_norm"]
        + w["human_footprint"] * df["hfi_norm"]
        + w["road_impact"]     * df["road_impact_norm"]
    )
    df["hqi"] = minmax_norm(df["raw_hqi"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    meta_path = os.path.join(PROJECT_ROOT, DATA_RAW, "xc_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Not found: {meta_path}\nRun xc_collector.py first.")
    df_meta = pd.read_csv(meta_path)
    log.info("Loaded %d recordings from %s", len(df_meta), meta_path)

    # FIX-1: clean dates before any GEE call
    orig = df_meta["date"].astype(str).copy()
    df_meta["date"] = df_meta["date"].apply(clean_date)
    n_fixed = (df_meta["date"] != orig).sum()
    if n_fixed:
        log.warning("Cleaned %d malformed dates.", n_fixed)

    n_pre_viirs = (df_meta["date"] < VIIRS_START_DATE).sum()
    n_pre_dmsp  = (df_meta["date"] < DMSP_START_DATE).sum()
    log.info("%d pre-VIIRS (→ DMSP), %d pre-DMSP (→ 0 nightlights)",
             n_pre_viirs, n_pre_dmsp)

    init_gee()
    _get_static_images()

    log.info("STEP 1a–d: Extracting GEE variables (batch=%d) …", BATCH_SIZE)
    df_gee = extract_all_gee(df_meta)

    log.info("STEP 1e: Road impact …")
    df_gee["recording_id"] = df_gee["recording_id"].astype(str)
    df_meta["id"]          = df_meta["id"].astype(str)
    df_merged = df_meta[["id", "lat", "lon"]].merge(
        df_gee, left_on="id", right_on="recording_id", how="left"
    )
    df_merged["road_impact_raw"] = extract_road_impact(df_merged)

    log.info("STEP 2–4: Normalise + HQI …")
    df_hqi = compute_hqi(df_merged)

    if "recording_id" not in df_hqi.columns and "id" in df_hqi.columns:
        df_hqi = df_hqi.rename(columns={"id": "recording_id"})

    out_cols = [
        "recording_id", "lat", "lon",
        "ndvi_raw", "naturalness_raw", "nightlights_raw",
        "hfi_raw", "road_impact_raw",
        "ndvi_norm", "naturalness_norm", "nightlights_norm",
        "hfi_norm", "road_impact_norm",
        "raw_hqi", "hqi",
    ]
    df_out = df_hqi[out_cols]

    out_path = os.path.join(PROJECT_ROOT, DATA_HQI, "hqi_scores.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    log.info("[✓] Saved → %s  (%d rows)", out_path, len(df_out))

    print("\n── HQI Summary ──────────────────────────────────────")
    print(df_out[["ndvi_norm","naturalness_norm","nightlights_norm",
                  "hfi_norm","road_impact_norm","raw_hqi","hqi"]]
          .describe().round(4).to_string())

    nan_report = df_out[out_cols].isna().sum()
    nan_report = nan_report[nan_report > 0]
    if nan_report.empty:
        log.info("✓ Zero NaN values in final output.")
    else:
        log.warning("Remaining NaN:\n%s", nan_report.to_string())

    if os.path.exists(PARTIAL_CSV):
        os.remove(PARTIAL_CSV)
        log.info("Removed checkpoint: %s", PARTIAL_CSV)


if __name__ == "__main__":
    main()