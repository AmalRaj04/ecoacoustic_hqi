"""
hqi_extractor.py
────────────────
Robust Habitat Quality Index (HQI) extractor using:
- Google Earth Engine
- ESA WorldCover
- MODIS NDVI
- VIIRS nightlights (+ safe fallback handling)
- Human Footprint Index
- OpenStreetMap road density via Overpass

Major fixes
------------
✓ FIX 1 — CSP/HM/GlobalHumanModification is ImageCollection
✓ FIX 2 — MODIS upgraded to 061
✓ FIX 3 — Proper _extract indentation
✓ FIX 4 — Safe handling when avg_rad missing
✓ FIX 5 — Invalid XC dates (1986-06-00 etc.)
✓ FIX 6 — One bad record no longer kills entire batch
✓ FIX 7 — overpy modern API fix
✓ FIX 8 — Faster retries / reduced runtime
✓ FIX 9 — Robust reduceRegion dictionary access
✓ FIX 10 — Safe Overpass node handling
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

import ee
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    HQI_WEIGHTS,
    HQI_BUFFER_RADIUS_M,
    DATA_RAW,
    DATA_HQI,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# overpy
# ──────────────────────────────────────────────────────────────────────────────

try:
    import overpy

    OVERPY_AVAILABLE = True

except ImportError:
    OVERPY_AVAILABLE = False
    warnings.warn(
        "overpy not installed — road impact disabled.",
        stacklevel=2,
    )

# ──────────────────────────────────────────────────────────────────────────────
# ENV
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
GEE_PROJECT = os.getenv("GEE_PROJECT", "")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 50
MODIS_SCALE_FACTOR = 0.0001
OVERPASS_DELAY_S = 1.0

PARTIAL_CSV = os.path.join(PROJECT_ROOT, DATA_HQI, "hqi_partial.csv")

# ESA WorldCover remap
LC_FROM = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
LC_TO_INT = [100, 80, 70, 30, 0, 40, 50, 50, 85, 90, 60]

_STATIC_IMAGES = {}

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════


def clean_date(date_str: str) -> str:
    """
    Fix invalid XC dates like:
    1986-06-00
    2001-00-00
    """

    try:
        parts = str(date_str).split("-")

        if len(parts) != 3:
            return "2000-01-01"

        y, m, d = parts

        if m == "00":
            m = "01"

        if d == "00":
            d = "01"

        return f"{y}-{m}-{d}"

    except Exception:
        return "2000-01-01"


def safe_get(dictionary, key):
    """
    Safely get value from ee.Dictionary.
    Returns None if key missing.
    """

    return ee.Algorithms.If(
        ee.Dictionary(dictionary).contains(key),
        ee.Dictionary(dictionary).get(key),
        None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GEE INIT
# ══════════════════════════════════════════════════════════════════════════════


def init_gee():

    if not GEE_PROJECT:
        raise EnvironmentError(
            "GEE_PROJECT missing in .env"
        )

    try:
        ee.Initialize(project=GEE_PROJECT)
        log.info("GEE initialised (project=%s)", GEE_PROJECT)

    except Exception as exc:
        raise RuntimeError(
            f"GEE init failed: {exc}"
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# Static images
# ══════════════════════════════════════════════════════════════════════════════


def _get_static_images():

    if _STATIC_IMAGES:
        return _STATIC_IMAGES

    # Naturalness
    _STATIC_IMAGES["naturalness"] = (
        ee.Image("ESA/WorldCover/v200/2021")
        .select("Map")
        .remap(LC_FROM, LC_TO_INT)
        .divide(100.0)
        .rename("naturalness")
    )

    # Human footprint
    _STATIC_IMAGES["hfi"] = (
        ee.ImageCollection("CSP/HM/GlobalHumanModification")
        .select("gHM")
        .first()
    )

    return _STATIC_IMAGES


# ══════════════════════════════════════════════════════════════════════════════
# GEE extraction
# ══════════════════════════════════════════════════════════════════════════════


def extract_gee_batch(records):

    static = _get_static_images()

    naturalness_img = static["naturalness"]
    hfi_img = static["hfi"]

    features = []

    for r in records:

        try:
            feat = ee.Feature(
                ee.Geometry.Point(
                    [float(r["lon"]), float(r["lat"])]
                ),
                {
                    "recording_id": str(r["id"]),
                    "date": clean_date(str(r["date"])),
                },
            )

            features.append(feat)

        except Exception:
            continue

    fc = ee.FeatureCollection(features)

    def _extract(feature):

        date = ee.Date(feature.get("date"))

        buffer = feature.geometry().buffer(
            HQI_BUFFER_RADIUS_M
        )

        d_start = date.advance(-30, "day")
        d_end = date.advance(30, "day")

        # ─────────────────────────────────────────────────────────────
        # NDVI
        # ─────────────────────────────────────────────────────────────

        ndvi_dict = (
            ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate(d_start, d_end)
            .select("NDVI")
            .mean()
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=250,
                maxPixels=int(1e7),
            )
        )

        ndvi_val = safe_get(ndvi_dict, "NDVI")

        # ─────────────────────────────────────────────────────────────
        # Naturalness
        # ─────────────────────────────────────────────────────────────

        nat_dict = (
            naturalness_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=10,
                maxPixels=int(1e9),
            )
        )

        nat_val = safe_get(nat_dict, "naturalness")

        # ─────────────────────────────────────────────────────────────
        # Nightlights
        # ─────────────────────────────────────────────────────────────

        nl_dict = (
            ee.ImageCollection(
                "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
            )
            .filterDate(d_start, d_end)
            .select("avg_rad")
            .mean()
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=500,
                maxPixels=int(1e7),
            )
        )

        nl_val = safe_get(nl_dict, "avg_rad")

        # ─────────────────────────────────────────────────────────────
        # Human footprint
        # ─────────────────────────────────────────────────────────────

        hfi_dict = (
            hfi_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=1000,
                maxPixels=int(1e7),
            )
        )

        hfi_val = safe_get(hfi_dict, "gHM")

        return feature.set(
            {
                "ndvi": ndvi_val,
                "naturalness": nat_val,
                "nightlights": nl_val,
                "hfi": hfi_val,
            }
        )

    result_info = fc.map(_extract).getInfo()

    rows = []

    for feat in result_info["features"]:

        p = feat["properties"]

        rows.append(
            {
                "recording_id": p.get("recording_id"),
                "ndvi_raw": p.get("ndvi"),
                "naturalness_raw": p.get("naturalness"),
                "nightlights_raw": p.get("nightlights"),
                "hfi_raw": p.get("hfi"),
            }
        )

    return rows


def extract_all_gee(df):

    records = df[
        ["id", "lat", "lon", "date"]
    ].to_dict("records")

    pending = records

    batches = [
        pending[i:i + BATCH_SIZE]
        for i in range(0, len(pending), BATCH_SIZE)
    ]

    all_rows = []

    for b_idx, batch in enumerate(
        tqdm(batches, desc="GEE batches", unit="batch")
    ):

        try:

            rows = extract_gee_batch(batch)
            all_rows.extend(rows)

            log.info(
                "Progress: %d / %d recordings processed",
                min((b_idx + 1) * BATCH_SIZE, len(records)),
                len(records),
            )

        except Exception as exc:

            log.error(
                "Batch %d failed: %s",
                b_idx + 1,
                exc,
            )

            for r in batch:

                all_rows.append(
                    {
                        "recording_id": str(r["id"]),
                        "ndvi_raw": np.nan,
                        "naturalness_raw": np.nan,
                        "nightlights_raw": np.nan,
                        "hfi_raw": np.nan,
                    }
                )

    return pd.DataFrame(all_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Road impact
# ══════════════════════════════════════════════════════════════════════════════


def haversine_m(lat1, lon1, lat2, lon2):

    R = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlam / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(min(a, 1.0)))


def _road_length_m(lat, lon, radius_m, api):

    d_deg = radius_m / 111000.0

    bbox = (
        lat - d_deg,
        lon - d_deg,
        lat + d_deg,
        lon + d_deg,
    )

    query = (
        f'[out:json][timeout:30];'
        f'(way["highway"]'
        f'({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}););'
        f'out body;>;out skel qt;'
    )

    try:

        result = api.query(query)

        total = 0.0

        for way in result.ways:

            try:
                nodes = way.get_nodes(resolve_missing=True)

                for i in range(len(nodes) - 1):

                    n1 = nodes[i]
                    n2 = nodes[i + 1]

                    total += haversine_m(
                        float(n1.lat),
                        float(n1.lon),
                        float(n2.lat),
                        float(n2.lon),
                    )

            except Exception:
                continue

        return total

    except Exception as exc:

        log.warning(
            "Overpass failed at (%.4f, %.4f): %s",
            lat,
            lon,
            exc,
        )

        return np.nan


def extract_road_impact(df):

    if not OVERPY_AVAILABLE:

        log.warning(
            "overpy unavailable — road impact disabled."
        )

        return np.zeros(len(df))

    try:

        api = overpy.Overpass(
            url="https://overpass-api.de/api/interpreter",
            max_retry_count=2,
            retry_timeout=5,
        )

    except Exception as exc:

        log.error(
            "Failed creating Overpass client: %s",
            exc,
        )

        return np.zeros(len(df))

    values = np.full(len(df), np.nan)

    for i, row in enumerate(
        tqdm(
            df.itertuples(),
            total=len(df),
            desc="Road impact",
            unit="rec",
        )
    ):

        values[i] = _road_length_m(
            row.lat,
            row.lon,
            HQI_BUFFER_RADIUS_M,
            api,
        )

        time.sleep(OVERPASS_DELAY_S)

    return values


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation
# ══════════════════════════════════════════════════════════════════════════════


def minmax_norm(series):

    lo = series.min()
    hi = series.max()

    if hi == lo:

        return pd.Series(
            np.where(series.isna(), np.nan, 0.5),
            index=series.index,
        )

    return (series - lo) / (hi - lo)


# ══════════════════════════════════════════════════════════════════════════════
# HQI
# ══════════════════════════════════════════════════════════════════════════════


def compute_hqi(df):

    raw_cols = [
        "ndvi_raw",
        "naturalness_raw",
        "nightlights_raw",
        "hfi_raw",
        "road_impact_raw",
    ]

    for col in raw_cols:

        n_missing = df[col].isna().sum()

        if n_missing:

            median = df[col].median()

            log.warning(
                "Imputing %d NaNs in %s",
                n_missing,
                col,
            )

            df[col] = df[col].fillna(median)

    df["ndvi_raw"] = (
        df["ndvi_raw"] * MODIS_SCALE_FACTOR
    )

    df["ndvi_norm"] = minmax_norm(df["ndvi_raw"])
    df["naturalness_norm"] = minmax_norm(df["naturalness_raw"])
    df["nightlights_norm"] = minmax_norm(df["nightlights_raw"])
    df["hfi_norm"] = minmax_norm(df["hfi_raw"])
    df["road_impact_norm"] = minmax_norm(df["road_impact_raw"])

    w = HQI_WEIGHTS

    df["raw_hqi"] = (
        w["ndvi"] * df["ndvi_norm"]
        + w["naturalness"] * df["naturalness_norm"]
        + w["nightlights"] * df["nightlights_norm"]
        + w["human_footprint"] * df["hfi_norm"]
        + w["road_impact"] * df["road_impact_norm"]
    )

    df["hqi"] = minmax_norm(df["raw_hqi"])

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():

    metadata_path = os.path.join(
        PROJECT_ROOT,
        DATA_RAW,
        "xc_metadata.csv",
    )

    if not os.path.exists(metadata_path):

        raise FileNotFoundError(
            f"Missing input: {metadata_path}"
        )

    df_meta = pd.read_csv(metadata_path)

    log.info(
        "Loaded %d recordings",
        len(df_meta),
    )

    init_gee()

    _get_static_images()

    log.info("Extracting GEE variables...")
    df_gee = extract_all_gee(df_meta)

    df_gee["recording_id"] = (
        df_gee["recording_id"].astype(str)
    )

    df_meta["id"] = (
        df_meta["id"].astype(str)
    )

    df_merged = df_meta[
        ["id", "lat", "lon"]
    ].merge(
        df_gee,
        left_on="id",
        right_on="recording_id",
        how="left",
    )

    log.info("Computing road impact...")
    df_merged["road_impact_raw"] = (
        extract_road_impact(df_merged)
    )

    log.info("Computing HQI...")
    df_hqi = compute_hqi(df_merged)

    output_cols = [
        "recording_id",
        "lat",
        "lon",
        "ndvi_raw",
        "naturalness_raw",
        "nightlights_raw",
        "hfi_raw",
        "road_impact_raw",
        "ndvi_norm",
        "naturalness_norm",
        "nightlights_norm",
        "hfi_norm",
        "road_impact_norm",
        "raw_hqi",
        "hqi",
    ]

    df_out = df_hqi[output_cols]

    out_path = os.path.join(
        PROJECT_ROOT,
        DATA_HQI,
        "hqi_scores.csv",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df_out.to_csv(out_path, index=False)

    log.info(
        "[✓] HQI saved → %s (%d rows)",
        out_path,
        len(df_out),
    )

    print("\n── HQI Summary ─────────────────────")

    print(
        df_out[
            [
                "ndvi_norm",
                "naturalness_norm",
                "nightlights_norm",
                "hfi_norm",
                "road_impact_norm",
                "raw_hqi",
                "hqi",
            ]
        ]
        .describe()
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()