"""
xc_collector.py
───────────────
Collect Great Tit (Parus major) recordings from the Xeno-Canto API v3,
apply geographic / seasonal / quality filters, and write a metadata CSV.

Run from the project root:
    python src/xc_collector.py

Or from any directory — the script adds the project root to sys.path so
config.py is always importable.
"""

import os
import sys
import time
import math

# ── Make project-root importable no matter where the script is invoked from ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    COUNTRIES,
    SEASON_START_MONTH,
    SEASON_END_MONTH,
    QUALITY_TIERS,
    SPECIES,
    DATA_RAW,
)

# ── Load API key from .env ────────────────────────────────────────────────────
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
XC_API_KEY = os.getenv("XC_API_KEY", "")
if not XC_API_KEY:
    raise EnvironmentError(
        "XC_API_KEY is not set. Add it to your .env file at the project root."
    )

# ── Constants ─────────────────────────────────────────────────────────────────
XC_ENDPOINT   = "https://xeno-canto.org/api/3/recordings"
PER_PAGE      = 500          # maximum allowed by the API
RETRY_LIMIT   = 3
RETRY_BACKOFF = 2.0          # seconds between retries

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_query() -> str:
    """
    Build an API v3 tag-based query string for Parus major at quality A/B/C.
    The XC API uses A > B > C > D > E ordering.
    q:">D" returns recordings with quality better than D, i.e. A, B, C.
    The query is returned as a plain string; it will be appended to the URL
    manually (not via requests params dict) to avoid double-encoding of
    operators like >, <, and quotation marks.
    """
    genus, epithet = SPECIES.split()
    # Spaces between tags replaced with '+'; quality range quoted as XC requires
    return f'gen:{genus.lower()}+sp:{epithet.lower()}+q:">D"'


def _fetch_page(query: str, page: int, session: requests.Session) -> dict:
    """
    Fetch a single page from the XC API, with retries on transient errors.

    IMPORTANT: We build the URL manually instead of passing a params dict to
    requests.  The XC API expects operators like >D and quoted strings such as
    q:">D" to appear *literally* in the query string.  When requests encodes
    a dict, it percent-encodes quotes and converts + to %2B, both of which
    cause the API to return a 400 error.
    """
    # Manually encode only the query value, preserving +, ", < and >
    from urllib.parse import quote
    encoded_query = quote(query, safe='":+<>') 
    url = (
        f"{XC_ENDPOINT}"
        f"?query={encoded_query}"
        f"&key={XC_API_KEY}"
        f"&per_page={PER_PAGE}"
        f"&page={page}"
    )
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = session.get(url, timeout=30)
            if not resp.ok:
                # Surface the actual API error message before raising
                try:
                    err_body = resp.json()
                    err_msg  = err_body.get("error", {}).get("message", resp.text)
                except ValueError:
                    err_msg = resp.text[:300]
                raise requests.HTTPError(
                    f"HTTP {resp.status_code}: {err_msg}", response=resp
                )
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == RETRY_LIMIT:
                raise RuntimeError(
                    f"API request failed after {RETRY_LIMIT} attempts: {exc}"
                ) from exc
            time.sleep(RETRY_BACKOFF * attempt)
    return {}  # unreachable


def _parse_month(date_str: str) -> int | None:
    """
    Extract the month integer from a date string like '2021-12-23'.
    Returns None if the string is missing or malformed.
    """
    if not date_str:
        return None
    parts = date_str.split("-")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    """Return float or None for empty / non-numeric coordinate strings."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fetch_all_recordings(query: str) -> list[dict]:
    """
    Page through the XC API and return every recording object as a raw dict.
    Shows a tqdm progress bar over pages.
    """
    session = requests.Session()

    # ── First request to discover total pages ─────────────────────────────
    print(f"\n[XC API] Query: {query}")
    first = _fetch_page(query, page=1, session=session)

    if "error" in first:
        raise RuntimeError(
            f"API returned an error: {first['error'].get('message', first['error'])}"
        )

    num_pages      = int(first.get("numPages", 1))
    num_recordings = first.get("numRecordings", "?")
    print(f"[XC API] Total recordings reported: {num_recordings}  |  Pages: {num_pages}")

    all_recordings: list[dict] = list(first.get("recordings", []))

    if num_pages > 1:
        for page in tqdm(range(2, num_pages + 1), desc="Fetching pages", unit="page"):
            data = _fetch_page(query, page=page, session=session)
            all_recordings.extend(data.get("recordings", []))
            time.sleep(0.25)   # be polite

    return all_recordings


def build_dataframe(raw: list[dict]) -> pd.DataFrame:
    """
    Convert raw API recording dicts into a tidy DataFrame with the fields
    required by the project spec.
    """
    rows = []
    for rec in raw:
        lat = _parse_float(rec.get("lat"))
        lon = _parse_float(rec.get("lon") or rec.get("lng"))  # API uses 'lon' in v3
        date_str = rec.get("date", "")
        month    = _parse_month(date_str)

        # also_species_list — join list to a pipe-separated string
        also = rec.get("also", [])
        also_str = "|".join(also) if isinstance(also, list) else str(also)

        rows.append({
            "id":               rec.get("id", ""),
            "lat":              lat,
            "lon":              lon,
            "date":             date_str,
            "month":            month,
            "country":          rec.get("cnt", ""),
            "quality":          rec.get("q", ""),
            "type":             rec.get("type", ""),
            "length":           rec.get("length", ""),
            "file_url":         rec.get("file", ""),
            "file_name":        rec.get("file-name", ""),
            "also_species_list": also_str,
        })

    return pd.DataFrame(rows)


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply all project filters and return (filtered_df, attrition_counts).
    Attrition is tracked cumulatively so that each stage is independent.
    """
    total_retrieved = len(df)

    # ── 1. Deduplicate by recording ID ───────────────────────────────────
    df = df.drop_duplicates(subset="id")
    after_dedup = len(df)

    # ── 2. GPS filter ─────────────────────────────────────────────────────
    mask_gps   = df["lat"].notna() & df["lon"].notna()
    removed_gps = (~mask_gps).sum()
    df = df[mask_gps].copy()

    # ── 3. Season filter  (month 3–6 inclusive) ───────────────────────────
    mask_season   = df["month"].between(SEASON_START_MONTH, SEASON_END_MONTH)
    removed_season = (~mask_season).sum()
    df = df[mask_season].copy()

    # ── 4. Country filter ─────────────────────────────────────────────────
    countries_lower = {c.lower() for c in COUNTRIES}
    mask_country    = df["country"].str.lower().isin(countries_lower)
    removed_country = (~mask_country).sum()
    df = df[mask_country].copy()

    # ── 5. Quality filter (keep only QUALITY_TIERS) ───────────────────────
    mask_quality    = df["quality"].isin(QUALITY_TIERS)
    removed_quality = (~mask_quality).sum()
    df = df[mask_quality].copy()

    attrition = {
        "total_retrieved":       total_retrieved,
        "removed_duplicates":    total_retrieved - after_dedup,
        "removed_missing_gps":   removed_gps,
        "removed_wrong_season":  removed_season,
        "removed_wrong_country": removed_country,
        "removed_wrong_quality": removed_quality,
        "final_usable":          len(df),
    }
    return df, attrition


def print_attrition_table(attrition: dict) -> None:
    """Pretty-print the attrition funnel."""
    width = 42
    print("\n" + "═" * width)
    print(f"{'ATTRITION TABLE':^{width}}")
    print("═" * width)
    rows = [
        ("Total retrieved from API",     attrition["total_retrieved"]),
        ("  − duplicate IDs removed",    attrition["removed_duplicates"]),
        ("  − missing GPS",              attrition["removed_missing_gps"]),
        ("  − outside season (Mar–Jun)", attrition["removed_wrong_season"]),
        ("  − outside target countries", attrition["removed_wrong_country"]),
        ("  − quality below tier C",     attrition["removed_wrong_quality"]),
        ("",                             None),
        ("✔ Final usable recordings",    attrition["final_usable"]),
    ]
    for label, value in rows:
        if value is None:
            print("─" * width)
        else:
            print(f"  {label:<36} {value:>4}")
    print("═" * width)


def print_quality_breakdown(df: pd.DataFrame) -> None:
    """Print per-quality-tier counts and percentages."""
    print("\n── Quality breakdown of final dataset ──")
    total = len(df)
    for q in ["A", "B", "C"]:
        count = (df["quality"] == q).sum()
        pct   = 100 * count / total if total else 0
        bar   = "█" * int(pct / 5)
        print(f"  Quality {q}: {count:>4}  ({pct:5.1f}%)  {bar}")
    print()


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """Ensure output directory exists, then write CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] CSV saved → {output_path}  ({len(df)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    query        = _build_query()
    raw_records  = fetch_all_recordings(query)

    df_raw       = build_dataframe(raw_records)
    df_final, attrition = apply_filters(df_raw)

    print_attrition_table(attrition)
    print_quality_breakdown(df_final)

    output_path  = os.path.join(PROJECT_ROOT, DATA_RAW, "xc_metadata.csv")
    save_csv(df_final, output_path)


if __name__ == "__main__":
    main()
