"""
extract_data.py
===============================================================================
Reads every Indian Union Budget PDF found in the "Budget Year Wise" folder,
extracts tables, normalises them into a single canonical schema, cleans the
data, merges all years together and saves the result to data/cleaned_budget.csv.

Design goals
------------
* Never crash. Corrupted / unreadable / empty PDFs are skipped with a warning.
* Prefer Camelot (as requested). If Camelot or its Ghostscript dependency is
  unavailable, fall back to pdfplumber so extraction still works.
* If real extraction produces nothing usable (very common with heterogeneous
  government PDFs), fall back to a clearly-labelled SAMPLE dataset so the
  dashboard is always demonstrable.

Run standalone to (re)build the dataset:
    python extract_data.py
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Folder that holds the year-wise PDF files (relative to this script).
PDF_FOLDER: Path = Path(__file__).parent / "Budget Year Wise"

# Where the cleaned, merged dataset is written.
OUTPUT_CSV: Path = Path(__file__).parent / "data" / "cleaned_budget.csv"

# The single schema every extracted table is normalised into.
# All monetary figures are expressed in Rupees crore (₹ crore).
CANONICAL_COLUMNS: List[str] = [
    "Year",
    "Ministry",
    "Department",
    "Scheme",
    "Revenue",
    "Capital",
    "Total",
]


# ---------------------------------------------------------------------------
# Locating PDF files
# ---------------------------------------------------------------------------

def find_pdf_files(folder: Path = PDF_FOLDER) -> List[Path]:
    """Return a sorted list of all *.pdf files inside ``folder``.

    Returns an empty list (and prints a warning) if the folder does not
    exist or contains no PDFs, instead of raising.
    """
    if not folder.exists():
        print(f"[warn] PDF folder not found: {folder}")
        return []
    pdfs = sorted(p for p in folder.glob("*.pdf") if p.is_file())
    if not pdfs:
        print(f"[warn] No PDF files found in: {folder}")
    return pdfs


def _year_from_filename(name: str) -> str:
    """Best-effort extraction of a budget year label from a filename.

    Handles patterns like '2022-23', '2022_23', '2022 24', '2013-14' or a
    lone '2019'. Falls back to 'Unknown' if nothing matches.
    """
    # 2022-23 / 2022_23 / 2022 23
    m = re.search(r"(20\d{2})[\s_\-](\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # A single 4-digit year.
    m = re.search(r"(20\d{2})", name)
    if m:
        yr = int(m.group(1))
        return f"{yr}-{str(yr + 1)[-2:]}"
    return "Unknown"


# ---------------------------------------------------------------------------
# Low-level table extraction (Camelot primary, pdfplumber fallback)
# ---------------------------------------------------------------------------

def _extract_tables_camelot(pdf_path: Path) -> List[pd.DataFrame]:
    """Extract tables with Camelot. Returns [] on any failure."""
    try:
        import camelot  # imported lazily so a missing install is non-fatal
    except Exception:  # ImportError or dependency (e.g. Ghostscript) issues
        return []

    frames: List[pd.DataFrame] = []
    # 'lattice' works well on ruled tables; 'stream' on whitespace-separated
    # ones. We try both and keep whatever we get.
    for flavor in ("lattice", "stream"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables = camelot.read_pdf(
                    str(pdf_path), pages="all", flavor=flavor
                )
            for t in tables:
                df = t.df
                if df is not None and not df.empty:
                    frames.append(df)
        except Exception as exc:  # noqa: BLE001 - never let one flavor kill us
            print(f"[warn] Camelot ({flavor}) failed on {pdf_path.name}: {exc}")
    return frames


def _extract_tables_pdfplumber(pdf_path: Path) -> List[pd.DataFrame]:
    """Extract tables with pdfplumber. Returns [] on any failure."""
    try:
        import pdfplumber
    except Exception:
        return []

    frames: List[pd.DataFrame] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                try:
                    for table in page.extract_tables() or []:
                        if table and len(table) > 1:
                            frames.append(pd.DataFrame(table))
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] pdfplumber page failed on "
                          f"{pdf_path.name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - corrupted / encrypted PDF, etc.
        print(f"[warn] pdfplumber could not open {pdf_path.name}: {exc}")
    return frames


def extract_tables(pdf_path: Path) -> List[pd.DataFrame]:
    """Extract raw tables from a single PDF, trying Camelot then pdfplumber."""
    frames = _extract_tables_camelot(pdf_path)
    if not frames:
        frames = _extract_tables_pdfplumber(pdf_path)
    return frames


# ---------------------------------------------------------------------------
# Normalisation: messy raw table -> canonical schema
# ---------------------------------------------------------------------------

def _to_number(value: object) -> float:
    """Convert a messy cell (e.g. '1,23,456.7', '(500)', '—') to a float.

    Returns NaN when the cell holds no usable number.
    """
    if value is None:
        return np.nan
    text = str(value).strip()
    if text in {"", "-", "—", "–", "NA", "N/A", "..", "…"}:
        return np.nan
    negative = text.startswith("(") and text.endswith(")")  # accounting style
    text = re.sub(r"[^\d.]", "", text)  # strip commas, ₹, spaces, brackets
    if text in {"", "."}:
        return np.nan
    try:
        num = float(text)
        return -num if negative else num
    except ValueError:
        return np.nan


def _looks_numeric(series: pd.Series, threshold: float = 0.6) -> bool:
    """True if at least ``threshold`` fraction of cells parse as numbers."""
    parsed = series.map(_to_number)
    non_null = parsed.notna().sum()
    return len(series) > 0 and (non_null / len(series)) >= threshold


def _normalize_table(df: pd.DataFrame, year: str) -> Optional[pd.DataFrame]:
    """Map an arbitrary extracted table onto the canonical schema.

    This is intentionally heuristic. Union Budget PDFs vary a great deal
    year to year, so the logic below is a pragmatic best-effort:

      * text columns  -> Ministry / Department / Scheme (by position)
      * numeric cols  -> Revenue / Capital / Total (by position/keywords)

    Returns None if the table has no recognisable structure. If your PDFs
    have a known consistent layout, this is the function to customise.
    """
    if df is None or df.empty or df.shape[1] < 2:
        return None

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Split columns into "text" vs "numeric" by content.
    text_cols, numeric_cols = [], []
    for col in df.columns:
        (numeric_cols if _looks_numeric(df[col]) else text_cols).append(col)

    # We need at least one label column and one numeric column to be useful.
    if not text_cols or not numeric_cols:
        return None

    out = pd.DataFrame()
    out["Year"] = [year] * len(df)

    # Assign the first three text columns to Ministry / Department / Scheme.
    labels = ["Ministry", "Department", "Scheme"]
    for i, label in enumerate(labels):
        out[label] = (
            df[text_cols[i]].astype(str).str.strip()
            if i < len(text_cols) else np.nan
        )

    # Try to identify Revenue / Capital by header keyword; else by position.
    rev_col = cap_col = tot_col = None
    for col in numeric_cols:
        low = col.lower()
        if rev_col is None and "revenue" in low:
            rev_col = col
        elif cap_col is None and "capital" in low:
            cap_col = col
        elif tot_col is None and "total" in low:
            tot_col = col
    remaining = [c for c in numeric_cols if c not in {rev_col, cap_col, tot_col}]
    if rev_col is None and remaining:
        rev_col = remaining.pop(0)
    if cap_col is None and remaining:
        cap_col = remaining.pop(0)

    out["Revenue"] = df[rev_col].map(_to_number) if rev_col else np.nan
    out["Capital"] = df[cap_col].map(_to_number) if cap_col else np.nan
    if tot_col:
        out["Total"] = df[tot_col].map(_to_number)
    else:
        out["Total"] = out["Revenue"].fillna(0) + out["Capital"].fillna(0)

    # Keep only rows that carry at least one real budget figure.
    out = out[out[["Revenue", "Capital", "Total"]].notna().any(axis=1)]
    return out if not out.empty else None


# ---------------------------------------------------------------------------
# Cleaning + merging
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise dtypes, fill gaps sensibly and drop empty rows."""
    df = df.reindex(columns=CANONICAL_COLUMNS)  # guarantee every column exists

    for col in ("Ministry", "Department", "Scheme"):
        df[col] = (
            df[col].astype(str).str.strip().replace({"nan": ""})
        )
        df[col] = df[col].replace("", "Unspecified")

    for col in ("Revenue", "Capital", "Total"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Recompute Total when it is missing but components exist.
    missing_total = df["Total"] <= 0
    df.loc[missing_total, "Total"] = (
        df.loc[missing_total, "Revenue"] + df.loc[missing_total, "Capital"]
    )

    df = df[df["Total"] > 0]                       # drop rows with no money
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_dataset(pdf_folder: Path = PDF_FOLDER) -> pd.DataFrame:
    """Extract, normalise and merge every PDF into one cleaned DataFrame."""
    all_frames: List[pd.DataFrame] = []
    pdf_files = find_pdf_files(pdf_folder)

    for pdf_path in pdf_files:
        year = _year_from_filename(pdf_path.name)
        print(f"[info] Processing {pdf_path.name}  (year={year})")
        try:
            raw_tables = extract_tables(pdf_path)
        except Exception as exc:  # noqa: BLE001 - absolute safety net
            print(f"[warn] Skipping {pdf_path.name}: {exc}")
            continue

        for raw in raw_tables:
            norm = _normalize_table(raw, year)
            if norm is not None:
                all_frames.append(norm)

    if not all_frames:
        print("[warn] No usable tables extracted. Using SAMPLE dataset so the "
              "dashboard still runs. Edit `_normalize_table` to match your "
              "PDFs, then re-run.")
        return generate_sample_dataset()

    merged = pd.concat(all_frames, ignore_index=True)
    return clean_dataframe(merged)


# ---------------------------------------------------------------------------
# Sample dataset (fallback so the app is always demonstrable)
# ---------------------------------------------------------------------------

def generate_sample_dataset() -> pd.DataFrame:
    """Return a small, clearly-labelled synthetic dataset (₹ crore).

    This is NOT real budget data. It exists only so the dashboard renders
    and can be explored before/if real extraction succeeds.
    """
    rng = np.random.default_rng(seed=42)

    years = [f"{y}-{str(y + 1)[-2:]}" for y in range(2013, 2024)]
    ministries = {
        "Ministry of Defence": ["Defence Services", "Defence (Civil)"],
        "Ministry of Finance": ["Revenue", "Expenditure", "Economic Affairs"],
        "Ministry of Home Affairs": ["Police", "Border Management"],
        "Ministry of Railways": ["Railway Board"],
        "Ministry of Rural Development": ["Rural Development", "Land Resources"],
        "Ministry of Health": ["Health & Family Welfare", "Health Research"],
        "Ministry of Education": ["School Education", "Higher Education"],
        "Ministry of Road Transport": ["Roads Wing"],
        "Ministry of Agriculture": ["Agriculture & Farmers Welfare"],
        "Ministry of Railways ": ["Railway Board"],
    }
    schemes = [
        "Capital Outlay", "Establishment", "Central Sector Scheme",
        "Centrally Sponsored Scheme", "Grants-in-Aid", "Subsidies",
        "Infrastructure", "Welfare Programme",
    ]

    rows = []
    for year in years:
        growth = 1.0 + 0.08 * (int(year[:4]) - 2013)  # gentle year-on-year rise
        for ministry, departments in ministries.items():
            for dept in departments:
                for scheme in rng.choice(schemes, size=2, replace=False):
                    revenue = float(rng.integers(500, 60000)) * growth
                    capital = float(rng.integers(100, 40000)) * growth
                    rows.append(
                        {
                            "Year": year,
                            "Ministry": ministry.strip(),
                            "Department": dept,
                            "Scheme": scheme,
                            "Revenue": round(revenue, 2),
                            "Capital": round(capital, 2),
                            "Total": round(revenue + capital, 2),
                        }
                    )

    return clean_dataframe(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Persistence + CLI
# ---------------------------------------------------------------------------

def save_dataset(df: pd.DataFrame, csv_path: Path = OUTPUT_CSV) -> None:
    """Write the dataset to CSV, creating the data folder if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[info] Saved {len(df):,} rows -> {csv_path}")


def main() -> None:
    """Build the dataset from PDFs (or sample) and save it to CSV."""
    print("=" * 70)
    print("INDIA UNION BUDGET ANALYSIS  --  data extraction")
    print("=" * 70)
    df = build_dataset()
    if df.empty:
        print("[warn] Final dataset is empty; writing sample data instead.")
        df = generate_sample_dataset()
    save_dataset(df)
    print("[done] Extraction complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - guarantee the script never dies
        print(f"[fatal] Unexpected error: {exc}", file=sys.stderr)
        # Even on catastrophic failure, leave a usable CSV behind.
        save_dataset(generate_sample_dataset())
