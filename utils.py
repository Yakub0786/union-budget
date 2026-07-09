"""
utils.py
===============================================================================
Reusable helpers for the Streamlit dashboard:

  * loading + caching the cleaned dataset
  * Indian-style number formatting
  * building filter option lists
  * applying combined filters (year -> ministry -> department -> scheme -> kw)
  * computing headline metrics
  * building every Plotly chart
  * exporting data to Excel bytes

Keeping this logic out of app.py keeps the UI file readable and the data
logic testable / reusable.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from extract_data import (
    CANONICAL_COLUMNS,
    OUTPUT_CSV,
    build_dataset,
    generate_sample_dataset,
    save_dataset,
)

# A colourway used consistently across every chart for a professional look.
COLOR_SEQUENCE: List[str] = px.colors.qualitative.Bold
PLOTLY_TEMPLATE: str = "plotly_dark"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: Path = OUTPUT_CSV) -> pd.DataFrame:
    """Load the cleaned dataset, building it on the fly if the CSV is absent.

    The returned DataFrame is always guaranteed to have CANONICAL_COLUMNS,
    so the rest of the app can rely on the schema.
    """
    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            # First run: try to build from PDFs, else fall back to sample.
            df = build_dataset()
            save_dataset(df, csv_path)
    except Exception:  # noqa: BLE001 - never let a bad CSV crash the app
        df = generate_sample_dataset()

    # Guarantee schema + dtypes.
    df = df.reindex(columns=CANONICAL_COLUMNS)
    for col in ("Revenue", "Capital", "Total"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ("Year", "Ministry", "Department", "Scheme"):
        df[col] = df[col].astype(str)
    return df


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_inr_crore(value: float) -> str:
    """Format a ₹-crore figure using the Indian grouping system.

    Example: 1234567.0 -> '₹ 12,34,567 Cr'
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "₹ 0 Cr"

    negative = value < 0
    whole = f"{abs(value):,.0f}"

    # Convert 1,234,567 (Western) to 12,34,567 (Indian) grouping.
    if "," in whole:
        integer = whole.replace(",", "")
        last3 = integer[-3:]
        rest = integer[:-3]
        if rest:
            groups = []
            while len(rest) > 2:
                groups.insert(0, rest[-2:])
                rest = rest[:-2]
            if rest:
                groups.insert(0, rest)
            whole = ",".join(groups) + "," + last3
        else:
            whole = last3

    return f"₹ {'-' if negative else ''}{whole} Cr"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def get_filter_options(df: pd.DataFrame, column: str) -> List[str]:
    """Return a sorted, de-duplicated list of options for a filter column."""
    if column not in df.columns or df.empty:
        return []
    return sorted(v for v in df[column].dropna().unique() if str(v).strip())


def apply_filters(
    df: pd.DataFrame,
    year: str = "All",
    ministry: str = "All",
    department: str = "All",
    scheme_query: str = "",
    keyword: str = "",
) -> pd.DataFrame:
    """Apply all filters together and return the matching subset.

    Filters combine with AND logic. 'All' / '' means "don't filter on this".
    Scheme and keyword are case-insensitive substring searches; keyword is
    matched across Ministry, Department and Scheme.
    """
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    if year and year != "All":
        mask &= df["Year"] == year
    if ministry and ministry != "All":
        mask &= df["Ministry"] == ministry
    if department and department != "All":
        mask &= df["Department"] == department
    if scheme_query.strip():
        mask &= df["Scheme"].str.contains(scheme_query.strip(), case=False,
                                          na=False)
    if keyword.strip():
        kw = keyword.strip()
        combined = (
            df["Ministry"].str.contains(kw, case=False, na=False)
            | df["Department"].str.contains(kw, case=False, na=False)
            | df["Scheme"].str.contains(kw, case=False, na=False)
        )
        mask &= combined

    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Return headline metrics for the metric cards."""
    if df.empty:
        return {
            "total_budget": 0.0,
            "revenue_budget": 0.0,
            "capital_budget": 0.0,
            "n_departments": 0,
            "n_ministries": 0,
        }
    return {
        "total_budget": float(df["Total"].sum()),
        "revenue_budget": float(df["Revenue"].sum()),
        "capital_budget": float(df["Capital"].sum()),
        "n_departments": int(df["Department"].nunique()),
        "n_ministries": int(df["Ministry"].nunique()),
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _empty_figure(message: str = "No data to display") -> go.Figure:
    """A styled placeholder figure shown when a filter yields no data."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=16, color="#888"))
    fig.update_layout(template=PLOTLY_TEMPLATE,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def pie_budget_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of Revenue vs Capital share of the total budget."""
    if df.empty:
        return _empty_figure()
    data = pd.DataFrame(
        {
            "Component": ["Revenue Budget", "Capital Budget"],
            "Amount": [df["Revenue"].sum(), df["Capital"].sum()],
        }
    )
    fig = px.pie(
        data, names="Component", values="Amount", hole=0.45,
        color_discrete_sequence=COLOR_SEQUENCE, template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(title="Budget Distribution (Revenue vs Capital)")
    return fig


def bar_top_ministries(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of the top-N ministries by total budget."""
    if df.empty:
        return _empty_figure()
    top = (
        df.groupby("Ministry", as_index=False)["Total"].sum()
        .sort_values("Total", ascending=False)
        .head(top_n)
        .sort_values("Total")  # ascending so the biggest bar is on top
    )
    fig = px.bar(
        top, x="Total", y="Ministry", orientation="h",
        color="Total", color_continuous_scale="Tealgrn",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(title=f"Top {top_n} Ministries by Budget",
                      xaxis_title="Total (₹ Cr)", yaxis_title="")
    return fig


def line_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of total budget across years."""
    if df.empty:
        return _empty_figure()
    trend = (
        df.groupby("Year", as_index=False)["Total"].sum().sort_values("Year")
    )
    fig = px.line(
        trend, x="Year", y="Total", markers=True,
        color_discrete_sequence=COLOR_SEQUENCE, template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(title="Budget Trend Across Years",
                      xaxis_title="Year", yaxis_title="Total (₹ Cr)")
    return fig


def department_comparison(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Grouped bar chart comparing Revenue vs Capital by department."""
    if df.empty:
        return _empty_figure()
    grp = (
        df.groupby("Department", as_index=False)[["Revenue", "Capital"]].sum()
        .sort_values(by="Revenue", ascending=False)
        .head(top_n)
    )
    melted = grp.melt(id_vars="Department", value_vars=["Revenue", "Capital"],
                      var_name="Component", value_name="Amount")
    fig = px.bar(
        melted, x="Department", y="Amount", color="Component",
        barmode="group", color_discrete_sequence=COLOR_SEQUENCE,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(title="Department Comparison (Revenue vs Capital)",
                      xaxis_title="", yaxis_title="Amount (₹ Cr)")
    fig.update_xaxes(tickangle=-40)
    return fig


def ministry_comparison(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Stacked bar chart comparing Revenue vs Capital by ministry."""
    if df.empty:
        return _empty_figure()
    grp = (
        df.groupby("Ministry", as_index=False)[["Revenue", "Capital"]].sum()
        .sort_values(by="Revenue", ascending=False)
        .head(top_n)
    )
    melted = grp.melt(id_vars="Ministry", value_vars=["Revenue", "Capital"],
                      var_name="Component", value_name="Amount")
    fig = px.bar(
        melted, x="Ministry", y="Amount", color="Component",
        barmode="stack", color_discrete_sequence=COLOR_SEQUENCE,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(title="Ministry Comparison (Revenue vs Capital)",
                      xaxis_title="", yaxis_title="Amount (₹ Cr)")
    fig.update_xaxes(tickangle=-40)
    return fig


def allocation_ranking(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Ranked bar chart of the largest individual allocations (scheme level)."""
    if df.empty:
        return _empty_figure()
    ranked = (
        df.assign(Label=df["Ministry"] + " — " + df["Scheme"])
        .groupby("Label", as_index=False)["Total"].sum()
        .sort_values("Total", ascending=False)
        .head(top_n)
        .sort_values("Total")
    )
    fig = px.bar(
        ranked, x="Total", y="Label", orientation="h",
        color="Total", color_continuous_scale="Sunsetdark",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(title=f"Top {top_n} Budget Allocations",
                      xaxis_title="Total (₹ Cr)", yaxis_title="")
    return fig


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Budget") -> bytes:
    """Serialise a DataFrame to .xlsx bytes for a Streamlit download button."""
    buffer = BytesIO()
    # XlsxWriter is preferred; openpyxl is the fallback engine.
    try:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    except Exception:  # noqa: BLE001
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()
