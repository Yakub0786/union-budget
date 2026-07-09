"""
app.py
===============================================================================
INDIA UNION BUDGET ANALYSIS SYSTEM  --  Streamlit dashboard.

Run with:
    streamlit run app.py

Everything data-related lives in utils.py / extract_data.py; this file is
purely the user interface: layout, filters, metric cards, charts, search,
interactive table and downloads.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import streamlit as st

import utils

# ---------------------------------------------------------------------------
# Page configuration + theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="India Union Budget Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light-touch CSS for a professional, responsive, dark-friendly look.
CUSTOM_CSS = """
<style>
  .main > div { padding-top: 1rem; }
  .metric-card {
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border: 1px solid #334155; border-radius: 14px;
      padding: 18px 20px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);
  }
  .metric-card h3 {
      margin: 0; font-size: 0.8rem; font-weight: 600;
      letter-spacing: .05em; text-transform: uppercase; color: #94a3b8;
  }
  .metric-card p {
      margin: 6px 0 0 0; font-size: 1.55rem; font-weight: 700; color: #f8fafc;
  }
  .app-title {
      font-size: 2.1rem; font-weight: 800; color: #f1f5f9; margin-bottom: 0;
  }
  .app-sub { color: #94a3b8; margin-top: 2px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data loading (spinner shown while building on first run)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading budget data…")
def get_data():
    """Load (and cache) the cleaned dataset."""
    return utils.load_data()


df_all = get_data()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<p class="app-title">📊 INDIA UNION BUDGET ANALYSIS SYSTEM</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Interactive analysis of Union Budget allocations '
    'across ministries, departments and years (figures in ₹ crore).</p>',
    unsafe_allow_html=True,
)

if df_all.empty:
    st.error("No budget data available. Add PDFs to the 'Budget Year Wise' "
             "folder and run `python extract_data.py`, then reload.")
    st.stop()

# Warn (once) if we are clearly on the synthetic sample dataset.
if df_all["Ministry"].str.contains("Ministry of").all() and \
        df_all["Year"].nunique() == 11 and len(df_all) < 600:
    st.info("ℹ️ Showing **sample data**. Place your budget PDFs in "
            "`Budget Year Wise/`, run `python extract_data.py`, and reload to "
            "analyse real figures.")


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("🔎 Filters")

years = ["All"] + utils.get_filter_options(df_all, "Year")
selected_year = st.sidebar.selectbox("Select Budget Year", years, index=0)

# Ministry options depend on the chosen year (dependent filters).
year_scoped = df_all if selected_year == "All" else \
    df_all[df_all["Year"] == selected_year]
ministries = ["All"] + utils.get_filter_options(year_scoped, "Ministry")
selected_ministry = st.sidebar.selectbox("Select Ministry", ministries, index=0)

# Department options depend on year + ministry.
min_scoped = year_scoped if selected_ministry == "All" else \
    year_scoped[year_scoped["Ministry"] == selected_ministry]
departments = ["All"] + utils.get_filter_options(min_scoped, "Department")
selected_department = st.sidebar.selectbox("Select Department", departments,
                                           index=0)

scheme_query = st.sidebar.text_input("Search by Scheme", "")
keyword = st.sidebar.text_input("Search by Keyword",
                                help="Matches Ministry, Department or Scheme.")

st.sidebar.markdown("---")
st.sidebar.caption("Filters combine together. Set a year, then narrow by "
                   "ministry, department, scheme or keyword.")

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

filtered = utils.apply_filters(
    df_all,
    year=selected_year,
    ministry=selected_ministry,
    department=selected_department,
    scheme_query=scheme_query,
    keyword=keyword,
)

if filtered.empty:
    st.warning("No records match your current filters. Try widening them.")


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------

metrics = utils.compute_metrics(filtered)


def metric_card(column, label: str, value: str) -> None:
    """Render a single styled metric card inside ``column``."""
    column.markdown(
        f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>',
        unsafe_allow_html=True,
    )


c1, c2, c3, c4, c5 = st.columns(5)
metric_card(c1, "Total Budget", utils.format_inr_crore(metrics["total_budget"]))
metric_card(c2, "Revenue Budget",
            utils.format_inr_crore(metrics["revenue_budget"]))
metric_card(c3, "Capital Budget",
            utils.format_inr_crore(metrics["capital_budget"]))
metric_card(c4, "Departments", f'{metrics["n_departments"]:,}')
metric_card(c5, "Ministries", f'{metrics["n_ministries"]:,}')

st.markdown("")  # spacer


# ---------------------------------------------------------------------------
# Visualisations (tabbed for a clean, responsive layout)
# ---------------------------------------------------------------------------

tab_overview, tab_compare, tab_ranking, tab_data = st.tabs(
    ["📈 Overview", "⚖️ Comparisons", "🏆 Rankings", "📋 Data"]
)

with tab_overview:
    left, right = st.columns(2)
    left.plotly_chart(utils.pie_budget_distribution(filtered),
                      use_container_width=True)
    right.plotly_chart(utils.bar_top_ministries(filtered),
                       use_container_width=True)
    # Trend uses the full dataset so the line still shows every year even
    # when a single year is selected in the sidebar.
    trend_source = df_all if selected_year != "All" else filtered
    st.plotly_chart(utils.line_trend(trend_source), use_container_width=True)

with tab_compare:
    st.plotly_chart(utils.department_comparison(filtered),
                    use_container_width=True)
    st.plotly_chart(utils.ministry_comparison(filtered),
                    use_container_width=True)

with tab_ranking:
    st.plotly_chart(utils.allocation_ranking(filtered),
                    use_container_width=True)

with tab_data:
    st.subheader(f"Filtered records ({len(filtered):,} rows)")
    st.dataframe(filtered, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("⬇️ Downloads")

d1, d2, d3 = st.columns(3)

d1.download_button(
    "Download Full CSV",
    data=df_all.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_budget.csv",
    mime="text/csv",
    use_container_width=True,
)

d2.download_button(
    "Download Full Excel",
    data=utils.to_excel_bytes(df_all),
    file_name="cleaned_budget.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

d3.download_button(
    "Download Filtered Data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_budget.csv",
    mime="text/csv",
    disabled=filtered.empty,
    use_container_width=True,
)

st.caption("Built with Streamlit · Pandas · Plotly · Camelot / pdfplumber")
