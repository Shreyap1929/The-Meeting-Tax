"""
The Meeting Tax — Streamlit app (initial version).

Reads pre-computed CSVs from data/processed/ (produced by the SQL rollup /
burn-score scripts in sql/) rather than talking to a database directly. This
keeps the deployment story simple: no DB credentials, no connection pooling,
just static CSVs that get refreshed by re-running the SQL + notebook
pipeline upstream.

Inputs (see STATUS SO FAR from the SQL/notebook phases):
- data/processed/meeting_costs.csv        — one row per meeting (cost only)
- data/processed/meeting_burn_scores.csv  — one row per meeting (cost + the
  1-5 Meeting Burn Score, 5 = worst offender)

Structure: small, composable functions (load_data / render_sidebar_filters /
apply_filters / compute_kpis / render_kpis / main) so later prompts can add
new sections (charts, tables, drill-downs) without needing to rewrite this
file — new render_* functions can just be added and called from main().

THIS ITERATION adds:
- Cost-by-department and Meeting Burn Score distribution charts (Altair,
  driven off the same filtered dataframe as the KPIs).
- A productivity-vs-hours scatter chart, mirroring the Pearson correlation
  analysis in notebooks/04_stats_correlation.ipynb, sourced from the
  separate per-employee-week data/processed/pulse_clean.csv.
- The interactive "what-if" centerpiece: a slider that models cutting
  recurring status_update meetings by X%, live-recalculating projected
  annual savings using the same annualization logic as the KPIs.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr

# =============================================================================
# CONSTANTS
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data" / "processed"

MEETING_COSTS_CSV = DATA_DIR / "meeting_costs.csv"
MEETING_BURN_SCORES_CSV = DATA_DIR / "meeting_burn_scores.csv"
PULSE_CLEAN_CSV = DATA_DIR / "pulse_clean.csv"

# meeting_burn_score is an NTILE(5) bucket, 5 = worst offender (see
# sql/meeting_value_score.sql). This is the threshold used for the
# "% of spend from high-burn meetings" KPI below.
HIGH_BURN_SCORE = 5

# The meeting_type the "what-if" slider targets — see
# sql/meeting_value_score.sql's header for why status_update is the
# canonical "this could have been an email" category.
WHATIF_TARGET_MEETING_TYPE = "status_update"
WHATIF_DEFAULT_CUT_PCT = 50

# Scatter charts get slow/heavy in the browser past a few thousand points,
# so the productivity chart plots a fixed random sample while still
# computing the Pearson correlation over the full (unsampled) filtered
# data. Seeded for a stable-looking chart across reruns.
SCATTER_SAMPLE_SIZE = 3000
SCATTER_SAMPLE_RANDOM_STATE = 42

# Columns from meeting_burn_scores.csv that aren't already in
# meeting_costs.csv — pulled in via a merge so the app works from one
# combined dataframe instead of juggling two.
BURN_ONLY_COLUMNS = [
    "meeting_id",
    "type_points",
    "attendee_bloat_ratio",
    "attendee_bloat_points",
    "recurring_points",
    "raw_burn_points",
    "meeting_burn_score",
]


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and merge the two processed CSVs into one per-meeting dataframe.

    Cached so re-running filters/widgets doesn't re-read the CSVs from disk
    on every Streamlit rerun (Streamlit reruns the whole script on every
    widget interaction).
    """
    costs = pd.read_csv(MEETING_COSTS_CSV, parse_dates=["meeting_date"])
    burn_scores = pd.read_csv(MEETING_BURN_SCORES_CSV, parse_dates=["meeting_date"])

    merged = costs.merge(burn_scores[BURN_ONLY_COLUMNS], on="meeting_id", how="left")
    merged["is_recurring"] = merged["is_recurring"].astype(bool)

    return merged


@st.cache_data
def load_pulse_data() -> pd.DataFrame:
    """Load the per-employee-per-week pulse survey data.

    This is a different grain (employee x week) than the per-meeting
    dataframe from load_data(), so it's kept as its own dataframe rather
    than merged in. It still shares a "department" column and a "week"
    date column, which lets the productivity chart reuse the same
    department/date-range sidebar filters (see filter_pulse_data below).
    """
    return pd.read_csv(PULSE_CLEAN_CSV, parse_dates=["week"])


# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
def render_sidebar_filters(df: pd.DataFrame) -> dict:
    """Render the sidebar widgets and return the selected filter values.

    Kept separate from apply_filters() so the widget-rendering concern
    (what the user sees) is decoupled from the filtering logic (what
    happens to the data) — makes both easier to extend/test independently.
    """
    st.sidebar.header("Filters")

    departments = sorted(df["department_name"].dropna().unique())
    selected_departments = st.sidebar.multiselect(
        "Department",
        options=departments,
        default=departments,
    )

    meeting_types = sorted(df["meeting_type"].dropna().unique())
    selected_meeting_types = st.sidebar.multiselect(
        "Meeting type",
        options=meeting_types,
        default=meeting_types,
    )

    min_date = df["meeting_date"].min().date()
    max_date = df["meeting_date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # st.date_input returns a single date until the user has picked both
    # ends of the range — fall back to the full range in that transient case
    # so the app doesn't error out mid-selection.
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    return {
        "departments": selected_departments,
        "meeting_types": selected_meeting_types,
        "start_date": start_date,
        "end_date": end_date,
    }


# =============================================================================
# FILTERING
# =============================================================================
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply the sidebar filter selections to the full dataframe."""
    mask = (
        df["department_name"].isin(filters["departments"])
        & df["meeting_type"].isin(filters["meeting_types"])
        & (df["meeting_date"].dt.date >= filters["start_date"])
        & (df["meeting_date"].dt.date <= filters["end_date"])
    )
    return df.loc[mask].copy()


def filter_pulse_data(pulse_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply the department + date-range sidebar filters to the pulse data.

    Pulse data has no meeting_type column (it's employee/week, not
    per-meeting), so only the department and date filters apply here.
    """
    mask = (
        pulse_df["department"].isin(filters["departments"])
        & (pulse_df["week"].dt.date >= filters["start_date"])
        & (pulse_df["week"].dt.date <= filters["end_date"])
    )
    return pulse_df.loc[mask].copy()


# =============================================================================
# KPIs
# =============================================================================
def _annualization_factor(df: pd.DataFrame) -> float:
    """Scale factor to project a date-range total up/down to a 365-day run-rate.

    THE BUG (root cause): this used to take `filters` (the sidebar widget's
    selected start/end dates) and compute days_in_range from THAT, while the
    cost total it scales is summed from a separate dataframe `df`. Those two
    happened to agree for this dataset (which has zero date gaps), but they
    are not the same source of truth — anything that makes the dataframe's
    actual populated date span diverge from the widget's selected boundary
    (sparser filters, data gaps, the widget's transient single-date state
    before both range ends are picked) would silently scale by the wrong
    number of days. THE FIX: derive days_in_range from `df["meeting_date"]`
    itself — the exact data being summed — so the denominator can never
    disagree with the numerator, regardless of what filters produced `df`.
    Shared by compute_kpis() and compute_whatif_savings() so both KPI cards
    and the what-if slider use identical annualization logic.
    """
    days_in_range = (df["meeting_date"].max() - df["meeting_date"].min()).days + 1
    days_in_range = max(days_in_range, 1)
    return 365.25 / days_in_range


def compute_kpis(df: pd.DataFrame, filters: dict) -> dict:
    """Compute headline KPI values from the (already filtered) dataframe.

    The date range picked in the sidebar can be shorter or longer than a
    full year, so the "annual projected spend" KPI is annualized: total
    cost observed in the selected window, scaled up/down to a 365-day
    run-rate.

    `filters` is accepted (but no longer used for annualization — see
    _annualization_factor()) purely to keep this function's signature
    stable for its caller in main().
    """
    total_cost = float(df["meeting_cost"].sum())
    annual_projected_spend = total_cost * _annualization_factor(df)

    high_burn_cost = float(df.loc[df["meeting_burn_score"] == HIGH_BURN_SCORE, "meeting_cost"].sum())
    pct_high_burn_spend = (high_burn_cost / total_cost * 100.0) if total_cost > 0 else 0.0

    return {
        "annual_projected_spend": annual_projected_spend,
        "pct_high_burn_spend": pct_high_burn_spend,
        "meetings_in_view": int(len(df)),
    }


# =============================================================================
# WHAT-IF: cut recurring status_update meetings
# =============================================================================
def compute_whatif_savings(df: pd.DataFrame, filters: dict, cut_percentage: float) -> dict:
    """Model cutting recurring status_update meetings by cut_percentage.

    Targets recurring status_update meetings specifically — per
    sql/meeting_value_score.sql, that combination (type_points=5,
    recurring_points=5) is the canonical "this could have been an email"
    case that also keeps repeating. Uses the same annualization logic as
    compute_kpis() so the projected savings is directly comparable to the
    "Total Annual Projected Meeting Spend" KPI.

    `filters` is accepted (but no longer used for annualization — see
    _annualization_factor()) purely to keep this function's signature
    stable for its caller in render_whatif_slider().
    """
    target_mask = (df["meeting_type"] == WHATIF_TARGET_MEETING_TYPE) & df["is_recurring"]
    target_df = df.loc[target_mask]

    if target_df.empty:
        # No recurring status_update meetings in the current filter
        # selection — nothing to annualize. Guards _annualization_factor()
        # against calling .min()/.max() on an empty date column.
        return {
            "n_target_meetings": 0,
            "annualized_target_cost": 0.0,
            "projected_annual_savings": 0.0,
        }

    # Annualize target_df's OWN date span, not df's — the denominator must
    # match the exact rows being summed (see _annualization_factor docstring).
    annualization_factor = _annualization_factor(target_df)
    annualized_target_cost = float(target_df["meeting_cost"].sum()) * annualization_factor
    projected_annual_savings = annualized_target_cost * (cut_percentage / 100.0)

    return {
        "n_target_meetings": int(len(target_df)),
        "annualized_target_cost": annualized_target_cost,
        "projected_annual_savings": projected_annual_savings,
    }


# =============================================================================
# RENDERING
# =============================================================================
def render_kpis(kpis: dict) -> None:
    """Render the top-of-page KPI cards using st.metric."""
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total Annual Projected Meeting Spend",
        f"${kpis['annual_projected_spend']:,.0f}",
    )
    col2.metric(
        f"Spend from High-Burn Meetings (Score {HIGH_BURN_SCORE})",
        f"{kpis['pct_high_burn_spend']:.1f}%",
    )
    col3.metric(
        "Meetings Analyzed",
        f"{kpis['meetings_in_view']:,}",
    )


def render_whatif_slider(df: pd.DataFrame, filters: dict, kpis: dict) -> None:
    """Render the interactive "what-if" centerpiece.

    Streamlit reruns the whole script on every widget interaction, so the
    slider is naturally "live": moving it triggers a rerun, which
    recomputes compute_whatif_savings() with the new percentage and
    redraws the metrics below — no manual callback wiring needed.
    """
    with st.container(border=True):
        st.subheader("🎚️ What If: Cut Recurring Status-Update Meetings?")
        st.caption(
            "Recurring status_update meetings are the single biggest "
            "\"this could have been an email\" category (see "
            "sql/meeting_value_score.sql). Drag the slider to model "
            "trimming them and see the projected annual savings update live."
        )
        cut_percentage = st.slider(
            "Cut recurring status_update meetings by:",
            min_value=0,
            max_value=100,
            value=WHATIF_DEFAULT_CUT_PCT,
            step=5,
            format="%d%%",
        )

        whatif = compute_whatif_savings(df, filters, cut_percentage)

        if whatif["n_target_meetings"] == 0:
            st.warning(
                "No recurring status_update meetings in the current filter "
                "selection — widen the Department/Meeting type filters in "
                "the sidebar to see projected savings."
            )
            return

        pct_of_annual_spend = (
            whatif["projected_annual_savings"] / kpis["annual_projected_spend"] * 100.0
            if kpis["annual_projected_spend"] > 0
            else 0.0
        )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "💰 Projected Annual Savings",
            f"${whatif['projected_annual_savings']:,.0f}",
        )
        col2.metric(
            "% of Total Annual Spend",
            f"{pct_of_annual_spend:.1f}%",
        )
        col3.metric(
            "Recurring Status-Updates In View",
            f"{whatif['n_target_meetings']:,}",
        )


def render_department_cost_chart(df: pd.DataFrame) -> None:
    """Render a horizontal bar chart of total meeting cost by department."""
    st.subheader("💸 Cost by Department")
    dept_cost = (
        df.groupby("department_name", as_index=False)["meeting_cost"]
        .sum()
        .sort_values("meeting_cost", ascending=False)
    )
    chart = (
        alt.Chart(dept_cost)
        .mark_bar(color="#2c3e50")
        .encode(
            x=alt.X("meeting_cost:Q", title="Total meeting cost ($)"),
            y=alt.Y("department_name:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("department_name:N", title="Department"),
                alt.Tooltip("meeting_cost:Q", title="Cost", format="$,.0f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def render_burn_score_distribution(df: pd.DataFrame) -> None:
    """Render a bar chart of meeting counts per Meeting Burn Score bucket."""
    st.subheader("🔥 Meeting Burn Score Distribution")
    burn_counts = (
        df["meeting_burn_score"]
        .value_counts()
        .reindex([1, 2, 3, 4, 5], fill_value=0)
        .rename_axis("meeting_burn_score")
        .reset_index(name="meeting_count")
    )
    chart = (
        alt.Chart(burn_counts)
        .mark_bar()
        .encode(
            x=alt.X("meeting_burn_score:O", title="Meeting Burn Score (5 = worst offender)"),
            y=alt.Y("meeting_count:Q", title="Number of meetings"),
            color=alt.Color(
                "meeting_burn_score:O",
                scale=alt.Scale(domain=[1, 2, 3, 4, 5], range=["#2ecc71", "#a3d977", "#f1c40f", "#e67e22", "#e74c3c"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("meeting_burn_score:O", title="Burn Score"),
                alt.Tooltip("meeting_count:Q", title="Meetings"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def render_productivity_scatter(pulse_df: pd.DataFrame) -> None:
    """Render the weekly meeting hours vs. self-reported productivity scatter.

    Mirrors notebooks/04_stats_correlation.ipynb: a Pearson correlation
    (computed over the full filtered data) plus a scatter + regression
    trend line (drawn from a fixed-size random sample, since Altair
    renders thousands of points sluggishly in-browser).

    The trend line is fit in Python (numpy.polyfit) rather than via
    Altair's transform_regression(): that transform runs client-side
    against Streamlit's Arrow-encoded chart data, and in this environment
    it corrupts the shared axis scale for the whole layered chart (every
    point silently disappears — a known bad interaction between Vega-Lite's
    regression transform and Arrow-backed datasets, not a data problem).
    Precomputing two line-endpoint rows sidesteps it entirely.
    """
    st.subheader("📉 Meeting Hours vs. Self-Reported Productivity")

    if pulse_df.empty or pulse_df["hours_in_meetings"].nunique() < 2:
        st.info("Not enough pulse-survey data in the current filters to compute a correlation.")
        return

    r, p_value = pearsonr(pulse_df["hours_in_meetings"], pulse_df["self_reported_productivity"])
    # Only pass the columns the chart actually needs — trims payload size and
    # avoids feeding the datetime/bool columns from pulse_df to Altair/Vega.
    chart_columns = ["department", "seniority", "hours_in_meetings", "self_reported_productivity"]
    sample_df = pulse_df[chart_columns].sample(
        n=min(len(pulse_df), SCATTER_SAMPLE_SIZE),
        random_state=SCATTER_SAMPLE_RANDOM_STATE,
    )

    scatter = (
        alt.Chart(sample_df)
        .mark_circle(opacity=0.25, size=25, color="#2c3e50")
        .encode(
            x=alt.X("hours_in_meetings:Q", title="Weekly hours in meetings"),
            y=alt.Y("self_reported_productivity:Q", title="Self-reported productivity (1-10)"),
            tooltip=[
                alt.Tooltip("department:N", title="Department"),
                alt.Tooltip("seniority:N", title="Seniority"),
                alt.Tooltip("hours_in_meetings:Q", title="Hours in meetings", format=".1f"),
                alt.Tooltip("self_reported_productivity:Q", title="Productivity"),
            ],
        )
    )

    slope, intercept = np.polyfit(pulse_df["hours_in_meetings"], pulse_df["self_reported_productivity"], 1)
    x_min = float(pulse_df["hours_in_meetings"].min())
    x_max = float(pulse_df["hours_in_meetings"].max())
    trend_df = pd.DataFrame(
        {
            "hours_in_meetings": [x_min, x_max],
            "self_reported_productivity": [slope * x_min + intercept, slope * x_max + intercept],
        }
    )
    trend = alt.Chart(trend_df).mark_line(color="firebrick").encode(
        x="hours_in_meetings:Q",
        y="self_reported_productivity:Q",
    )

    st.altair_chart((scatter + trend).properties(height=350), use_container_width=True)
    p_display = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    st.caption(
        f"Pearson r = {r:.3f} (p {p_display}, n = {len(pulse_df):,}). Weak negative "
        "correlation is expected — see notebooks/04_stats_correlation.ipynb for the "
        "full regression and the correlation-not-causation caveats. Chart is drawn "
        f"from a random sample of up to {SCATTER_SAMPLE_SIZE:,} points for rendering "
        "speed; the r/p values above use all filtered rows."
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="The Meeting Tax", page_icon="🗓️", layout="wide")

    st.title("🗓️ The Meeting Tax")
    st.caption(
        "How much do meetings actually cost, which ones are the worst offenders, "
        "and is meeting load related to productivity and burnout?"
    )

    df = load_data()
    filters = render_sidebar_filters(df)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        st.warning("No meetings match the current filters. Try widening your selection.")
        return

    kpis = compute_kpis(filtered_df, filters)
    render_kpis(kpis)
    st.caption(
        "KPIs reflect the current sidebar filters. Annual spend is annualized "
        "from the selected date range, not a flat sum, so it stays comparable "
        "across different-length windows."
    )

    st.divider()
    # The live-demo centerpiece: gets its own full-width row, right after the
    # KPIs, so it's the first interactive thing a viewer sees/touches.
    render_whatif_slider(filtered_df, filters, kpis)

    st.divider()
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        render_department_cost_chart(filtered_df)
    with chart_col2:
        render_burn_score_distribution(filtered_df)

    st.divider()
    pulse_df = load_pulse_data()
    filtered_pulse_df = filter_pulse_data(pulse_df, filters)
    render_productivity_scatter(filtered_pulse_df)

    st.divider()
    # Placeholder for the next iteration: cost trend-over-time chart, the
    # top-10 Burn Score offenders table, and any further drill-downs get
    # added here as new render_*() functions called from main().
    st.info("More drill-down tables are coming in the next iteration of this app.")


if __name__ == "__main__":
    main()
