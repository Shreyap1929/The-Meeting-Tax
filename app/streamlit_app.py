"""The Meeting Tax Streamlit dashboard."""

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
ALL_OPTION = "All"

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

MEETING_COSTS_REQUIRED_COLUMNS = {
    "meeting_id",
    "meeting_date",
    "department_name",
    "meeting_type",
    "meeting_cost",
    "is_recurring",
}
BURN_SCORES_REQUIRED_COLUMNS = set(BURN_ONLY_COLUMNS)
PULSE_REQUIRED_COLUMNS = {
    "week",
    "department",
    "hours_in_meetings",
    "self_reported_productivity",
}


class DataValidationError(ValueError):
    """Raised when required data files or columns are not available/valid."""


def format_currency_compact(value: float) -> str:
    """Format currency in compact form for dashboard readability."""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs_value >= 10_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def format_currency_full(value: float) -> str:
    """Format currency with full separators."""
    return f"${value:,.0f}"


def format_p_value(p_value: float) -> str:
    """Format p-values for concise statistical reporting."""
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def interpret_correlation(r_value: float) -> str:
    """Return a concise interpretation based on Pearson's r."""
    strength = abs(r_value)
    if strength < 0.2:
        strength_text = "Very weak"
    elif strength < 0.4:
        strength_text = "Weak"
    elif strength < 0.6:
        strength_text = "Moderate"
    elif strength < 0.8:
        strength_text = "Strong"
    else:
        strength_text = "Very strong"

    if r_value > 0:
        direction = "positive"
    elif r_value < 0:
        direction = "negative"
    else:
        direction = "no linear"

    if direction == "no linear":
        return "No linear association between meeting hours and self-reported productivity."
    return f"{strength_text} {direction} association between meeting hours and self-reported productivity."


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str], dataset_name: str) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise DataValidationError(f"{dataset_name} is missing required columns: {missing_str}")


def _normalize_multiselect_selection(selected_values: list[str], all_values: list[str]) -> list[str]:
    """Resolve 'All' + specific choices into an intuitive effective selection."""
    if not all_values:
        return []
    if ALL_OPTION in selected_values and len(selected_values) > 1:
        selected_values = [value for value in selected_values if value != ALL_OPTION]
    if not selected_values or ALL_OPTION in selected_values:
        return all_values
    return selected_values


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and merge the two processed CSVs into one per-meeting dataframe."""
    missing_files = [
        str(path)
        for path in (MEETING_COSTS_CSV, MEETING_BURN_SCORES_CSV)
        if not path.exists()
    ]
    if missing_files:
        missing_str = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing processed meeting data file(s): {missing_str}")

    costs = pd.read_csv(MEETING_COSTS_CSV)
    burn_scores = pd.read_csv(MEETING_BURN_SCORES_CSV)

    _validate_required_columns(costs, MEETING_COSTS_REQUIRED_COLUMNS, "meeting_costs.csv")
    _validate_required_columns(burn_scores, BURN_SCORES_REQUIRED_COLUMNS, "meeting_burn_scores.csv")

    costs["meeting_date"] = pd.to_datetime(costs["meeting_date"], errors="coerce")
    if costs["meeting_date"].isna().any():
        raise DataValidationError("meeting_costs.csv has invalid or missing values in meeting_date.")

    merged = costs.merge(burn_scores[BURN_ONLY_COLUMNS], on="meeting_id", how="left")
    merged["is_recurring"] = merged["is_recurring"].astype(bool)
    merged["meeting_cost"] = pd.to_numeric(merged["meeting_cost"], errors="coerce")

    if merged["meeting_cost"].isna().any():
        raise DataValidationError("meeting_costs.csv has invalid or missing values in meeting_cost.")

    return merged


@st.cache_data
def load_pulse_data() -> pd.DataFrame:
    """Load the per-employee-per-week pulse survey data."""
    if not PULSE_CLEAN_CSV.exists():
        raise FileNotFoundError(f"Missing processed pulse data file: {PULSE_CLEAN_CSV}")

    pulse_df = pd.read_csv(PULSE_CLEAN_CSV)
    _validate_required_columns(pulse_df, PULSE_REQUIRED_COLUMNS, "pulse_clean.csv")
    pulse_df["week"] = pd.to_datetime(pulse_df["week"], errors="coerce")
    if pulse_df["week"].isna().any():
        raise DataValidationError("pulse_clean.csv has invalid or missing values in week.")
    if "seniority" not in pulse_df.columns:
        pulse_df["seniority"] = "Unknown"

    return pulse_df


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
    st.sidebar.caption("Default is organization-wide view across all data.")

    departments = sorted(df["department_name"].dropna().unique())
    selected_departments_raw = st.sidebar.multiselect(
        "Department",
        options=[ALL_OPTION] + departments,
        default=[ALL_OPTION],
    )
    selected_departments = _normalize_multiselect_selection(selected_departments_raw, departments)

    meeting_types = sorted(df["meeting_type"].dropna().unique())
    selected_meeting_types_raw = st.sidebar.multiselect(
        "Meeting type",
        options=[ALL_OPTION] + meeting_types,
        default=[ALL_OPTION],
    )
    selected_meeting_types = _normalize_multiselect_selection(selected_meeting_types_raw, meeting_types)

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
    annualization_factor = _annualization_factor(df)
    annual_projected_spend = total_cost * annualization_factor

    high_burn_cost = float(df.loc[df["meeting_burn_score"] == HIGH_BURN_SCORE, "meeting_cost"].sum())
    annualized_high_burn_spend = high_burn_cost * annualization_factor
    pct_high_burn_spend = (high_burn_cost / total_cost * 100.0) if total_cost > 0 else 0.0

    return {
        "annual_projected_spend": annual_projected_spend,
        "annualized_high_burn_spend": annualized_high_burn_spend,
        "high_burn_spend": high_burn_cost,
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
        "Projected Annual Meeting Spend",
        format_currency_compact(kpis["annual_projected_spend"]),
    )
    col1.caption(f"Full value: {format_currency_full(kpis['annual_projected_spend'])}")

    col2.metric(
        f"High-Burn Spend (Score {HIGH_BURN_SCORE})",
        format_currency_compact(kpis["annualized_high_burn_spend"]),
    )
    col2.caption(
        f"Share of spend: {kpis['pct_high_burn_spend']:.1f}% "
        f"({format_currency_full(kpis['annualized_high_burn_spend'])} annualized)"
    )

    col3.metric(
        "Meetings in View",
        f"{kpis['meetings_in_view']:,}",
    )


def render_whatif_slider(df: pd.DataFrame, filters: dict, kpis: dict) -> None:
    """Render the interactive what-if section."""
    with st.container(border=True):
        st.subheader("What If: Cut Recurring Status-Update Meetings?")
        st.caption(
            "Estimate potential savings by reducing recurring status-update meetings."
        )
        cut_percentage = st.slider(
            "Reduction in recurring status-update meetings",
            min_value=0,
            max_value=100,
            value=WHATIF_DEFAULT_CUT_PCT,
            step=5,
            format="%d%%",
        )

        whatif = compute_whatif_savings(df, filters, cut_percentage)

        pct_of_annual_spend = (
            whatif["projected_annual_savings"] / kpis["annual_projected_spend"] * 100.0
            if kpis["annual_projected_spend"] > 0
            else 0.0
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Reduction",
            f"{cut_percentage}%",
        )
        col2.metric(
            "Projected Annual Savings",
            format_currency_compact(whatif["projected_annual_savings"]),
        )
        col2.caption(f"Full value: {format_currency_full(whatif['projected_annual_savings'])}")
        col3.metric(
            "Share of Annual Spend",
            f"{pct_of_annual_spend:.1f}%",
        )
        col4.metric(
            "Recurring Status-Update Meetings Affected",
            f"{whatif['n_target_meetings']:,}",
        )
        st.caption(
            "Illustrative scenario assuming savings scale linearly with the selected reduction."
        )

        if whatif["n_target_meetings"] == 0:
            st.info(
                "No recurring status-update meetings match the current filters, so projected savings are $0."
            )


def render_department_cost_chart(df: pd.DataFrame) -> None:
    """Render a horizontal bar chart of total meeting cost by department."""
    st.subheader("Meeting Spend by Department")
    dept_cost = (
        df.groupby("department_name", as_index=False)["meeting_cost"]
        .sum()
        .sort_values("meeting_cost", ascending=False)
    )
    if dept_cost.empty:
        st.info("No department spend data is available for the selected filters.")
        return

    chart = (
        alt.Chart(dept_cost)
        .mark_bar(color="#2c3e50")
        .encode(
            x=alt.X(
                "meeting_cost:Q",
                title="Meeting spend",
                axis=alt.Axis(format="$~s"),
            ),
            y=alt.Y("department_name:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("department_name:N", title="Department"),
                alt.Tooltip("meeting_cost:Q", title="Cost", format="$,.0f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def render_burn_score_distribution(df: pd.DataFrame) -> None:
    """Render a bar chart of meeting counts per Meeting Burn Score bucket."""
    st.subheader("Meeting Burn Score Distribution")
    st.caption(
        "Meeting Burn Score combines meeting type, relative attendee bloat, and recurrence "
        "to identify meetings with higher potential for inefficiency."
    )
    with st.expander("How is Burn Score calculated?"):
        st.markdown(
            "- Meeting type contribution: **40%**\n"
            "- Attendee bloat contribution: **35%**\n"
            "- Recurrence contribution: **25%**\n"
            "- Final score is grouped into **5 relative buckets** using `NTILE(5)`.\n\n"
            "**5 = highest relative burn**, **1 = lowest relative burn**."
        )

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
            x=alt.X("meeting_burn_score:O", title="Meeting Burn Score (1 = lowest, 5 = highest)"),
            y=alt.Y("meeting_count:Q", title="Number of meetings", axis=alt.Axis(format="d")),
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
        .properties(height=320)
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
    st.subheader("Meeting Hours vs. Self-Reported Productivity")

    if (
        pulse_df.empty
        or pulse_df["hours_in_meetings"].nunique() < 2
        or pulse_df["self_reported_productivity"].nunique() < 2
    ):
        st.info("Not enough pulse-survey variation in the current filters to compute correlation.")
        return

    pulse_for_stats = pulse_df[["hours_in_meetings", "self_reported_productivity"]].dropna()
    if len(pulse_for_stats) < 3:
        st.info("Not enough pulse-survey records in the current filters to compute correlation.")
        return

    r, p_value = pearsonr(
        pulse_for_stats["hours_in_meetings"],
        pulse_for_stats["self_reported_productivity"],
    )

    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("Pearson r", f"{r:.3f}")
    stat_col2.metric("p-value", format_p_value(p_value))
    stat_col3.markdown(f"**Interpretation**\n\n{interpret_correlation(r)}")

    st.caption("Correlation does not imply causation.")

    # Only pass the columns the chart actually needs — trims payload size and
    # avoids feeding the datetime/bool columns from pulse_df to Altair/Vega.
    chart_columns = ["department", "seniority", "hours_in_meetings", "self_reported_productivity"]
    chart_df = pulse_df[chart_columns].dropna(subset=["hours_in_meetings", "self_reported_productivity"])
    sample_df = chart_df.sample(
        n=min(len(chart_df), SCATTER_SAMPLE_SIZE),
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

    slope, intercept = np.polyfit(
        pulse_for_stats["hours_in_meetings"],
        pulse_for_stats["self_reported_productivity"],
        1,
    )
    x_min = float(pulse_for_stats["hours_in_meetings"].min())
    x_max = float(pulse_for_stats["hours_in_meetings"].max())
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
    st.caption(
        f"Statistics above use all filtered pulse rows (n = {len(pulse_for_stats):,}). "
        "Chart is drawn "
        f"from a random sample of up to {SCATTER_SAMPLE_SIZE:,} points for rendering "
        "performance."
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="The Meeting Tax", page_icon="🗓️", layout="wide")

    st.title("The Meeting Tax")
    st.subheader("Quantifying the financial and productivity impact of workplace meetings.")
    st.caption(
        "Explore meeting costs, identify high-burn meeting patterns, and estimate "
        "potential savings from reducing recurring meetings."
    )
    st.caption("Analysis is based on processed meeting and pulse-survey data.")

    try:
        df = load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Please ensure the processed CSV files are present in data/processed.")
        return
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        st.error(f"Unable to parse processed meeting data: {exc}")
        return
    except DataValidationError as exc:
        st.error(str(exc))
        return

    filters = render_sidebar_filters(df)
    filtered_df = apply_filters(df, filters)

    if filtered_df.empty:
        st.warning("No meetings match the current filters. Try widening your selections.")
        return

    kpis = compute_kpis(filtered_df, filters)
    render_kpis(kpis)
    st.caption(
        "Projected annual meeting spend is an analytical estimate annualized from "
        "the actual date span of records in the filtered view."
    )

    st.divider()
    render_whatif_slider(filtered_df, filters, kpis)

    st.divider()
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        render_department_cost_chart(filtered_df)
    with chart_col2:
        render_burn_score_distribution(filtered_df)

    st.divider()
    try:
        pulse_df = load_pulse_data()
    except FileNotFoundError as exc:
        st.warning(str(exc))
        st.info("Productivity analysis is unavailable until pulse_clean.csv is present.")
        pulse_df = pd.DataFrame()
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        st.warning(f"Unable to parse pulse_clean.csv: {exc}")
        pulse_df = pd.DataFrame()
    except DataValidationError as exc:
        st.warning(str(exc))
        pulse_df = pd.DataFrame()

    if pulse_df.empty:
        st.info("No pulse-survey data is available for productivity analysis.")
    else:
        filtered_pulse_df = filter_pulse_data(pulse_df, filters)
        render_productivity_scatter(filtered_pulse_df)

    st.divider()
    with st.expander("About this analysis"):
        st.markdown(
            "- Meeting costs are analyzed using processed meeting data.\n"
            "- Meeting Burn Score combines meeting type, attendee bloat, and recurrence.\n"
            "- The what-if model estimates potential savings from reducing recurring status-update meetings.\n"
            "- Productivity analysis uses Pearson correlation between meeting hours and self-reported productivity.\n"
            "- Correlation is not proof of causation."
        )


if __name__ == "__main__":
    main()
