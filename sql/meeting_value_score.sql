-- =============================================================================
-- meeting_value_score.sql
-- "The Meeting Tax" — the Meeting Burn Score
--
-- DEPENDENCY: this script reads from the `meeting_costs` VIEW created by
-- sql/meeting_cost_rollup.sql. Run that script first (it is idempotent —
-- DROP VIEW IF EXISTS / CREATE VIEW — so re-running it here is safe too).
-- We deliberately do NOT redefine the cost logic here, to keep one source
-- of truth for "what a meeting costs" and a separate, composable layer for
-- "how bad is this meeting" scoring on top of it.
--
-- Compatible with PostgreSQL and SQLite (both support NTILE() and other
-- window functions used below; no dialect-specific syntax used).
--
-- WHAT IS THE "MEETING BURN SCORE"?
-- -----------------------------------------------------------------------
-- An invented (not industry-standard) 1-5 score, 5 = worst offender,
-- modeled loosely on RFM (Recency/Frequency/Monetary) scoring: several
-- independent signals are each converted to a 1-5 sub-score, combined
-- with weights into a single composite, then re-bucketed into a final
-- 1-5 score via NTILE so the output distribution is even and comparable
-- across meeting types/departments (same intent as RFM's use of NTILE
-- quintiles instead of raw dollar amounts).
--
-- The three signals, why each was picked, and how it's weighted:
--
-- 1. MEETING TYPE ("replaceability" points, weight 40%)
--    The single strongest predictor of whether a meeting is inherently
--    wasteful is *what kind* of meeting it is — some categories are
--    almost always replaceable by an async update; others structurally
--    require live, synchronous attendance to produce value.
--      status_update    -> 5  (near-always replaceable by Slack/email/doc;
--                               this is the canonical "this meeting could
--                               have been an email" case, per the task
--                               requirement that these skew high-burn)
--      all_hands        -> 4  (broadcast-style; large audience multiplies
--                               cost, and most content could be pre-recorded,
--                               but live Q&A has some real value)
--      brainstorm       -> 2  (creative ideation benefits genuinely from
--                               real-time back-and-forth)
--      decision_making  -> 2  (reaching consensus usually needs live
--                               discussion; treated as similarly low-burn
--                               to brainstorm)
--      one_on_one       -> 1  (highest value-per-dollar: small attendee
--                               count, direct relationship/coaching value,
--                               essentially never "should have been an email")
--    This is a fixed CASE mapping, not data-derived, because it encodes a
--    business judgment about meeting purpose — the kind of assumption that
--    should be stated explicitly and be easy to argue with in an interview.
--
-- 2. ATTENDEE BLOAT ("overstaffed-for-its-type" points, weight 35%)
--    A meeting is worse the more it overshoots the *typical* attendee
--    count for its own type — e.g. a status_update with 15 people is a
--    much bigger red flag than a status_update with 3. We compute:
--        attendee_bloat_ratio = attendee_count / avg(attendee_count for
--                                                     that meeting_type)
--    then rank all meetings by that ratio using NTILE(5), so the score
--    reflects "how much of an outlier is this meeting's headcount versus
--    its own type's norm" rather than raw headcount (which would just
--    reward/penalize all_hands meetings for being large by nature).
--    This is the direct cost-driver signal: extra attendees multiply
--    duration_hours x hourly_rate linearly, so bloat is real dollars.
--
-- 3. IS_RECURRING (compounding points, weight 25%)
--    A recurring meeting's cost is not a one-off; it repeats every
--    week/cycle for as long as the series runs, so its *effective* annual
--    cost exposure is far higher than a single instance suggests. We
--    treat this as a binary step function (5 if recurring, 1 if not)
--    rather than a graded score, because the compounding risk is a
--    structural property of the meeting (does it recur or not?), not a
--    matter of degree. It's weighted lowest of the three (25%) because,
--    for cost purposes, meeting_costs already reflects the true expanded
--    row-per-occurrence data — this factor is about behavioral risk
--    ("this bad habit repeats") rather than an additional cost multiplier
--    on top of what's already summed in the rollups.
--
-- Weights (0.40 / 0.35 / 0.25) sum to 1.0 and were chosen to rank
-- "what kind of meeting is this" and "is it overstaffed" as the two
-- dominant, cost-correlated signals, with recurrence as a meaningful but
-- secondary tiebreaker. These are business judgment calls, not derived
-- from the data — flag them as tunable if challenged in an interview.
-- =============================================================================

-- -----------------------------------------------------------------------
-- STEP 1: meeting-type norms — the average attendee_count per
-- meeting_type, used as the baseline "what's normal for this type" in
-- the bloat ratio below.
-- -----------------------------------------------------------------------
WITH type_norms AS (
    SELECT
        meeting_type,
        AVG(attendee_count) AS avg_attendees_for_type
    FROM meeting_costs
    GROUP BY meeting_type
),

-- -----------------------------------------------------------------------
-- STEP 2: per-meeting raw signal points — one row per meeting, with the
-- three component sub-scores (each roughly on a 1-5 scale) computed
-- side by side for transparency/debuggability.
-- -----------------------------------------------------------------------
signal_points AS (
    SELECT
        mc.meeting_id,
        mc.meeting_date,
        mc.department_id,
        mc.meeting_type,
        mc.organizer_seniority,
        mc.is_recurring,
        mc.duration_hours,
        mc.attendee_count,
        mc.meeting_cost,

        -- Signal 1: replaceability points by meeting_type (see header).
        CASE mc.meeting_type
            WHEN 'status_update'   THEN 5
            WHEN 'all_hands'       THEN 4
            WHEN 'brainstorm'      THEN 2
            WHEN 'decision_making' THEN 2
            WHEN 'one_on_one'      THEN 1
            ELSE 3  -- safety net if a new meeting_type value is ever added
        END AS type_points,

        -- Raw bloat ratio, carried through so it can be inspected/audited
        -- alongside the bucketed score below.
        ROUND(mc.attendee_count * 1.0 / tn.avg_attendees_for_type, 2) AS attendee_bloat_ratio,

        -- Signal 3: recurrence points — binary step function (see header).
        CASE WHEN mc.is_recurring THEN 5 ELSE 1 END AS recurring_points

    FROM meeting_costs mc
    JOIN type_norms tn ON tn.meeting_type = mc.meeting_type
),

-- -----------------------------------------------------------------------
-- STEP 3: bucket the bloat ratio into 1-5 points via NTILE, then compute
-- the weighted composite. Kept as a separate step from STEP 2 because
-- NTILE needs to see the full attendee_bloat_ratio distribution first.
-- -----------------------------------------------------------------------
weighted_scores AS (
    SELECT
        sp.*,
        NTILE(5) OVER (ORDER BY sp.attendee_bloat_ratio) AS attendee_bloat_points,
        -- Composite = weighted sum of the three 1-5 sub-scores.
        -- Weights: type 40%, bloat 35%, recurrence 25% (see header).
        ROUND(
            (sp.type_points * 0.40)
            + (NTILE(5) OVER (ORDER BY sp.attendee_bloat_ratio) * 0.35)
            + (sp.recurring_points * 0.25),
            3
        ) AS raw_burn_points
    FROM signal_points sp
)

-- -----------------------------------------------------------------------
-- STEP 4: final output — re-bucket the composite into an even 1-5 scale
-- via NTILE (mirrors RFM's use of quintiles) so "5" always means "worst
-- 20% of meetings by burn score" regardless of how the raw weighted sum
-- is distributed. Joined back to full per-meeting cost/context columns
-- from meeting_costs so this table is self-sufficient for reporting.
-- -----------------------------------------------------------------------
SELECT
    meeting_id,
    meeting_date,
    department_id,
    meeting_type,
    organizer_seniority,
    is_recurring,
    duration_hours,
    attendee_count,
    meeting_cost,
    type_points,
    attendee_bloat_ratio,
    attendee_bloat_points,
    recurring_points,
    raw_burn_points,
    NTILE(5) OVER (ORDER BY raw_burn_points) AS meeting_burn_score
FROM weighted_scores
ORDER BY meeting_burn_score DESC, meeting_cost DESC;

-- =============================================================================
-- HEADLINE FINDING QUERY
-- Top 10 "most expensive, highest burn score, recurring" meetings.
-- This is the money quote for the README / interview: these are meetings
-- that are (a) provably expensive in dollars, (b) scored as structurally
-- wasteful by type + overstaffing, and (c) repeating — i.e. still bleeding
-- money every single week this query is re-run.
-- =============================================================================
WITH type_norms AS (
    SELECT
        meeting_type,
        AVG(attendee_count) AS avg_attendees_for_type
    FROM meeting_costs
    GROUP BY meeting_type
),
signal_points AS (
    SELECT
        mc.meeting_id,
        mc.meeting_date,
        mc.department_id,
        mc.meeting_type,
        mc.organizer_seniority,
        mc.is_recurring,
        mc.duration_hours,
        mc.attendee_count,
        mc.meeting_cost,
        CASE mc.meeting_type
            WHEN 'status_update'   THEN 5
            WHEN 'all_hands'       THEN 4
            WHEN 'brainstorm'      THEN 2
            WHEN 'decision_making' THEN 2
            WHEN 'one_on_one'      THEN 1
            ELSE 3
        END AS type_points,
        ROUND(mc.attendee_count * 1.0 / tn.avg_attendees_for_type, 2) AS attendee_bloat_ratio,
        CASE WHEN mc.is_recurring THEN 5 ELSE 1 END AS recurring_points
    FROM meeting_costs mc
    JOIN type_norms tn ON tn.meeting_type = mc.meeting_type
),
weighted_scores AS (
    SELECT
        sp.*,
        NTILE(5) OVER (ORDER BY sp.attendee_bloat_ratio) AS attendee_bloat_points,
        ROUND(
            (sp.type_points * 0.40)
            + (NTILE(5) OVER (ORDER BY sp.attendee_bloat_ratio) * 0.35)
            + (sp.recurring_points * 0.25),
            3
        ) AS raw_burn_points
    FROM signal_points sp
),
scored_meetings AS (
    SELECT
        meeting_id,
        meeting_date,
        department_id,
        meeting_type,
        organizer_seniority,
        is_recurring,
        duration_hours,
        attendee_count,
        meeting_cost,
        NTILE(5) OVER (ORDER BY raw_burn_points) AS meeting_burn_score
    FROM weighted_scores
)
SELECT
    meeting_id,
    meeting_date,
    department_id,
    meeting_type,
    organizer_seniority,
    duration_hours,
    attendee_count,
    meeting_cost,
    meeting_burn_score
FROM scored_meetings
WHERE is_recurring = TRUE
  AND meeting_burn_score = 5
ORDER BY meeting_cost DESC
LIMIT 10;

-- =============================================================================
-- OPTIONAL VALIDATION QUERY (not required by spec, but useful ammunition in
-- an interview): average cost per burn-score bucket. If the scoring design
-- is doing its job, meeting_burn_score should correlate monotonically with
-- avg_meeting_cost — i.e. score 5 meetings should, on average, cost more
-- than score 1 meetings, even though cost itself is not a direct input to
-- the score. Uncomment to run standalone against the full scored set.
-- -----------------------------------------------------------------------
-- WITH type_norms AS (
--     SELECT meeting_type, AVG(attendee_count) AS avg_attendees_for_type
--     FROM meeting_costs GROUP BY meeting_type
-- ),
-- signal_points AS (
--     SELECT mc.*, tn.avg_attendees_for_type,
--         CASE mc.meeting_type
--             WHEN 'status_update' THEN 5 WHEN 'all_hands' THEN 4
--             WHEN 'brainstorm' THEN 2 WHEN 'decision_making' THEN 2
--             WHEN 'one_on_one' THEN 1 ELSE 3 END AS type_points,
--         ROUND(mc.attendee_count * 1.0 / tn.avg_attendees_for_type, 2) AS attendee_bloat_ratio,
--         CASE WHEN mc.is_recurring THEN 5 ELSE 1 END AS recurring_points
--     FROM meeting_costs mc JOIN type_norms tn ON tn.meeting_type = mc.meeting_type
-- ),
-- weighted_scores AS (
--     SELECT sp.*, NTILE(5) OVER (ORDER BY sp.attendee_bloat_ratio) AS attendee_bloat_points,
--         ROUND((sp.type_points * 0.40) + (NTILE(5) OVER (ORDER BY sp.attendee_bloat_ratio) * 0.35)
--             + (sp.recurring_points * 0.25), 3) AS raw_burn_points
--     FROM signal_points sp
-- )
-- SELECT NTILE(5) OVER (ORDER BY raw_burn_points) AS meeting_burn_score,
--        COUNT(*) AS n_meetings,
--        ROUND(AVG(meeting_cost), 2) AS avg_meeting_cost
-- FROM weighted_scores
-- GROUP BY meeting_burn_score
-- ORDER BY meeting_burn_score;

-- =============================================================================
-- EXPORTING TO CSV FOR CHARTING (next phase / Streamlit)
-- -----------------------------------------------------------------------
-- Pick whichever matches your workflow; all three produce the same shape
-- (the STEP 4 output columns above, i.e. one row per meeting with
-- meeting_burn_score included):
--
-- 1) psql (PostgreSQL):
--      \copy (SELECT * FROM (<STEP 1-4 query above>) t) \
--        TO 'exports/meeting_burn_scores.csv' WITH CSV HEADER;
--
-- 2) sqlite3 CLI:
--      sqlite3 meeting_tax.db
--      .headers on
--      .mode csv
--      .output exports/meeting_burn_scores.csv
--      <paste the STEP 1-4 query here, ending in ;>
--      .output stdout
--
-- 3) Python (works against either backend, and is the most convenient if
--    the Streamlit app is already using pandas/SQLAlchemy):
--      import pandas as pd, sqlite3  # or sqlalchemy.create_engine for Postgres
--      conn = sqlite3.connect("meeting_tax.db")
--      df = pd.read_sql(open("sql/meeting_value_score.sql").read().split(";")[0], conn)
--      df.to_csv("exports/meeting_burn_scores.csv", index=False)
--
-- Recommended file name: exports/meeting_burn_scores.csv — one row per
-- meeting, so it can be filtered/pivoted directly in the Streamlit app
-- without another join.
-- =============================================================================
