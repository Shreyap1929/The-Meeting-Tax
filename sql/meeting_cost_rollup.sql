-- =============================================================================
-- meeting_cost_rollup.sql
-- "The Meeting Tax" — meeting cost calculation and rollups
--
-- Compatible with PostgreSQL and SQLite (no dialect-specific syntax used).
--
-- COST MODEL / ASSUMPTIONS
-- -----------------------------------------------------------------------
-- 1. hourly_rate is DERIVED from each attendee's annual_salary, not read
--    from employees.hourly_rate or meeting_attendees.hourly_rate/row_cost.
--    Formula:  hourly_rate = annual_salary / 1920.0
--    (1920 = 48 working weeks x 40 hours, a standard "working hours/year"
--    denominator). This keeps a single, auditable source of truth for cost,
--    independent of whatever pre-populated the meeting_attendees columns.
-- 2. Per-meeting cost = duration_hours x attendee_count x hourly_rate.
--    Because attendees on the same meeting can have different salaries,
--    this is computed at the attendee-row grain (duration_hours x each
--    attendee's own hourly_rate) and then SUMmed per meeting. That sum is
--    mathematically identical to duration_hours x attendee_count x
--    hourly_rate when every attendee shares one rate, and is the correct
--    generalization when they don't.
-- 3. "Annual" cost = the total cost of every meeting row currently in the
--    meetings table. The dataset is assumed to represent one calendar
--    year of activity (recurring meetings are already expanded into one
--    row per occurrence). If your dataset spans a different number of
--    months, divide the totals below by (months_covered / 12.0) to
--    normalize to a true annual run-rate.
-- =============================================================================

-- -----------------------------------------------------------------------
-- STEP 0: rebuild the reusable view so this script is safely re-runnable.
-- A VIEW (rather than a bare CTE) lets every rollup query below reference
-- the same per-meeting cost logic without repeating the CTE text.
-- -----------------------------------------------------------------------
DROP VIEW IF EXISTS meeting_costs;

CREATE VIEW meeting_costs AS
WITH attendee_costs AS (
    -- Grain: one row per (meeting, attendee). This is where hourly_rate
    -- is derived and multiplied by the meeting's duration in hours.
    SELECT
        m.meeting_id,
        m.meeting_date,
        m.department_id,
        m.meeting_type,
        m.organizer_seniority,
        m.is_recurring,
        m.duration_mins / 60.0                    AS duration_hours,
        ma.employee_id,
        e.annual_salary / 1920.0                   AS hourly_rate,
        (m.duration_mins / 60.0) * (e.annual_salary / 1920.0) AS attendee_row_cost
    FROM meetings m
    JOIN meeting_attendees ma ON ma.meeting_id = m.meeting_id
    JOIN employees e         ON e.employee_id = ma.employee_id
)
-- Grain: one row per meeting. Collapse attendee rows into a single
-- meeting_cost = duration_hours x SUM(attendee hourly_rate).
SELECT
    meeting_id,
    meeting_date,
    department_id,
    meeting_type,
    organizer_seniority,
    is_recurring,
    duration_hours,
    COUNT(employee_id)          AS attendee_count,
    ROUND(SUM(attendee_row_cost), 2) AS meeting_cost
FROM attendee_costs
GROUP BY
    meeting_id,
    meeting_date,
    department_id,
    meeting_type,
    organizer_seniority,
    is_recurring,
    duration_hours;

-- =============================================================================
-- ROLLUP 1: total annual cost by department
-- =============================================================================
SELECT
    d.department_id,
    d.department_name,
    COUNT(mc.meeting_id)                                   AS total_meetings,
    ROUND(SUM(mc.duration_hours), 2)                       AS total_hours,
    ROUND(SUM(mc.meeting_cost), 2)                         AS total_annual_cost,
    ROUND(SUM(mc.meeting_cost) / NULLIF(COUNT(mc.meeting_id), 0), 2) AS avg_cost_per_meeting,
    ROUND(
        100.0 * SUM(mc.meeting_cost) / NULLIF(SUM(SUM(mc.meeting_cost)) OVER (), 0),
        2
    )                                                       AS pct_of_org_total
FROM meeting_costs mc
JOIN departments d ON d.department_id = mc.department_id
GROUP BY d.department_id, d.department_name
ORDER BY total_annual_cost DESC;

-- =============================================================================
-- ROLLUP 2: total annual cost by meeting_type
-- =============================================================================
SELECT
    mc.meeting_type,
    COUNT(mc.meeting_id)                                   AS total_meetings,
    ROUND(SUM(mc.duration_hours), 2)                       AS total_hours,
    ROUND(SUM(mc.meeting_cost), 2)                         AS total_annual_cost,
    ROUND(SUM(mc.meeting_cost) / NULLIF(COUNT(mc.meeting_id), 0), 2) AS avg_cost_per_meeting,
    ROUND(
        100.0 * SUM(mc.meeting_cost) / NULLIF(SUM(SUM(mc.meeting_cost)) OVER (), 0),
        2
    )                                                       AS pct_of_org_total
FROM meeting_costs mc
GROUP BY mc.meeting_type
ORDER BY total_annual_cost DESC;

-- =============================================================================
-- ROLLUP 3: total annual cost by organizer_seniority
-- =============================================================================
SELECT
    mc.organizer_seniority,
    COUNT(mc.meeting_id)                                   AS total_meetings,
    ROUND(SUM(mc.duration_hours), 2)                       AS total_hours,
    ROUND(SUM(mc.meeting_cost), 2)                         AS total_annual_cost,
    ROUND(SUM(mc.meeting_cost) / NULLIF(COUNT(mc.meeting_id), 0), 2) AS avg_cost_per_meeting,
    ROUND(
        100.0 * SUM(mc.meeting_cost) / NULLIF(SUM(SUM(mc.meeting_cost)) OVER (), 0),
        2
    )                                                       AS pct_of_org_total
FROM meeting_costs mc
GROUP BY mc.organizer_seniority
ORDER BY total_annual_cost DESC;

-- =============================================================================
-- HEADLINE NUMBER: grand total projected annual meeting spend, org-wide
-- =============================================================================
SELECT
    COUNT(meeting_id)              AS total_meetings,
    ROUND(SUM(duration_hours), 2)  AS total_hours,
    ROUND(SUM(meeting_cost), 2)    AS org_total_annual_meeting_spend
FROM meeting_costs;
