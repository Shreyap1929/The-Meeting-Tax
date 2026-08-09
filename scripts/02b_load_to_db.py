from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[1]
MEETINGS_CSV = ROOT / "data" / "processed" / "meetings_clean.csv"
PULSE_CSV = ROOT / "data" / "processed" / "pulse_clean.csv"
SCHEMA_SQL = ROOT / "sql" / "schema.sql"


SQLITE_SCHEMA = """
PRAGMA foreign_keys = OFF;
DROP TABLE IF EXISTS pulse_weekly;
DROP TABLE IF EXISTS meeting_attendees;
DROP TABLE IF EXISTS meetings;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;

CREATE TABLE departments (
    department_id INTEGER PRIMARY KEY,
    department_name TEXT NOT NULL UNIQUE
);

CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    department_id INTEGER NOT NULL,
    seniority TEXT NOT NULL,
    hourly_rate REAL NOT NULL,
    annual_salary REAL NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

CREATE TABLE meetings (
    meeting_id INTEGER PRIMARY KEY,
    meeting_date TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    meeting_type TEXT NOT NULL,
    duration_mins INTEGER NOT NULL,
    attendee_count INTEGER NOT NULL,
    organizer_seniority TEXT NOT NULL,
    is_recurring INTEGER NOT NULL,
    recurring_series_id TEXT,
    is_disproportionately_expensive INTEGER NOT NULL,
    duration_was_imputed INTEGER NOT NULL,
    duration_outlier_flag INTEGER NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

CREATE TABLE meeting_attendees (
    meeting_id INTEGER NOT NULL,
    employee_id INTEGER NOT NULL,
    attendee_seniority TEXT NOT NULL,
    attendee_department_id INTEGER NOT NULL,
    hourly_rate REAL NOT NULL,
    row_cost REAL NOT NULL,
    PRIMARY KEY (meeting_id, employee_id),
    FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
    FOREIGN KEY (attendee_department_id) REFERENCES departments(department_id)
);

CREATE TABLE pulse_weekly (
    employee_id INTEGER NOT NULL,
    week_start TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    seniority TEXT NOT NULL,
    hours_in_meetings REAL NOT NULL,
    self_reported_productivity INTEGER NOT NULL,
    self_reported_burnout INTEGER NOT NULL,
    productivity_was_imputed INTEGER NOT NULL,
    burnout_was_imputed INTEGER NOT NULL,
    PRIMARY KEY (employee_id, week_start),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);
PRAGMA foreign_keys = ON;
"""


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def execute_schema(engine) -> None:
    if engine.dialect.name == "sqlite":
        with engine.raw_connection() as raw_conn:
            raw_conn.executescript(SQLITE_SCHEMA)
            raw_conn.commit()
        return

    schema_sql = SCHEMA_SQL.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in schema_sql.split(";") if stmt.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def build_tables(meetings_raw: pd.DataFrame, pulse_raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    meetings_raw = meetings_raw.copy()
    pulse_raw = pulse_raw.copy()

    meetings_raw["date"] = pd.to_datetime(meetings_raw["date"]).dt.date
    pulse_raw["week"] = pd.to_datetime(pulse_raw["week"]).dt.date

    bool_cols_meetings = [
        "is_recurring",
        "is_disproportionately_expensive",
        "duration_was_imputed",
        "duration_outlier_flag",
    ]
    for col in bool_cols_meetings:
        meetings_raw[col] = as_bool(meetings_raw[col])

    for col in ["productivity_was_imputed", "burnout_was_imputed"]:
        pulse_raw[col] = as_bool(pulse_raw[col])

    dept_values = sorted(
        set(meetings_raw["department"].dropna().unique())
        | set(meetings_raw["attendee_department"].dropna().unique())
        | set(pulse_raw["department"].dropna().unique())
    )
    departments = pd.DataFrame(
        {
            "department_id": range(1, len(dept_values) + 1),
            "department_name": dept_values,
        }
    )
    dept_id_map = dict(zip(departments["department_name"], departments["department_id"]))

    pulse_employee = pulse_raw[["employee_id", "department", "seniority"]].drop_duplicates("employee_id")
    meeting_employee = (
        meetings_raw[["employee_id", "attendee_department", "attendee_seniority", "hourly_rate"]]
        .drop_duplicates("employee_id")
        .rename(
            columns={
                "attendee_department": "department_meeting",
                "attendee_seniority": "seniority_meeting",
            }
        )
    )

    employees = pulse_employee.merge(meeting_employee, on="employee_id", how="left")
    employees["department_name"] = employees["department_meeting"].fillna(employees["department"])
    employees["seniority"] = employees["seniority_meeting"].fillna(employees["seniority"])
    employees["department_id"] = employees["department_name"].map(dept_id_map).astype(int)
    employees["hourly_rate"] = employees["hourly_rate"].astype(float).round(2)
    employees["annual_salary"] = (employees["hourly_rate"] * 1920.0).round(2)
    employees = employees[["employee_id", "department_id", "seniority", "hourly_rate", "annual_salary"]]

    meetings = (
        meetings_raw[
            [
                "meeting_id",
                "date",
                "department",
                "meeting_type",
                "duration_mins",
                "attendee_count",
                "organizer_seniority",
                "is_recurring",
                "recurring_series_id",
                "is_disproportionately_expensive",
                "duration_was_imputed",
                "duration_outlier_flag",
            ]
        ]
        .drop_duplicates("meeting_id")
        .rename(columns={"date": "meeting_date"})
    )
    meetings["department_id"] = meetings["department"].map(dept_id_map).astype(int)
    meetings["duration_mins"] = meetings["duration_mins"].round().astype(int)
    meetings["attendee_count"] = meetings["attendee_count"].astype(int)
    meetings = meetings[
        [
            "meeting_id",
            "meeting_date",
            "department_id",
            "meeting_type",
            "duration_mins",
            "attendee_count",
            "organizer_seniority",
            "is_recurring",
            "recurring_series_id",
            "is_disproportionately_expensive",
            "duration_was_imputed",
            "duration_outlier_flag",
        ]
    ]

    meeting_attendees = meetings_raw[
        [
            "meeting_id",
            "employee_id",
            "attendee_seniority",
            "attendee_department",
            "hourly_rate",
            "row_cost",
        ]
    ].copy()
    meeting_attendees["attendee_department_id"] = meeting_attendees["attendee_department"].map(dept_id_map).astype(int)
    meeting_attendees["hourly_rate"] = meeting_attendees["hourly_rate"].astype(float).round(2)
    meeting_attendees["row_cost"] = meeting_attendees["row_cost"].astype(float).round(2)
    meeting_attendees = meeting_attendees.drop(columns=["attendee_department"])
    meeting_attendees = meeting_attendees.drop_duplicates(["meeting_id", "employee_id"])

    pulse_weekly = pulse_raw.rename(columns={"week": "week_start"}).copy()
    pulse_weekly["department_id"] = pulse_weekly["department"].map(dept_id_map).astype(int)
    pulse_weekly["hours_in_meetings"] = pulse_weekly["hours_in_meetings"].astype(float).round(2)
    pulse_weekly["self_reported_productivity"] = pulse_weekly["self_reported_productivity"].astype(int)
    pulse_weekly["self_reported_burnout"] = pulse_weekly["self_reported_burnout"].astype(int)
    pulse_weekly = pulse_weekly[
        [
            "employee_id",
            "week_start",
            "department_id",
            "seniority",
            "hours_in_meetings",
            "self_reported_productivity",
            "self_reported_burnout",
            "productivity_was_imputed",
            "burnout_was_imputed",
        ]
    ].drop_duplicates(["employee_id", "week_start"])

    return {
        "departments": departments,
        "employees": employees,
        "meetings": meetings,
        "meeting_attendees": meeting_attendees,
        "pulse_weekly": pulse_weekly,
    }


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set.\n"
            "Example PostgreSQL:\n"
            "  set DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/meeting_tax\n"
            "SQLite fallback:\n"
            "  set DATABASE_URL=sqlite:///meeting_tax.db"
        )

    # SQLite fallback example if PostgreSQL is not available locally:
    # set DATABASE_URL=sqlite:///meeting_tax.db
    engine = create_engine(database_url, future=True)

    meetings_raw = pd.read_csv(MEETINGS_CSV)
    pulse_raw = pd.read_csv(PULSE_CSV)
    tables = build_tables(meetings_raw, pulse_raw)

    execute_schema(engine)

    with engine.begin() as conn:
        tables["departments"].to_sql("departments", conn, if_exists="append", index=False)
        tables["employees"].to_sql("employees", conn, if_exists="append", index=False)
        tables["meetings"].to_sql("meetings", conn, if_exists="append", index=False)
        tables["meeting_attendees"].to_sql("meeting_attendees", conn, if_exists="append", index=False)
        tables["pulse_weekly"].to_sql("pulse_weekly", conn, if_exists="append", index=False)

    print("Load complete.")
    for table_name in ["departments", "employees", "meetings", "meeting_attendees", "pulse_weekly"]:
        print(f"{table_name}: {len(tables[table_name]):,}")


if __name__ == "__main__":
    main()
