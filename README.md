# 🗓️ The Meeting Tax

**The Meeting Tax** is an interactive Streamlit dashboard that analyzes the financial cost and potential productivity impact of workplace meetings.

> How much do meetings actually cost, which ones are the worst offenders, and is meeting load related to productivity and burnout?

---

## 📊 Features

### 💰 KPI Overview
At a glance, the dashboard surfaces:
- **Total Annual Projected Meeting Spend** — annualized from the selected date range (not a flat sum), so it stays comparable across different-length windows
- **Spend from High-Burn Meetings (Score 5)** — the share of total spend coming from the worst-offending meetings
- **Meetings Analyzed** — total meeting count in the current filtered view

All KPIs update live based on the sidebar filters.

### 🎚️ What-If: Cut Recurring Status-Update Meetings
Recurring `status_update` meetings are the single biggest "this could have been an email" category (see `sql/meeting_value_score.sql`). A slider lets you model trimming these meetings by 0–100% and see the impact update live:
- **Projected Annual Savings**
- **% of Total Annual Spend**
- **Recurring Status-Updates In View**

### 🏢 Cost by Department
A horizontal bar chart comparing total meeting cost across departments (Engineering, Sales, Operations, Marketing, Finance, HR).

### 🔥 Meeting Burn Score Distribution
Meetings are assigned a **Burn Score from 1 to 5**, where 5 represents the worst offenders. This chart shows how many meetings fall into each score bucket.

### 📉 Meeting Hours vs. Self-Reported Productivity
A scatter plot (with trend line) of weekly meeting hours against self-reported productivity (1–10), plus the Pearson correlation coefficient and p-value calculated across all filtered rows.

> **Note:** Correlation does not imply causation. The chart itself is rendered from a random sample of up to 3,000 points for speed, but the reported r/p values use the full filtered dataset. See `notebooks/04_stats_correlation.ipynb` for the full regression and caveats.

### 🧰 Filters
All views respond to a shared sidebar:
- **Department** — multi-select (Engineering, Finance, HR, Marketing, Operations, Sales)
- **Meeting Type** — multi-select (all_hands, brainstorm, decision_making, one_on_one, status_update)
- **Date Range** — custom range picker

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Streamlit
- Altair
- SQL
- Jupyter Notebook

---

## 📁 Project Structure

```text
The Meeting Tax/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── processed/
│       ├── meeting_costs.csv
│       ├── meeting_burn_scores.csv
│       └── pulse_clean.csv
│
├── notebooks/
├── sql/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "The Meeting Tax"
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit application**
```bash
streamlit run app/streamlit_app.py
```

The dashboard will open in your browser.

---

## 📌 Analysis

The dashboard provides interactive filtering by **Department**, **Meeting Type**, and **Date Range**. Selected filters dynamically update the KPIs, meeting cost breakdowns, burn score distribution, and the productivity scatter plot.

The productivity analysis uses employee-week pulse data and calculates the Pearson correlation between weekly meeting hours and self-reported productivity.

> **Note:** Correlation does not imply causation. The productivity analysis shows an observed relationship and does not establish that meetings directly cause changes in productivity.

---
## Output 
<img width="1912" height="808" alt="image" src="https://github.com/user-attachments/assets/ae9b4448-44c6-45af-ba56-c1bbe475f45c" />
<img width="1917" height="533" alt="image" src="https://github.com/user-attachments/assets/a7ac21a1-1820-4212-95c7-fe45d7d021da" />
<img width="1904" height="893" alt="image" src="https://github.com/user-attachments/assets/f4422c6c-e4bf-4a0c-91e2-cf518fbc30f7" />


## 🔮 Future Improvements

- Meeting cost trends over time
- Top costly meetings leaderboard
- Detailed meeting drill-down tables *(flagged in-app as coming in the next iteration)*
- Additional productivity and burnout analysis
- Automated data refresh
- Cloud deployment
