# core/llm_sql.py
from typing import Dict, Any, List
import json
import os

# If you prefer litellm, you can swap imports easily.
from openai import OpenAI

ChartType = str  # "bar" | "line" | "area" | "scatter" | "pie"

# core/llm_sql.py (only the prompt & few-shots changed here)
SYSTEM_PROMPT = """You are a senior analytics engineer.
You will receive:
- A user question.
- A SCHEMA JSON containing tables/columns.
- A SEMANTICS JSON with optional hints: {numeric, categorical, datelike, computed_fields[]}.

Return STRICT JSON ONLY in this exact shape (no extra text):

{
  "sql": "SELECT ...",
  "chart": {"type":"bar|line|area|scatter|pie","x":"<col>","y":"<col>","series":""},
  "summary": "<1-2 sentence takeaway>"
}

Rules:
- Target dialect is DUCKDB. Use only DuckDB-compatible SQL.
- Use the EXACT table name(s) from the provided schema JSON. Do not invent tables.
- SQL must be SELECT-only (no DDL/DML).
- If there is no LIMIT, add LIMIT 1000.
- When extracting year-month, prefer one of:
  • strftime('%Y-%m', CAST(date_col AS DATE))  -- string label
  • date_trunc('month', CAST(date_col AS DATE)) -- DATE bucket (good for time series)
- Prefer GROUP BY when the question asks for “by …”.
- If there is a date-like column and the question asks for trends, prefer a LINE chart with the date-like column on X.
- If one categorical + one numeric, prefer BAR.
- If two numeric columns, prefer SCATTER.
- If the question asks for a breakdown by a categorical field (e.g., “by segment/region/product”), set chart.series to that field so multi-series visuals are possible.
- PIE is acceptable for small categorical distributions (<=10).
- You MAY use computed_fields from SEMANTICS (by inlining their expression in SELECT) when relevant.
- Do NOT invent columns. Always alias computed expressions.

Important:
- Do NOT invent columns or tables.
- Use semantic hints only if they clearly apply.
"""

FEW_SHOT = [
    {
        "q": "Show total of a numeric field by a category",
        "schema_hint": {
            "tables": {"mytable": {"columns": [
                {"name": "category", "dtype": "VARCHAR"},
                {"name": "value", "dtype": "BIGINT"}
            ]}},
            "semantics": {"categorical": ["category"], "numeric": ["value"], "datelike": [], "computed_fields": []}
        },
        "a": {
            "sql": "SELECT category, SUM(value) AS total_value FROM mytable GROUP BY category ORDER BY total_value DESC",
            "chart": {"type": "bar", "x": "category", "y": "total_value", "series": ""},
            "summary": "Categories ranked by total value."
        }
    },
    {
        "q": "Show trend of a numeric field over time",
        "schema_hint": {
            "tables": {"mytable": {"columns": [
                {"name": "ts", "dtype": "DATE"},
                {"name": "metric", "dtype": "BIGINT"}
            ]}},
            "semantics": {"categorical": [], "numeric": ["metric"], "datelike": ["ts"], "computed_fields": []}
        },
        "a": {
            "sql": "SELECT date_trunc('month', CAST(ts AS DATE)) AS month, SUM(metric) AS total_metric FROM mytable GROUP BY month ORDER BY month",
            "chart": {"type": "line", "x": "month", "y": "total_metric", "series": ""},
            "summary": "Monthly trend of the metric."
        }
    },
    {
        "q": "Use a computed measure if relevant",
        "schema_hint": {
            "tables": {"mytable": {"columns": [
                {"name": "region", "dtype": "VARCHAR"},
                {"name": "quantity", "dtype": "BIGINT"},
                {"name": "price", "dtype": "BIGINT"}
            ]}},
            "semantics": {"categorical": ["region"], "numeric": ["quantity","price"], "datelike": [], "computed_fields": [
                {"name": "revenue", "expr": "quantity * price"}
            ]}
        },
        "a": {
            "sql": "SELECT region, SUM(quantity * price) AS revenue FROM mytable GROUP BY region ORDER BY revenue DESC",
            "chart": {"type": "bar", "x": "region", "y": "revenue", "series": ""},
            "summary": "Regions ranked by computed revenue."
        }
    },
    {
    "q": "Monthly trend broken down by category",
    "schema_hint": {
        "tables": {
            "mytable": {
                "columns": [
                    {"name": "ts", "dtype": "DATE"},
                    {"name": "category", "dtype": "VARCHAR"},
                    {"name": "value", "dtype": "DOUBLE"}
                ]
            }
        },
        "semantics": {
            "categorical": ["category"],
            "numeric": ["value"],
            "datelike": ["ts"]
        }
    },
    "a": {
        "sql": "SELECT date_trunc('month', CAST(ts AS DATE)) AS month, category, SUM(value) AS total_value FROM mytable GROUP BY month, category ORDER BY month, category LIMIT 1000",
        "chart": {"type": "line", "x": "month", "y": "total_value", "series": "category"},
        "summary": "Monthly trend with one line per category."
    }
}

]



def _build_user_prompt(question: str, schema_json: Dict[str, Any]) -> str:
    return json.dumps({
        "question": question,
        "schema": schema_json,
        "few_shot": FEW_SHOT
    }, ensure_ascii=False)

def _parse_json(s: str) -> Dict[str, Any]:
    # Try to extract the first JSON object in the response
    s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return JSON.")
    return json.loads(s[start:end+1])

def generate_sql_and_chart(
    question: str,
    schema_json: Dict[str, Any],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Returns dict: {"sql": str, "chart": {...}, "summary": str}
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or secrets.")
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(question, schema_json)},
        ],
        temperature=0.2,
    )

    raw = completion.choices[0].message.content or ""
    data = _parse_json(raw)

    # Normalize fields
    data.setdefault("chart", {})
    ch = data["chart"]
    ch.setdefault("type", "bar")
    ch.setdefault("x", "")
    ch.setdefault("y", "")
    ch.setdefault("series", "")

    data.setdefault("sql", "")
    data.setdefault("summary", "")

    return data
