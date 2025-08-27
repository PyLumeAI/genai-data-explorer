import pandas as pd
import re
LIKELY_DATE_PAT = re.compile(r"(date|time|timestamp|dt)$", re.I)
LIKELY_QTY_PAT = re.compile(r"(qty|quantity|units?)$", re.I)
LIKELY_PRICE_PAT = re.compile(r"(price|amount|amt|rate|cost)$", re.I)
LIKELY_DISCOUNT_PAT = re.compile(r"(discount|disc)$", re.I)
LIKELY_CATEGORY_PAT = re.compile(r"(category|segment|type|class|region|dept|product|name)$", re.I)


def duck_schema_preview(con, table_name: str) -> pd.DataFrame:
    # DESCRIBE on a SELECT returns column_name / column_type for DuckDB
    return con.execute(f"DESCRIBE SELECT * FROM {table_name} LIMIT 0").fetchdf()

def quick_profile(con, table_name: str, limit: int = 5) -> pd.DataFrame:
    # small peek of data for UX context
    return con.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchdf()
def _infer_semantics(cols_df: pd.DataFrame) -> dict:
    """
    Infer basic roles: numeric, categorical, date-like; also propose computed fields.
    """
    numeric, categorical, datelike = [], [], []
    candidates_qty, candidates_price, candidates_discount = [], [], []

    for _, r in cols_df.iterrows():
        col = str(r["column_name"])
        typ = str(r["column_type"]).upper()
        # duckdb DESCRIBE returns types like VARCHAR, BIGINT, DOUBLE, DATE, TIMESTAMP
        if any(t in typ for t in ["INT", "DEC", "DOUB", "REAL", "NUM"]):
            numeric.append(col)
        elif "DATE" in typ or "TIMESTAMP" in typ or LIKELY_DATE_PAT.search(col):
            datelike.append(col)
        else:
            categorical.append(col)

        if LIKELY_QTY_PAT.search(col):
            candidates_qty.append(col)
        if LIKELY_PRICE_PAT.search(col):
            candidates_price.append(col)
        if LIKELY_DISCOUNT_PAT.search(col):
            candidates_discount.append(col)

    # Propose computed fields only if we have clear pairs
    computed = []
    for q in candidates_qty:
        for p in candidates_price:
            if q != p:
                computed.append({"name": "revenue", "expr": f"{q} * {p}"})
                # only one reasonable pair is enough
                break
        if computed:
            break

    # Discounted revenue if both price/amount and discount present
    if candidates_discount and candidates_price and candidates_qty:
        q = candidates_qty[0]; p = candidates_price[0]; d = candidates_discount[0]
        computed.append({"name": "net_revenue", "expr": f"({q} * {p}) - {d}"})

    # Some friendly generic categorical suggestions if none matched
    if not categorical:
        # fall back: treat non-numeric as category
        for _, r in cols_df.iterrows():
            col = str(r["column_name"])
            typ = str(r["column_type"]).upper()
            if not any(t in typ for t in ["INT", "DEC", "DOUB", "REAL", "NUM", "DATE", "TIMESTAMP"]):
                categorical.append(col)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datelike": datelike,
        "computed_fields": computed
    }

def schema_for_prompt(con, table_name: str) -> dict:
    cols_df = con.execute(f"DESCRIBE SELECT * FROM {table_name} LIMIT 0").fetchdf()
    cols = [{"name": r["column_name"], "dtype": str(r["column_type"])} for _, r in cols_df.iterrows()]
    semantics = _infer_semantics(cols_df)
    return {"tables": {table_name: {"columns": cols}}}
