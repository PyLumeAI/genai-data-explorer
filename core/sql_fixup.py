# core/sql_fixups.py
import re

def duckdb_fixups(sql: str) -> str:
    s = sql

    # DATE_FORMAT(...) -> strftime('%Y-%m', CAST(col AS DATE)) pattern is tricky to generalize,
    # but we at least map the function name so it's not an unknown symbol.
    s = re.sub(r'\bdate_format\s*\(', 'strftime(', s, flags=re.I)

    # If strftime appears as strftime(col, 'fmt'), swap to strftime('fmt', col)
    s = re.sub(
        r"strftime\s*\(\s*([A-Za-z_][\w]*)\s*,\s*'([^']+)'\s*\)",
        r"strftime('\2', \1)",
        s,
        flags=re.I
    )

    # Ensure CAST to DATE inside strftime('fmt', col)
    s = re.sub(
        r"strftime\('([^']+)',\s*([A-Za-z_][\w]*)\)",
        r"strftime('\1', CAST(\2 AS DATE))",
        s,
        flags=re.I
    )

    return s
