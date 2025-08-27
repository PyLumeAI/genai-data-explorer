# core/sql_guard.py
from typing import Tuple
import re

DISALLOWED = (
    "insert", "update", "delete", "drop", "alter", "create",
    "truncate", "merge", "grant", "revoke", "call", "execute",
)

def is_select_only(sql: str) -> bool:
    # Strip comments and whitespace
    s = re.sub(r"--.*?$|/\*.*?\*/", "", sql, flags=re.S).strip().lower()
    # Must start with "select" and not contain disallowed verbs
    if not s.startswith("select"):
        return False
    return not any(f" {tok} " in f" {s} " for tok in DISALLOWED)

def ensure_limit(sql: str, default_limit: int = 1000) -> str:
    s = sql.strip().rstrip(";")
    # If user already limited, keep it; else add
    if re.search(r"\blimit\s+\d+\b", s, flags=re.I):
        return s
    return f"{s} LIMIT {default_limit}"

def sanitize_sql(sql: str, default_limit: int = 1000) -> Tuple[bool, str, str]:
    """
    Returns (ok, sql_or_empty, error_message_or_empty)
    """
    if not is_select_only(sql):
        return False, "", "Only SELECT queries are allowed."
    return True, ensure_limit(sql, default_limit), ""
