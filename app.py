import os
import uuid
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import vl_convert as vlc  # for chart export
from datetime import datetime
from connectors.duck_conn import get_duck
from core.schema_introspect import (
    duck_schema_preview,
    quick_profile,
    schema_for_prompt,
    schema_for_prompt_multi,   # <-- multi-table schema with join hints
)
from core.llm_sql import generate_sql_and_chart
from core.sql_fixups import duckdb_fixups
from core.sql_guard import sanitize_sql
import hashlib, io

# ----- Apply model-suggested chart defaults per new result -----
ALLOWED_CHARTS = {"bar", "line", "area", "scatter", "pie"}

def _apply_model_chart_defaults(df_res: pd.DataFrame, chart_suggestion: dict):
    """Reset chart_type/x/y to the model suggestion for the *current* result."""
    if df_res is None or df_res.empty:
        return

    all_cols = list(df_res.columns)
    num_cols = [c for c in all_cols if np.issubdtype(df_res[c].dtype, np.number)]
    cat_cols = [c for c in all_cols if c not in num_cols]

    # pull from suggestion
    ctype = (chart_suggestion or {}).get("type", "bar")
    x = (chart_suggestion or {}).get("x", "")
    y = (chart_suggestion or {}).get("y", "")
    s = (chart_suggestion or {}).get("series", "")

    # validate / fallback
    ctype_final = ctype if ctype in ALLOWED_CHARTS else "bar"
    x_final = x if x in all_cols else (cat_cols[0] if cat_cols else (all_cols[0] if all_cols else ""))
    y_final = y if y in num_cols else (num_cols[0] if num_cols else "")

    # apply to session state
    st.session_state.chart_type = ctype_final
    st.session_state.chart_x = x_final
    st.session_state.chart_y = y_final
    # default series only if it’s a valid non-numeric column
    st.session_state.chart_series = s if (s in cat_cols) else ""


# ----------------- Page header -----------------
st.set_page_config(page_title="GenAI Data Explorer", page_icon="📊", layout="wide")
st.title("🧪 GenAI Data Explorer")
st.caption("Chat with your data. Multi-file upload → LLM → Safe SQL → Table + Chart + Summary. Stateful UI so nothing ‘resets’ on interaction.")

# ----------------- Query history helpers -----------------
def init_history():
    if "query_history" not in st.session_state:
        st.session_state.query_history = []  # list[dict]

def append_history(question: str, sql: str, rows: int, chart: dict):
    init_history()
    st.session_state.query_history.append({
        "id": str(uuid.uuid4())[:8],
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "sql": sql,
        "rows": rows,
        "chart": chart
    })

def history_df():
    init_history()
    return pd.DataFrame(st.session_state.query_history)

def clear_history():
    st.session_state.query_history = []

# ----------------- Session state init -----------------
def init_state():
    ss = st.session_state
    # Multiple tables now: {table_name: DataFrame}
    ss.setdefault("tables", {})                # dict[str, pd.DataFrame]
    ss.setdefault("llm_result", None)          # last LLM JSON ({sql, chart, summary})
    ss.setdefault("sql_safe", None)            # sanitized SQL string
    ss.setdefault("df_result", None)           # last query result DataFrame
    # chart UI state
    ss.setdefault("chart_type", "")
    ss.setdefault("chart_x", "")
    ss.setdefault("chart_y", "")
    ss.setdefault("chart_series", "")
    ss.setdefault("chart_log_y", False)
    ss.setdefault("chart_sort_desc", True)
    ss.setdefault("chart_top_n", 20)
    # reset confirm flag
    ss.setdefault("show_reset_confirm", False)
    # uploaded files tracking (to avoid re-uploads if same file)
    ss.setdefault("ingested_files", {})   # {file_hash: {"name": original_name, "table": tname}}
    ss.setdefault("uploader_nonce", 0)    # forces file_uploader to remount empty after reset
    ss.setdefault("last_question", "")    # To store last question asked
    



init_state()

# ----------------- DuckDB connection -----------------
@st.cache_resource
def _conn():
    return get_duck()

con = _conn()

# ----------------- Sidebar: multi-file upload + controls -----------------
st.sidebar.header("Data Source")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True,
    key=f"uploader_multi_{st.session_state.uploader_nonce}",
)

def _sanitize_table_name(name: str) -> str:
    import re, os as _os
    stem = _os.path.splitext(name)[0]
    t = re.sub(r"[^A-Za-z0-9_]+", "_", stem.strip().lower())
    if not t:
        t = "table"
    return t

def _file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

# If new files uploaded this run, ingest only if not already seen (by hash)
if uploaded_files:
    for f in uploaded_files:
        try:
            raw = f.getvalue()  # bytes; safe to call multiple reruns
            h = _file_hash(raw)
            if h in st.session_state.ingested_files:
                # already loaded this exact file content; skip
                continue

            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            st.sidebar.error(f"Failed to read {f.name}: {e}")
            continue

        base_name = _sanitize_table_name(f.name)
        tname = base_name
        i = 2
        while tname in st.session_state.tables:
            tname = f"{base_name}_{i}"
            i += 1

        st.session_state.tables[tname] = df
        st.session_state.ingested_files[h] = {"name": f.name, "table": tname}


def _unregister_all_tables():
    # Best-effort unregister (duckdb has .unregister; if not, ignore)
    for tname in list(st.session_state.tables.keys()):
        try:
            con.unregister(tname)  # type: ignore[attr-defined]
        except Exception:
            # Fallback: no-op; register() overwrites, and reset clears our dict anyway
            pass

def reset_app():
    # Unregister from DuckDB
    _unregister_all_tables()
    # Clear our state
    st.session_state.tables = {}
    st.session_state.ingested_files = {}
    st.session_state.llm_result = None
    st.session_state.sql_safe = None
    st.session_state.df_result = None
    st.session_state.chart_type = ""
    st.session_state.chart_x = ""
    st.session_state.chart_y = ""
    st.session_state.chart_series = ""
    st.session_state.chart_log_y = False
    st.session_state.chart_sort_desc = True
    st.session_state.chart_top_n = 20
    clear_history()
    # Important: bump uploader nonce so file_uploader remounts clean
    st.session_state.uploader_nonce += 1
    # Hide confirm & rerun
    st.session_state.show_reset_confirm = False


# Two-step Reset in sidebar
st.sidebar.markdown("---")
st.sidebar.header("App Controls")

if not st.session_state.show_reset_confirm:
    if st.sidebar.button("🔄 Reset App", use_container_width=True, key="reset_step1"):
        st.session_state.show_reset_confirm = True
        st.rerun()
else:
    st.sidebar.info("This will clear uploads, results, chart settings, and history.")
    c_yes, c_no = st.sidebar.columns(2)
    with c_yes:
        if st.button("✅ Yes, reset", use_container_width=True, key="reset_yes"):
            # for k in list(st.session_state.keys()):
            #     del st.session_state[k]
            reset_app()
            # _unregister_all_tables()
            # clear_history()
            # Important: bump uploader nonce so file_uploader remounts clean
            # st.session_state.uploader_nonce += 1
            # Hide confirm & rerun
            # st.session_state.show_reset_confirm = False
            st.rerun()
    with c_no:
        if st.button("❌ Cancel", use_container_width=True, key="reset_no"):
            st.session_state.show_reset_confirm = False
            st.rerun()

# Mini history in sidebar
st.sidebar.markdown("### History (mini)")
mini = history_df()
if mini.empty:
    st.sidebar.caption("No queries yet.")
else:
    st.sidebar.dataframe(mini.tail(5)[["ts","question","rows"]], use_container_width=True, height=170)

# ----------------- Register all tables each run -----------------
if not st.session_state.tables:
    st.info("Upload one or more CSV files in the sidebar to begin.")
    st.stop()

for tname, df in st.session_state.tables.items():
    con.register(tname, df)

# ----------------- Show registered tables + schemas -----------------
with st.expander("📚 Registered Tables & Schemas", expanded=True):
    st.write(f"**Tables:** {', '.join(st.session_state.tables.keys())}")
    to_delete = None
    for tname in list(st.session_state.tables.keys()):
        st.markdown(f"**`{tname}` schema:**")
        st.dataframe(duck_schema_preview(con, tname), use_container_width=True)
        # remove button per table
        if st.button(f"Remove `{tname}`", key=f"rm_{tname}"):
            to_delete = tname
    if to_delete:
        # also delete from ingested_files (by matching table)
        del st.session_state.tables[to_delete]
        for h, meta in list(st.session_state.ingested_files.items()):
            if meta["table"] == to_delete:
                del st.session_state.ingested_files[h]
        st.success(f"Removed table `{to_delete}`.")
        st.rerun()

with st.expander("📄 Ingested Files (dedup)"):
    if st.session_state.ingested_files:
        st.table(pd.DataFrame([
            {"file": v["name"], "table": v["table"], "hash": k[:8]}
            for k, v in st.session_state.ingested_files.items()
        ]))
    else:
        st.caption("No files ingested yet.")

# ----------------- Sample preview (first table only for brevity) -----------------
first_table = list(st.session_state.tables.keys())[0]
st.subheader("Sample (first table)")
st.caption(f"Showing a few rows from `{first_table}`")
st.dataframe(quick_profile(con, first_table), use_container_width=True)

# ----------------- Natural language -----------------
st.subheader("🧠 Ask in natural language")
q = st.text_input("Ask about your data (e.g., 'Average price by product', 'Monthly trend of total amount', 'Revenue by segment')", key="question_input")

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
if not OPENAI_KEY:
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

c1, c2 = st.columns([1, 3])
with c1:
    run_llm = st.button(
        "Generate Answer",
        use_container_width=True,
        disabled=(not bool(q) or not bool(st.session_state.tables))
    )
with c2:
    show_sql_toggle = st.toggle("Show SQL before run", value=True, key="show_sql_toggle")

if run_llm:
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY not configured. Add it in .streamlit/secrets.toml or env.")
        st.stop()
    with st.spinner("Thinking…"):
        # If 2+ tables, build multi-table schema with join hints; else fallback to single
        table_names = list(st.session_state.tables.keys())
        if len(table_names) >= 2:
            schema_json = schema_for_prompt_multi(con, table_names)
        else:
            schema_json = schema_for_prompt(con, table_names[0])

        os.environ["OPENAI_API_KEY"] = OPENAI_KEY
        result = generate_sql_and_chart(q, schema_json)

        fixed_sql = duckdb_fixups(result["sql"])
        ok, safe_sql, err = sanitize_sql(fixed_sql, default_limit=1000)
        if not ok:
            st.error(f"Blocked query: {err}")
            st.code(result["sql"], language="sql")
            st.stop()

        try:
            df_res = con.execute(safe_sql).fetchdf()
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.code(safe_sql, language="sql")
            st.stop()

        st.session_state.llm_result = result
        st.session_state.sql_safe = safe_sql
        st.session_state.df_result = df_res
        append_history(question=q, sql=safe_sql, rows=len(df_res), chart=result.get("chart", {})) 

        # NEW: reset chart defaults to model suggestion for this new result
        _apply_model_chart_defaults(df_res, (result or {}).get("chart", {}))


# ----------------- Render persisted result -----------------
if st.session_state.df_result is not None:
    if st.session_state.sql_safe and st.session_state.get("show_sql_toggle", True):
        with st.expander("Generated SQL"):
            st.code(st.session_state.sql_safe, language="sql")

    if st.session_state.llm_result:
        with st.expander("Model suggestion (chart)", expanded=False):
            st.json(st.session_state.llm_result.get("chart", {}))

    st.subheader("Result")
    st.dataframe(st.session_state.df_result, use_container_width=True)

    # ------------- Visualization (stateful) -------------
    st.subheader("Visualization")
    ch = (st.session_state.llm_result or {}).get("chart", {})
    x_suggest = ch.get("x")
    y_suggest = ch.get("y")
    ctype_suggest = ch.get("type", "bar")

    all_cols = list(st.session_state.df_result.columns)
    num_cols = [c for c in all_cols if np.issubdtype(st.session_state.df_result[c].dtype, np.number)]
    cat_cols = [c for c in all_cols if c not in num_cols]

    if not st.session_state.chart_x:
        st.session_state.chart_x = x_suggest if x_suggest in all_cols else (cat_cols[0] if cat_cols else (all_cols[0] if all_cols else ""))
    if not st.session_state.chart_y:
        st.session_state.chart_y = y_suggest if y_suggest in num_cols else (num_cols[0] if num_cols else "")
    if "chart_type" not in st.session_state or st.session_state.chart_type not in ["bar","line","area","scatter","pie"]:
        st.session_state.chart_type = ctype_suggest if ctype_suggest in ["bar","line","area","scatter","pie"] else "bar"

    with st.expander("Chart controls", expanded=True):
        st.session_state.chart_type = st.selectbox(
            "Chart type",
            options=["bar", "line", "area", "scatter", "pie"],
            index=["bar", "line", "area", "scatter", "pie"].index(st.session_state.chart_type),
            key="chart_type_select"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.session_state.chart_x = st.selectbox(
                "X axis (category/date/numeric)",
                options=all_cols,
                index=all_cols.index(st.session_state.chart_x) if st.session_state.chart_x in all_cols else 0,
                key="chart_x_select"
            )
        with col_b:
            y_options = num_cols or [""]
            default_y_idx = (y_options.index(st.session_state.chart_y) if st.session_state.chart_y in y_options else 0)
            st.session_state.chart_y = st.selectbox(
                "Y axis (numeric)",
                options=y_options,
                index=default_y_idx,
                key="chart_y_select"
            )

        # Existing X/Y selects ...
        col_c0, col_d0 = st.columns(2)
        with col_c0:
            # series selection (optional, categorical only)
            series_options = ["" ] + cat_cols  # empty = no series
            default_series_idx = series_options.index(st.session_state.chart_series) if st.session_state.chart_series in series_options else 0
            st.session_state.chart_series = st.selectbox(
                "Series (optional, categorical = multiple lines/bars)",
                options=series_options,
                index=default_series_idx,
                key="chart_series_select"
            )

        
        col_c, col_d, col_e = st.columns([1,1,1])
        with col_c:
            st.session_state.chart_log_y = st.checkbox("Log scale (Y)", value=st.session_state.chart_log_y, key="chart_log_y_check")
        with col_d:
            st.session_state.chart_sort_desc = st.checkbox("Sort descending by Y", value=st.session_state.chart_sort_desc, key="chart_sort_desc_check")
        with col_e:
            st.session_state.chart_top_n = st.number_input("Top N (for bar/pie)", min_value=5, max_value=1000, value=int(st.session_state.chart_top_n), step=5, key="chart_top_n_input")

    sel_x       = st.session_state.chart_x
    sel_y       = st.session_state.chart_y
    chart_type  = st.session_state.chart_type
    log_y       = st.session_state.chart_log_y
    sort_desc   = st.session_state.chart_sort_desc
    top_n       = int(st.session_state.chart_top_n)

    if sel_x and sel_y and sel_x in all_cols and sel_y in all_cols:
        plot_df = st.session_state.df_result.copy()

        # For bar/pie, aggregate duplicates on X (sum Y) and keep Top N
        if chart_type in ("bar", "pie"):
            if plot_df.groupby(sel_x, as_index=False)[sel_y].size().shape[0] != plot_df[sel_x].nunique():
                plot_df = plot_df.groupby(sel_x, as_index=False)[sel_y].sum()
            plot_df = plot_df.sort_values(sel_y, ascending=not sort_desc).head(top_n)

        series   = st.session_state.chart_series or None
        tooltips = list(plot_df.columns)

        if chart_type == "line":
            enc = alt.Chart(plot_df).mark_line().encode(
                x=alt.X(sel_x, sort=None),                
                y=alt.Y(sel_y, scale=alt.Scale(type="log") if log_y else alt.Undefined),
                color=alt.Color(series, legend=alt.Legend(title=series)) if series else alt.Undefined,
                tooltip=tooltips
            )
        elif chart_type == "area":
            enc = alt.Chart(plot_df).mark_area().encode(
                x=alt.X(sel_x, sort=None),
                y=alt.Y(sel_y, scale=alt.Scale(type="log") if log_y else alt.Undefined),
                tooltip=tooltips
            )
        elif chart_type == "scatter":
            enc = alt.Chart(plot_df).mark_point().encode(
                x=alt.X(sel_x, sort=None),
                y=alt.Y(sel_y, scale=alt.Scale(type="log") if log_y else alt.Undefined),
                tooltip=tooltips
            )
        elif chart_type == "pie":
            enc = alt.Chart(plot_df).mark_arc().encode(
                theta=alt.Theta(field=sel_y, type='quantitative'),
                color=alt.Color(field=sel_x, type='nominal', sort=None),
                tooltip=tooltips
            )
        else:  # bar
            enc = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X(sel_x, sort=None),
                y=alt.Y(sel_y, scale=alt.Scale(type="log") if log_y else alt.Undefined),
                tooltip=tooltips
            )

        chart = enc.properties(height=380)
        st.altair_chart(chart, use_container_width=True)

        dl_col1, dl_col2, _ = st.columns([1,1,2])
        with dl_col1:
            png_bytes = vlc.vegalite_to_png(chart.to_dict(), scale=2)
            st.download_button("🖼️ PNG", data=png_bytes, file_name="chart.png", mime="image/png", key="dl_png_btn")
        with dl_col2:
            svg_text = vlc.vegalite_to_svg(chart.to_dict())
            st.download_button("🧩 SVG", data=svg_text, file_name="chart.svg", mime="image/svg+xml", key="dl_svg_btn")
    else:
        st.info("Select valid X/Y columns to render a chart.")

    if (st.session_state.llm_result or {}).get("summary"):
        st.success(st.session_state.llm_result["summary"])

    # ------------- Query History -------------
    st.markdown("---")
    st.subheader("🗂️ Query History")

    hist = history_df()
    if hist.empty:
        st.info("No queries yet. Ask something above to build history.")
    else:
        hist = hist.iloc[::-1].reset_index(drop=True)
        st.dataframe(hist[["ts","question","rows","id"]], use_container_width=True)

        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            chosen_id = st.selectbox("Re-run a query", options=hist["id"].tolist())
        with c2:
            if st.button("▶️ Re-run"):                
                sql_to_run = hist.loc[hist["id"] == chosen_id, "sql"].iloc[0]
                question   = hist.loc[hist["id"] == chosen_id, "question"].iloc[0]
                chart      = hist.loc[hist["id"] == chosen_id, "chart"].iloc[0]

                try:
                    df_res2 = con.execute(sql_to_run).fetchdf()
                    # Update the results
                    st.session_state.sql_safe = sql_to_run
                    st.session_state.df_result = df_res2

                    # Update question in UI + state
                    st.session_state.last_question    = question
                    st.session_state["last_question"] = question
                    

                    st.session_state.llm_result = chart
                    _apply_model_chart_defaults(df_res2, {})
                    # st.success(f"Re-ran query {chosen_id} ({len(df_res2)} rows). Scroll up to see the result.")
                    # reset extra toggles each re-run
                    st.session_state.chart_log_y = False
                    st.session_state.chart_sort_desc = True
                    st.session_state.chart_top_n = 20
                    st.session_state.chart_series = ""
                    st.session_state.chart_type = chart["type"] if chart and chart.get("type") in ALLOWED_CHARTS else "bar"
                    st.session_state.chart_x = chart["x"] if chart and "x" in chart else ""
                    st.session_state.chart_y = chart["y"] if chart and "y" in chart else ""
                    st.session_state.chart_series = chart["series"] if chart and "series" in chart else ""
                    st.rerun()
                except Exception as e:
                    st.error(f"Re-run failed: {e}")
                    st.code(sql_to_run, language="sql")

        with c3:
            csv_bytes = hist.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download history (CSV)", data=csv_bytes, file_name="query_history.csv", mime="text/csv")

        with st.expander("Danger zone"):
            col_yes, col_no = st.columns(2)
            if col_yes.button("🗑️ Clear history (Yes)"):
                clear_history()
                st.rerun()
            if col_no.button("❌ Cancel"):
                st.info("History not cleared.")
