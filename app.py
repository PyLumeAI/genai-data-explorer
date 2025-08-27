import streamlit as st
import pandas as pd
import altair as alt
from connectors.duck_conn import get_duck
from core.schema_introspect import duck_schema_preview, quick_profile, schema_for_prompt
from core.llm_sql import generate_sql_and_chart
from core.sql_fixups import duckdb_fixups
from core.sql_guard import sanitize_sql
import os
import numpy as np
import vl_convert as vlc  # for chart export
from datetime import datetime
import uuid




st.set_page_config(page_title="GenAI Data Explorer", page_icon="📊", layout="wide")
st.title("🧪 GenAI Data Explorer")
st.caption("Chat with your CSV. LLM → Safe SQL → Table + Chart + Summary. Stateful UX so nothing ‘resets’ on interactions.")

# ------------ Query History helpers (session-scoped) ------------
def init_history():
    if "query_history" not in st.session_state:
        st.session_state.query_history = []  # list[dict]

def append_history(question: str, sql: str, rows: int):
    init_history()
    st.session_state.query_history.append({
        "id": str(uuid.uuid4())[:8],
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "sql": sql,
        "rows": rows,
    })

def history_df():
    init_history()
    return pd.DataFrame(st.session_state.query_history)

def clear_history():
    st.session_state.query_history = []


# ---------- Session State Init ----------
def init_state():
    ss = st.session_state
    ss.setdefault("uploaded_df", None)          # stores the uploaded CSV as a DataFrame
    ss.setdefault("table_name", "data")         # current table name
    ss.setdefault("llm_result", None)           # last LLM JSON ({sql, chart, summary})
    ss.setdefault("sql_safe", None)             # sanitized SQL string
    ss.setdefault("df_result", None)            # last query result DataFrame
    # chart UI state
    ss.setdefault("chart_type", "")
    ss.setdefault("chart_x", "")
    ss.setdefault("chart_y", "")
    ss.setdefault("chart_log_y", False)
    ss.setdefault("chart_sort_desc", True)
    ss.setdefault("chart_top_n", 20)

init_state()

# ---------- DuckDB connection (cached) ----------
@st.cache_resource
def _conn():
    return get_duck()

con = _conn()

# ---------- Sidebar: CSV upload & table name ----------
st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="uploader")
table_name = st.sidebar.text_input("Register as table", value=st.session_state.table_name, key="table_name_input")

# --- Reset (two-step Yes/No) in sidebar ---
st.sidebar.markdown("---")
st.sidebar.subheader("App Controls")

# initialize the flag once
if "show_reset_confirm" not in st.session_state:
    st.session_state.show_reset_confirm = False

if not st.session_state.show_reset_confirm:
    # Step 1: user clicks Reset App -> show confirm buttons
    if st.sidebar.button("🔄 Reset App", use_container_width=True, key="reset_step1"):
        st.session_state.show_reset_confirm = True
        st.rerun()
else:
    # Step 2: confirmation UI
    st.sidebar.info("This will clear the upload, results, and chart settings.")
    col_yes, col_no = st.sidebar.columns(2)
    with col_yes:
        if st.button("✅ Yes, reset", use_container_width=True, key="reset_yes"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    with col_no:
        if st.button("❌ Cancel", use_container_width=True, key="reset_no"):
            st.session_state.show_reset_confirm = False
            st.rerun()

# --- Mini history in sidebar ---
st.sidebar.markdown("### History (mini)")
mini = history_df()
if mini.empty:
    st.sidebar.caption("No queries yet.")
else:
    # show last 5
    st.sidebar.dataframe(mini.tail(5)[["ts","rows"]], use_container_width=True, height=170)

# If a new file is uploaded, read and persist it
if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    st.session_state.uploaded_df = df_up
    st.session_state.table_name = table_name

# If we have a persisted dataframe, (re)register it every run
if st.session_state.uploaded_df is not None:
    con.register(st.session_state.table_name, st.session_state.uploaded_df)
    st.sidebar.success(f"Registered table: {st.session_state.table_name} ({len(st.session_state.uploaded_df):,} rows)")
else:
    st.info("Upload a CSV in the sidebar to begin.")
    st.stop()

# ---------- Schema + sample ----------
st.subheader("Schema")
st.dataframe(duck_schema_preview(con, st.session_state.table_name), use_container_width=True)

st.subheader("Sample")
st.dataframe(quick_profile(con, st.session_state.table_name), use_container_width=True)

# ---------- Natural language ----------
st.subheader("🧠 Ask in natural language")
q = st.text_input("Ask about your data (e.g., 'Monthly sales report' or 'Average price by product')", key="question_input")

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
if not OPENAI_KEY:
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

c1, c2 = st.columns([1,3])
with c1:
    run_llm = st.button("Generate Answer", use_container_width=True, disabled=st.session_state.uploaded_df is None or not bool(q))
with c2:
    show_sql_toggle = st.toggle("Show SQL before run", value=True, key="show_sql_toggle")

if run_llm:
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY not configured. Add it in .streamlit/secrets.toml or env.")
        st.stop()
    with st.spinner("Thinking…"):
        schema_json = schema_for_prompt(con, st.session_state.table_name)
        os.environ["OPENAI_API_KEY"] = OPENAI_KEY
        result = generate_sql_and_chart(q, schema_json)

        # # Optional: show the model's suggestion for transparency
        # with st.expander("Model suggestion (chart)", expanded=False):
        #     st.json(result.get("chart", {}))

        fixed_sql = duckdb_fixups(result["sql"])
        ok, safe_sql, err = sanitize_sql(fixed_sql, default_limit=1000)
        if not ok:
            st.error(f"Blocked query: {err}")
            st.code(result["sql"], language="sql")
            st.stop()

        # Execute and persist results in session state
        try:
            df_res = con.execute(safe_sql).fetchdf()
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.code(safe_sql, language="sql")
            st.stop()

        st.session_state.llm_result = result
        st.session_state.sql_safe = safe_sql
        st.session_state.df_result = df_res

        # log this run
        append_history(question=q, sql=safe_sql, rows=len(df_res))

# If we have a persisted result, render it (survives reruns like downloads / control changes)
if st.session_state.df_result is not None:
    if st.session_state.sql_safe and st.session_state.get("show_sql_toggle", True):
        with st.expander("Generated SQL"):
            st.code(st.session_state.sql_safe, language="sql")
    
    # --- Persisted model suggestion expander ---
    if st.session_state.llm_result:
        with st.expander("Model suggestion (chart)", expanded=False):
            st.json(st.session_state.llm_result.get("chart", {}))

    st.subheader("Result")
    st.dataframe(st.session_state.df_result, use_container_width=True)

    # -----------------------
    # Visualization (stateful)
    # -----------------------
    st.subheader("Visualization")

    # Suggested fields from model if present
    ch = (st.session_state.llm_result or {}).get("chart", {})
    x_suggest = ch.get("x")
    y_suggest = ch.get("y")
    ctype_suggest = ch.get("type", "bar")

    all_cols = list(st.session_state.df_result.columns)
    num_cols = [c for c in all_cols if np.issubdtype(st.session_state.df_result[c].dtype, np.number)]
    cat_cols = [c for c in all_cols if c not in num_cols]

    # Defaults (only when state is empty)
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
            # if no numeric columns, keep empty
            y_options = num_cols or [""]
            default_y_idx = (y_options.index(st.session_state.chart_y) if st.session_state.chart_y in y_options else 0)
            st.session_state.chart_y = st.selectbox(
                "Y axis (numeric)",
                options=y_options,
                index=default_y_idx,
                key="chart_y_select"
            )

        col_c, col_d, col_e = st.columns([1,1,1])
        with col_c:
            st.session_state.chart_log_y = st.checkbox("Log scale (Y)", value=st.session_state.chart_log_y, key="chart_log_y_check")
        with col_d:
            st.session_state.chart_sort_desc = st.checkbox("Sort descending by Y", value=st.session_state.chart_sort_desc, key="chart_sort_desc_check")
        with col_e:
            st.session_state.chart_top_n = st.number_input("Top N (for bar/pie)", min_value=5, max_value=1000, value=int(st.session_state.chart_top_n), step=5, key="chart_top_n_input")

    sel_x = st.session_state.chart_x
    sel_y = st.session_state.chart_y
    chart_type = st.session_state.chart_type
    log_y = st.session_state.chart_log_y
    sort_desc = st.session_state.chart_sort_desc
    top_n = int(st.session_state.chart_top_n)

    if sel_x and sel_y and sel_x in all_cols and sel_y in all_cols:
        plot_df = st.session_state.df_result.copy()

        # For bar/pie, aggregate duplicates on X (sum Y) and keep Top N
        if chart_type in ("bar", "pie"):
            # If multiple rows per X, groupby sum
            if plot_df.groupby(sel_x, as_index=False)[sel_y].size().shape[0] != plot_df[sel_x].nunique():
                plot_df = plot_df.groupby(sel_x, as_index=False)[sel_y].sum()
            plot_df = plot_df.sort_values(sel_y, ascending=not sort_desc).head(top_n)

        tooltips = list(plot_df.columns)

        if chart_type == "line":
            enc = alt.Chart(plot_df).mark_line().encode(
                x=alt.X(sel_x, sort=None),
                y=alt.Y(sel_y, scale=alt.Scale(type="log") if log_y else alt.Undefined),
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

        # Downloads (won't lose state because df + selections live in session_state)
        dl_col1, dl_col2, dl_col3 = st.columns([1,1,2])
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

    st.markdown("---")
    st.subheader("🗂️ Query History")

    hist = history_df()
    if hist.empty:
        st.info("No queries yet. Ask something above to build history.")
    else:
        # Recent first
        hist = hist.iloc[::-1].reset_index(drop=True)
        st.dataframe(hist[["ts","question","rows","id"]], use_container_width=True)

        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            # Re-run by ID (keeps your current data/table)
            chosen_id = st.selectbox("Re-run a query", options=hist["id"].tolist())
        with c2:
            if st.button("▶️ Re-run"):
                sql_to_run = hist.loc[hist["id"] == chosen_id, "sql"].iloc[0]
                try:
                    df_res2 = con.execute(sql_to_run).fetchdf()
                    st.session_state.sql_safe = sql_to_run
                    st.session_state.df_result = df_res2
                    st.success(f"Re-ran query {chosen_id} ({len(df_res2)} rows). Scroll up to see the result.")
                except Exception as e:
                    st.error(f"Re-run failed: {e}")
                    st.code(sql_to_run, language="sql")

        with c3:
            csv_bytes = hist.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download history (CSV)", data=csv_bytes, file_name="query_history.csv", mime="text/csv")

        # Clear history with confirmation
        with st.expander("Danger zone"):
            col_yes, col_no = st.columns(2)
            if col_yes.button("🗑️ Clear history (Yes)"):
                clear_history()
                st.experimental_set_query_params()  # noop to force rerun without losing file
                st.rerun()
            if col_no.button("❌ Cancel"):
                st.info("History not cleared.")

    st.download_button(
        "Download results as CSV",
        st.session_state.df_result.to_csv(index=False),
        file_name="answer_results.csv",
        mime="text/csv",
        key="dl_res_csv"
    )


