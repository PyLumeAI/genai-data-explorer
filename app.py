import streamlit as st
import pandas as pd
import altair as alt
from connectors.duck_conn import get_duck
from core.schema_introspect import duck_schema_preview, quick_profile, schema_for_prompt
from core.llm_sql import generate_sql_and_chart
from core.sql_fixup import duckdb_fixups
from core.sql_guard import sanitize_sql
import os
import numpy as np
import vl_convert as vlc  # for chart export


st.set_page_config(page_title="GenAI Data Explorer (Baseline)", page_icon="📊", layout="wide")
st.title("🧪 GenAI Data Explorer — Baseline")
st.caption("Step 1: CSV → DuckDB → Schema preview + simple chart. LLM coming next.")

# ---- DuckDB connection (cached) ----
@st.cache_resource
def _conn():
    return get_duck()
con = _conn()

# ---- Sidebar: CSV upload & table name ----
st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
table_name = st.sidebar.text_input("Register as table", value="data")

if uploaded:
    df = pd.read_csv(uploaded)
    # Register the dataframe as a DuckDB view bound to this df
    con.register(table_name, df)
    st.sidebar.success(f"Registered table: {table_name} ({len(df):,} rows)")

    # ---- Schema + sample preview ----
    st.subheader("Schema")
    st.dataframe(duck_schema_preview(con, table_name), use_container_width=True)

    st.subheader("Sample")
    st.dataframe(quick_profile(con, table_name), use_container_width=True)

    # ---- Simple exploration without LLM ----
    st.subheader("Quick Explore (no LLM yet)")

    st.markdown("---")
    st.subheader("🧠 Ask in natural language")

    q = st.text_input("Ask about your data (e.g., 'Average price by product in 2023')")

    # Read API key from Streamlit secrets or environment
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    if not OPENAI_KEY:
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    ask_col1, ask_col2 = st.columns([1, 3])
    with ask_col1:
        run_llm = st.button("Generate Answer", use_container_width=True, disabled=not (uploaded and q))
    with ask_col2:
        show_sql_toggle = st.toggle("Show SQL before run", value=True)

    if run_llm:
        if not OPENAI_KEY:
            st.error("OPENAI_API_KEY not configured. Add it in .streamlit/secrets.toml or env.")
            st.stop()

        with st.spinner("Thinking…"):
            # Build schema JSON for the prompt
            schema_json = schema_for_prompt(con, table_name)
            # Temporarily set env for llm_sql
            os.environ["OPENAI_API_KEY"] = OPENAI_KEY
            result = generate_sql_and_chart(q, schema_json)

            # Apply dialect fixups first (e.g., DATE_FORMAT -> strftime, correct arg order, add CAST)
            fixed_sql = duckdb_fixups(result["sql"])

            # Then sanitize (SELECT-only + LIMIT)
            ok, safe_sql, err = sanitize_sql(fixed_sql, default_limit=1000)
            if not ok:
                st.error(f"Blocked query: {err}")
                st.code(result["sql"], language="sql")
                st.stop()

        if show_sql_toggle:
            with st.expander("Generated SQL"):
                st.code(safe_sql, language="sql")

        # Execute
        try:
            df_res = con.execute(safe_sql).fetchdf()
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.code(safe_sql, language="sql")
            st.stop()

        # Show results
        st.subheader("Result")
        st.dataframe(df_res, use_container_width=True)

        # Chart (best-effort from LLM; fall back if invalid)
        ch = result.get("chart", {})
        x, y, ctype = ch.get("x"), ch.get("y"), ch.get("type", "bar")

        def _valid_cols(cols: list[str]) -> list[str]:
            return [c for c in cols if c in df_res.columns]

        valid_x = _valid_cols([x])
        valid_y = _valid_cols([y])
        if x in valid_x and y in valid_y:
            if ctype == "line":
                chart = alt.Chart(df_res).mark_line().encode(x=x, y=y, tooltip=list(df_res.columns))
            elif ctype == "area":
                chart = alt.Chart(df_res).mark_area().encode(x=x, y=y, tooltip=list(df_res.columns))
            elif ctype == "scatter":
                chart = alt.Chart(df_res).mark_point().encode(x=x, y=y, tooltip=list(df_res.columns))
            elif ctype == "pie" and len(df_res) <= 30:
                chart = alt.Chart(df_res).mark_arc().encode(
                    theta=alt.Theta(field=y, type="quantitative"),
                    color=alt.Color(field=x, type="nominal"),
                    tooltip=list(df_res.columns)
                )
            else:
                chart = alt.Chart(df_res).mark_bar().encode(x=x, y=y, tooltip=list(df_res.columns))
            st.altair_chart(chart.properties(height=360), use_container_width=True)
        else:
            st.info("The model's chart fields didn't match the result. Showing table only.")

        if result.get("summary"):
            st.success(result["summary"])

        st.download_button("Download results as CSV", df_res.to_csv(index=False),
                        file_name="answer_results.csv", mime="text/csv")



    # # pick columns for a chart
    # numeric_cols = [c for c, t in zip(df.columns, df.dtypes) if pd.api.types.is_numeric_dtype(t)]
    # cat_cols = [c for c in df.columns if c not in numeric_cols]

    # col1, col2 = st.columns(2)
    # with col1:
    #     x = st.selectbox("X axis", options=cat_cols + numeric_cols, index=0 if cat_cols else 0)
    # with col2:
    #     y = st.selectbox("Y axis (numeric)", options=numeric_cols, index=0 if numeric_cols else 0)

    # if x and y and len(df) > 0:
    #     # Aggregate if x is categorical; else just show scatter
    #     if x in cat_cols:
    #         q = f"SELECT {x} AS x, AVG({y}) AS y FROM {table_name} GROUP BY {x} ORDER BY y DESC LIMIT 1000"
    #         res = con.execute(q).fetchdf()
    #         chart = alt.Chart(res).mark_bar().encode(x="x:N", y="y:Q", tooltip=list(res.columns))
    #     else:
    #         q = f"SELECT {x} AS x, {y} AS y FROM {table_name} LIMIT 1000"
    #         res = con.execute(q).fetchdf()
    #         chart = alt.Chart(res).mark_point().encode(x="x:Q", y="y:Q", tooltip=list(res.columns))
    #     st.altair_chart(chart.properties(height=360), use_container_width=True)

    #     with st.expander("SQL used"):
    #         st.code(q, language="sql")

    #     st.download_button("Download results CSV", res.to_csv(index=False), "explore_results.csv", "text/csv")
else:
    st.info("Upload a CSV in the sidebar to begin.")
