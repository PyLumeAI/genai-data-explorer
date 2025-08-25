# 🧪 GenAI Data Explorer

[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-orange)](https://streamlit.io/)
[![DuckDB](https://img.shields.io/badge/Engine-DuckDB-yellow)](https://duckdb.org/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI-blue)](https://platform.openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive **Generative AI app** that lets you **chat with your data**.  
Upload a CSV or connect to a database → Ask questions in natural language → Get SQL, tables, charts, and insights automatically.

👉 **Live Demo (coming soon)** | 👉 **Case Study on [PyLumeAI Website](https://pylumeai.com/projects/genai-data-explorer.html)**

---

## 📖 Overview
Exploring data typically requires SQL knowledge or BI tools.  
This app allows **non-technical users** to query and visualize their data by simply asking questions in plain English.

---

## 🎯 Problem Statement
- Business users often have raw CSVs or databases but **can’t write SQL**.  
- Data teams are overloaded with **ad-hoc analysis requests**.  
- Existing BI tools are rigid, require setup, and lack conversational interfaces.  

---

## 🛠️ Solution Approach
- Upload **CSV** (or connect to a read-only SQL database).  
- The app inspects schema and builds a **compact profile** of columns.  
- An **LLM translates questions → safe SQL**.  
- Guardrails ensure **only SELECT queries** (no DROP/UPDATE/DELETE).  
- Query runs via DuckDB/Postgres → returns a table.  
- The app suggests a **chart type** and generates a **visualization**.  
- A **summary insight** is generated for context.  
- Results can be **downloaded as CSV**.

---

## ✨ Features
- 🔍 **Ask in plain English**: “Show top 10 customers by revenue”  
- 📊 **Smart visualizations**: bar, line, scatter, histogram, pie  
- 🛡 **Safe SQL only**: validated with guardrails (`sqlglot`)  
- ⚡ **Works instantly**: DuckDB runs queries directly on CSVs/Parquet  
- 🔄 **Optional Postgres connector** for enterprise demo  
- 💾 **Download results as CSV**  
- 📝 **SQL preview toggle** before execution  
- 🌓 Dark/light mode friendly  

---

## 📊 Example Screens

### Ask a Question
<img src="docs/images/ui_query.png" alt="Query UI" width="600"/>

### Table + Chart + Insight
<img src="docs/images/ui_chart.png" alt="Chart Output" width="600"/>

---

## 📂 Repository Structure
```
genai-data-explorer/
│
├── app.py # Main Streamlit app
├── requirements.txt # Dependencies
├── core/ # Core logic
│ ├── llm_sql.py # LLM prompt + JSON output
│ ├── sql_guard.py # SQL validation & LIMIT enforcement
│ ├── schema_introspect.py # Schema summary for LLM
│ ├── chart_suggest.py # Chart type heuristics
│ └── summarizer.py # Insight text generation
├── connectors/ # Data connectors
│ ├── duck_conn.py # DuckDB connector
│ └── pg_conn.py # Postgres connector (optional)
├── sample_data/ # Demo datasets
│ ├── sales.csv
│ └── customers.csv
├── docs/images/ # Screenshots for README
│ ├── ui_query.png
│ └── ui_chart.png
└── README.md
```

---

## 🧰 Tech Stack
- **Frontend:** Streamlit  
- **LLM:** OpenAI GPT / Azure OpenAI / Hugging Face (pluggable)  
- **Query Engine:** DuckDB (CSV/Parquet) + optional Postgres connector  
- **Visualization:** Altair (safe, declarative)  
- **Validation:** sqlglot (SQL parsing & dialect handling)  

---

## 🚀 Quickstart

```bash
git clone https://github.com/PyLumeAI/genai-data-explorer.git
cd genai-data-explorer
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```
## 🔑 Configure API Keys
```toml
OPENAI_API_KEY = "sk-..."
# Optional Postgres
PG_URI = "postgresql+psycopg2://user:pass@host:5432/dbname"
```
## ▶️ Run the App
```bash
streamlit run app.py
```
## 🔍 Use Cases
- Business analysts exploring sales/customer data without SQL
- Finance teams reviewing expense trends
- Startups querying their user activity CSV exports
- Consultants demoing quick insights to clients

📜 License
Distributed under the MIT License. 

✨ About PyLumeAI

PyLumeAI builds data engineering pipelines and AI-powered applications.
This is the second showcase project, after PaySim Fraud Analytics
.

👉 Visit: https://pylumeai.com

👉 Contact: contact.pylumeai@gmail.com