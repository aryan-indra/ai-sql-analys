import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
import re
from viz_utils import run_sql_to_df
from advanced_viz import auto_advanced_viz

# =========================================================
#                   PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SQL Query Assistant (Advanced Viz)",
    page_icon="📊",
    layout="wide"
)

# =========================================================
#              INITIALIZE DATABASE & SCHEMA
# =========================================================
if "db" not in st.session_state:
    with st.spinner("Connecting to database..."):
        db_uri = "postgresql+psycopg2://postgres:indra@localhost:5432/test_db"
        st.session_state.db = SQLDatabase.from_uri(db_uri)
        st.session_state.schema_text = st.session_state.db.get_table_info()
        st.session_state.table_names = st.session_state.db.get_usable_table_names()
        st.session_state.table_list = ", ".join(st.session_state.table_names)

        # Schema map
        schema_map = {
            "users": {"primary_key": "user_id"},
            "addresses": {
                "primary_key": "address_id",
                "foreign_keys": {"user_id": "users.user_id"}
            },
            "categories": {"primary_key": "category_id"},
            "products": {
                "primary_key": "product_id",
                "foreign_keys": {"category_id": "categories.category_id"}
            },
            "product_tags": {
                "primary_key": "product_id, tag",
                "foreign_keys": {"product_id": "products.product_id"}
            },
            "reviews": {
                "primary_key": "review_id",
                "foreign_keys": {
                    "product_id": "products.product_id",
                    "user_id": "users.user_id"
                }
            },
            "orders": {
                "primary_key": "order_id",
                "foreign_keys": {
                    "user_id": "users.user_id",
                    "address_id": "addresses.address_id"
                }
            },
            "order_items": {
                "primary_key": "order_id, product_id",
                "foreign_keys": {
                    "order_id": "orders.order_id",
                    "product_id": "products.product_id"
                }
            },
            "payments": {
                "primary_key": "payment_id",
                "foreign_keys": {
                    "order_id": "orders.order_id",
                    "paid_by": "users.user_id"
                }
            },
            "vw_order_summary": {"primary_key": "order_id"}
        }

        schema_map_text = "\n".join([
            f"{table}: PK({info['primary_key']})" +
            (f", FK({', '.join([f'{fk}->{ref}' for fk, ref in info.get('foreign_keys', {}).items()])})"
             if info.get("foreign_keys") else "")
            for table, info in schema_map.items()
        ])

        # SYSTEM PROMPT
        st.session_state.system_prompt = f"""
You are an expert SQL generator for PostgreSQL databases.

AVAILABLE TABLES:
{st.session_state.table_list}

CRITICAL RULES:
- Only use these exact table names.
- Never guess or invent table names.
- Always match schema exactly.

TABLE RELATIONSHIPS:
{schema_map_text}

FULL SCHEMA (tables and columns):
{st.session_state.schema_text}

BUSINESS SEMANTICS (VERY IMPORTANT):

1. "Revenue", "total revenue", "sales amount", "total sales":
   - By default, interpret this as the SUM of payments.amount
   - Use ONLY payments where payments.status = 'success'
   - So: revenue = SUM(payments.amount) with payments.status = 'success'

2. "City", "revenue by city", "sales by city":
   - City is the shipping/billing city stored in addresses.city
   - orders.address_id references addresses.address_id
   - To group revenue by city, you MUST:
     - JOIN orders o ON p.order_id = o.order_id
     - JOIN addresses a ON o.address_id = a.address_id
     - GROUP BY a.city

   Example (for "total revenue by city"):
   SELECT
     a.city,
     SUM(p.amount) AS total_revenue
   FROM payments p
   JOIN orders o    ON p.order_id = o.order_id
   JOIN addresses a ON o.address_id = a.address_id
   WHERE p.status = 'success'
   GROUP BY a.city;

3. "Total spent per user", "amount each user spent":
   - Use payments.amount with status = 'success'
   - JOIN payments p -> orders o -> users u
   - GROUP BY u.user_id (or u.email/full_name as needed)

4. "Order total" or "order value":
   - orders.total_amount is the precomputed total for each order.
   - order_items has unit_price, quantity, discount if you need line-level details.

5. When the user asks for "total revenue each user has spent", "total spend per user",
or similar:

   - Use payments.amount with payments.status = 'success'.
   - Always SELECT identifying user fields (at least users.user_id and users.full_name).
   - GROUP BY those user fields.
   - Prefer LEFT JOIN from users if the user might want to include users with zero spend.

6. When a question asks for stats "per X" (per product, per user, per category, per city, etc.):

- ALWAYS include the identifier column of X (e.g., product_id, name).
- ALWAYS GROUP BY that identifier.
- NEVER return a single aggregated value unless explicitly asked.
- If reviews are involved, LEFT JOIN reviews to include unrated products.

GENERAL BEHAVIOR:
- Prefer using payments.amount with payments.status = 'success' when user asks about revenue/sales/spend.
- Use addresses.city for anything grouped by city or location.
- If a question is ambiguous, assume these defaults.
- IF a user asks for results "by city" or "grouped by city" AND they include phrases like:
  "even if zero" or "show 0" or "all cities" or "include cities without revenue/orders",
  THEN you MUST use a LEFT JOIN starting from addresses to ensure all cities appear:
  addresses a
  LEFT JOIN orders o ON a.address_id = o.address_id
  LEFT JOIN payments p ON p.order_id = o.order_id AND p.status = 'success'

Your job:
- Generate ONLY valid PostgreSQL SQL queries.
- NEVER return explanations, notes, or text descriptions.
- Return ONLY the SQL query, no explanations.

OUTPUT FORMAT:
- Return a SINGLE SQL statement only.
- Do NOT wrap it in ``` code fences.
- Do NOT add any explanation or comments.
- Do NOT prepend text like "Here is the query:".
"""

# =========================================================
#                 INITIALIZE LLM
# =========================================================
if "llm" not in st.session_state:
    with st.spinner("Loading LLM..."):
        st.session_state.llm_model = st.session_state.get('selected_model', "llama3.2")
        st.session_state.llm = ChatOllama(
            model=st.session_state.llm_model,
            temperature=0,
        )


# =========================================================
#               TABLE NAME VALIDATOR
# =========================================================
def validate_and_correct_table_names(sql, valid_tables):
    corrections = {
        "product_review": "reviews",
        "product_reviews": "reviews",
        "review": "reviews",
        "product": "products",
        "user": "users",
        "category": "categories",
        "address": "addresses",
        "order": "orders",
        "order_item": "order_items",
        "orderitem": "order_items",
        "payment": "payments",
        "product_tag": "product_tags",
        "producttag": "product_tags"
    }

    table_pattern = r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    matches = re.findall(table_pattern, sql, re.IGNORECASE)

    corrected_sql = sql
    corrections_made = []

    for match in matches:
        tbl = match.lower()

        if tbl not in [t.lower() for t in valid_tables]:
            if tbl in corrections:
                corrected = corrections[tbl]
                corrected_sql = re.sub(r"\b" + re.escape(match) + r"\b", corrected, corrected_sql, flags=re.IGNORECASE)
                corrections_made.append(f"{match} → {corrected}")
            else:
                for valid in valid_tables:
                    if tbl in valid.lower() or valid.lower() in tbl:
                        corrected_sql = re.sub(
                            r"\b" + re.escape(match) + r"\b",
                            valid,
                            corrected_sql,
                            flags=re.IGNORECASE
                        )
                        corrections_made.append(f"{match} → {valid}")
                        break

    return corrected_sql, corrections_made


# =========================================================
#                SQL GENERATION LOGIC
# =========================================================
def generate_sql(llm, system_prompt, user_question, valid_tables, max_retries=2):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            "User request:\n"
            f"{user_question.strip()}\n\n"
            "Return ONLY a single valid PostgreSQL SELECT query, nothing else."
        ))
    ]

    for attempt in range(max_retries):
        response = llm.invoke(messages)
        generated_sql = response.content

        cleaned_sql = generated_sql.strip()
        if cleaned_sql.startswith("```sql"):
            cleaned_sql = cleaned_sql[6:]
        if cleaned_sql.startswith("```"):
            cleaned_sql = cleaned_sql[3:]
        if cleaned_sql.endswith("```"):
            cleaned_sql = cleaned_sql[:-3]
        cleaned_sql = cleaned_sql.strip()

        corrected_sql, corrections = validate_and_correct_table_names(
            cleaned_sql, valid_tables
        )

        if corrections and attempt < max_retries - 1:
            error_feedback = (
                f"The previous SQL used invalid table names. "
                f"Valid tables are: {', '.join(valid_tables)}. "
                f"Regenerate the SQL using ONLY these exact table names."
            )
            messages.append(SystemMessage(content=error_feedback))
            continue

        return corrected_sql, corrections if corrections else []

    return corrected_sql, corrections if corrections else []


# =========================================================
#              HISTORY STATE
# =========================================================
if "query_history" not in st.session_state:
    st.session_state.query_history = []


# =========================================================
#                    MAIN UI LAYOUT
# =========================================================
st.markdown("## 🧠 SQL Query Assistant")
st.caption("Ask questions in natural language. I’ll generate and run SQL on your PostgreSQL DB, then visualize the results.")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📊 Database")
    st.write(f"**Tables:** {len(st.session_state.table_names)}")
    with st.expander("View tables"):
        for t in st.session_state.table_names:
            st.write(f"• `{t}`")

    st.divider()
    st.header("⚙️ Settings")
    show_sql = st.checkbox("Show generated SQL", True)

    model_options = {
        "sqlcoder:latest": "SQLCoder (Best for SQL)",
        "llama3.2": "Llama 3.2",
        "mistral": "Mistral",
        "qwen2.5-coder": "Qwen2.5 Coder"
    }

    selected_model = st.selectbox(
        "LLM model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )

    if selected_model != st.session_state.get("selected_model", "llama3.2"):
        st.session_state.selected_model = selected_model
        st.session_state.llm = ChatOllama(model=selected_model, temperature=0)
        st.rerun()

    st.divider()
    with st.expander("Advanced / Dev"):
        if st.button("♻️ Reset session"):
            st.session_state.clear()
            st.rerun()
        if st.button("🗑️ Clear history"):
            st.session_state.query_history = []
            st.rerun()

# ---------------- MAIN CHAT INPUT ----------------
user_query = st.chat_input("Ask a question about your database...")

latest_entry = None

if user_query:
    st.session_state.query_history.append({
        "question": user_query,
        "sql": None,
        "result": None,
        "error": None
    })
    latest_entry = st.session_state.query_history[-1]

    # User bubble
    with st.chat_message("user"):
        st.write(user_query)

    # -------- Generate SQL --------
    with st.chat_message("assistant"):
        with st.spinner("Generating SQL..."):
            sql, corrections = generate_sql(
                st.session_state.llm,
                st.session_state.system_prompt,
                user_query,
                st.session_state.table_names
            )
            latest_entry["sql"] = sql

            if corrections:
                st.warning("⚠️ Auto-corrected table names: " + ", ".join(corrections))

            if not sql or not sql.strip():
                err_msg = "Model did not generate a valid SQL query. Please try rephrasing your question."
                latest_entry["error"] = err_msg
                st.error(err_msg)
            else:
                if show_sql:
                    st.markdown("**Generated SQL:**")
                    st.code(sql, language="sql")

                # -------- Execute SQL --------
                with st.spinner("Executing query & building visuals..."):
                    try:
                        df = run_sql_to_df(st.session_state.db, sql)
                        latest_entry["result"] = df

                        if df is None or df.empty:
                            st.info("(No data found)")
                        else:
                            auto_advanced_viz(df)

                    except Exception as e:
                        err = str(e)
                        latest_entry["error"] = err
                        st.error(err)

# If no new query this run, still render previous conversation
if not user_query and st.session_state.query_history:
    st.markdown("---")

# ---------------- HISTORY DISPLAY ----------------
if st.session_state.query_history:
    st.markdown("### 📜 Conversation history")

    # Show older turns (except the very latest if we already showed it in chat bubbles)
    # We'll render everything as chat-style, but without re-running viz logic
    for i, entry in enumerate(st.session_state.query_history):
        # For the current run, if user_query is not None,
        # we've already shown the last entry as chat_message above.
        if user_query and i == len(st.session_state.query_history) - 1:
            continue

        with st.chat_message("user"):
            st.write(entry["question"])

        with st.chat_message("assistant"):
            if entry.get("error"):
                st.error(entry["error"])
            if entry.get("sql") and show_sql:
                with st.expander("View SQL"):
                    st.code(entry["sql"], language="sql")
            if entry.get("result") is not None:
                if isinstance(entry["result"], pd.DataFrame):
                    with st.expander("View data"):
                        st.dataframe(entry["result"], use_container_width=True)
                else:
                    st.text(str(entry["result"]))
