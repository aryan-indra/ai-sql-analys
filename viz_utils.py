import pandas as pd
from sqlalchemy import text
import streamlit as st

def run_sql_to_df(db, sql: str) -> pd.DataFrame:
    """Execute query via SQLDatabase engine and return a DataFrame."""
    engine = db._engine
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn)


def auto_visualize(df: pd.DataFrame):
    """Auto generate a basic visualization for results."""
    if df.empty:
        st.info("No data to visualize.")
        return

    st.subheader("📋 Data Table")
    st.dataframe(df, use_container_width=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 1) Time-series
    if datetime_cols and numeric_cols:
        time = datetime_cols[0]
        value = numeric_cols[0]
        st.subheader(f"📈 Time Series: {value} over {time}")
        sorted_df = df.sort_values(by=time)
        st.line_chart(sorted_df.set_index(time)[value])
        return

    # 2) Category vs numeric → bar chart
    if object_cols and numeric_cols:
        cat = object_cols[0]
        num = numeric_cols[0]
        st.subheader(f"📊 {num} grouped by {cat}")
        st.bar_chart(df[[cat, num]].set_index(cat))
        return

    # 3) Two numeric columns → scatter
    if len(numeric_cols) >= 2:
        x, y = numeric_cols[:2]
        st.subheader(f"📈 Scatter Plot: {y} vs {x}")
        st.scatter_chart(df[[x, y]])
        return

    st.info("No suitable chart type detected; showing only table.")
