import pandas as pd
import plotly.express as px
import streamlit as st


def _detect_column_roles(df: pd.DataFrame):
    """Infer candidate time, category, and numeric columns."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return datetime_cols, object_cols, numeric_cols


def _time_series_viz(df: pd.DataFrame, time_col: str, value_col: str):
    st.subheader(f"📈 {value_col} over time ({time_col})")
    df_sorted = df.sort_values(by=time_col)
    fig = px.line(df_sorted, x=time_col, y=value_col, markers=True)
    st.plotly_chart(fig, use_container_width=True)


def _bar_or_treemap_viz(df: pd.DataFrame, cat_cols, num_col: str):
    main_cat = cat_cols[0]
    st.subheader(f"📊 {num_col} by {main_cat}")
    agg = df.groupby(main_cat, as_index=False)[num_col].sum()
    fig_bar = px.bar(agg, x=main_cat, y=num_col)
    st.plotly_chart(fig_bar, use_container_width=True)

    if len(cat_cols) >= 2:
        st.subheader("🧩 Treemap")
        cat_hierarchy = cat_cols[:2]
        agg2 = df.groupby(cat_hierarchy, as_index=False)[num_col].sum()
        fig_tree = px.treemap(agg2, path=cat_hierarchy, values=num_col)
        st.plotly_chart(fig_tree, use_container_width=True)


def _scatter_and_pairs_viz(df: pd.DataFrame, numeric_cols):
    if len(numeric_cols) < 2:
        return

    x, y = numeric_cols[:2]
    st.subheader(f"📈 Scatter: {y} vs {x}")
    fig_scatter = px.scatter(df, x=x, y=y)
    st.plotly_chart(fig_scatter, use_container_width=True)

    if len(numeric_cols) >= 3:
        st.subheader("🔍 Pairwise relationships (scatter matrix)")
        fig_matrix = px.scatter_matrix(df, dimensions=numeric_cols[:4])
        st.plotly_chart(fig_matrix, use_container_width=True)


def auto_advanced_viz(df: pd.DataFrame):
    """Advanced auto-viz: table + time-series + bar/treemap + scatter."""
    if df is None or df.empty:
        st.info("No data to visualize.")
        return

    st.subheader("📋 Data")
    st.dataframe(df, use_container_width=True)

    datetime_cols, cat_cols, numeric_cols = _detect_column_roles(df)

    if datetime_cols and numeric_cols:
        _time_series_viz(df, datetime_cols[0], numeric_cols[0])

    if cat_cols and numeric_cols:
        _bar_or_treemap_viz(df, cat_cols, numeric_cols[0])

    if len(numeric_cols) >= 2:
        _scatter_and_pairs_viz(df, numeric_cols)
