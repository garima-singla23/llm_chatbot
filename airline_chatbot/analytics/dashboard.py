# pages/1_Analytics.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from analytics.tracker import get_stats

st.set_page_config(page_title="AirAssist Analytics", page_icon="📊", layout="wide")
st.title("📊 AirAssist — Session Analytics")

stats = get_stats()

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Queries", stats["total_queries"])
col2.metric("Agentic", stats["agentic_queries"],
            f"{stats['agentic_pct']}% of total")
col3.metric("FAQ", stats["faq_queries"])
col4.metric("Avg Latency", f"{stats['avg_response_ms']}ms")
col5.metric("Tool Calls", sum(t["cnt"] for t in stats["tool_usage"]))

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Intent Distribution")
    if stats["intent_distribution"]:
        df = pd.DataFrame(stats["intent_distribution"])
        fig = px.pie(df, values="cnt", names="intent",
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No queries yet. Start chatting to see data.")

with col_right:
    st.subheader("Tool Usage")
    if stats["tool_usage"]:
        df = pd.DataFrame(stats["tool_usage"])
        fig = px.bar(df, x="tool_name", y="cnt",
                     color="cnt",
                     color_continuous_scale="Blues",
                     labels={"cnt": "Calls", "tool_name": "Tool"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white",
                          xaxis_tickangle=-30,
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No tool calls yet.")

st.subheader("Recent Queries")
if stats["recent_queries"]:
    df = pd.DataFrame(stats["recent_queries"])
    df["response_time_ms"] = df["response_time_ms"].apply(
        lambda x: f"{x}ms" if x else "-")
    st.dataframe(df[["timestamp","user_message","intent","response_time_ms"]],
                 use_container_width=True, hide_index=True)
else:
    st.info("No recent queries.")

if st.button("🔄 Refresh"):
    st.rerun()