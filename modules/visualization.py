import streamlit as st
import plotly.express as px

def display_summary(report):
    st.header("📌 Summary Report")
    for key, value in report.items():
        if isinstance(value, dict):
            with st.expander(f"{key}"):
                st.json(value)
        else:
            st.write(f"**{key}**: {value}")

def display_detailed_issues(report):
    st.header("🔎 Detailed Issues")
    if "missing_values" in report:
        missing_df = {k: v for k, v in report["missing_values"].items() if v > 0}
        if missing_df:
            st.subheader("Missing Values")
            fig = px.bar(x=list(missing_df.keys()), y=list(missing_df.values()), labels={'x': 'Column', 'y': 'Missing Count'})
            st.plotly_chart(fig)

    if "token_frequency" in report:
        st.subheader("Most Frequent Tokens")
        fig = px.bar(x=list(report["token_frequency"].keys()), y=list(report["token_frequency"].values()), labels={'x': 'Token', 'y': 'Count'})
        st.plotly_chart(fig)

    if "language_distribution" in report and isinstance(report["language_distribution"], dict):
        st.subheader("Detected Languages")
        fig = px.pie(names=list(report["language_distribution"].keys()), values=list(report["language_distribution"].values()))
        st.plotly_chart(fig)
