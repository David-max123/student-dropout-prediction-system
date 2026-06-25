import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

st.title("📈 Advanced Analytics Dashboard")

# Load Dataset
df = pd.read_csv("data/student_dataset.csv")

# =========================
# KPI CARDS
# =========================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Students",
    len(df)
)

col2.metric(
    "Dropout Rate",
    f"{df['Dropout'].mean()*100:.2f}%"
)

col3.metric(
    "Average CGPA",
    f"{df['CGPA'].mean():.2f}"
)

col4.metric(
    "Average Attendance",
    f"{df['Attendance'].mean():.1f}%"
)

st.markdown("---")

# =========================
# DROPOUT BY DEPARTMENT
# =========================

st.subheader("🏫 Dropout Rate by Department")

dept_dropout = (
    df.groupby("Department")["Dropout"]
    .mean()
    .reset_index()
)

fig1 = px.bar(
    dept_dropout,
    x="Department",
    y="Dropout",
    color="Dropout",
    title="Department Risk Ranking"
)

st.plotly_chart(fig1, use_container_width=True)

# =========================
# CGPA ANALYSIS
# =========================

st.subheader("🎓 CGPA vs Dropout")

fig2 = px.box(
    df,
    x="Dropout",
    y="CGPA",
    color="Dropout"
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# ATTENDANCE ANALYSIS
# =========================

st.subheader("📚 Attendance vs Dropout")

fig3 = px.box(
    df,
    x="Dropout",
    y="Attendance",
    color="Dropout"
)

st.plotly_chart(fig3, use_container_width=True)

# =========================
# FINANCIAL STRESS
# =========================

st.subheader("💰 Financial Stress Analysis")

fig4 = px.histogram(
    df,
    x="Financial_Stress",
    color="Dropout",
    barmode="group"
)

st.plotly_chart(fig4, use_container_width=True)

# =========================
# MENTAL HEALTH
# =========================

st.subheader("🧠 Mental Health Impact")

fig5 = px.box(
    df,
    x="Dropout",
    y="Mental_Health",
    color="Dropout"
)

st.plotly_chart(fig5, use_container_width=True)

# =========================
# CARRYOVERS
# =========================

st.subheader("📑 Carryovers vs Dropout")

fig6 = px.histogram(
    df,
    x="Carryovers",
    color="Dropout",
    barmode="group"
)

st.plotly_chart(fig6, use_container_width=True)

# =========================
# PERFORMANCE
# =========================

st.subheader("🏆 Performance Distribution")

fig7 = px.histogram(
    df,
    x="Performance",
    color="Dropout",
    barmode="group"
)

st.plotly_chart(fig7, use_container_width=True)

# =========================
# CORRELATION HEATMAP
# =========================

st.subheader("🔥 Correlation Heatmap")

corr = df.select_dtypes(
    include=["int64", "float64"]
).corr()

fig8 = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="Viridis"
)

fig8.update_layout(
    height=800,
    title="Feature Correlation Heatmap"
)

st.plotly_chart(fig8, use_container_width=True)

st.subheader("🎯 Features Most Related to Dropout")

dropout_corr = (
    corr["Dropout"]
    .sort_values(ascending=False)
    .drop("Dropout")
)

fig9 = px.bar(
    x=dropout_corr.values,
    y=dropout_corr.index,
    orientation="h",
    title="Correlation with Dropout"
)

st.plotly_chart(fig9, use_container_width=True)

st.markdown("---")

st.markdown("""
<div style='text-align:center;color:gray;'>

Student Dropout Prediction System

Final Year Project

Developed by NWOSU DAVID (22-10026)

Department of Computer Science

</div>
""", unsafe_allow_html=True)