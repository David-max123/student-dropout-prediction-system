import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

st.title("📊 Exploratory Data Analysis (EDA) Dashboard")

st.markdown("""
This dashboard presents the Exploratory Data Analysis (EDA) carried out on the
student dataset before model development. The visualizations help in understanding
the distribution of student records, identifying patterns, examining relationships
between variables, and determining the factors associated with student dropout.
""")

# Load Dataset
df = pd.read_csv("data/student_dataset.csv")

st.subheader("📋 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Number of Records:** {len(df)}")
    st.write(f"**Number of Features:** {df.shape[1]-1}")
    st.write(f"**Target Variable:** Dropout")

with col2:
    st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
    st.write(f"**Duplicate Records:** {df.duplicated().sum()}")

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

st.write("""
This visualization compares the dropout rate across different academic
departments to identify departments with higher dropout risk.
""")

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

st.write("""
This chart compares students' CGPA with their dropout status to determine
whether academic performance influences dropout.
""")

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

st.write("""
This visualization examines whether attendance levels are associated
with student dropout.
""")

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

st.write("""
This visualization shows how financial stress is distributed among
students and its relationship with dropout.
""")

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

st.write("""
This chart analyses the influence of students' mental health scores
on dropout status.
""")

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

st.write("""
This visualization compares the number of carryover courses
among students based on dropout status.
""")

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

st.write("""
This chart illustrates the distribution of students' academic
performance and its relationship with dropout.
""")

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

st.write("""
The correlation heatmap illustrates the strength of relationships
between numerical variables within the dataset.
""")

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

st.write("""
This chart ranks variables according to how strongly they are
correlated with student dropout.
""")

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

st.subheader("📝 EDA Summary")

st.success("""
The exploratory data analysis showed that student dropout is influenced by
multiple academic and non-academic factors, including CGPA, attendance,
financial stress, mental health, carryovers, and academic performance.

These findings guided the selection of relevant features for machine
learning model development and evaluation.
""")

st.markdown("""
<div style='text-align:center;color:gray;'>

Student Dropout Prediction System

Final Year Project

Developed by NWOSU DAVID (22-10026)

Department of Computer Science

</div>
""", unsafe_allow_html=True)