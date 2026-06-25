import streamlit as st

st.set_page_config(
    page_title="Student Dropout Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

st.sidebar.title("🎓 Student Dropout System")

st.sidebar.info("""
AI-Powered Student Performance
and Dropout Risk Prediction
""")

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.title("🎓 Student Dropout Prediction System")

st.markdown("""
### AI-Powered Student Performance & Dropout Risk Analysis

This system uses Machine Learning to:

- Predict student dropout risk
- Analyze academic performance
- Identify high-risk students
- Provide intervention recommendations
- Visualize institutional trends
""")

# Load dataset
df = pd.read_csv("data/student_dataset.csv")

# Load model results
results = joblib.load("models/model_results.pkl")

best_model = max(
    results,
    key=lambda x: results[x]["Accuracy"]
)

best_accuracy = (
    results[best_model]["Accuracy"] * 100
)

st.markdown("---")

# ==========================
# KPI CARDS
# ==========================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "👨‍🎓 Total Students",
        len(df)
    )

with col2:
    st.metric(
        "⚠️ Dropout Rate",
        f"{df['Dropout'].mean()*100:.2f}%"
    )

with col3:
    st.metric(
        "🎯 Best Accuracy",
        f"{best_accuracy:.2f}%"
    )

with col4:
    st.metric(
        "🤖 Best Model",
        best_model
    )

with col5:
    st.metric(
        "📈 AUC Score",
        "0.856"
    )

st.success(
    f"✅ System Status: Active | Best Model: {best_model} | Accuracy: {best_accuracy:.2f}%"
)

st.markdown("---")

# ==========================
# QUICK INSIGHTS
# ==========================

col1, col2 = st.columns(2)

with col1:

    fig1 = px.pie(
        df,
        names="Dropout",
        title="Student Dropout Distribution"
    )
    fig1.update_layout(
    template="plotly_dark"
)

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with col2:

    dept = (
        df.groupby("Department")["Dropout"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        dept,
        x="Department",
        y="Dropout",
        color="Dropout",
        title="Department Risk Ranking"
    )

    fig2.update_layout(
        template="plotly_dark",
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

st.markdown("---")

# ==========================
# ACADEMIC PERFORMANCE
# ==========================

st.subheader("📚 Academic Performance Overview")

fig3 = px.histogram(
    df,
    x="CGPA",
    color="Dropout",
    title="CGPA Distribution"
)
fig3.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.subheader("📈 Attendance Overview")

fig4 = px.box(
    df,
    x="Dropout",
    y="Attendance",
    color="Dropout"
)

fig4.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

st.markdown("---")

# ==========================
# SYSTEM SUMMARY
# ==========================

st.subheader("🤖 Model Summary")

st.success(
    f"""
    Best Performing Model: {best_model}

    Accuracy: {best_accuracy:.2f}%

    The machine learning system analyzes student attributes,
    academic performance, attendance patterns, financial stress,
    and behavioral indicators to estimate dropout risk.
    """
)

st.markdown("---")

st.subheader("📋 Dataset Overview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

st.subheader("📊 Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

st.markdown("---")

st.markdown("""
<div style='text-align:center;color:gray;'>

Student Dropout Prediction System

Final Year Project

Developed by NWOSU DAVID (22-10026)

Department of Computer Science

</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.caption(
    "Student Dropout Prediction System | Final Year Project | Machine Learning Based Early Warning System"
)