import streamlit as st

st.title("ℹ️ About This Project")

st.markdown("""
<div style="
background: linear-gradient(90deg,#2563eb,#7c3aed);
padding:25px;
border-radius:15px;
text-align:center;
color:white;">
<h2>Student Dropout Prediction System</h2>
<p>AI-Powered Academic Risk Analysis & Early Intervention Platform</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
# Student Dropout Prediction System

This project uses Machine Learning techniques to predict the likelihood of student dropout based on academic, financial, behavioral, and personal factors.

The system helps educational institutions identify at-risk students early and implement timely interventions.
""")

st.markdown("---")

st.subheader("🎯 Project Objectives")

st.markdown("""
- Predict student dropout risk
- Analyze student academic performance
- Identify high-risk students
- Support institutional decision making
- Recommend intervention strategies
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", "2,000")

with col2:
    st.metric("Models Tested", "5")

with col3:
    st.metric("Best Accuracy", "81.25%")

st.subheader("📊 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Students", "2000")

with col2:
    st.metric("Features", "17")

st.markdown("""
Dataset features include:

- Age
- Gender
- Department
- Level
- CGPA
- Attendance
- Assignment Score
- Midterm Score
- Exam Score
- Study Hours
- Financial Stress
- Family Income
- Scholarship Status
- Sleep Hours
- Mental Health
- Carryovers
- Performance
""")

st.markdown("---")

st.subheader("🔄 System Workflow")

st.markdown("""
1. Student Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Machine Learning Prediction
5. Risk Classification
6. Intervention Recommendation
7. Analytics & Reporting
""")

st.subheader("🤖 Machine Learning Models Used")

st.markdown("""
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
""")

st.success("""
Best Model: Gradient Boosting

Accuracy: 81.25%

AUC Score: 0.856
""")

st.markdown("---")

st.subheader("🛠 Technologies Used")

st.markdown("""
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly
- Joblib
""")

st.markdown("---")

st.markdown("""
<div style='text-align:center;color:gray;'>

Student Dropout Prediction System

Final Year Project (2026)

Developed by NWOSU DAVID (22-10026)

Department of Computer Science

</div>
""", unsafe_allow_html=True)