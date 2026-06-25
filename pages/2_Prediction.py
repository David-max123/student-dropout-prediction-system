import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time


st.title("🎯 Student Dropout Prediction")

# Load model and encoders
model = joblib.load("models/best_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")

st.sidebar.header("Enter Student Information")

# =========================
# INPUTS
# =========================

age = st.sidebar.number_input("Age", 16, 40, 20)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

department = st.sidebar.selectbox(
    "Department",
    [
        "Computer Science",
        "Software Engineering",
        "Cyber Security",
        "Information Technology",
        "Electrical Engineering",
        "Mechanical Engineering",
        "Civil Engineering",
        "Accounting",
        "Economics",
        "Business Administration",
        "Mass Communication",
        "Biochemistry",
        "Microbiology",
        "Nursing",
        "Medicine"
    ]
)

level = st.sidebar.selectbox(
    "Level",
    [100, 200, 300, 400, 500]
)

cgpa = st.sidebar.number_input(
    "CGPA",
    min_value=0.0,
    max_value=5.0,
    value=3.0,
    step=0.01
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    0,
    100,
    75
)

assignment = st.sidebar.slider(
    "Assignment Score",
    0,
    100,
    70
)

midterm = st.sidebar.slider(
    "Midterm Score",
    0,
    100,
    70
)

exam = st.sidebar.slider(
    "Exam Score",
    0,
    100,
    70
)

study_hours = st.sidebar.slider(
    "Study Hours",
    0,
    12,
    4
)

financial_stress = st.sidebar.slider(
    "Financial Stress",
    1,
    10,
    5
)

family_income = st.sidebar.selectbox(
    "Family Income",
    ["Low", "Medium", "High"]
)

scholarship = st.sidebar.selectbox(
    "Scholarship",
    ["Yes", "No"]
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours",
    0,
    12,
    7
)

mental_health = st.sidebar.slider(
    "Mental Health",
    1,
    10,
    6
)

carryovers = st.sidebar.slider(
    "Carryovers",
    0,
    10,
    0
)

performance = st.sidebar.selectbox(
    "Performance",
    ["Poor", "Fair", "Good", "Excellent"]
)

# =========================
# PREDICTION
# =========================

if st.button("Predict Risk"):
    

    # Encode categorical values
    gender_encoded = encoders["Gender"].transform([gender])[0]
    department_encoded = encoders["Department"].transform([department])[0]
    family_income_encoded = encoders["Family_Income"].transform([family_income])[0]
    scholarship_encoded = encoders["Scholarship"].transform([scholarship])[0]
    performance_encoded = encoders["Performance"].transform([performance])[0]

    input_data = pd.DataFrame([[
        age,
        gender_encoded,
        department_encoded,
        level,
        cgpa,
        attendance,
        assignment,
        midterm,
        exam,
        study_hours,
        financial_stress,
        family_income_encoded,
        scholarship_encoded,
        sleep_hours,
        mental_health,
        carryovers,
        performance_encoded
    ]], columns=[
        "Age",
        "Gender",
        "Department",
        "Level",
        "CGPA",
        "Attendance",
        "Assignment",
        "Midterm",
        "Exam",
        "Study_Hours",
        "Financial_Stress",
        "Family_Income",
        "Scholarship",
        "Sleep_Hours",
        "Mental_Health",
        "Carryovers",
        "Performance"
    ])

    with st.spinner("🤖 AI is analyzing student profile..."):
        time.sleep(1)
        prob = model.predict_proba(input_data)[0][1]

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Dropout Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 60], "color": "gold"},
                {"range": [60, 100], "color": "tomato"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Risk category
    # Risk category
        # Risk category
    if prob < 0.30:

        st.balloons()

        st.markdown(
            f"""
            <div style="
            background:#14532d;
            color:white;
            padding:25px;
            border-radius:15px;
            text-align:center;
            font-size:28px;
            font-weight:bold;
            animation: popIn 0.5s ease;">
            🟢 LOW DROPOUT RISK<br>
            Probability: {prob:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )

    elif prob < 0.60:

        st.markdown(
            f"""
            <div style="
            background:#92400e;
            color:white;
            padding:25px;
            border-radius:15px;
            text-align:center;
            font-size:28px;
            font-weight:bold;
            animation: popIn 0.5s ease;">
            🟡 MEDIUM DROPOUT RISK<br>
            Probability: {prob:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div style="
            background:#991b1b;
            color:white;
            padding:25px;
            border-radius:15px;
            text-align:center;
            font-size:28px;
            font-weight:bold;
            animation: popIn 0.5s ease;">
            🔴 HIGH DROPOUT RISK<br>
            Probability: {prob:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("📋 Prediction Summary")

    if prob >= 0.60:

        st.error(
            "The student is at high risk of dropping out. Immediate intervention is recommended."
        )

    elif prob >= 0.30:

        st.warning(
            "The student shows moderate dropout risk. Academic monitoring is advised."
        )

    else:

        st.success(
            "The student appears academically stable. Continue normal monitoring."
        )

    st.markdown("---")

    st.subheader("Possible Risk Factors")

    reasons = []

    if cgpa < 2.5:
        reasons.append("Low CGPA")

    if attendance < 60:
        reasons.append("Poor Attendance")

    if financial_stress > 7:
        reasons.append("High Financial Stress")

    if carryovers >= 3:
        reasons.append("Too Many Carryovers")

    if mental_health < 4:
        reasons.append("Poor Mental Health")

    if study_hours < 2:
        reasons.append("Insufficient Study Hours")

    if len(reasons) == 0:
        st.success("No major risk factors detected.")
    else:
        for reason in reasons:
            st.write("•", reason)

    st.markdown("---")

    st.subheader("Recommendations")

    recommendations = []

    if attendance < 75:
        recommendations.append("Improve attendance")

    if cgpa < 3.0:
        recommendations.append("Attend academic support sessions")

    if financial_stress > 7:
        recommendations.append("Apply for financial aid")

    if mental_health < 5:
        recommendations.append("Visit student counselling services")

    if study_hours < 3:
        recommendations.append("Increase study hours")

    if carryovers > 2:
        recommendations.append("Meet an academic adviser")

    if len(recommendations) == 0:
        st.success("Student is performing well.")
    else:
        for rec in recommendations:
            st.write("✅", rec)

            st.markdown("---")

    result_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Department": [department],
        "Level": [level],
        "CGPA": [cgpa],
        "Attendance": [attendance],
        "Risk Probability": [round(prob * 100, 2)]
    })

    st.download_button(
        "📥 Download Result",
        result_df.to_csv(index=False),
        "prediction_result.csv",
        "text/csv"
    )

st.markdown("""
<div style='text-align:center;color:gray;'>

Student Dropout Prediction System

Final Year Project

Developed by NWOSU DAVID (22-10026)

Department of Computer Science

</div>
""", unsafe_allow_html=True)

