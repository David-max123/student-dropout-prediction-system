import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

departments = [
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

genders = ["Male", "Female"]
income_levels = ["Low", "Medium", "High"]

def performance_from_cgpa(cgpa):
    if cgpa >= 4.5:
        return "Excellent"
    elif cgpa >= 3.5:
        return "Good"
    elif cgpa >= 2.5:
        return "Fair"
    else:
        return "Poor"
    
def generate_student(student_id):

    # -------------------------
    # BASE STUDENT INFO
    # -------------------------
    age = random.randint(16, 30)
    gender = random.choice(genders)
    department = random.choice(departments)
    level = random.choice([100, 200, 300, 400, 500])

    income = random.choice(income_levels)
    scholarship = random.choice(["Yes", "No"])

    # -------------------------
    # FINANCIAL STRESS EFFECT
    # -------------------------
    if income == "Low":
        financial_stress = random.randint(6, 10)
    elif income == "Medium":
        financial_stress = random.randint(3, 7)
    else:
        financial_stress = random.randint(1, 5)

    # -------------------------
    # STUDY HABITS (AFFECTED BY STRESS)
    # -------------------------
    study_hours = round(random.uniform(1, 8) - (financial_stress * 0.2), 1)
    study_hours = max(0.5, study_hours)

    # -------------------------
    # ATTENDANCE (RELATED TO DISCIPLINE)
    # -------------------------
    attendance = random.randint(50, 100) - (financial_stress * 2)
    attendance = max(40, min(100, attendance))

    # -------------------------
    # ACADEMIC SCORES (DEPEND ON STUDY + ATTENDANCE)
    # -------------------------
    base_score = study_hours * 10 + attendance * 0.3

    assignment = max(30, min(100, base_score + random.randint(-10, 10)))
    midterm = max(30, min(100, base_score + random.randint(-15, 15)))
    exam = max(30, min(100, base_score + random.randint(-20, 20)))

    # -------------------------
    # CGPA CALCULATION
    # -------------------------
    cgpa = round(((assignment + midterm + exam) / 3) / 20, 2)
    cgpa = max(1.0, min(5.0, cgpa))

    # -------------------------
    # MENTAL HEALTH EFFECT
    # -------------------------
    mental_health = random.randint(1, 10) - int(financial_stress / 2)
    mental_health = max(1, min(10, mental_health))

    sleep_hours = round(random.uniform(4, 9) - (financial_stress * 0.1), 1)
    sleep_hours = max(3, sleep_hours)

    # -------------------------
    # CARRYOVERS (DEPEND ON CGPA)
    # -------------------------
    if cgpa < 2.5:
        carryovers = random.randint(3, 5)
    elif cgpa < 3.5:
        carryovers = random.randint(1, 3)
    else:
        carryovers = random.randint(0, 1)

    # -------------------------
    # PERFORMANCE LABEL
    # -------------------------
    performance = performance_from_cgpa(cgpa)

    # -------------------------
    # DROP OUT LOGIC (REALISTIC RELATIONSHIP)
    # -------------------------
    dropout_risk = 0

    if cgpa < 2.5:
        dropout_risk += 35
    elif cgpa < 3.0:
        dropout_risk += 20

    if attendance < 60:
        dropout_risk += 25

    if financial_stress > 7:
        dropout_risk += 20

    if carryovers >= 3:
        dropout_risk += 20

    if mental_health < 4:
        dropout_risk += 15

    if study_hours < 2:
        dropout_risk += 10

    dropout_risk += random.randint(-5, 5)

    dropout_risk = max(0, min(100, dropout_risk))

    dropout = 1 if random.randint(1, 100) <= dropout_risk else 0

    # -------------------------
    # RETURN STUDENT RECORD
    # -------------------------
    return {
        "Student_ID": student_id,
        "Age": age,
        "Gender": gender,
        "Department": department,
        "Level": level,
        "CGPA": cgpa,
        "Attendance": attendance,
        "Assignment": assignment,
        "Midterm": midterm,
        "Exam": exam,
        "Study_Hours": study_hours,
        "Financial_Stress": financial_stress,
        "Family_Income": income,
        "Scholarship": scholarship,
        "Sleep_Hours": sleep_hours,
        "Mental_Health": mental_health,
        "Carryovers": carryovers,
        "Performance": performance,
        "Dropout": dropout
    }

students = []

for i in range(1, 2001):
    students.append(generate_student(i))

df = pd.DataFrame(students)

df.to_csv("data/student_dataset.csv", index=False)

print("Dataset regenerated successfully!")
print(df.head())
print("Shape:", df.shape)

