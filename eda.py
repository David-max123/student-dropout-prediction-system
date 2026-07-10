# ==========================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# Student Dropout Prediction System
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Create folder to save EDA figures
# ----------------------------------------------------------

os.makedirs("eda_results", exist_ok=True)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("data/student_dataset.csv")

# ===========================
# DATASET OVERVIEW
# ===========================

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Dataset Shape : {df.shape}")
print(f"Rows          : {df.shape[0]}")
print(f"Columns       : {df.shape[1]}")
print(f"Target Column : Dropout")

print("\nDuplicate Rows:", df.duplicated().sum())

print("\n")

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ===========================
# DATA DICTIONARY
# ===========================

print("=" * 60)
print("DATA DICTIONARY")
print("=" * 60)

data_dictionary = pd.DataFrame({
    "Variable": [
        "Student_ID",
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
        "Performance",
        "Dropout"
    ],
    "Description": [
        "Unique student identification number",
        "Student's age",
        "Student's gender",
        "Academic department",
        "Current level of study",
        "Cumulative Grade Point Average",
        "Attendance percentage",
        "Assignment score",
        "Midterm examination score",
        "Final examination score",
        "Average daily study hours",
        "Financial stress level",
        "Family income category",
        "Scholarship status",
        "Average sleep hours per day",
        "Mental health rating",
        "Number of carryover courses",
        "Overall academic performance",
        "Target variable indicating dropout status"
    ]
})

print(data_dictionary)

print("\n")

# ----------------------------------------------------------
# Dataset Preview
# ----------------------------------------------------------

print("\nFIRST FIVE RECORDS")
print(df.head())

print("\nLAST FIVE RECORDS")
print(df.tail())

# ----------------------------------------------------------
# Dataset Shape
# ----------------------------------------------------------

print("\nDATASET SHAPE")
print(df.shape)

# ----------------------------------------------------------
# Dataset Information
# ----------------------------------------------------------

print("\nDATASET INFORMATION")
df.info()

# ===========================
# DATA TYPE INTERPRETATION
# ===========================

print("=" * 60)
print("DATA TYPE INTERPRETATION")
print("=" * 60)

print("""
Interpretation:

• The dataset contains both numerical and categorical variables.

• Numerical variables include Age, Level, CGPA, Attendance,
  Assignment, Midterm, Exam, Study_Hours, Financial_Stress,
  Sleep_Hours, Mental_Health and Carryovers.

• Categorical variables include Gender, Department,
  Family_Income, Scholarship and Performance.

• Student_ID is used only as an identifier and is not
  included as a predictive feature.

• The target variable is Dropout, which indicates whether
  a student dropped out (1) or did not drop out (0).

• Before model training, all categorical variables were
  transformed into numerical values using Label Encoding.
""")

print("\n")

# ----------------------------------------------------------
# Summary Statistics
# ----------------------------------------------------------

print("\nSUMMARY STATISTICS")
print(df.describe())

# ===========================
# SUMMARY STATISTICS INTERPRETATION
# ===========================

print("=" * 60)
print("SUMMARY STATISTICS INTERPRETATION")
print("=" * 60)

print("""
Interpretation:

• The summary statistics provide information about the
  distribution of the numerical variables in the dataset.

• It shows the minimum, maximum, mean and standard deviation
  of each numerical feature.

• The variables have different value ranges. For example,
  CGPA ranges from 0 to 5, while Attendance, Assignment,
  Midterm and Exam range from 0 to 100.

• Because the numerical variables have different scales,
  StandardScaler was applied during preprocessing to
  standardize the data before training the machine
  learning models.

• Standardizing the features helps improve model
  performance by ensuring that variables with larger
  numerical values do not dominate those with smaller values.
""")

print("\n")

# ----------------------------------------------------------
# Missing Values
# ----------------------------------------------------------

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\n")

# ===========================
# MISSING VALUE INTERPRETATION
# ===========================

print("=" * 60)
print("MISSING VALUE INTERPRETATION")
print("=" * 60)

if df.isnull().sum().sum() == 0:
    print("""
Interpretation:

• No missing values were found in the dataset.

• This indicates that the dataset is complete and
  no data imputation or record deletion was required.

• The dataset was therefore suitable for
  machine learning model training.
""")
else:
    print("""
Interpretation:

• Missing values were detected in the dataset.

• These values should be handled through
  data cleaning techniques before model training.
""")

print("\n")

# ----------------------------------------------------------
# Duplicate Records
# ----------------------------------------------------------

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# ==========================================================
# SECTION 1: CATEGORICAL VARIABLE DISTRIBUTIONS
# ==========================================================

# -----------------------------
# Dropout Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Dropout", data=df)
plt.title("Dropout Distribution")
plt.xlabel("Dropout Status")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("eda_results/dropout_distribution.png", dpi=300)
plt.show()
print("""
Interpretation:

The dropout distribution shows the number of students who
dropped out and those who did not. This helps determine
whether the dataset is balanced for classification.
""")

print("\n")

# -----------------------------
# Gender Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("eda_results/gender_distribution.png", dpi=300)
plt.show()
print("""
Interpretation:

The gender distribution illustrates the proportion of male
and female students in the dataset, providing insight into
the demographic composition of the data.
""")

print("\n")

# -----------------------------
# Department Distribution
# -----------------------------
plt.figure(figsize=(10,5))
sns.countplot(x="Department", data=df)
plt.title("Department Distribution")
plt.xlabel("Department")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_results/department_distribution.png", dpi=300)
plt.show()
print("""
Interpretation:

This chart shows how students are distributed across
different academic departments. It helps identify whether
some departments have more student records than others.
""")

print("\n")

# -----------------------------
# Level Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Level", data=df)
plt.title("Level Distribution")
plt.xlabel("Academic Level")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("eda_results/level_distribution.png", dpi=300)
plt.show()

# ==========================================================
# SECTION 2: NUMERICAL VARIABLE DISTRIBUTIONS
# ==========================================================

# -----------------------------
# CGPA Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["CGPA"], bins=10, kde=True)
plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("eda_results/cgpa_distribution.png", dpi=300)
plt.show()
print("""
Interpretation:

The CGPA distribution shows the spread of students'
academic performance. This helps identify whether most
students have low, average, or high CGPAs.
""")

print("\n")

# -----------------------------
# Attendance Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["Attendance"], bins=10, kde=True)
plt.title("Attendance Distribution")
plt.xlabel("Attendance (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("eda_results/attendance_distribution.png", dpi=300)
plt.show()

print("""
Interpretation:

The attendance distribution shows students' attendance
patterns. Since attendance is an important predictor of
academic success, this visualization helps understand its
variation across the dataset.
""")

print("\n")

# -----------------------------
# Study Hours Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["Study_Hours"], bins=10, kde=True)
plt.title("Study Hours Distribution")
plt.xlabel("Study Hours")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("eda_results/study_hours_distribution.png", dpi=300)
plt.show()

# -----------------------------
# Exam Score Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["Exam"], bins=10, kde=True)
plt.title("Exam Score Distribution")
plt.xlabel("Exam Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("eda_results/exam_score_distribution.png", dpi=300)
plt.show()

# ==========================================================
# SECTION 3: CORRELATION ANALYSIS
# ==========================================================

# Select only numerical columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(12,8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig("eda_results/correlation_heatmap.png", dpi=300)

plt.show()

# ==========================================================
# SECTION 4: OUTLIER DETECTION
# ==========================================================

# -----------------------------
# CGPA Boxplot
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(y=df["CGPA"])
plt.title("Boxplot of CGPA")
plt.tight_layout()
plt.savefig("eda_results/cgpa_boxplot.png", dpi=300)
plt.show()

# -----------------------------
# Attendance Boxplot
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(y=df["Attendance"])
plt.title("Boxplot of Attendance")
plt.tight_layout()
plt.savefig("eda_results/attendance_boxplot.png", dpi=300)
plt.show()

# -----------------------------
# Study Hours Boxplot
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(y=df["Study_Hours"])
plt.title("Boxplot of Study Hours")
plt.tight_layout()
plt.savefig("eda_results/study_hours_boxplot.png", dpi=300)
plt.show()

# ==========================================================
# SECTION 5: EDA SUMMARY
# ==========================================================

print("\n" + "="*70)
print("EDA SUMMARY")
print("="*70)

print(f"Dataset Shape: {df.shape}")
print(f"Number of Columns: {df.shape[1]}")
print(f"Number of Rows: {df.shape[0]}")

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nTarget Variable Distribution:")
print(df["Dropout"].value_counts())

print("\nKey Findings:")

print("""
• The dataset contains 2,000 student records and 19 variables.

• No missing values or duplicate records were found,
  indicating that the dataset is complete and suitable
  for machine learning.

• Both numerical and categorical variables are present.
  The categorical variables were later encoded using
  Label Encoding.

• The numerical features have different value ranges,
  which justified the use of StandardScaler before
  model training.

• The visualizations provided insight into the
  distribution of student performance, attendance,
  departmental enrolment and dropout status.

• The exploratory analysis confirmed that the dataset
  is suitable for training machine learning models
  to predict student dropout risk.
""")

print("\nEDA Completed Successfully!")
print("All figures have been saved in the 'eda_results' folder.")
print("="*70)