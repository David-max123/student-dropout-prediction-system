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

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

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

# ----------------------------------------------------------
# Summary Statistics
# ----------------------------------------------------------

print("\nSUMMARY STATISTICS")
print(df.describe())

# ----------------------------------------------------------
# Missing Values
# ----------------------------------------------------------

print("\nMISSING VALUES")
print(df.isnull().sum())

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

print("\nEDA Completed Successfully!")
print("All figures have been saved in the 'eda_results' folder.")
print("="*70)