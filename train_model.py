"""
==========================================================
Student Dropout Prediction System
Model Training Script

This script:
1. Loads the dataset
2. Preprocesses the data
3. Trains multiple ML models
4. Compares their performance
5. Saves the best model
==========================================================
"""

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/student_dataset.csv")

print(df.head())

print("\nDataset Shape:", df.shape)

# ==========================
# Encode Text Columns
# ==========================

label_encoders = {}

# Automatically detect categorical columns
categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    label_encoders[column] = encoder

# ==========================
# Features and Target
# ==========================

X = df.drop(columns=[
    "Student_ID",
    "Dropout"
])
print(X.columns.tolist())

y = df["Dropout"]

# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

# ==========================
# Machine Learning Models
# ==========================

models = {

    "Logistic Regression": Pipeline([

        ("scaler", StandardScaler()),

        ("model", LogisticRegression(max_iter=3000))

    ]),

    "Decision Tree": DecisionTreeClassifier(

        random_state=42

    ),

    "Random Forest": RandomForestClassifier(

        n_estimators=200,

        random_state=42

    ),

    "Gradient Boosting": GradientBoostingClassifier(

        random_state=42

    ),

    "Support Vector Machine": Pipeline([

        ("scaler", StandardScaler()),

        ("model", SVC(probability=True))

    ])

}

# ==========================
# Train Models
# ==========================

results = {}

best_accuracy = 0

best_model = None

best_name = ""

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    results[name] = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1

    }

    print(results[name])
if accuracy > best_accuracy:

    best_accuracy = accuracy

    best_model = model

    best_name = name

    best_predictions = prediction

    best_probabilities = probabilities
     
# Save predictions from best model
best_predictions = best_model.predict(X_test)

joblib.dump(
    best_predictions,
    "models/best_predictions.pkl"
)

joblib.dump(
    best_probabilities,
    "models/best_probabilities.pkl"
)

joblib.dump(
    y_test,
    "models/y_test.pkl"
)

# ==========================
# Save Best Model
# ==========================

joblib.dump(best_model, "models/best_model.pkl")

joblib.dump(label_encoders, "models/label_encoders.pkl")

# Save model comparison metrics
joblib.dump(
    results,
    "models/model_results.pkl"
)

# Save feature names
joblib.dump(
    X.columns.tolist(),
    "models/feature_names.pkl"
)   

print("\n==============================")

print("Best Model:", best_name)

print("Accuracy:", round(best_accuracy * 100, 2), "%")

print("==============================")