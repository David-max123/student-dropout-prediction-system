import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.title("🤖 Model Evaluation Dashboard")

# Load saved results
results = joblib.load("models/model_results.pkl")

# Convert dictionary to dataframe
df_results = pd.DataFrame(results).T

# Convert to percentage
df_display = (df_results * 100).round(2)

st.subheader("📊 Model Performance Comparison")

st.dataframe(
    df_display,
    use_container_width=True
)

# ==========================
# Best Model
# ==========================

best_model = df_display["Accuracy"].idxmax()
best_accuracy = df_display["Accuracy"].max()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "🏆 Best Model",
        best_model
    )

with col2:
    st.metric(
        "🎯 Accuracy",
        f"{best_accuracy:.2f}%"
    )

st.markdown("---")

# ==========================
# Accuracy Comparison
# ==========================

st.subheader("📈 Accuracy Comparison")

fig1 = px.bar(
    df_display.reset_index(),
    x="index",
    y="Accuracy",
    color="Accuracy",
    title="Model Accuracy Ranking"
)

fig1.update_layout(
    xaxis_title="Model",
    yaxis_title="Accuracy (%)"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# ==========================
# Precision Comparison
# ==========================

st.subheader("🎯 Precision Comparison")

fig2 = px.bar(
    df_display.reset_index(),
    x="index",
    y="Precision",
    color="Precision"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ==========================
# Recall Comparison
# ==========================

st.subheader("🔍 Recall Comparison")

fig3 = px.bar(
    df_display.reset_index(),
    x="index",
    y="Recall",
    color="Recall"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ==========================
# F1 Score Comparison
# ==========================

st.subheader("⚖️ F1 Score Comparison")

fig4 = px.bar(
    df_display.reset_index(),
    x="index",
    y="F1",
    color="F1"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# ==========================
# Radar Chart
# ==========================

st.subheader("🕸️ Model Metrics Radar Chart")

radar_data = df_display.reset_index()

fig5 = px.line_polar(
    radar_data,
    r="Accuracy",
    theta="index",
    line_close=True
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# ==========================
# FEATURE IMPORTANCE
# ==========================

st.markdown("---")
st.subheader("🎯 Feature Importance Analysis")

try:

    model = joblib.load("models/best_model.pkl")

    feature_names = joblib.load(
        "models/feature_names.pkl"
    )

    # Gradient Boosting supports feature_importances_
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig6 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Features Influencing Student Dropout Prediction"
    )

    fig6.update_layout(
        height=700
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

except Exception as e:

    st.warning(
        f"Feature importance unavailable: {e}"
    )

# ==========================
# CONFUSION MATRIX
# ==========================

from sklearn.metrics import confusion_matrix
import numpy as np

st.markdown("---")
st.subheader("🔲 Confusion Matrix")

try:

    y_test = joblib.load(
        "models/y_test.pkl"
    )

    predictions = joblib.load(
        "models/best_predictions.pkl"
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual No Dropout",
            "Actual Dropout"
        ],
        columns=[
            "Predicted No Dropout",
            "Predicted Dropout"
        ]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )

    fig_cm.update_layout(
        height=500
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

except Exception as e:

    st.error(
        f"Confusion Matrix Error: {e}"
    )   

# ==========================
# ROC CURVE & AUC SCORE
# ==========================

from sklearn.metrics import roc_curve, auc

st.markdown("---")
st.subheader("📈 ROC Curve & AUC Score")

try:

    y_test = joblib.load(
        "models/y_test.pkl"
    )

    probabilities = joblib.load(
        "models/best_probabilities.pkl"
    )

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    auc_score = auc(
        fpr,
        tpr
    )

    roc_df = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr
    })

    fig_roc = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve (AUC = {auc_score:.3f})"
    )

    # Reference diagonal
    fig_roc.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Guess"
    )

    st.plotly_chart(
        fig_roc,
        use_container_width=True
    )

    st.metric(
        "🎯 AUC Score",
        f"{auc_score:.3f}"
    )

except Exception as e:

    st.error(
        f"ROC Curve Error: {e}"
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