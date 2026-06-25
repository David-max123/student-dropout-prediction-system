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

st.markdown("---")

st.caption(
    "Student Dropout Prediction System | Final Year Project | Machine Learning Based Early Warning System"
)