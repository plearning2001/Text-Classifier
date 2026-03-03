import streamlit as st
import pickle
import nltk
from utils.text_transformation import transform_text

# ----------------------------
# Safe NLTK Download
# ----------------------------
def download_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords")
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_resources()

# ----------------------------
# Load Model
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

THRESHOLD = 0.25  # Tuned for high recall

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered"
)

# ----------------------------
# Header
# ----------------------------
st.title("📱 Intelligent SMS Spam Detector")
st.markdown(
    """
Built using **TF-IDF + Multinomial Naive Bayes**  
Optimized for **High Recall** to catch more spam messages.
"""
)

st.divider()

# ----------------------------
# Input Section
# ----------------------------
st.subheader("Test an SMS Message")

input_text = st.text_area(
    "Enter message:",
    placeholder="Example: Congratulations! You have won a free prize. Call now!"
)

col1, col2 = st.columns(2)

with col1:
    predict_clicked = st.button("🔍 Analyze Message")

with col2:
    example_clicked = st.button("🧪 Load Example")

if example_clicked:
    input_text = "Free entry in a weekly competition to win cash prizes. Text WIN to 80086 now!"
    st.text_area("Example Message:", input_text, height=100)

if predict_clicked:

    if input_text.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_text = transform_text(input_text)
        vector_input = vectorizer.transform([transformed_text])
        spam_proba = model.predict_proba(vector_input)[0][1]
        prediction = int(spam_proba > THRESHOLD)

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 Spam Detected")
        else:
            st.success("✅ Not Spam")

        st.metric("Spam Probability", f"{spam_proba:.2f}")

        if 0.20 <= spam_proba <= 0.40:
            st.caption("⚠️ Borderline probability range.")

st.divider()

# ----------------------------
# Compact Info Section
# ----------------------------
with st.expander("About this Model"):
    st.markdown("""
**Algorithm Used:** Multinomial Naive Bayes  
**Feature Engineering:** TF-IDF (Term Frequency – Inverse Document Frequency)  
**Goal:** Prioritize catching spam (high recall tuning)

**Scope:**  
Designed for SMS-style spam detection.

**Limitations:**  
May not perfectly classify long email marketing content.  
Performance depends on patterns present in training data.
""")

# ----------------------------
# Footer
# ----------------------------
st.caption("Built by Pranav Kode • Machine Learning • NLP • Model Deployment")