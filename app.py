import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from utils.text_transformation import transform_text

# download only once (safe to keep)
nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ----------------------------
# Load model and vectorizer
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))




# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Spam Classifier")

st.title("📩 SMS / Email Spam Classifier")

input_text = st.text_area("Enter your message here")

if st.button("Predict"):

    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # preprocess
        transformed_text = transform_text(input_text)

        # vectorize
        vector_input = vectorizer.transform([transformed_text])

        # predict
        prediction = model.predict(vector_input)[0]

        st.subheader("Result")

        if prediction == 1:
            st.error("🚨 Spam message")
        else:
            st.success("✅ Not spam (Ham)")
