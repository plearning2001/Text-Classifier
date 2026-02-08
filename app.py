import streamlit as st
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from utils.text_transformation import transform_text

def download_nltk_resources():
    import nltk

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
