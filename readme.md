## SMS classifier to general text classifier

Steps:

- Lowercasing
- Tokenization
- Removal of non-alphanumeric tokens
- Stop-word removal
- Stemming using `PorterStemmer`

The cleaned tokens are joined back into a single string before vectorization.

---

## 🧪 Methods and Concepts Used

### Natural Language Processing

- Tokenization (NLTK)
- Stop-word removal
- Stemming
- Bag of Words / TF-IDF vectorization

### Machine Learning

- Scikit-learn classifiers
- Train-test split with stratification
- Model evaluation using accuracy and precision

---

## 📊 Visualization & Analysis

- Correlation heatmap on engineered numerical features
- Most frequent words visualization using:
  - `Counter`
  - Bar plots
  - Word clouds

---

## 📈 Model Performance

The trained model achieves high accuracy and high precision on the validation set.  
(Exact results are available in the training notebook.)

---

## ⚠️ Important Note (Limitation)

The model is trained mainly on **SMS-style spam data**.  
It performs best on promotional and marketing type spam messages.

It may not generalize well to:

- phishing emails
- account-security alerts
- enterprise email scams

This limitation is due to the dataset distribution.

---

## 💻 Streamlit Web App

The application allows the user to:

- enter a message
- click a predict button
- see whether the message is spam or not

---

## ▶ How to Run the Project

### 1. Clone the repository

git clone <your-repository-url>
cd Text-Classifier
python -m venv venv
venv\Scripts\Activate.ps1 / venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
