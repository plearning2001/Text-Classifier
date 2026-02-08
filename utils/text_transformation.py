import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import  PorterStemmer


# ----------------------------
# Text preprocessing function
# ----------------------------

def transform_text(text):
    ps = PorterStemmer()

    # Convert in lowercase
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    # Use set below to avoid nested loops
    stop_words = set(stopwords.words("english"))

    y = []
    for t in tokens:
        # if special chars then don't append
        # if t.isalnum() and t not in stopwords.words('english') and t not in string.punctuation:
        # if t.isalnum() and t not in stopwords.words('english'): # no need of string.punctuation
        if t.isalnum() and t not in stop_words: # no need to stopwords list
            y.append(ps.stem(t))
        
    return " ".join(y)