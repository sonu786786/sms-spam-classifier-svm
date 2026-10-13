import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# NLTK setup
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocessing helper
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title="SMS Spam Detector (SVM)", layout="centered")
st.title("📩 SMS Spam Classifier using Linear SVM")
st.write("Support Vector Machine (LinearSVC) & TF-IDF NLP Pipeline")

# Load artifacts
try:
    model = joblib.load('svm_spam_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model files (.pkl) nahi mili! Pehle notebook ka Step 10 run karein.")
    st.stop()

# User Input
user_input = st.text_area("Write your message here:", placeholder="e.g. WINNER! You have won a $1000 cash prize...")

if st.button("Classify Message"):
    if user_input.strip() == "":
        st.warning("Pehle koi message type karein!")
    else:
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned])
        
        # Prediction & Margin distance
        pred = model.predict(vec)[0]
        decision_val = model.decision_function(vec)[0]

        if pred == 1:
            st.error(f"🚨 **SPAM DETECTED** (Margin score: {decision_val:.2f})")
        else:
            st.success(f"✅ **LEGITIMATE (HAM)** (Margin score: {decision_val:.2f})")