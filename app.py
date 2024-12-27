import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Prediction function
def predict_news(text):
    transformed_text = vectorizer.transform([text])  # Ensure vectorizer is fitted
    prediction = model.predict(transformed_text)
    return 'Spam News' if prediction == 1 else 'Legitimate News'

# Streamlit UI
st.title("Spam News Detection")
st.text("This app classifies news as Spam or Legitimate")

input_text = st.text_area("Enter news text to classify:")

if st.button("Classify"):
    if input_text.strip():
        result = predict_news(input_text)
        st.success(f'The news is classified as: {result}')
    else:
        st.warning("Please enter some text to classify.")
