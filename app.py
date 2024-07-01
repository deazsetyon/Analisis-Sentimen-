# import streamlit as st
# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load trained model and TfidfVectorizer
# nb_model = joblib.load('nb_model.pkl')
# tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# # Streamlit app title and input
# st.title("GenZ Sentiment Checker")
# st.markdown("---")
# # Define CSS style for border
# st.markdown(
#     """
#     <style>
#     .stImage {
#         border: 2px solid #4682B4;
#         border-radius: 5px;
#         padding: 5px;
#     }
#     .stTextInput {
#         border: 2px solid #4682B4;
#         border-radius: 5px;
#         padding: 8px;
#         margin-bottom: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
# st.markdown("Enter a sentence and we'll predict if it's positive or negative!")

# text = st.text_input("Please Enter Your Sentence:")

# # Function to preprocess input text and make predictions
# def predict_sentiment(text):
#     # Transform the input text using the loaded TfidfVectorizer
#     vectorized_text = tfidf_vectorizer.transform([text])

#     # Predict sentiment
#     nb_prediction = nb_model.predict(vectorized_text)

#     # Map prediction to 'negatif' or 'positif'
#     if nb_prediction[0] == 0:
#         return 'negatif'
#     else:
#         return 'positif'

# # Display predictions
# if text:
#     st.markdown("---")
#     st.write("## Prediction:")
#     nb_sentiment = predict_sentiment(text)
#     st.write(f"Naive Bayes Sentiment: **{nb_sentiment}**")

import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and TfidfVectorizer
nb_model = joblib.load('nb_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Custom CSS (sama seperti sebelumnya)
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #0072ff, #00c6ff);
    color: #000;
    font-family: Arial, sans-serif;
}
.stApp {
    background-color: white;
    border-radius: 10px;
    padding: 30px;
    margin: 10% auto;
    width: 70%;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}
h1 {
    color: #0062cc;
    text-align: center;
    margin-bottom: 30px;
}
.stTextInput > div > div > input {
    border-radius: 1rem;
}
.stButton > button {
    width: 45%;
    border: none;
    border-radius: 1rem;
    padding: 10px;
    background: #dc3545;
    font-weight: 600;
    color: #fff;
    cursor: pointer;
    margin: 10px 2.5%;
}
.prediction {
    margin-top: 20px;
}
.prediction h3 {
    font-size: 18px;
    color: #0062cc;
    font-weight: 600;
    margin-bottom: 5px;
}
.prediction p {
    font-size: 18px;
    color: #000;
    font-weight: 400;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Function to preprocess input text and make predictions
def predict_sentiment(text):
    vectorized_text = tfidf_vectorizer.transform([text])
    nb_prediction = nb_model.predict(vectorized_text)
    return 'positif' if nb_prediction[0] == 1 else 'negatif'

# App layout
st.title("Analisis Sentimen Komentar Instagram Pemilu 2024")

col1, col2 = st.columns(2)

with col1:
    text = st.text_area("Your Comments *", height=150)
    if st.button("Predict NB"):
        if text:
            nb_sentiment = predict_sentiment(text)
            with col2:
                st.markdown("<div class='prediction'>", unsafe_allow_html=True)
                st.markdown("<h3>Hasil Prediksi NB</h3>", unsafe_allow_html=True)
                st.markdown(f"<p>{nb_sentiment.capitalize()}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Silakan masukkan komentar terlebih dahulu.")
