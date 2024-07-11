import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import nltk


st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0; /* Light gray background */
        margin: 0; /* Remove default margin for body */
        padding: 0; /* Remove default padding for body */
    }
    .st-bw {
        background-color: #eeeeee; /* White background for widgets */
    }
    .st-cq {
        background-color: #cccccc; /* Gray background for chat input */
        border-radius: 10px; /* Add rounded corners */
        padding: 8px 12px; /* Add padding for input text */
        color: black; /* Set text color */
    }
    .st-cx {
        background-color: white; /* White background for chat messages */
    }
    .sidebar .block-container {
        background-color: #f0f0f0; /* Light gray background for sidebar */
        border-radius: 10px; /* Add rounded corners */
        padding: 10px; /* Add some padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True
)

nltk.download('wordnet')
nltk.download('stopwords')

with open("./bookgenremodel.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

def cleantext(text):
    text = re.sub("'\''","",text)
    text = re.sub("[^a-zA-Z]"," ",text)
    text = ' '.join(text.split())
    text = text.lower()
    return text

def removestopwords(text):
    stop_words = set(stopwords.words('english'))
    removedstopword = [word for word in text.split() if word not in stop_words]
    return ' '.join(removedstopword)

def lematizing(sentence):
    lemma = WordNetLemmatizer()
    stemSentence = ""
    for word in sentence.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem
        stemSentence += " "
    
    stemSentence = stemSentence.strip()
    return stemSentence

def stemming(sentence):
    stemmer = PorterStemmer()
    stemmed_sentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemmed_sentence += stem
        stemmed_sentence += " "

    stemmed_sentence = stemmed_sentence.strip()
    return stemmed_sentence

def test(text, model, tfidf_vectorizer):
    text = cleantext(text)
    text = removestopwords(text)
    text = lematizing(text)
    text = stemming(text)

    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)

    newmapper = {0: 'fantasy', 1: 'science', 2: 'crime', 3: 'history', 4: 'horror', 5: 'thriller'}

    return newmapper[predicted[0]]

def predict_genre(book_summary):
    if not book_summary:
        st.warning("Please Enter a synopsis of the novel.")
    else:

        progress_placeholder = st.empty()
        progress_placeholder.info("Making predictions...")

        cleaned_summary = cleantext(book_summary)

        with open("./tfidfvector.pkl", 'rb') as file:
            vectorizer = pickle.load(file)

        vectorized_summary = vectorizer.transform([cleaned_summary])

        with open("./bookgenremodel.pkl", 'rb') as file:
            loaded_model = pickle.load(file)

        prediction = loaded_model.predict(vectorized_summary)

        newmapper = {0: 'fantasy', 1: 'science', 2: 'crime', 3: 'history', 4: 'horror', 5: 'thriller'}
        predicted_genre = newmapper[prediction[0]]

        progress_placeholder.empty()

        st.write("Genre Prediction Results")
        st.title(predicted_genre)
        st.success("Prediction complete!")

st.markdown("""
    <div style='display: flex; align-items: center; gap: 15px;'>
        <h1 style='margin: 0;'>Novel genre prediction</h1>
    </div>
""", unsafe_allow_html=True)

book_summary = st.text_area("write a synopsis of the novel:")

if st.button("Genre Prediction"):
    predict_genre(book_summary)
