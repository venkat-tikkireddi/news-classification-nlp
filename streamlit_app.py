import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('wordnet')

# Load the model and other necessary files

filename = 'models/news_classification_SGDClassified_71percent_model.sav'
SGD_model = pickle.load(open(filename, 'rb'))
vector_filename = 'models/SGD_tfidf_vectorizer.sav'
TFIDF_Vectorizer = pickle.load(open(vector_filename, 'rb'))
label_encoder_pickle = 'models/SGD_label_encoder.sav'
label_encoder = pickle.load(open(label_encoder_pickle, 'rb'))

# Define the preprocessing function
chichewa = ['i', 'ine', 'wanga', 'inenso', 'ife', 'athu', 'athu', 'tokha', 'inu', 'ndinu','iwe ukhoza', 'wako','wekha','nokha','iye','wake','iyemwini','icho','ndi','zake','lokha','iwo','awo','iwowo','chiyani','amene', 'uyu', 'uyo', 'awa', "ndili", 'ndi', 'ali','anali','khalani','akhala','kukhala',' Khalani nawo','wakhala','anali','chitani','amachita','kuchita', 'a', 'an', 'pulogalamu ya', 'ndi', 'koma', 'ngati', 'kapena', 'chifkwa', 'monga', 'mpaka', 'pamene', 'wa', 'pa ',' by','chifukwa' 'ndi','pafupi','kutsutsana','pakati','kupyola','nthawi', 'nthawi','kale','pambuyo','pamwamba', 'pansipa', 'kuti', 'kuchokera', 'mmwamba', 'pansi', 'mu', 'kunja', 'kuyatsa', 'kuchoka', 'kutha', 'kachiwiri', 'kupitilira','kenako',' kamodzi','apa','apo','liti','pati','bwanji','onse','aliyense','onse','aliyense', 'ochepa', 'zambiri', 'ambiri', 'ena', 'otero', 'ayi', 'kapena', 'osati', 'okha', 'eni', 'omwewo', 'kotero',' kuposa','nawonso',' kwambiri','angathe','ndidzatero','basi','musatero', 'musachite',' muyenera', 'muyenera kukhala','tsopano', 'sali', 'sindinathe','​​sanachite','satero','analibe', 'sanatero','sanachite','sindinatero','ayi','si', 'ma', 'sizingatheke','mwina','sayenera', 'osowa','osafunikira', 'shan' , 'nenani', 'sayenera', 'sanali', 'anapambana', 'sangachite', 'sanakonde', 'sangatero']
wn = WordNetLemmatizer()

def text_preprocessing(review):
    if not isinstance(review, str):
        review = str(review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [wn.lemmatize(word) for word in review if not word in chichewa]
    review = ' '.join(review)
    return review

# Streamlit app
st.title("News Classification App")
description = st.text_input("Enter news description:")

if st.button("Predict"):
    if description:
        processed_description = text_preprocessing(description)
        pred_test = pd.DataFrame({'description': [processed_description]})
        pred_test = TFIDF_Vectorizer.transform(pred_test['description']).toarray()
        prediction = SGD_model.predict(pred_test)
        predicted_category = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Category: {predicted_category}")
    else:
        st.warning("Please enter a news description.")