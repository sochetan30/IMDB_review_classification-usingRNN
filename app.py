import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense


word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

model=load_model("SimpleRNN-IMDB.h5")


## Helpr Fucntion
def decode_reveiw(encoded_reveiw):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in encoded_reveiw])

def preprocess_text(text):
    words =text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review= sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Prediction function
def predict_sentiment(reveiw):
    preprocessed_input = preprocess_text(reveiw)
    prediction=model.predict(preprocessed_input)


    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]

import streamlit as st
## Streamlit app
st.title("IMDB MOvie reveiew Sentiment Analysis")
st.write("Enter a moveiw reveiw to classify it as a positive or negative")

# User Input
user_input =st.text_area("Movie Reveiw")

if st.button("Classify"):
    preprocess_input= preprocess_text(user_input)

    # Make Prediction:
    prediction=model.predict(preprocess_input)

    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
    st.write(f"Movie Reveiw sentiment is : {sentiment}")
else:
    st.write("Please enter a movie review")