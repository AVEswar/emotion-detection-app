import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# MODEL_PATH = 'emotion_detection.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

model = load_model('emotion_detection_model.keras')

with open(TOKENIZER_PATH, 'rb') as handle:
  tokenizer = pickle.load(handle)

with open(LABEL_ENCODER_PATH, 'rb') as handle:
  label_encoder = pickle.load(handle)

MAX_SEQUENCE_LENGTH = model.input_shape[1]

def predict_emotion(text):
  input_sequence = tokenizer.texts_to_sequences([text])
  padded_input_sequence = pad_sequences(input_sequence, maxlen = MAX_SEQUENCE_LENGTH)

  prediction = model.predict(padded_input_sequence)
  predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
  
  return predicted_label[0]

st.title("Emotion Detection App")
st.write("Enter some text to find out the emotion behind it!")

user_input = st.text_area("Type your text here")
if st.button("Predict Emotion",icon = '🪄'):
  if user_input:
    emotion = predict_emotion(user_input)
    st.write(f"The predicted emotion is: **{emotion}**")
  else:
    st.write("Please enter some text to analyze")