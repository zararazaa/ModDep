import streamlit as st
import joblib


def predict_with_model(model, user_input):
  prediction = model.predict([user_input])
return prediction[0]

def main():
  st.title('Obesity Classification')
  st.info('Tugas Sebelum UTS')

