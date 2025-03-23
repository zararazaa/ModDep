import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
RAW_DATASET_PATH = "ObesityDataSet_raw_and_data_sinthetic.csv"
MODEL_PATH = "trained_model.pkl"
NORMALIZER_PATH = "normalizer.pkl"
ENCODING_PATH = "encoding.pkl"

#to show raw data
def load_data():
    return pd.read_csv(RAW_DATASET_PATH)

#loading trained model
def load_model():
    return joblib.load(MODEL_PATH)

# loading the normalizer
def load_normalizer():
    return joblib.load(NORMALIZER_PATH)

# loading the encoder
def load_encoders():
    return joblib.load(ENCODING_PATH)

df = load_data()
model = load_model()
normalizer = load_normalizer()
encoders = load_encoders()

binary_encoders = encoders.get("binary_encoders", {})  
one_hot_encoders = encoders.get("one_hot_encoders", {})  

feature_columns = df.columns[:-1] 

class Modeling:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)
        except Exception as e:
            st.error(f"Error loading model: {e}")

    def predict(self, processed_input):
        if self.model is None:
            st.error("Model is not loaded. Please train and save the model first.")
            return None
        return self.model.predict(processed_input), self.model.predict_proba(processed_input)

modeling = Modeling(MODEL_PATH)

def preprocessing(user_input):
    df_input = pd.DataFrame([user_input], columns=feature_columns)

    for col in binary_encoders:
            if col in df_input.columns:
                df_input[col] = binary_encoders[col].transform([df_input[col][0]])
    
    for col in one_hot_encoders:
        if col in df_input.columns:
            encoded_df = pd.DataFrame(
                one_hot_encoders[col].transform([[df_input[col][0]]]),
                columns=one_hot_encoders[col].get_feature_names_out([col])
            )
            df_input = df_input.drop(col, axis=1).join(encoded_df)
        
    df_input = normalizer.transform(df_input)
    return df_input




def main():
    st.title("Obesity Classification App")
    st.info("2702353221 - Zara Abigail Budiman - Tugas Sebelum UTS - Machine Learning OOP Implementation")

    if st.subheader("Raw Data"):
        st.info("this is raw data")
        st.dataframe(df)

    # data visualization
    st.subheader("Data Visualization")
    data = st.selectbox("Select Data :3", df.columns)  # Includes all columns (features + target)
    fig, ax = plt.subplots()
    sns.histplot(df[data], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # User Input Section
    st.subheader("User Input")
    user_input = {}

    for feature in feature_columns:
        if df[feature].dtype == 'object':
            user_input[feature] = st.selectbox(f"Select {feature}", df[feature].unique())
        else:
            min_val, max_val = float(df[feature].min()), float(df[feature].max())

            if feature == "Height":  
                value = st.slider(f"Enter {feature}", min_val, max_val, (min_val + max_val) / 2)
                user_input[feature] = round(value,2)
                
            else:  
                user_input[feature] = st.slider(f"Enter {feature}", int(min_val), int(max_val), int((min_val + max_val) / 2), step=1)

    st.write("User Input Data:", user_input)

    if st.button("Predict"):
        processed_input = preprocessing(user_input)  
        prediction, probability = modeling.predict(processed_input)

        if prediction is not None:
            st.success(f"Predicted Category: {prediction[0]}")
            prob_df = pd.DataFrame(probability, columns=modeling.model.classes_)
            st.subheader("Prediction Probabilities")
            st.dataframe(prob_df)
            
if __name__ == "__main__":
    main()
