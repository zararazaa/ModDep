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



feature_columns = df.columns[:-1]

def preprocessing(user_input):
    df_input = pd.DataFrame([user_input], columns=feature_columns)

    # encoding
    for col in df_input.columns:
        if col in binary_encoding:  # binary
            df_input[col] = binary_encoding[col].transform([df_input[col][0]])  
        elif col in one_hot_encoding: 
            encoded_df = pd.DataFrame( #ohe
                one_hot_encoding[col].transform([[df_input[col][0]]]),
                columns=one_hot_encoding[col].get_feature_names_out([col])
            )
            df_input = df_input.drop(col, axis=1).join(encoded_df)

    
    df_input = normalizer.transform(df_input)
    return df_input


def main():
    st.title("Obesity Classification App")
    st.info("2702353221 - Zara Abigail Budiman - Tugas Sebelum UTS - Machine Learning OOP Implementation")

    if st.checkbox("Raw Data"):
        st.dataframe(df)

    # data visualization
    st.subheader("Data Visualization")
    data = st.selectbox("Data :3", df)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # User Input Section
    st.subheader("User Input")
    user_input = {}

    for feature in feature_columns:
        if df[feature].dtype == 'object':
            user_input[feature] = st.selectbox(f"Select {feature}", df[feature].unique())
        else:
            min_val, max_val = float(df[feature].min()), float(df[feature].max())
            
            if feature == "Height":  # Ensure Age is an integer input
                user_input[feature] = st.slider(f"Enter {feature}", min_val, max_val, (min_val + max_val) / 2)
            else:  
                user_input[feature] = st.slider(f"Enter {feature}", int(min_val), int(max_val), int((min_val + max_val) / 2), step=1)


    st.write("User Input Data:", user_input)

    if st.button("Predict"):
        processed_input = preprocess_input(user_input)
        prediction = model.predict(processed_input)
        probability = model.predict_proba(processed_input)

        st.success(f"Predicted Category: {prediction[0]}")

        # Show probability for each class
        prob_df = pd.DataFrame(probability, columns=model.classes_)
        st.subheader("Prediction Probabilities")
        st.dataframe(prob_df)

if __name__ == "__main__":
    main()
