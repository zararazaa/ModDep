import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


RAW_DATASET_PATH = "ObesityDataSet_raw_and_data_sinthetic.csv"
MODEL_PATH = "trained_model.pkl"
NORMALIZER_PATH = "normalizer.pkl"
ENCODING_PATH = "encoding.pkl"

# Load the trained model and preprocessing tools

def load_data():
    return pd.read_csv(RAW_DATASET_PATH)


def load_model():
    return joblib.load(MODEL_PATH)


def load_normalizer():
    return joblib.load(NORMALIZER_PATH)


def load_encoders():
    return joblib.load(ENCODING_PATH)

class ObesityClassifier:
    def __init__(self):
        self.model = load_model()
        self.normalizer = load_normalizer()
        self.encoders = load_encoders()
        self.df = load_data()
        self.feature_columns = self.df.columns[:-1]  # Exclude target column

    def preprocess_input(self, user_input):
        df_input = pd.DataFrame([user_input], columns=self.feature_columns)


        for col in df_input.columns:
            if col in self.encoders["label_encoders"]:  
                df_input[col] = self.encoders["label_encoders"][col].transform(df_input[col])
            elif col in self.encoders["one_hot_encoders"]:
                encoded_df = pd.DataFrame(
                    self.encoders["one_hot_encoders"][col].transform(df_input[[col]]),
                    columns=self.encoders["one_hot_encoders"][col].get_feature_names_out([col])
                )
                df_input = df_input.drop(col, axis=1).join(encoded_df)

        # Apply normalization
        df_input = self.normalizer.transform(df_input)
        return df_input

    def predict(self, user_input):
        processed_input = self.preprocess_input(user_input)
        prediction = self.model.predict(processed_input)
        probability = self.model.predict_proba(processed_input)
        return prediction[0], probability

# Streamlit UI
def main():
    st.title("Obesity Classification App")
    st.info("Tugas Sebelum UTS - Machine Learning OOP Implementation")

    # Load classifier
    classifier = ObesityClassifier()

    # Display dataset
    if st.checkbox("Show Raw Data"):
        st.dataframe(classifier.df)

    # Data Visualization
    st.subheader("Data Visualization")
    selected_feature = st.selectbox("Select feature for distribution plot", classifier.feature_columns)
    fig, ax = plt.subplots()
    sns.histplot(classifier.df[selected_feature], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # Get user input
    st.subheader("User Input")
    user_input = {}

    for feature in classifier.feature_columns:
        if classifier.df[feature].dtype == 'object':
            user_input[feature] = st.selectbox(f"Select {feature}", classifier.df[feature].unique())
        else:
            min_val, max_val = float(classifier.df[feature].min()), float(classifier.df[feature].max())  # Ensure values exist
            
            if feature == "Height":
                user_input[feature] = st.slider(f"Enter {feature}", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
            else:
                user_input[feature] = st.slider(f"Enter {feature}", min_value=int(min_val), max_value=int(max_val), value=int((min_val + max_val) / 2), step=1)
    

    # Show input data
    st.write("User Input Data:", user_input)

    # Predict
    if st.button("Predict"):
        prediction, probability = classifier.predict(user_input)
        st.success(f"Predicted Category: {prediction}")

        # Show probabilities
        prob_df = pd.DataFrame(probability, columns=classifier.model.classes_)
        st.subheader("Prediction Probabilities")
        st.dataframe(prob_df)

if __name__ == "__main__":
    main()
