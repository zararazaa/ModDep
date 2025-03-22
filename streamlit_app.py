import streamlit as st
import pandas as pd
import joblib

RAW_DATASET_PATH = "ObesityDataSet_raw_and_data_sinthetic.csv"
MODEL_PATH = "trained_model.pkl"
NORMALIZER_PATH = "normalizer.pkl"
ENCODING_PATH = "encoding.pkl"

# Load the trained model
def load_model():
    return joblib.load(MODEL_PATH)


def load_normalizer():
    return joblib.load(NORMALIZER_PATH)

def load_encoders():
    return joblib.load(ENCODING_PATH)


def load_data():
    return pd.read_csv(RAW_DATASET_PATH)

def preprocess_input(user_input, normalizer, encoders, feature_columns):
    df_input = pd.DataFrame([user_input], columns=feature_columns)

    for col in df_input.columns:
        if col in encoders["label_encoders"]:  # If binary categorical column
            df_input[col] = encoders["label_encoders"][col].transform(df_input[col])
        elif col in encoders["one_hot_encoders"]:  # If multi-category column
            encoded_df = pd.DataFrame(
                encoders["one_hot_encoders"][col].transform(df_input[[col]]),
                columns=encoders["one_hot_encoders"][col].get_feature_names_out([col])
            )
            df_input = df_input.drop(col, axis=1).join(encoded_df)

    # Apply normalization
    df_input = normalizer.transform(df_input)

    return df_input

# Streamlit UI
def main():
    st.title("Obesity Classification")
    st.info("Tugas Sebelum UTS")

    # Load dataset and models
    df = load_data()
    model = load_model()
    normalizer = load_normalizer()
    encoders = load_encoders()

    st.success("Dataset, Model, Normalizer, and Encoders Loaded Successfully!")

    # Select feature columns
    feature_columns = df.columns[:-1]  # Assuming the last column is the target
    selected_features = st.multiselect("Select input features", feature_columns, default=feature_columns)

    if selected_features:
        # Get user input
        user_input = []
        for feature in selected_features:
            value = st.number_input(f"Enter value for {feature}", float(df[feature].min()), float(df[feature].max()))
            user_input.append(value)

        if st.button("Predict"):
            # Preprocess input (apply encoding and normalization)
            processed_input = preprocess_input(user_input, normalizer, encoders, selected_features)

            # Make prediction
            prediction = model.predict(processed_input)
            st.success(f"Predicted Category: {prediction[0]}")

if __name__ == "__main__":
    main()
