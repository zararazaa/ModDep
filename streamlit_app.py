import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DATASET_PATH = "ObesityDataSet_raw_and_data_sinthetic.csv"
NORMALIZER_PATH = "normalizer.pkl"
ENCODING_PATH = "encoding.pkl"

#to show raw data
def load_data():
    return pd.read_csv(RAW_DATASET_PATH)

#loading trained model
# def load_model():
#     model = joblib.load(MODEL_PATH)
#     return model

def load_model(filename):
    model = joblib.load(filename)
    return model

# loading the normalizer
def load_normalizer():
    return joblib.load(NORMALIZER_PATH)

# loading the encoder
def load_encoders():
    return joblib.load(ENCODING_PATH)

df = load_data()
normalizer = load_normalizer()
encoders = load_encoders()

binary_encoders = encoders.get("binary_encoders", {})  
one_hot_encoders = encoders.get("one_hot_encoders", {})  

feature_columns = df.columns[:-1] 


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

def predict_with_model(model, processed_input):
    prediction = model.predict([processed_input])
    return prediction[0]


def main():
    st.title("Obesity Classification App")
    st.info("2702353221 - Zara Abigail Budiman - Tugas Sebelum UTS - Machine Learning OOP Implementation")

    if st.subheader("Raw Data"):
        st.info("this is raw data")
        st.dataframe(df)
        
    # Data Visualization
    st.subheader("Data Visualization")
    
    # Select features for scatter plot
    x_axis = "Height"  # X-axis
    y_axis = "Weight"  # Y-axis
    hue_column = "NObeyesdad"  # Target variable (for color grouping)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue_column, palette="Set2", alpha=0.7, edgecolor="black")
    
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.title("Height vs Weight by Obesity Category")
    plt.legend(title="Obesity Levels", bbox_to_anchor=(1, 1), loc="upper left")
    
    # Display in Streamlit
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
        model_filename = 'trained_model.pkl'
        model=load_model(model_filename)
        processed_input = preprocessing(user_input)  
        prediction = predict_with_model(model, processed_input)
        st.write('The prediction output is:', prediction)


            
if __name__ == "__main__":
    main()
