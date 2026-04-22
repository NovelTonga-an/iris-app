import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- 1. LOAD THE BRAIN ---
# Make sure 'iris_model.joblib' is in the same GitHub folder
model = joblib.load('iris_model.joblib')

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸")

# --- 3. SIDEBAR LAYOUT ---
st.sidebar.header("User Input Parameters")
st.sidebar.write("Adjust the sliders to input flower measurements.")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

input_df = user_input_features()

st.title("🌸 Iris Flower Classification App")

# Display the inputs in a nice table
st.subheader('Selected Input Measurements')
st.write(pd.DataFrame(input_df, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']))

# --- 5. PREDICTION LOGIC ---
if st.button('Predict Flower Species'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    species = ['Setosa', 'Versicolor', 'Virginica']
    result = species[prediction[0]]

    # Display Result
    st.subheader('Prediction Result')
    st.success(f"The AI model predicts this is a **{result}**.")
    
    # Show confidence levels
    st.subheader('Prediction Confidence')
    for i, s in enumerate(species):
        st.write(f"{s}: {prediction_proba[0][i]*100:.2f}%")
