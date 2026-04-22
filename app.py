import streamlit as st
import joblib
import numpy as np

model = joblib.load('iris_model.joblib')

st.title("🌸 Iris Flower Predictor")

# Sliders for user input
s_len = st.slider("Sepal Length", 4.0, 8.0, 5.0)
s_wid = st.slider("Sepal Width", 2.0, 4.5, 3.0)
p_len = st.slider("Petal Length", 1.0, 7.0, 1.5)
p_wid = st.slider("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predict"):
    data = np.array([[s_len, s_wid, p_len, p_wid]])
    prediction = model.predict(data)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Result: {species[prediction[0]]}")