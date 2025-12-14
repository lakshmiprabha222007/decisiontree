import streamlit as st
import pickle
import numpy as np

# Load trained Decision Tree model
with open("trained_decision_tree_pickle.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Decision Tree Prediction App")

st.write("Enter feature values to get prediction")

# 🔢 Change number of inputs according to your dataset
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)

if st.button("Predict"):
    input_data = np.array([[feature1, feature2]])
    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")
