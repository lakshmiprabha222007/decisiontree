import streamlit as st
import pickle
import numpy as np

# Load trained Decision Tree model
with open("trained_decision_tree_pickle.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Decision Tree Prediction App")

# Get number of features model expects
n_features = model.n_features_in_

st.write(f"Model expects {n_features} input features")

# Create inputs dynamically
inputs = []
for i in range(n_features):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

if st.button("Predict"):
    input_data = np.array([inputs])
    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")
