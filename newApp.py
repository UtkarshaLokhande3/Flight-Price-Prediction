import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the label encoders
@st.cache_resource
def load_label_encoders():
    with open("label_encoders.pkl", "rb") as file:
        encoders = pickle.load(file)
    return encoders

encoders = load_label_encoders()

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open("FPP_Final.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Function to encode input data using the loaded encoders
def encode_input_data(data, encoders):
    encoded_data = data.copy()
    for col, encoder in encoders.items():
        if col in data:
            try:
                encoded_data[col] = encoder.transform([data[col]])[0]
            except ValueError:
                st.error(f"Invalid value for {col}: '{data[col]}'. Please select a valid option.")
                return None
        else:
            st.warning(f"Missing input for {col}. Using default value.")
            encoded_data[col] = encoder.transform([encoder.classes_[0]])[0]  # Default to the first class
    return encoded_data

# Set up the Streamlit app
st.title("Flight Price Prediction")

# Input features for prediction
st.subheader("Input flight details:")
airline = st.selectbox("Airline", encoders['airline'].classes_)
source_city = st.selectbox("Source City", encoders['source_city'].classes_)
departure_time = st.selectbox("Departure Time", encoders['departure_time'].classes_)
stops = st.number_input("Number of Stops", min_value=0, max_value=5, value=0, step=1)
arrival_time = st.selectbox("Arrival Time", encoders['arrival_time'].classes_)
destination_city = st.selectbox("Destination City", encoders['destination_city'].classes_)
class_type = st.selectbox("Class Type", options=['Business', 'Economy'])
duration = st.number_input("Duration (in hours)", min_value=0.0, max_value=72.0, step=0.1)
days_left = st.number_input("Days Left for Journey", min_value=0, max_value=365)

# Prepare input data
input_data = {
    'airline': airline,
    'source_city': source_city,
    'departure_time': departure_time,
    'stops': stops,
    'arrival_time': arrival_time,
    'destination_city': destination_city,
    'class_type': class_type,  # Keep as is for encoding
    'duration': duration,
    'days_left': days_left,
}

# Add a button for prediction
if st.button("Predict Flight Price"):
    # Encode the input data
    input_data['class_type'] = 1 if class_type == 'Business' else 2  # Manually encode class_type
    encoded_data = encode_input_data(input_data, encoders)

    if encoded_data:
        # Convert to DataFrame
        input_df = pd.DataFrame([encoded_data])

        # Ensure all data is numeric
        st.write("Encoded Input Data for Debugging:")
        st.write(input_df)

        # Make predictions
        try:
            prediction = model.predict(input_df)
            st.subheader("Predicted Flight Price")
            st.write(f"₹ {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
