import joblib
import pickle
import streamlit as st

# Load the model and label encoders
model = joblib.load("D:/Doc/Flight Price Prediction File/FPP_Final.joblib", mmap_mode='r')
label_encoders = pickle.load(open("D:/Doc/Flight Price Prediction File/label_encoders.pkl", 'rb'))

# Streamlit app
st.title("Flight Price Prediction")

# Collecting form data
stops = st.number_input("Number of Stops", min_value=0, max_value=5, value=0, step=1)
class_type = st.selectbox("Class Type", options=['Business', 'Economy'])
duration = st.number_input("Duration (in hours)", min_value=0.0, value=1.0, step=0.1)
days_left = st.number_input("Days Left to Departure", min_value=0, value=30, step=1)

# Airline selection
airline = st.selectbox("Airline", options=['GO_FIRST', 'IndiGo', 'Air_India', 'SpiceJet', 'Vistara', 'AirAsia'])

# Source city selection
source = st.selectbox("Source City", options=['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Bangalore', 'Hyderabad', 'Ahmedabad'])

# Destination city selection
destination = st.selectbox("Destination City", options=['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Bangalore', 'Hyderabad', 'Ahmedabad'])

# Arrival time selection
arrival_time = st.selectbox("Arrival Time", options=['12_PM_6_PM', '6_AM_12_PM', 'After_6_PM', 'Before_6_AM'])

# Departure time selection
departure_time = st.selectbox("Departure Time", options=['12_PM_6_PM', '6_AM_12_PM', 'After_6_PM', 'Before_6_AM'])

# Debugging: Print available classes in label encoders
st.write("Available classes in label encoders:")
for feature_name, encoder in label_encoders.items():
    st.write(f"{feature_name}: {encoder.classes_}")

# Function to encode features and handle missing labels
def encode_feature(feature_name, feature_value):
    if feature_name in label_encoders:
        encoder = label_encoders[feature_name]
        if feature_value in encoder.classes_:
            return encoder.transform([feature_value])[0]
        else:
            st.warning(f"Unexpected value '{feature_value}' for '{feature_name}'. Using default encoding.")
            # Handle unseen values, e.g., using the index for 'unknown' class or setting to a default value
            return encoder.transform(['unknown'])[0] if 'unknown' in encoder.classes_ else -1
    else:
        st.error(f"Label encoder for '{feature_name}' not found.")
        return -1

# Encoding features
airline_encoded = encode_feature('airline', airline)
source_encoded = encode_feature('source_city', source)
destination_encoded = encode_feature('destination_city', destination)
arrival_encoded = encode_feature('arrival_time', arrival_time)
departure_encoded = encode_feature('departure_time', departure_time)

# Prepare the feature vector for prediction
feature_vector = [
    stops,
    1 if class_type == 'Business' else 2,  # Convert class_type to numeric
    duration,
    days_left,
    airline_encoded,
    source_encoded,
    destination_encoded,
    arrival_encoded,
    departure_encoded,
]

# Make the prediction
if st.button("Predict"):
    try:
        prediction = model.predict([feature_vector])
        output = round(prediction[0], 2)
        st.success(f"Your Flight price is Rs. {output}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
