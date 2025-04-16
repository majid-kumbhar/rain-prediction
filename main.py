import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Mapping for location encoding
location_data = {
    'new york': 11, 'los angeles': 10, 'chicago': 2, 'houston': 7, 'phoenix': 13,
    'philadelphia': 12, 'san antonio': 14, 'san diego': 15, 'dallas': 4,
    'san jose': 17, 'austin': 0, 'jacksonville': 9, 'fort worth': 6,
    'columbus': 3, 'indianapolis': 8, 'charlotte': 1, 'san francisco': 16,
    'seattle': 18, 'denver': 5, 'washington dc': 19
}

# Title
st.markdown(
    "<h1 style='text-align: center; color: #4682B4;'>🌧️ Rain Prediction App 🌧️</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #6A5ACD;'>Predict the likelihood of rain based on weather parameters</p>",
    unsafe_allow_html=True,
)

# Sidebar for inputs
st.sidebar.markdown("<h2 style='color: #4682B4;'>Enter Weather Details:</h2>", unsafe_allow_html=True)

location_input = st.sidebar.selectbox("Select Location:", options=sorted(location_data.keys()))
temperature = st.sidebar.number_input("Temperature (°F):", step=0.1)
humidity = st.sidebar.number_input("Humidity (%):", step=0.1)
wind_speed = st.sidebar.number_input("Wind Speed (km/h):", step=0.1)
precipitation = st.sidebar.number_input("Precipitation (mm):", step=0.1)
cloud_cover = st.sidebar.number_input("Cloud Cover (%):", step=0.1)
pressure = st.sidebar.number_input("Pressure (hPa):", step=0.1)

# Predict button
if st.sidebar.button("🌤️ Predict Rain"):
    # Encode location
    location = location_data.get(location_input.lower(), 0)
    input_features = np.array([[location, temperature, humidity, wind_speed, precipitation, cloud_cover, pressure]])

    # Make a probability prediction
    probabilities = model.predict_proba(input_features)[0]
    prob_rain = probabilities[1] * 100  # Assuming the "Rain" class is the second class (index 1)

    # Display the result
    st.markdown("<h2 style='text-align: center; color: #4682B4;'>Prediction Results</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<h3 style='text-align: center; color: {'#FF6347' if prob_rain >= 50 else '#32CD32'};'>The chances of rain are: {prob_rain:.2f}%</h3>",
        unsafe_allow_html=True,
    )

    # Progress bar for visual representation
    st.progress(int(prob_rain))
else:
    st.markdown("<h4 style='text-align: center;'>Enter details in the sidebar and click 'Predict Rain'</h4>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<footer style='text-align: center; font-size: 0.9em; color: gray;'>Built with ❤️ using Streamlit</footer>",
    unsafe_allow_html=True,
)   