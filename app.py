import streamlit as st
import pickle
import pandas as pd

# Load the full pipeline
model = pickle.load(open('california_knn_pipeline.pkl', 'rb'))
st.title('California Housing Price Predictor')
st.write('Enter the details below to get a predicted house value.')

st.subheader('Property details')

col1, col2 = st.columns(2)
with col1:
    MedInc     = st.number_input('Median Income',      value=3.5)
    HouseAge   = st.number_input('House Age',           value=25.0)
    AveRooms   = st.number_input('Average Rooms',       value=5.2)
    AveBedrms  = st.number_input('Average Bedrooms',    value=1.1)
with col2:
    Population = st.number_input('Population',          value=1200.0)
    AveOccup   = st.number_input('Average Occupancy',   value=2.8)
    Latitude   = st.number_input('Latitude',            value=34.1)
    Longitude  = st.number_input('Longitude',           value=-118.3)

if st.button('Predict House Value'):
    input_data = pd.DataFrame([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]], columns=model.feature_names_in_)
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: ${prediction[0] * 100:.0f},000")