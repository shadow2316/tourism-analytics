# TOURISM ML DASHBOARD
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("🌍 Tourism Experience Analytics")
st.markdown("**Predict Ratings & Recommend Attractions**")

# Load models + data
@st.cache_data
def load_models():
    rf_reg = joblib.load('rf_reg.pkl')
    rf_clf = joblib.load('rf_clf.pkl')
    master_df = pd.read_csv('master_df.csv')
    return rf_reg, rf_clf, master_df

rf_reg, rf_clf, master_df = load_models()

# User Input
continent = st.selectbox("Continent", [1,2,3])
region = st.selectbox("Region", [10,11,12])
country = st.selectbox("Country", [100,101,102])
city = st.selectbox("City", [1000,1001,1002])
month = st.slider("Visit Month", 1,12,7)
year = st.selectbox("Year", [2018,2019])
mode = st.selectbox("Travel Mode", [1,2,3,4])  # 1=Business,etc
attr_type = st.selectbox("Attraction Type", [10,11,12])

if st.button("🚀 PREDICT"):
    # Predict Rating
    features = [[continent, region, country, city, month, year, mode, attr_type]]
    rating = rf_reg.predict(features)[0]
    
    st.success(f"⭐ **Predicted Rating: {rating:.1f}/5**")
    
    # Recommend (Top similar from master_df)
    similar = master_df.iloc[0:5][['CityId', 'AttractionTypeId', 'Rating']]
    st.subheader("🔥 Top 5 Recommendations")
    st.dataframe(similar)
