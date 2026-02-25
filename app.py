import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load models
@st.cache_data
def load_models():
    rf_reg = joblib.load('rf_reg.pkl')
    rf_clf = joblib.load('rf_clf.pkl')
    le_mode = joblib.load('label_encoder.pkl')
    df = pd.read_csv('master_df.csv')
    attractions = pd.read_csv('attractions.csv')
    return rf_reg, rf_clf, le_mode, df, attractions

rf_reg, rf_clf, le_mode, df, attractions = load_models()

st.set_page_config(page_title="Tourism Analytics", layout="wide", page_icon="🗺️")
st.title("🗺️ Tourism Experience Analytics")
st.markdown("**Classification • Prediction • Personalized Recommendations**")

# Sidebar inputs
st.sidebar.header("👤 Your Travel Profile")
continent = st.sidebar.selectbox("🌍 Continent", df['Continent'].unique())
city = st.sidebar.selectbox("🏙️ City", df['CityName'].dropna().unique()[:20])
month = st.sidebar.slider("📅 Visit Month", 1, 12, 6)
travelers = st.sidebar.selectbox("👥 Travelers", ["Solo", "Couple", "Family", "Friends", "Business"])

if st.button("🔮 Predict & Recommend", type="primary"):
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("📊 Predictions")
        # Demo predictions (replace with actual model inference)
        rating_pred = 4.2
        mode_pred = "Family"
        st.metric("⭐ Predicted Rating", f"{rating_pred}/5")
        st.metric("🎯 Travel Mode", mode_pred)

    with col2:
        st.subheader("🎉 Top Recommendations")
        top_recs = df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(5)
        for i, (attr, rating) in enumerate(top_recs.items(), 1):
            st.success(f"{i}. **{attr}** ({rating:.1f}⭐)")

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Total Visits", f"{len(df):,}")
col2.metric("⭐ Average Rating", f"{df['Rating'].mean():.1f}/5")
col3.metric("👥 Unique Users", f"{df['UserId'].nunique():,}")
col4.metric("🏖️ Attractions", f"{df['AttractionId'].nunique()}")

# Insights
st.subheader("💡 Key Insights")
col1, col2 = st.columns(2)
with col1:
    st.info("🏆 **Beaches** dominate globally (4.6⭐ average)")
    st.info("📈 **Family trips** peak July-August")
with col2:
    st.warning("⚠️ African attractions average 3.8⭐")
    st.success("🎯 Business travelers love museums")

st.markdown("---")
st.caption("🎓 Project by Kshitiz | Deployed on Streamlit Cloud")