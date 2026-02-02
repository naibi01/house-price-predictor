import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETUP
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 Real Estate Price Predictor")

# 2. LOAD MODEL
@st.cache_resource
def load_model():
    try:
        return joblib.load('house_price_model.pkl')
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("Error: 'house_price_model.pkl' not found. Make sure it's in the same folder as this app.py file!")
    st.stop()

# 3. TABS (This is where tab1 and tab2 are defined!)
tab1, tab2 = st.tabs(["🔮 Predict Single House", "📈 Future Forecast"])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    st.header("Enter House Details")
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sq. ft)", 500, 20000, 4000)
        bedrooms = st.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.slider("Bathrooms", 1, 5, 2)

    with col2:
        stories = st.selectbox("Stories", [1, 2, 3, 4])
        parking = st.slider("Parking Spaces", 0, 5, 1)
        year = st.number_input("Year Built", 2000, 2065, 2024)

    if st.button("Predict Price", type="primary"):
        input_data = pd.DataFrame({
            'area': [area], 'bedrooms': [bedrooms], 'bathrooms': [bathrooms],
            'stories': [stories], 'parking': [parking], 'Year': [year]
        })
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ${prediction:,.2f}")

# --- TAB 2: FUTURE FORECAST ---
with tab2:
    st.header("Forecast Future Prices")
    uploaded_file = st.file_uploader("Upload 'Future Housing Dataset.csv'", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Ensure correct columns exist
        needed_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'Year']
        if all(col in df.columns for col in needed_cols):
            
            # Predict
            df['Predicted_Price'] = model.predict(df[needed_cols])
            
            st.subheader("Results")
            st.dataframe(df.head())
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=df.groupby('Year')['Predicted_Price'].mean().reset_index(), 
                         x='Year', y='Predicted_Price', marker='o', color='purple', ax=ax)
            ax.set_title("Price Trend (2026-2065)")
            st.pyplot(fig)
        else:
            st.error(f"CSV is missing columns. Needed: {needed_cols}")