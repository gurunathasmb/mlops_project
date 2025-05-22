import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('mobile phone price prediction.csv')

    # Clean Price column
    df['Price'] = df['Price'].astype(str).str.replace(',', '', regex=False).astype(float)

    # Features and target
    X = df.drop(columns=['Price'])
    y = df['Price']

    # Columns to clean
    columns_to_clean = [
        'Ram', 'Battery', 'Display', 'Camera', 'External_Memory',
        'Inbuilt_memory', 'Screen_resolution', 'Processor'
    ]

    for col in columns_to_clean:
        X[col] = X[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X['fast_charging_wattage'] = X['fast_charging'].astype(str).str.extract(r'(\d+)W').astype(float)

    # Drop rows with missing values
    columns_to_check_for_nan = columns_to_clean + ['fast_charging_wattage']
    X.dropna(subset=columns_to_check_for_nan, inplace=True)
    y = y[X.index]
    X = X.drop(columns=['fast_charging'])

    return X, y

def train_model(X, y):
    categorical_features = ['company', 'Processor_name', 'No_of_sim']
    numerical_features = [
        'Rating', 'Spec_score', 'Ram', 'Battery', 'Display',
        'Camera', 'External_Memory', 'Android_version',
        'Inbuilt_memory', 'fast_charging_wattage', 'Screen_resolution', 'Processor'
    ]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X, y)
    return model

# Load data and train model once
X, y = load_data()
model = train_model(X, y)

# --- Streamlit UI ---
st.title("📱 Mobile Price Predictor")

st.write("Enter the mobile specifications to predict the price:")

# Form
with st.form("prediction_form"):
    Rating = st.number_input("Rating", min_value=0.0, step=0.1)
    Spec_score = st.number_input("Spec Score", min_value=0.0, step=0.1)
    No_of_sim = st.selectbox("Number of SIMs", X['No_of_sim'].unique())
    Ram = st.number_input("RAM (MB)", min_value=0.0, step=1.0)
    Battery = st.number_input("Battery (mAh)", min_value=0.0, step=1.0)
    Display = st.number_input("Display (inches)", min_value=0.0, step=0.1)
    Camera = st.number_input("Camera (MP)", min_value=0.0, step=1.0)
    External_Memory = st.number_input("External Memory (GB)", min_value=0.0, step=1.0)
    Android_version = st.number_input("Android Version", min_value=1.0, step=0.1)
    Inbuilt_memory = st.number_input("Inbuilt Memory (GB)", min_value=0.0, step=1.0)
    fast_charging_wattage = st.number_input("Fast Charging (W)", min_value=0.0, step=1.0)
    Screen_resolution = st.number_input("Screen Resolution (pixels)", min_value=0.0, step=1.0)
    Processor = st.number_input("Processor Speed (GHz)", min_value=0.0, step=0.1)
    company = st.selectbox("Company", X['company'].unique())
    Processor_name = st.selectbox("Processor Name", X['Processor_name'].unique())

    submitted = st.form_submit_button("Predict Price")

    if submitted:
        input_data = pd.DataFrame([{
            'Rating': Rating,
            'Spec_score': Spec_score,
            'No_of_sim': No_of_sim,
            'Ram': Ram,
            'Battery': Battery,
            'Display': Display,
            'Camera': Camera,
            'External_Memory': External_Memory,
            'Android_version': Android_version,
            'Inbuilt_memory': Inbuilt_memory,
            'fast_charging_wattage': fast_charging_wattage,
            'Screen_resolution': Screen_resolution,
            'Processor': Processor,
            'company': company,
            'Processor_name': Processor_name
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Mobile Price: ₹{prediction:,.2f}")
