import pickle
from datetime import datetime
import openpyxl
import pandas as pd
import streamlit as st
import math
import category_encoders
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def main(x=None):
    st.title("Car Price Predictor")

    # Add your GUI components here
    st.write("Predict the Price of Your Used Car")

    brandModelDF = pd.read_excel('BrandModel.xlsx')
    companies = brandModelDF.columns.tolist()
    brand = st.selectbox("Select your Car Company:", companies)

    models = brandModelDF[brand].tolist()
    models = [x for x in models if isinstance(x, (str))]
    model = st.selectbox("Select your Car Brand:", models)

    modelVariantDF = pd.read_excel('ModelVariant.xlsx')
    variants = modelVariantDF[model].values.tolist()
    variants = [x for x in variants if isinstance(x, (str))]
    variant = st.selectbox("Select your Car Variant:", variants)

    current_year = datetime.now().year
    year_range = list(range(current_year, 1999, -1))
    year = st.selectbox("Select a year:", year_range)

    fuel = st.selectbox("Select Type of Fuel:", ["Petrol", "Diesel", "LPG", "CNG"])
    seller_type = st.selectbox("Select Type of Fuel:", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Select Type of Transmission:", ["Manual", "Automatic"])
    owner = st.selectbox("Select Type of Owner:", ["First Owner", "Second Owner", "Third Owner", "Fourth Owner"])
    mileage = st.number_input("Enter the Mileage", min_value=1, max_value=100, value=20)
    km_driven = st.number_input("Enter the Kms Driven by the Car:", min_value=0, max_value=1000000, value=0)
    seat = st.number_input("Enter the Seating Capacity", min_value=1, max_value=10, value=5)
    engine = st.number_input("Enter the Engine Power", max_value=3000)


    submit_button = st.button("Predict")

    target_encoder = pickle.load(open('encoder.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    modelPredictor = pickle.load(open('model.pkl', 'rb'))
    name = brand + " " + model + " " + variant
        
    if submit_button:
        transFeatures = ['brand', 'model', 'variant', 'name', 'fuel', 'seller_type', 'transmission', 'owner']
        int_list = ["year", "km_driven", "seats", "engine"]
        float_list = ['mileage']

        def transform(df):
            try:
                df[transFeatures] = target_encoder.transform(df[transFeatures])
            except Exception as e:
                print(e)

            for i in int_list:
                df[i] = df[i].astype("int64")

            for j in float_list:
                df[j] = df[j].astype("float")

        def make_prediction(df):
            X = df.drop(['model', 'variant'], axis=1)
            X = scaler.transform(X)

            pred = modelPredictor.predict(X)
            return pred

        data = pd.DataFrame([[brand, model, variant, name, fuel, seller_type, transmission, owner, year, km_driven, seat, mileage, engine]], columns=['brand', 'model', 'variant', 'name', 'fuel', 'seller_type', 'transmission', 'owner', 'year', 'km_driven', 'seats', 'mileage', 'engine'])

        # Transforming Data
        transform(data)
        # Making Prediction
        pred = make_prediction(data)

        st.write(f"Price of this will be : {pred[0]}")
        # Perform any action or computation here


if __name__ == "__main__":
    main()
