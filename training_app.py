from src.pipeline import (data_obj, model_obj, transformer)
import streamlit as st
import pandas as pd
import pickle as pkl

# Load the data for the dropdowns
data = pkl.load(open(data_obj, 'rb'))

# Load the trained model and transformer
model = pkl.load(open(model_obj, 'rb'))
transformer = pkl.load(open(transformer, "rb"))

st.title("HEART STROKE PREDICTION")

# Age column
age = st.number_input("AGE", step=1, value=1)
age = int(age)

# Sex column
hypertension = st.selectbox('HYPER_TENSION (0->NO AND 1->YES)', data['hypertension'].unique())

# Heart disease column
heart_disease = st.selectbox("HEART DISEASE (0->NO AND 1->YES)", data['heart_disease'].unique())

# Ever married
ever_married = st.selectbox("EVER MARRIED", data['ever_married'].unique())

# Work type column
work_type = st.selectbox('WORK TYPE', data['work_type'].unique())

# Avg glucose level
avg_glucose_level = st.number_input('AVERAGE GLUCOSE LEVEL')

# BMI
bmi = st.number_input('BMI (BODY MASS INDEX)')

# Smoking status
smoking_status = st.selectbox('SMOKING STATUS (0->NO AND 1->YES)', data['smoking_status'].unique())

# Button for prediction
if st.button('PREDICT STROKE'):
    # Create a DataFrame with the input data
    query_data = {'age': [age],
                  'hypertension': [hypertension],
                  'heart_disease': [heart_disease],
                  'ever_married': [ever_married],
                  'work_type': [work_type],
                  'avg_glucose_level': [avg_glucose_level],
                  'bmi': [bmi],
                  'smoking_status': [smoking_status]}

    df = pd.DataFrame(query_data)

    # Perform the same transformations as during training
    df = transformer.transform(df)

    # Query point
    y_pred = model.predict(df)

    # Display the prediction
    if y_pred[0] == 1:
        st.header(f"PERSON MAY GET HEART STROKE")
    else:
        st.header(f"NOTHING TO WORRY, THIS PERSON IS FINE")
