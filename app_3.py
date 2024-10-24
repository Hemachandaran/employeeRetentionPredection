import streamlit as st
import joblib
import pandas as pd
from pre_processing import Preprocessing

# Load the model
model = joblib.load('lightgbm_model.pkl')

# Function to encode categorical variables if needed
def encode_input_data(input_data):
    df = pd.DataFrame(input_data, columns=[
        'enrollee_id', 'city', 'city_development_index', 'gender',
        'relevent_experience', 'enrolled_university', 'education_level',
        'major_discipline', 'experience', 'company_size',
        'company_type', 'last_new_job', 'training_hours'
    ])

    pr = Preprocessing(df)
    in_df = pr.handle_nulls()
    in_df = pr.encode_features()
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(in_df)
    
    return df_encoded.values

# User inputs
st.title("Employee Retention Prediction")

enrollee_id = st.number_input("Enrollee ID", min_value=1)
city = st.selectbox("Select City", [
    'city_103', 'city_40', 'city_21', 'city_115', 'city_162',
    'city_176', 'city_160', 'city_46', 'city_61', 'city_114',
    'city_13', 'city_159', 'city_102', 'city_67', 'city_100',
    'city_16', 'city_71', 'city_104', 'city_64', 'city_101',
    'city_83', 'city_105', 'city_73', 'city_75', 'city_41',
    'city_11', 'city_93', 'city_90', 'city_36', 'city_20',
    'city_57', 'city_152', 'city_19', 'city_65', 'city_74',
    'city_173', 'city_136', 'city_98', 'city_97', 'city_50',
    'city_138', 'city_82', 'city_157', 'city_89','city_150',
    # Add remaining cities...
])
city_development_index = st.number_input("City Development Index", min_value=0.0, max_value=1.0, step=0.01)
gender = st.selectbox("Select Gender", ['Male', 'Not_specified', 'Female', 'Other'])
relevant_experience = st.selectbox("Relevant Experience", ['Has relevant experience', 'No relevant experience'])
enrolled_university = st.selectbox("Enrolled University", ['no_enrollment', 
                                                             'Full time course',
                                                             'none',
                                                             'Part time course'])
education_level = st.selectbox("Education Level", ['Graduate', 
                                                    'Masters',
                                                    'High School',
                                                    'Other',
                                                    "PhD",
                                                    "Primary School"])
major_discipline = st.selectbox("Major Discipline", ['STEM','Business Degree',
                                                      "Not_Specified",
                                                      "Arts",
                                                      "Humanities",
                                                      "No Major",
                                                      "Other"])
experience = st.selectbox("Years of Experience", ['>20','15','5','<1','11','13','7',
                                                    '17','2','16','1','4','10','14',
                                                    "18","19","12","3","6","9","8","20"])
company_size = st.selectbox("Company Size", ['NS','50-99','<10','10000+',
                                               "5000-9999","1000-4999",
                                               "10/49","100-500","500-999"])
company_type = st.selectbox("Company Type", ['not_specified','Pvt Ltd',
                                               "Funded Startup",
                                               "Early Stage Startup",
                                               "Other",
                                               "Public Sector",
                                               "NGO"])
last_new_job = st.selectbox("Last New Job Duration", ['1','>4','never','4','3','2','Not Specified'])
training_hours = st.number_input("Training Hours", min_value=0)

# Button to make prediction
if st.button('Predict'):
    # Prepare input data for prediction
    input_data = [[
        enrollee_id, city, city_development_index, gender,
        relevant_experience, enrolled_university, education_level,
        major_discipline, experience, company_size,
        company_type, last_new_job, training_hours
    ]]
    
    # Encode input data
    encoded_input = encode_input_data(input_data)

    # Make prediction
    try:
        prediction = model.predict(encoded_input)
        st.write(f"The predicted class is: {prediction[0]}")
    except Exception as e:
        st.write(f"Error during prediction: {e}")