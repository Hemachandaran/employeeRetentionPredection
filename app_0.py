import streamlit as st
import joblib  # Use joblib for loading LightGBM models
import pickle
from pre_processing import Preprocessing
import torch
import torch.nn as nn
import pandas as pd

# Load the LightGBM model using joblib
try:
    lightgbm_model = joblib.load('lightgbm_model.pkl')
except Exception as e:
    st.error("Error loading LightGBM model: " + str(e))

# Load the pickled text model and vectorizer
try:
    with open('text_model.pkl', 'rb') as model_file:
        model_weights = pickle.load(model_file)

    with open('vectorizer_text.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    st.error("Error loading text model or vectorizer: " + str(e))

def encode_input_data(input_data):
    df = pd.DataFrame(input_data, columns=['enrollee_id', 'city', 'city_development_index', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size','company_type','last_new_job','training_hours'])

    pr = Preprocessing(df)
    in_df = pr.encode_features(df)
    return in_df.values

# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)  # Output layer for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialize the neural network model
input_dim = 472  # Adjust this based on your TF-IDF features
model = SimpleNN(input_dim=input_dim)
model.load_state_dict(model_weights)
model.eval()

# Function for automatic classification
def automatic_classification(input_text):
    features_tfidf = loaded_vectorizer.transform([input_text]).toarray()
    features_tensor = torch.FloatTensor(features_tfidf)

    with torch.no_grad():
        output = model(features_tensor)
        _, predicted_class = torch.max(output.data, 1)

    return predicted_class.item()

# Function for manual classification using LightGBM
def manual_classification(features):
    features_df = pd.DataFrame([features])
    pr = Preprocessing()
    encode_df = (features_df)

    # Debugging outputs
    st.write("Shape of input data:", features_df.shape)
    st.write("Input data:", features_df)
    st.write("Shape of encoded input data:", encode_df.shape)
    st.write("Encoded input data:", encode_df)

    print(f"""
          ----------------------------------
          shape: {features_df.shape}\n
          features_df: \n
          {features_df.columns.to_list()}
          {features_df.values}\n
          shape: {encode_df.shape}\n
          data : {encode_df}\n
          -------------------------------- 
          """)

    # Ensure encode_df is not empty
    if encode_df.empty:
        st.error("Encoded data is empty. Please check your input.")
        return

    # Check for missing values
    if encode_df.isnull().values.any():
        st.error("Input data contains missing values.")
        return

    # Make prediction
    prediction = lightgbm_model.predict(encode_df)
    return prediction
    
    
    
    

# Streamlit app layout
st.title("Classification App")

classification_type = st.selectbox("Choose classification type:", ["Automatic", "Manual"])

if classification_type == "Automatic":
    input_text = st.text_area("Enter text for classification:")
    
    if st.button("Classify"):
        if input_text:
            result = automatic_classification(input_text)
            st.success(f"Prediction: {result}")
        else:
            st.error("Please enter some text to classify.")

elif classification_type == "Manual":
    # Get user inputs for each feature
    enrollee_id = st.text_input("Enrollee ID:")
    
    city_development_index = st.number_input("City Development Index:", min_value=0.0, format="%.2f")
    
    city = st.selectbox("City:", ['city_103', 'city_40', 'city_21', 'city_115', 'city_162',
                                   'city_176', 'city_160', 'city_46', 'city_61', 'city_114',
                                   'city_13', 'city_159', 'city_102', 'city_67', 'city_100',
                                   'city_16', 'city_71', 'city_104', 'city_64', 'city_101',
                                   'city_83', 'city_105', 'city_73', 'city_75', 'city_41',
                                   'city_11', 'city_93', 'city_90', 'city_36', 'city_20',
                                   'city_57', 'city_152', 'city_19', 'city_65', 'city_74',
                                   'city_173', 'city_136', 'city_98', 'city_97', 'city_50',
                                   'city_138', 'city_82', 'city_157', 'city_89'])
    
    gender = st.selectbox("Gender:", ['Male', 'Not_specified', 'Female', 'Other'])
    
    relevant_experience = st.selectbox("Relevant Experience:", ['Has relevant experience',
                                                                'No relevant experience'])
    
    enrolled_university = st.selectbox("Enrolled University:", ['no_enrollment',
                                                                'Full time course',
                                                                'none',
                                                                'Part time course'])
    
    education_level = st.selectbox("Education Level:", ['Graduate',
                                                         'Masters',
                                                         'High School',
                                                         'Other',
                                                         'Phd',
                                                         'Primary School'])
    
    major_discipline = st.selectbox("Major Discipline:", ['STEM',
                                                           'Business Degree',
                                                           'Not_Specified',
                                                           'Arts',
                                                           'Humanities',
                                                           'No Major',
                                                           'Other'])
    
    experience = st.selectbox("Experience:", ['>20','15','5','<1','11','13','7','17','2','16','1','4','10','14','18','19','12','3','6','9','8','20'])
    
    company_size = st.selectbox("Company Size:", ['NS','50-99','<10','10000+','5000-9999',
                                                   '1000-4999','10/49','100-500','500-999'])
    
    company_type = st.selectbox("Company Type:", ['not_specified','Pvt Ltd','Funded Startup',
                                                   'Early Stage Startup','Other',
                                                   'Public Sector','NGO'])
    
    last_new_job = st.selectbox("Last New Job:", ['1','>4','never','4','3','2','Not Specified'])

    training_hours = st.number_input("Training Hours", min_value=0)

    if st.button("Classify"):
        # Prepare features dictionary including all required fields
        features = {
            "enrollee_id": enrollee_id,
            "city": city,
            "city_development_index": city_development_index,
            "gender": gender,
            "relevent_experience": relevant_experience,
            "enrolled_university": enrolled_university,
            "education_level": education_level,
            "major_discipline": major_discipline,
            "experience": experience,
            "company_size": company_size,
            "company_type": company_type,
            "last_new_job": last_new_job,
            "training_hours": training_hours
        }
        
        # Check if all required fields are filled before classifying
        if any(value == "" for value in features.values()):
            st.error("Please fill in all fields.")
        else:
            result = manual_classification(features)
            st.success(f"Prediction: {result[0]}")