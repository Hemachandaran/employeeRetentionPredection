import streamlit as st
import pickle
import torch
import torch.nn as nn
import numpy as np

# Load the pickled model and vectorizer
with open('text_model.pkl', 'rb') as model_file:
    model_weights = pickle.load(model_file)

with open('vectorizer_text.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Define the neural network class (same as in your training script)
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

# Initialize the model
input_dim = 472    # Adjust this based on your TF-IDF features
model = SimpleNN(input_dim=input_dim)
model.load_state_dict(model_weights)
model.eval()

# Function to classify new sentences based on combined features
def classify_features(features):
    features_tfidf = loaded_vectorizer.transform([features]).toarray()
    features_tensor = torch.FloatTensor(features_tfidf)

    with torch.no_grad():
        output = model(features_tensor)
        _, predicted_class = torch.max(output.data, 1)

    return predicted_class.item()

# Streamlit UI
st.title("Job Candidate Classification")
st.write("Enter candidate details below:")

# Input fields for user to fill in
city = st.text_input("City")
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
relevent_experience = st.selectbox("Relevant Experience", ["Yes", "No"])
enrolled_university = st.selectbox("Enrolled University", ["No Enrollment", "Part-time", "Full-time"])
education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
major_discipline = st.selectbox("Major Discipline", ["STEM", "Business", "Arts", "Humanities"])
experience = st.text_input("Experience (in years)")
company_size = st.selectbox("Company Size", ["Small (<10)", "Medium (10-50)", "Large (>50)"])
company_type = st.selectbox("Company Type", ["Private", "Public"])
last_new_job = st.text_input("Last New Job (in years)")
training_hours = st.number_input("Training Hours")

# Button to classify
if st.button("Classify"):
    # Prepare combined features string
    combined_features = f"{city} {gender} {relevent_experience} {enrolled_university} {education_level} {major_discipline} {experience} {company_size} {company_type} {last_new_job} {training_hours}"
    
    # Classify the input features
    prediction = classify_features(combined_features)

    # Display result
    if prediction == 0:
        st.success("The candidate is likely to be hired.")
    else:
        st.error("The candidate is unlikely to be hired.")