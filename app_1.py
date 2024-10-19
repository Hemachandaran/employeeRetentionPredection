import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
import streamlit as st

# Preprocessing class (as defined previously)
class Preprocessing:
    # ... [Include the entire Preprocessing class here] ...

# Load your dataset from the CSV file and preprocess it.
data_path = "aug_train.csv"  # Update this path if necessary.
data = pd.read_csv(data_path)

# Create an instance of the Preprocessing class and run preprocessing.
preprocessor = Preprocessing(data)
processed_df = preprocessor.preprocess()

# Prepare features and target variable for modeling.
X_combined_features = processed_df.drop(columns=['target'])  
y_encoded_target = processed_df['target']

# Convert combined features into TF-IDF format (if necessary).
X_combined_features['combined_features'] = (
    X_combined_features['city'].astype(str) + ' ' +
    X_combined_features['gender'].astype(str) + ' ' +
    X_combined_features['relevent_experience'].astype(str) + ' '
    # Add other relevant columns as needed...
)

# Define features and target variable again after preprocessing if necessary.
X_final_features = X_combined_features['combined_features']
y_final_target = LabelEncoder().fit_transform(y_encoded_target)

# Split the dataset into training and testing sets.
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final_features, y_final_target, test_size=0.2, random_state=42)

# Convert text to TF-IDF features.
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf_final = vectorizer.fit_transform(X_train_final).toarray()
X_test_tfidf_final = vectorizer.transform(X_test_final).toarray()

# Convert to PyTorch tensors.
X_train_tensor_final = torch.FloatTensor(X_train_tfidf_final)
X_test_tensor_final = torch.FloatTensor(X_test_tfidf_final)
y_train_tensor_final = torch.LongTensor(y_train_final)
y_test_tensor_final = torch.LongTensor(y_test_final)

# Define the neural network model with Batch Normalization.
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)  
        self.dropout1 = nn.Dropout(0.5)
        
    def forward(self,x):
      x=torch.relu(self.bn1(self.fc1(x)))
      x=self.dropout1(x)
      return x

# Initialize the model.
model=SimpleNN(input_dim=X_train_tfidf_final.shape[1])
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=0.001)

# Training the model with more epochs.
num_epochs=40  
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs=model(X_train_tensor_final)
    
    loss=criterion(outputs,y_train_tensor_final)
    loss.backward()
    
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
      print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model's accuracy on the training set.
model.eval()
with torch.no_grad():
    train_outputs=model(X_train_tensor_final)
    _, train_predicted_classes=torch.max(train_outputs.data ,1)

train_accuracy=accuracy_score(y_train_tensor_final.numpy(), train_predicted_classes.numpy())
print(f'Training Accuracy: {train_accuracy:.2f}')

# Evaluate the model's accuracy on the test set.
with torch.no_grad():
    test_outputs=model(X_test_tensor_final)
    _, predicted_classes=torch.max(test_outputs.data ,1)

test_accuracy=accuracy_score(y_test_tensor_final.numpy(), predicted_classes.numpy())
roc_auc=roc_auc_score(y_test_tensor_final.numpy(),torch.softmax(test_outputs ,dim=1)[:,1].numpy())
conf_matrix=confusion_matrix(y_test_tensor_final.numpy(),predicted_classes.numpy())

print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'ROC-AUC: {roc_auc:.2f}')

# Display confusion matrix.
ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification report for test set.
print("Classification Report:")
print(classification_report(y_test_tensor_final.numpy(),predicted_classes.numpy()))

# Function to classify new sentences based on combined features.
def classify_features(features):
    features_tfidf=vectorizer.transform([features]).toarray()
    features_tensor=torch.FloatTensor(features_tfidf)
    
    with torch.no_grad():
      output=model(features_tensor)
      _, predicted_class=torch.max(output.data ,1)
      
    return predicted_class.item()

# Streamlit UI
st.title('Job Change Prediction')
st.write('Choose input method:')
input_method = st.radio("Select Input Method:", ('Professional Summary', 'Manual Input'))

if input_method == 'Professional Summary':
    st.subheader('Professional Summary')
    summary_input = st.text_area("Edit your professional summary here:", 
                                  "A seasoned professional with over 20 years of experience in the STEM field...")
else:
    st.subheader('Manual Input')
    
    city_input = st.selectbox('City:', options=['city_103', 'city_40', 'city_21', 'city_115'])  # Add all city options here
    gender_input = st.selectbox('Gender:', options=['Male', 'Female', 'Other'])
    relevant_experience_input = st.selectbox('Relevant Experience:', options=['Has relevent experience', 'No relevent experience'])
    enrolled_university_input = st.selectbox('Enrolled University:', options=['no_enrollment', 'Full time course', 'Part time course'])
    education_level_input = st.selectbox('Education Level:', options=['Graduate', 'Masters', 'High School'])
    major_discipline_input = st.selectbox('Major Discipline:', options=['STEM', 'Humanities', 'Business Degree'])
    
    experience_input = st.text_input('Experience (in years):')
    
    company_size_input = st.selectbox('Company Size:', options=['<10', '10/49', '50-99', '100-500'])
    
    company_type_input = st.selectbox('Company Type:', options=['Pvt Ltd', 'Funded Startup', 'Public Sector'])
    
    last_new_job_input = st.selectbox('Last New Job:', options=['Not Specified', '<1 year', '>4 years'])
    
    training_hours_input = st.number_input('Training Hours:', min_value=0)

if st.button('Predict'):
    if input_method == 'Professional Summary':
        new_features = summary_input
    else:
        new_features=f"{city_input} {gender_input} {relevant_experience_input} {enrolled_university_input} {education_level_input} {major_discipline_input} {experience_input} {company_size_input} {company_type_input} {last_new_job_input} {training_hours_input}"
    
    prediction_result=classify_features(new_features)
    
    if prediction_result==1:
        st.success('The model predicts that you are looking for a job change.')
    else:
        st.success('The model predicts that you are not looking for a job change.')

        