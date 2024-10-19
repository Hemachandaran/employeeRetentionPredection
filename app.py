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

class Preprocessing:
    def __init__(self, df):
        self.df = df
        self.imputed_df = None
        self.balance_df = None
        self.featureEncoded_df = None
        self.target_mean_feature = []
        self.categorical_features_indices = []

    def handle_nulls(self):
        """Fill null values in specific columns."""
        self.df["enrolled_university"] = self.df["enrolled_university"].fillna("none")
        self.df["education_level"] = self.df["education_level"].fillna("Other")

        mode = self.df["experience"].mode()[0]
        self.df["experience"] = self.df["experience"].fillna(mode)

        self.df["last_new_job"] = self.df["last_new_job"].fillna("Not Specified")
        self.df['major_discipline'] = self.df['major_discipline'].fillna("Not_Specified")
        self.df["gender"] = self.df["gender"].fillna("Not_specified")
        self.df["company_size"] = self.df["company_size"].fillna("NS")
        self.df["company_type"] = self.df["company_type"].fillna("not_specified")
        
        # Store the imputed DataFrame
        self.imputed_df = self.df
        return self.df

    def encode_features(self):
        """Encode categorical features with target mean."""
        features = ['gender', "enrolled_university", "major_discipline",
                    "education_level", "company_type", "city"]

        for i, feature in enumerate(features):
            # Calculate mean target for each category and map it to the feature
            self.target_mean_feature.append(self.df.groupby(feature)['target'].mean())
            self.df[feature] = self.df[feature].map(self.target_mean_feature[i])

        rel_exp={'Has relevent experience':1,'No relevent experience':0}
        # Map relevant experience to binary values.
        self.df["relevent_experience"] = self.df["relevent_experience"].map(rel_exp)

        # Map company size categories to numerical values using map().
        size_mapping = {
            'NS': 1, '<10': 2, '10/49': 3,
            '50-99': 4, '100-500': 5, '500-999': 6,
            '1000-4999': 7, '5000-9999': 8, '10000+': 9
        }
        self.df["company_size"] = self.df["company_size"].map(size_mapping)

        # Map last new job categories to numerical values.
        u = self.df["last_new_job"].unique().tolist()
        u.sort()
        un = [2, 3, 4, 5, 6, 0, 1]

        for i, j in zip(u, un):
            self.df["last_new_job"] = self.df["last_new_job"].replace(i, j)

        # Map experience categories to numerical values.
        experience_mapping = {
            '<1': 0, '1': 1, '2': 2,
            '3': 3, '4': 4, '5': 5,
            '6': 6, '7': 7, '8': 8,
            '9': 9, '10': 10, '11': 11,
            '12': 12, '13': 13,
            '14': 14, '15': 15,
            '16': 16, '17': 17,
            '18': 18, '19': 19,
            '20': 20,'>20': 21,
        }
        
        for i,j in zip(experience_mapping.keys(), experience_mapping.values()):
            self.df["experience"] = self.df["experience"].replace(i,j)

        # Store the encoded DataFrame
        self.featureEncoded_df = self.df
        return self.df

    def balance_data(self):
        """Balance the dataset using SMOTENC."""
        X = self.df.drop("target", axis=1)
        y = self.df["target"]

        categorical_features_indices = [
            self.df.columns.get_loc(col) for col in [
                'city', 'gender', 'relevent_experience',
                'enrolled_university', 'education_level',
                'major_discipline', 'experience',
                'company_size', 'company_type',
                'last_new_job'
            ]
        ]

        smote_nc = SMOTENC(categorical_features=categorical_features_indices, random_state=42)
        
        X_resampled, y_resampled = smote_nc.fit_resample(X, y)

        # Concatenate the resampled data back into a DataFrame
        self.balance_df = pd.concat([X_resampled, y_resampled], axis=1)

        # Store the balanced DataFrame back into self.df
        self.df = self.balance_df
        return self.balance_df

    def preprocess(self):
        """Run all preprocessing steps."""
        print("Handling null values...")
        self.handle_nulls()
        
        print("Encoding features...")
        self.encode_features()
        
        print("Balancing data...")
        return self.balance_data()

# Load your dataset from the CSV file and preprocess it.
data_path = "aug_train.csv" # Update this path if necessary.
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

# Example usage:
st.title('Job Change Prediction')
st.write('Enter details below:')

city_input = st.selectbox('City:', options=['city_103', 'city_40', ...]) # Add all city options here
gender_input = st.selectbox('Gender:', options=['Male', 'Female', 'Other'])
relevant_experience_input = st.selectbox('Relevant Experience:', options=['Has relevent experience', 'No relevent experience'])
enrolled_university_input = st.selectbox('Enrolled University:', options=['no_enrollment', ...]) # Add options here
education_level_input = st.selectbox('Education Level:', options=['Graduate', ...]) # Add options here
major_discipline_input = st.selectbox('Major Discipline:', options=['STEM', ...]) # Add options here
experience_input=st.text_input('Experience (in years):')
company_size_input=st.selectbox('Company Size:', options=['<10', ...]) # Add options here
company_type_input=st.selectbox('Company Type:', options=['Pvt Ltd', ...]) # Add options here
last_new_job_input=st.selectbox('Last New Job:', options=['Not Specified', ...]) # Add options here
training_hours_input=st.number_input('Training Hours:', min_value=0)

if st.button('Predict'):
    new_features=f"{city_input} {gender_input} {relevant_experience_input} {enrolled_university_input} {education_level_input} {major_discipline_input} {experience_input} {company_size_input} {company_type_input} {last_new_job_input} {training_hours_input}"
    
    prediction_result=classify_features(new_features)
    
    if prediction_result==1:
      st.success('The model predicts that you are looking for a job change.')
    else:
      st.success('The model predicts that you are not looking for a job change.')
