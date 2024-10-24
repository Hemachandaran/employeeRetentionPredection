# text_class/model_and_preprocessing.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from pre_processing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

# Define the Simple Neural Network model
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

# Class for combining features and preprocessing
class comb_fet:
    def __init__(self, data):
        self.data = data

    def prepare_features(self):
        self.data['combined_features'] = (
            self.data['city'].fillna('') + ' ' +
            self.data['gender'].fillna('') + ' ' +
            self.data['relevent_experience'].fillna('') + ' ' +
            self.data['enrolled_university'].fillna('') + ' ' +
            self.data['education_level'].fillna('') + ' ' +
            self.data['major_discipline'].fillna('') + ' ' +
            self.data['experience'].fillna('') + ' ' +
            self.data['company_size'].fillna('') + ' ' +
            self.data['company_type'].fillna('') + ' ' +
            self.data['last_new_job'].fillna('') + ' ' +
            self.data['training_hours'].astype(str).fillna('')
        )
        return LabelEncoder().fit_transform(self.data['target'])

# Class to handle model training and feature classification
class ModelHandler:
    def __init__(self):
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def train_model(self, df, target, num_epochs=78):
        # Load the dataset and preprocess it
        data = df

        # Assuming Preprocessing is defined elsewhere
        pr = Preprocessing(data)
        data = pr.handle_nulls()
        data = pr.handle_imbalance()
        
        # Prepare features and target variable
        x = comb_fet(data)
        x.prepare_features()
        
        X = x.data['combined_features']
        y = LabelEncoder().fit_transform(x.data[target])
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert text to TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = self.vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = self.vectorizer.transform(X_test).toarray()

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_tfidf)
        X_test_tensor = torch.FloatTensor(X_test_tfidf)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)

        # Initialize the model and optimizer
        self.model = SimpleNN(input_dim=X_train_tfidf.shape[1])
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training the model
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            outputs = self.model(X_train_tensor)

            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model's accuracy on the test set
        self.evaluate_model(X_test_tensor, y_test_tensor)

    def evaluate_model(self, X_test_tensor, y_test_tensor):
        """Evaluate the trained model on the test set."""
        
        self.model.eval()
        
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor)
            _, predicted_classes = torch.max(test_outputs.data, 1)

        accuracy = accuracy_score(y_test_tensor.numpy(), predicted_classes.numpy())
        
        roc_auc = roc_auc_score(y_test_tensor.numpy(), torch.softmax(test_outputs, dim=1)[:, 1].numpy())
        
        print(f'Accuracy: {accuracy:.2f}')
        print(f'ROC-AUC: {roc_auc:.2f}')

        conf_matrix = confusion_matrix(y_test_tensor.numpy(), predicted_classes.numpy())
        
        ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

    def classify_features(self, features):
        
        with open('vectorizer_text.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        features_tfidf = vectorizer.transform([features]).toarray()
        
        features_tensor = torch.FloatTensor(features_tfidf)

        with torch.no_grad():
            output = self.model(features_tensor)  # Use the shared model
            _, predicted_class = torch.max(output.data, 1)

        return predicted_class.item()
    
    def save_model(self):
         
        """Save the model and vectorizer to disk using pickle."""
        with open('text_model.pkl', 'wb') as model_file:
            pickle.dump(self.model.state_dict(), model_file)

        with open('vectorizer_text.pkl', 'wb') as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)

"""# Example usage:
if __name__ == "__main__":
    # Load your dataset here (df), ensure it has a target column.
    df = pd.read_csv('your_dataset.csv')
    
    handler = ModelHandler()
    handler.train_model(df, target='your_target_column')
    
    result = handler.classify_features("Some input text for classification")
    print(f'Predicted class: {result}')"""