import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt




def text_model(data: pd.DataFrame, target_feature: str, num_epochs: int = 10):
    # Prepare the features by concatenating relevant columns into a single string
    data['combined_features'] = (
        data['city'].astype(str).fillna('') + ' ' +
        data['gender'].astype(str).fillna('') + ' ' +
        data['relevent_experience'].astype(str).fillna('') + ' ' +
        data['enrolled_university'].astype(str).fillna('') + ' ' +
        data['education_level'].astype(str).fillna('') + ' ' +
        data['major_discipline'].astype(str).fillna('') + ' ' +
        data['experience'].astype(str).fillna('') + ' ' +
        data['company_size'].astype(str).fillna('') + ' ' +
        data['company_type'].astype(str).fillna('') + ' ' +
        data['last_new_job'].astype(str).fillna('') + ' ' +
        data['training_hours'].astype(str).fillna('')
    )

    # Define features and target variable
    X = data['combined_features']
    y = LabelEncoder().fit_transform(data[target_feature])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_tfidf)
    X_test_tensor = torch.FloatTensor(X_test_tfidf)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    # Define the neural network model
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

    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_dim=X_train_tfidf.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)

        # Compute loss and backpropagate
        loss = criterion(outputs, y_train_tensor)
        loss.backward()

        # Update weights
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model's accuracy on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted_classes = torch.max(test_outputs.data, 1)

    # Calculate metrics
    accuracy = accuracy_score(y_test_tensor.numpy(), predicted_classes.numpy())
    roc_auc = roc_auc_score(y_test_tensor.numpy(), torch.softmax(test_outputs, dim=1)[:, 1].numpy())
    conf_matrix = confusion_matrix(y_test_tensor.numpy(), predicted_classes.numpy())

    print(f'Accuracy: {accuracy:.2f}')
    print(f'ROC-AUC: {roc_auc:.2f}')

    # Display confusion matrix
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Function to classify new sentences based on combined features
    def classify_features(features):
        features_tfidf = vectorizer.transform([features]).toarray()
        features_tensor = torch.FloatTensor(features_tfidf)

        with torch.no_grad():
            output = model(features_tensor)
            _, predicted_class = torch.max(output.data, 1)

        return predicted_class.item()

    return classify_features

# Example usage:
# df = pd.read_csv('aug_train.csv')
# classifier_function = train_and_evaluate_model(df, target_feature='target')
# result = classifier_function("New input features here")