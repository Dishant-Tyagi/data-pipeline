# ml.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Scikit-learn ML models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import torch
import torch.nn as nn
import torch.optim as optim

def choose_model():
    models = {
        "1": "Logistic Regression",
        "2": "Linear Regression",
        "3": "Decision Tree (Classifier)",
        "4": "Decision Tree (Regressor)",
        "5": "Random Forest (Classifier)",
        "6": "Random Forest (Regressor)",
        "7": "Gradient Boosting",
        "8": "Support Vector Classifier",
        "9": "Support Vector Regressor",
        "10": "Naive Bayes",
        "11": "KNN Classifier",
        "12": "Neural Network (TensorFlow)",
        "13": "Neural Network (PyTorch)"
    }
    print("\nAvailable Models:")
    for k, v in models.items():
        print(f"{k}. {v}")
    
    choice = input("\nChoose a model by number: ")
    return models.get(choice, None)


def train_model(df, target_column):
    model_choice = choose_model()
    if not model_choice:
        print("[ERROR] Invalid choice.")
        return

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = None

    # Traditional ML models
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Decision Tree (Classifier)":
        model = DecisionTreeClassifier()
    elif model_choice == "Decision Tree (Regressor)":
        model = DecisionTreeRegressor()
    elif model_choice == "Random Forest (Classifier)":
        model = RandomForestClassifier()
    elif model_choice == "Random Forest (Regressor)":
        model = RandomForestRegressor()
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier()
    elif model_choice == "Support Vector Classifier":
        model = SVC()
    elif model_choice == "Support Vector Regressor":
        model = SVR()
    elif model_choice == "Naive Bayes":
        model = GaussianNB()
    elif model_choice == "KNN Classifier":
        model = KNeighborsClassifier()

    # Deep Learning (TensorFlow)
    elif model_choice == "Neural Network (TensorFlow)":
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        loss, acc = model.evaluate(X_test, y_test)
        print(f"[INFO] TensorFlow NN Accuracy: {acc:.4f}")
        model.save("saved_models/tensorflow_nn.h5")
        return model

    # Deep Learning (PyTorch)
    elif model_choice == "Neural Network (PyTorch)":
        class Net(nn.Module):
            def __init__(self, input_size):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x

        input_size = X_train.shape[1]
        net = Net(input_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        for epoch in range(10):
            optimizer.zero_grad()
            outputs = net(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        torch.save(net.state_dict(), "saved_models/pytorch_nn.pth")
        print("[INFO] PyTorch NN model saved.")
        return net

    if model:  # For sklearn models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("[INFO] Model trained successfully.")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Confusion matrix visualization
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()

        joblib.dump(model, "saved_models/sklearn_model.pkl")
        print("[INFO] Model saved to saved_models/sklearn_model.pkl")
        return model
