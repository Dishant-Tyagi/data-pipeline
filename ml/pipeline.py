import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
from models import get_available_models

def run_ml_pipeline(csv_path):
    print(f"[INFO] Loading AI-filtered data from {csv_path}")
    df = pd.read_csv(csv_path)

    print("\n[INFO] Columns in dataset:", list(df.columns))
    target_col = input("Enter the target column name: ").strip()

    if target_col not in df.columns:
        raise ValueError("Target column not found in dataset!")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical target if needed
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Show available models
    models = get_available_models()
    print("\nAvailable Models:")
    for i, model_name in enumerate(models.keys(), start=1):
        print(f"{i}. {model_name}")

    choice = int(input("\nSelect a model by number: "))
    model_name = list(models.keys())[choice - 1]
    ModelClass = models[model_name]

    print(f"\n[INFO] Training model: {model_name}")
    model = ModelClass()
    model.fit(X_train, y_train)

    # Prediction
    preds = model.predict(X_test)

    # Evaluate
    if "Regressor" in model_name or "Regression" in model_name or "SVR" in model_name:
        mse = mean_squared_error(y_test, preds)
        print(f"[RESULT] MSE: {mse}")
        plt.scatter(y_test, preds)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title(f"{model_name} - True vs Predicted")
        plt.savefig("output/ml_results.png")
        plt.show()
    else:
        acc = accuracy_score(y_test, preds)
        print(f"[RESULT] Accuracy: {acc}")
        print(classification_report(y_test, preds))
    
    # Save trained model
    model_path = f"output/{model_name.replace(' ', '_')}.pkl"
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved at {model_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ml/pipeline.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    run_ml_pipeline(csv_path)
