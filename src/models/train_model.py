import pandas as pd
import os
import pickle
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

def main():
    input_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    X_train = pd.read_csv(os.path.join(input_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).values.ravel()

    # Load best parameters
    with open(os.path.join(model_dir, "best_params.pkl"), "rb") as f:
        best_params = pickle.load(f)

    # Train model
    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)

    # Save trained model
    dump(model, os.path.join(model_dir, "trained_model.joblib"))

    print("Model trained and saved to models/trained_model.joblib")

if __name__ == "__main__":
    main()