import pandas as pd
import os
import json
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score

def main():
    input_dir = "./data/processed"
    model_dir = "./models"
    metrics_dir = "./metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # Load data
    X_test = pd.read_csv(os.path.join(input_dir, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv")).values.ravel()

    # Load model
    model = load(os.path.join(model_dir, "trained_model.joblib"))

    # Predict
    predictions = model.predict(X_test)
    pd.DataFrame({"prediction": predictions}).to_csv("./data/predictions.csv", index=False)

    # Evaluate
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    scores = {"mse": mse, "r2": r2}

    # Save metrics
    with open(os.path.join(metrics_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=4)

    print("Evaluation complete. Metrics saved to metrics/scores.json.")

if __name__ == "__main__":
    main()