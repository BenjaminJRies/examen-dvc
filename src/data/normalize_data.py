import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def main():
    input_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
    os.makedirs(input_dir, exist_ok=True)
    X_train_path = os.path.join(input_dir, "X_train.csv")
    X_test_path = os.path.join(input_dir, "X_test.csv")

    # Load data
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    # Fit scaler on training data and transform both train and test
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Save scaled data
    X_train_scaled.to_csv(os.path.join(input_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(input_dir, "X_test_scaled.csv"), index=False)

if __name__ == "__main__":
    main()