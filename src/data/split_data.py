import pandas as pd
from sklearn.model_selection import train_test_split
import os

def save_splits(X_train, X_test, y_train, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    splits = {
        "X_train.csv": X_train,
        "X_test.csv": X_test,
        "y_train.csv": y_train,
        "y_test.csv": y_test
    }
    for filename, df in splits.items():
        df.to_csv(os.path.join(output_dir, filename), index=False)

def main():
    input_path = os.path.join(os.path.dirname(__file__), '../../data/raw_data/raw.csv')
    output_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
    df = pd.read_csv(input_path, sep=",")
    # Drop the Date column if it exists
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    X = df.drop(columns=["silica_concentrate"])
    y = df["silica_concentrate"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    save_splits(X_train, X_test, y_train, y_test, output_dir)

if __name__ == "__main__":
    main()