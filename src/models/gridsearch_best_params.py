import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def main():
    input_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    os.makedirs(model_dir, exist_ok=True)

    X_train = pd.read_csv(os.path.join(input_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).values.ravel()

    # Define model and parameter grid
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)

    # Save best parameters
    best_params = grid_search.best_params_
    with open(os.path.join(model_dir, "best_params.pkl"), "wb") as f:
        pickle.dump(best_params, f)

    print("Best parameters found and saved:", best_params)

if __name__ == "__main__":
    main()