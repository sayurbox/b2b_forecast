import os
import pandas as pd
from typing import Dict, Tuple, List
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import joblib
import logging
import train_model.helper as hlp

FWD_LAG_DAYS = 2

logger = logging.getLogger(__name__)
logger.info('Hello from xgb_model.py')

def train_models(sku_lagged_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, XGBRegressor]:
    # Create a dictionary to store the trained models
    models = {}

    # Define the hyperparameter grid for XGBRegressor
    param_grid = {
            'max_depth': [3, 4, 5],
            'eta':[0.01, 0.05, 0.1],
            'min_child_weight': [1, 2, 3],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500, 1000],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.5, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.7, 1 ]

        }


    # Train a separate model for each SKU DataFrame in sku_lagged_data_dict
    for sku, df in sku_lagged_data_dict.items():
        # Split the data into training and testing sets
        y_col = 'item_qty'
        y_cols  = [f"{y_col}_fwd_{lag}" for lag in range(1, FWD_LAG_DAYS)]
        y_cols.insert(0, y_col)
        x_drop = ['sku_number','average_price', 'order_count',  'returning_customer', 'new_customer'] 

        X_train, X_test, y_train, y_test = hlp.split_and_transform(df, y_cols, x_drop,
            test_size=0.2,
            random_state=42
        )

        # Initialize the XGBRegressor
        model = XGBRegressor(objective='reg:squarederror', n_jobs=-1)

        # Perform a grid search to find the best hyperparameters
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Print the best hyperparameters found by the grid search
        print(f"Best hyperparameters for SKU {sku}: {grid_search.best_params_}")

        # Train a new model with the best hyperparameters on the entire training set
        model = XGBRegressor(objective='reg:squarederror', **grid_search.best_params_)
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        score = model.score(X_test, y_test)
        print(f"R^2 score for SKU {sku}: {score}")

        # Add the trained model to the dictionary
        models[sku] = [model, score]

    return models

def filter_and_save_models_dep(models, output_dir, threshold=0.4):
    """
    Evaluate the models on the training set, print the R-squared for each SKU above
    the given threshold, and save the passing models to a subfolder called 'pass'.
    """
    # Create the 'pass' directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each model and evaluate it on the training set
    for sku, (model, score) in models.items():
        # If the R-squared is above the threshold, print the SKU and score and save the model
        if score > threshold:
            print(f"SKU {sku}: R-squared = {score}")
            model_filename = f"{sku}.pkl"
            model_file_path = os.path.join(output_dir, f"{sku}_model.joblib")
            joblib.dump(model, model_file_path)



