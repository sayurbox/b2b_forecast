import os
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import joblib
import logging

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import train_model.helper as hlp
from data_prep.preprocess_data import generate_predict_data
FWD_LAG_DAYS = 2



logger = logging.getLogger(__name__)

logger.info('Hello from module1')


"""

def train_models(sku_lagged_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, ElasticNet]:

 #   Train an Elastic Net regression model for each SKU DataFrame in sku_lagged_data_dict.
  #  Returns a dictionary of trained models keyed by SKU.
    
    # Initialize an empty dictionary to store the trained models
    models = {}

    # Train a separate model for each SKU DataFrame in sku_lagged_data_dict
    for sku, df in sku_lagged_data_dict.items():
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop('item_qty', axis=1), df['item_qty'], test_size=0.2, random_state=42)

        # Initialize and fit the model
        model = ElasticNet()
        model.fit(X_train, y_train)

        # Compute the R^2 score on the test set
        r2_score = model.score(X_test, y_test)

        # Add the trained model to the dictionary
        models[sku] = model

        # Print the R^2 score for the current SKU
        print(f"R^2 score for SKU {sku}: {r2_score}")

    return models


"""


def train_models(sku_lagged_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, ElasticNet]:
    # Create a dictionary to store the trained models
    models = {}

    # Define the hyperparameter grid for ElasticNet
    param_grid = {
           'alpha': [0.1, 1, 5, 10],
           'l1_ratio': [0.1, 0.3, 0.5, 0.9],
           'max_iter': [10000, 50000, 100000, 200000]
       }

    y_col = 'item_qty'
    y_cols  = [f"{y_col}_fwd_{lag}" for lag in range(1, FWD_LAG_DAYS)]
    y_cols.insert(0, y_col)
    x_drop = ['average_price', 'order_count',  'returning_customer', 'new_customer'] 

    # Train a separate model for each SKU DataFrame in sku_lagged_data_dict
    for sku, df in sku_lagged_data_dict.items():
        # Split the data into training and testing sets


        X_train, X_test, y_train, y_test = hlp.split_and_transform(df, y_cols, x_drop,
            test_size=0.2,
            random_state=42
        )
        #debug code
        nan_indices = np.argwhere(np.isnan(y_test))
        print(nan_indices)
        nan_indices = np.argwhere(np.isnan(X_train))
        print(nan_indices)
        # Initialize the model
        model = ElasticNet()

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
        model = ElasticNet(**grid_search.best_params_)
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





# add predict method that takes loads model for the given SKU in function arg from the model directory in arguments
# the functon also takes a dataframe of input values and rwturns perdiction results
# the function should use the scaler saved earlier. 
def predict_sku_model(sku, input_df, model_dir, scaler_dir):
    model_path = os.path.join(model_dir, f"{sku}_model.joblib")
    #model_path = f"{model_dir}/{sku}_model.joblib"
    scaler_path = os.path.join(scaler_dir, "scaler.joblib")
    # Load the model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # TODO lot more wotk needed here
    processed_data = generate_predict_data(input_df)
 
    # Scale the input data using the previously saved scaler
    scaled_input = scaler.transform(processed_data)

    # Make predictions using the loaded model and scaled data
    predictions = model.predict(scaled_input)

    return predictions
