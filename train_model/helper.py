import os
import pandas as pd
from typing import Dict, Tuple, List
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import joblib
import logging

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def split_and_transform(df, y_cols: List[str], test_size=0.2, random_state=None) -> Tuple:
    # Separate the features (X) and target (y) variables
    X = df.drop(y_cols, axis=1)
    y = df.loc[:, y_cols]
    y.fillna(0, inplace=True)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Initialize the scaler and fit it to the training set
    scaler = StandardScaler()
    '''
    It's generally a good practice to fit the scaler on the training data only and then use it to 
    transform both the training and testing data. This is because in real-world scenarios, 
    you won't have access to the testing data during the model development phase, so you need to 
    simulate this by holding out a portion of the training data for testing. By fitting the scaler 
    on the training data only, you're ensuring that your model is not "cheating" by learning 
    information from the testing data.
    '''
    scaler.fit(X_train)
    
    # important to save scaler
    joblib.dump(scaler, 'model_output/scaler.joblib')
    # Transform the training and testing sets using the scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Return the transformed data and target variables
    return X_train_scaled, X_test_scaled, y_train, y_test


def filter_and_save_models(models, output_dir, threshold=0.4):
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