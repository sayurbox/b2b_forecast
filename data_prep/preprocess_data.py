# data_prep/preprocess_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing  import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder

HIST_LAG_DAYS = 10
FWD_LAG_DAYS = 2
lagged_cols = ['average_price', 'order_count', 'item_qty', 'returning_customer', 'new_customer']
date_column = 'orderDate'
def preprocess_data(data):
    """
    Preprocess the input data, e.g., handle missing values, normalize or standardize variables, etc.
    """
    # Preprocess the data and return the preprocessed DataFrame
    # ...
    # dont need sku_description
    #todo remove  this line after fixing pisang report
    data = data.drop(['sku_description'], axis=1)

 #   data = normalize_cols(data, lagged_cols)

    sku_data_dict = group_by_sku(data)
    filtered_sku_data_dict = filter_skus(sku_data_dict)
    # Assume sku_data_dict is a dictionary with SKUs as keys and DataFrames as values
    
    sku_lagged_data_dict = {}
    
    for sku, data in filtered_sku_data_dict.items():
        #fill any missing sales rows with zero
        filled_data = fill_missing_sales_dates(data, 'orderDate', 'sku_number')
        # Generate lagged columns for this SKU's DataFrame
        lagged_data = generate_lagged_columns(filled_data, 'orderDate' ,lagged_cols, HIST_LAG_DAYS)
        lagged_data = add_fwd_sales_cols(lagged_data, 'orderDate', 'item_qty')
        lagged_data = lagged_data.drop('orderDate', axis = 1)
        sku_lagged_data_dict[sku] = lagged_data.copy()
        # Do something with the resulting lagged DataFrame, such as save it to disk
        # ...
    return   sku_lagged_data_dict  

#create missing date ranges for a given SKU. fill zero for the missing dates



def fill_missing_sales_dates(data, date_column, sku_column):
    # Make sure the 'date' column is of datetime type
    data[date_column] = pd.to_datetime(data[date_column])

    # Get the complete date range
    date_range = pd.date_range(start=data[date_column].min(), end=data[date_column].max())

    # Create a new DataFrame with the complete date range for the current SKU
    sku_date_range = pd.DataFrame({date_column: date_range, sku_column: data[sku_column].iloc[0]})

    # Merge the new DataFrame with the original SKU data, filling in missing sales values with zeros
    merged_data = pd.merge(sku_date_range, data, on=[date_column, sku_column], how='left').fillna(0)
    merged_data['average_price'] = merged_data['average_price'].replace(0, np.nan).ffill().fillna(0)
    return merged_data



#ignore the SKUs that dont have enough sales history
def filter_skus(sku_data_dict):
    """
    Filter out all SKUs that have less than 30 data points.
    """
    new_sku_data_dict = {}
    for sku, df in sku_data_dict.items():
        if len(df) >= 30:
            new_sku_data_dict[sku] = df
    return new_sku_data_dict
    
def group_by_sku(df):
    """
    Group a DataFrame by SKU and return a dictionary of DataFrames.
    """
    # Group the DataFrame by SKU
    grouped = df.groupby('sku_number')

    # Create a dictionary to hold the resulting DataFrames
    result = {}

    # Loop through the groups and create a DataFrame for each one
    for sku, group in grouped:
        result[sku] = group.copy()

    return result

"""
def generate_lagged_columns(df, date_column, lag_cols):
    
    Generate lagged columns for a given list of column names based on the date column.
    The lagged values are from the previous orderdate value.
    
    # Ensure the data is sorted by SKU and the date column in ascending order
    df = df.sort_values(by=[date_column])

    # Create a copy of the DataFrame with the lagged columns added
    lagged_df = df.copy()
    for lag in range(1, HIST_LAG_DAYS):
        # Add a lagged column for each column in the list
        for col_name in lag_cols:
            new_col_name = f"{col_name}_lag{lag}"
            lagged_df[new_col_name] = lagged_df[col_name].shift(lag)

    # Drop any rows with missing values (-1 values represent missing values in our dataset)
    lagged_df = lagged_df[lagged_df != -1].dropna()

    return lagged_df
"""



def generate_lagged_columns(df, date_column, col_names, num_legs):
    """
    Generate lagged columns and day-of-week encoding for the given column names.
    """
    # Ensure the data is sorted by SKU and the date column in ascending order
    df = df.sort_values(by=[date_column])
    # Create a copy of the DataFrame to avoid modifying the original
    lagged_df = df.copy()

    # Create a list to store the lagged DataFrames
    lagged_dfs = []

    # Generate HIST_LAG_DAYS lagged columns for each column name
    for col_name in col_names:
        lagged_cols = [lagged_df[col_name].shift(lag) for lag in range(1, num_legs)]
        lagged_df_subset = pd.concat(lagged_cols, axis=1)
        lagged_df_subset.columns = [f"{col_name}_lag_{lag}" for lag in range(1, num_legs)]

        # Add the lagged DataFrame to the list
        lagged_dfs.append(lagged_df_subset)
    # Add day-of-week encoding for each row in the subset DataFrame
    dates_subset = lagged_df[date_column].reset_index(drop=True)
    day_of_week = dates_subset.dt.dayofweek.values.reshape(-1, 1)
    enc = OneHotEncoder(sparse=False)
    day_of_week_enc = enc.fit_transform(day_of_week)
    day_of_week_enc_cols = [f"dow_{dow}" for dow in range(7)]
    day_of_week_enc_df = pd.DataFrame(day_of_week_enc, columns=day_of_week_enc_cols)

    # Concatenate all the lagged DataFrames together
    lagged_df_concat = pd.concat(lagged_dfs, axis=1)

    # Concatenate the original DataFrame with all the lagged DataFrames and day-of-week encoding DataFrame
    result = pd.concat([df, lagged_df_concat, day_of_week_enc_df], axis=1).iloc[num_legs:-FWD_LAG_DAYS]

    return result
def shift_predict_values(lagged_df, col_names, num_legs):
    df_to_modify = lagged_df
    for col_name in col_names:
        df_to_modify[f"{col_name}_lag_{num_legs}"] = np.nan
        for i in range(1, num_legs):
            curr_lag = num_legs - i +1
            prev_leg = curr_lag -1
            df_to_modify[f"{col_name}_lag_{curr_lag}"] = df_to_modify[f"{col_name}_lag_{(prev_leg)}"]
        df_to_modify.drop(col_name, axis = 1, inplace = True )

    return df_to_modify    


def generate_predict_data(input_df, dateToPredict):
    input_df[date_column] = pd.to_datetime(input_df[date_column]).dt.date
    input_df = input_df.sort_values(by=date_column)
    predict_df = input_df[input_df[date_column] < dateToPredict.date()]
   # if len(predict_df) < 20:

        #raise ValueError("Input dataframe size should be at least 20.")    

    # Sort DataFrame by date column
    
    filled_data = fill_missing_sales_dates(predict_df, date_column, 'sku_number')
    numLegs = HIST_LAG_DAYS -1
    lagged_data = generate_lagged_columns(filled_data, date_column ,lagged_cols, numLegs)
    lagged_data = shift_predict_values(lagged_data, lagged_cols, numLegs)
    lagged_data = lagged_data.drop(['sku_number','orderDate','sku_description' ], axis = 1)
    # Extract row with latest date
    predict_data = lagged_data.tail(1)

    return predict_data


def add_fwd_sales_cols(df, date_column, sales_col):
    """
    Add forward sales columns to a DataFrame.
    """
    # Sort the DataFrame by the date column
    df = df.sort_values(by=[date_column])

    # Add forward sales columns for each lag
    for lag in range(1, FWD_LAG_DAYS):
        fwd_sales_col_name = f"{sales_col}_fwd_{lag}"
        df[fwd_sales_col_name] = df[sales_col].shift(-lag)

    return df