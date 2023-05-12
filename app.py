import os, io
import json
import pandas as pd
from data_prep.load_data import CsvLoader, DatabaseLoader
import data_prep.preprocess_data as prsdt
import train_model.enr_model as enr
import train_model.xgb_model as xgb
import train_model.helper as hlp

import logging
import logging.config

# Set up the logger
# Load the logging configuration from file
logging.config.fileConfig('logging.conf')

# Instantiate the logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.handlers.TimedRotatingFileHandler('logs.log', when='midnight', backupCount=7))


logger.info('Hello world!')

# module1.py
import logging

logger = logging.getLogger(__name__)

logger.info('Hello from module1')

logger.debug("some shit going on")


def train_model_enr():
    # Load data
    df = data_loader.load_raw_data()
    processed_data = prsdt.preprocess_data(df)
    # Do further processing with DataFrame
    # ...
    trained_models_dict = enr.train_models(processed_data)


    

    # Define the output directory elastic net
    output_dir_enr = "model_output/enr"
    message_stream = io.StringIO()
    
    hlp.filter_and_save_models(trained_models_dict, output_dir_enr, 0.4)




def train_model_xgb():
#####
    df = data_loader.load_raw_data()
    processed_data = prsdt.preprocess_data(df)
    # Do further processing with DataFrame
    # ...
    xgb_trained_models_dict = xgb.train_models(processed_data)

    # Define the output directory for XGB
    output_dir_xgb = "model_output/xgb"
 #   message_stream = io.StringIO()
    
    hlp.filter_and_save_models(xgb_trained_models_dict, output_dir_xgb, 0.4)


# LEts test our models outside the training dataset

def generate_forecast(model):
    test_file_path = 'data/testingData.csv'
    test_data_loader = CsvLoader(test_file_path)
    output_dir = "model_output"
    modelDir = os.path.join(output_dir, model)
    result_file = os.path.join(modelDir, "trainedSKU.json")
    trained_model_dict = {}
    with open(result_file, 'r') as f: 
        trained_model_dict = json.load(f)
    #trained_model_dict  = pd.read_json(result_file)

        # Load data
    df = test_data_loader.load_raw_data()
    sku_data_dict = prsdt.group_by_sku(df)
        # Assume sku_data_dict is a dictionary with SKUs as keys and DataFrames as values
        
    sku_lagged_data_dict = {}
    scaler_dir = 'model_output'
        
    for sku, data in sku_data_dict.items():
            if str(sku) not in trained_model_dict:
                print (f"model not tained for - {sku}")
                continue
            if(len(data) < prsdt.HIST_LAG_DAYS):
                print("not enough sales days for  SKU-", sku )
                continue

            prediction = enr.predict_sku_model(sku, data, output_dir_enr, scaler_dir )
            #filled_data = prsdt.fill_missing_sales_dates(data, 'orderDate', 'sku_number')
            # Generate lagged columns for this SKU's DataFrame
            #lagged_data = prsdt.generate_lagged_columns(filled_data, 'orderDate' ,prsdt.lagged_cols, prsdt.HIST_LAG_DAYS)
            #prepared_data = hlp.generate_predict_data(filled_data)


if __name__ == '__main__':
    # Load environment variables
    file_path = 'data/trainingData.csv'
    db_host = os.getenv('DB_HOST')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_name = os.getenv('DB_NAME')

    # Instantiate data loader based on environment
    if file_path:
        data_loader = CsvLoader(file_path)
    elif all([db_host, db_user, db_password, db_name]):
        engine_str = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
        data_loader = DatabaseLoader(engine_str)
    else:
        raise ValueError('No valid data source specified.')
#    train_model_enr ()
    train_model_xgb()
    generate_forecast('xgb')
