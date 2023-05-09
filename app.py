import os, io
from data_prep.load_data import CsvLoader, DatabaseLoader
from data_prep.preprocess_data import preprocess_data
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

    # Load data
    df = data_loader.load_raw_data()
    processed_data = preprocess_data(df)
    # Do further processing with DataFrame
    # ...
    trained_models_dict = enr.train_models(processed_data)


    
   
    # Define the output directory elastic net
    output_dir_enr = "model_output/enr"
    message_stream = io.StringIO()
     
    hlp.filter_and_save_models(trained_models_dict, output_dir_enr, 0.4, message_stream)

#    xgb_trained_models_dict = xgb.train_models(processed_data)

    # Define the output directory for XGB
    output_dir_xgb = "model_output/xgb"
 #   message_stream = io.StringIO()
    
#    hlp.filter_and_save_models(xgb_trained_models_dict, output_dir_xgb, 0.4, message_stream)
