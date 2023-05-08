# data_prep/load_data.py


import pandas as pd


from sqlalchemy import create_engine

# Set up the database connection settings
db_settings = {
    'drivername': 'mysql+pymysql',
    'username': 'your_username',
    'password': 'your_password',
    'host': 'your_host',
    'port': 'your_port',
    'database': 'your_database'
}

# Create the engine with the connection settings
#engine = create_engine(f"{db_settings['drivername']}://{db_settings['username']}:{db_settings['password']}@{db_settings['host']}:{db_settings['port']}/{db_settings['database']}")


class RawDataLoader:

    def load_raw_data(self):
        """
        Load raw data from a file or database.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class CsvLoader(RawDataLoader):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def load_raw_data(self):
        """
        Load raw data from a CSV file.
        """
        df = pd.read_csv(self.file_path)
        return df
    # load input data for prediction
    def load_raw_predict_data(self, filepath):
        return pd.read_csv(filepath)


class DatabaseLoader(RawDataLoader):
    def __init__(self, db_connection_string, table_name):
        self.db_connection_string = db_connection_string
        self.table_name = table_name

    def load_raw_data(self):
        """
        Load raw data from a database table.
        """
        conn = create_engine(self.db_connection_string)
        query = f"SELECT * FROM {self.table_name};"
        return pd.read_sql(query, conn)