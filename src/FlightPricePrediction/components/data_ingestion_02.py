import sqlite3
import pandas as pd
import csv
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.custom_logging import logger
from utils.custom_exception import CustomException


class DataIngestion:
    
    def __init__(self):
        self.logger = logger
        self.log_path = 'log\logging_info.log'
        self.custom_exception = CustomException
        self.db_name = 'artifact\dboperation\FlightPricePrediction.db'
        self.table_name = 'FlightPricePrediction'
        os.makedirs('artifact\dataingestion', exist_ok=True)
        self.output_csv_path = 'artifact\dataingestion\data_from_db.csv'

    def export_data_to_csv(self):
        try:

            """
            Export all data from an SQLite database table to a CSV file.
            
            Parameters:
            - db_name: The SQLite database file name.
            - table_name: The table name to export data from.
            - output_csv_path: Path to save the CSV file.
            """
            # Stage 1: Connect to the SQLite database
            self.logger(self.log_path,logging.INFO, 'Entering the Data Ingestion module')
            self.logger(self.log_path,logging.INFO, 'Connecting to the SQLite database')
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Stage 2: Fetch all column headers (field names) from the table
            self.logger(self.log_path,logging.INFO, 'Fetching all column headers from the table')
            cursor.execute(f"PRAGMA table_info({self.table_name});")
            headers = [column[1] for column in cursor.fetchall()]  # Get column names from the PRAGMA table_info
            
            # Stage 3: Fetch all rows of data from the table
            self.logger(self.log_path,logging.INFO, 'Fetching all rows of data from the table')
            cursor.execute(f"SELECT * FROM {self.table_name};")
            rows = cursor.fetchall()

            # Stage 4: Write data to CSV
            self.logger(self.log_path,logging.INFO, 'Writing data to CSV')
            with open(self.output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(headers)  # Write headers as the first row
                writer.writerows(rows)    # Write data rows

            # Close the connection
            conn.close()
            self.logger(self.log_path,logging.INFO, 'Exiting the Data Ingestion module')

            # return the data
            df = pd.read_csv(self.output_csv_path)
            return df
        
        except Exception as e:
            self.logger(self.log_path,logging.ERROR,CustomException(e,sys))
            raise CustomException(e,sys)
