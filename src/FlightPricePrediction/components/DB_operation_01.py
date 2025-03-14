
import sqlite3
import csv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.custom_logging import logger
from utils.custom_exception import CustomException
import logging


log_path = 'log\logging_info.log'


class dBOperation:


    def __init__(self):
        self.data_path = 'artifact\data\data.csv'
        self.db_name = 'artifact\dboperation\FlightPricePrediction.db'
        self.table_name = 'FlightPricePrediction'
        self.logger = logger
        self.log_path = 'log\logging_info.log'

    def infer_column_type(self,value):
        """
        Infers the type of a column based on its value:
        - TEXT if it's a string
        - INTEGER if it's an integer
        - REAL if it's a floating-point number
        """
        if value.isdigit():  # Checks if value is an integer
            return 'INTEGER'
        try:
            float(value)  # Try converting to float
            return 'REAL'
        except ValueError:
            return 'TEXT'
    
    def create_table_and_upload_data(self):
        try:

            # Stage 1: Connect to the SQLite database
            logger(self.log_path, logging.INFO, 'Connecting to the SQLite database.')

            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Stage 2: Create the table based on the CSV headers and inferred column types
            logger(self.log_path, logging.INFO, 'Creating the table and uploading data.')
            with open(self.data_path, 'r') as csv_file:
                reader = csv.reader(csv_file)
                headers = next(reader)  # Read the first row as the column headers

                # Initialize a list to store the inferred column types
                column_types = []

                # Read the first few rows to infer types for each column
                for i, row in enumerate(reader):
                    if i > 5:  # Use first 6 rows for type inference
                        break
                    for j, value in enumerate(row):
                        inferred_type = self.infer_column_type(value)
                        if len(column_types) <= j:
                            column_types.append(inferred_type)
                        else:
                            # Ensure the column type is the most specific possible
                            existing_type = column_types[j]
                            if existing_type != inferred_type and existing_type != 'TEXT' and inferred_type != 'TEXT':
                                column_types[j] = 'REAL'  # Prioritize REAL > INTEGER > TEXT
                
                # Create the table with the inferred column types
                create_table_sql = f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
                for i, header in enumerate(headers):
                    create_table_sql += f"{header} {column_types[i]}, "
                create_table_sql = create_table_sql.rstrip(', ') + ");"  # Remove the trailing comma and add closing parenthesis

                cursor.execute(create_table_sql)
                logger(self.log_path, logging.INFO, 'Table created successfully.')

            # Stage 3: Upload data into the table
            logger(self.log_path, logging.INFO, 'Uploading data into the table.')
            with open(self.data_path, 'r') as csv_file:
                reader = csv.reader(csv_file)
                next(reader)  # Skip the header row
                for row in reader:
                    # Prepare SQL to insert data into the table
                    insert_sql = f"INSERT INTO {self.table_name} ({', '.join(headers)}) VALUES ({', '.join(['?'] * len(row))});"
                    cursor.execute(insert_sql, row)
                logger(self.log_path, logging.INFO, 'Data uploaded successfully.')
            # Commit the transaction and close the connection
            conn.commit()
            conn.close()
        except Exception as e:
            logger(self.log_path, logging.ERROR, CustomException(e,sys))
            raise CustomException(e,sys)



'''if __name__ == '__main__':
    db = dBOperation()
    db.create_table_and_upload_data()
    print('Data uploaded successfully!')'''