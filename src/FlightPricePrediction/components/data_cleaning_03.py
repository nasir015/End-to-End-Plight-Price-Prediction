import pandas as pd
import os
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.custom_logging import logger
from utils.custom_exception import CustomException
from utils.common import *
from src.FlightPricePrediction.components.data_ingestion_02 import DataIngestion



class DataCleaning:

    def __init__(self):
        self.data = DataIngestion().export_data_to_csv()
        self.log_path = 'log\logging_info.log'
        os.makedirs('artifact\data_cleaning', exist_ok=True)

    def data_clining(self):
        try:
            # Date of Journey column
            self.data['Date_of_Journey'] = pd.to_datetime(self.data['Date_of_Journey'])
            
            # Arrival_Time column
            self.data[['Arrival_Time', 'Arrival_Date']] = self.data['Arrival_Time'].str.split(' ', expand=True, n=1)

            # Keep only the time part
            self.data['Arrival_Time'] = self.data['Arrival_Time'].fillna('00:00:00')

            # Drop the 'Arrival_Date' column as it is not needed
            self.data.drop(columns=['Arrival_Date'], inplace=True)

            # duration column
            def convert_to_minutes(duration):
                hours = 0
                minutes = 0
                
                # Check if the duration contains hours and minutes (e.g., '3h 34m')
                if 'h' in duration and 'm' in duration:
                    hours_part, minutes_part = duration.split('h')
                    minutes_part = minutes_part.replace('m', '').strip()
                    
                    hours = int(hours_part.strip())  # Extract the hours
                    minutes = int(minutes_part.strip())  # Extract the minutes
                    
                # Check if the duration contains only hours (e.g., '3h')
                elif 'h' in duration:
                    hours = int(duration.replace('h', '').strip())
                    minutes = 0
                
                # Check if the duration contains only minutes (e.g., '34m')
                elif 'm' in duration:
                    minutes = int(duration.replace('m', '').strip())
                    
                # Convert hours to minutes and sum with the minutes
                total_minutes = (hours * 60) + minutes
                return total_minutes

            # Apply the function to the Duration column and create a new column for total minutes
            self.data['Duration(minute)'] = self.data['Duration'].apply(convert_to_minutes)

            # Drop the original Duration column
            self.data.drop(columns=['Duration'], inplace=True)

            # remove the Additional_Info column 
            self.data.drop(columns=['Additional_Info'], inplace=True)

            # Route column
            self.data['Route']= self.data['Route'].str.replace('?', '->')


            export_to_csv(self.data, 'artifact\data_cleaning\data_cleaned.csv')

            # create dataframes for the cleaned data
            return self.data
        
        except Exception as e:
            logger(self.log_path,logging.ERROR, CustomException(e,sys))
            raise CustomException(e,sys)



