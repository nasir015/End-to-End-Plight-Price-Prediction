import pandas as pd
import os
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.custom_logging import logger
from utils.custom_exception import CustomException
from utils.common import *
from src.FlightPricePrediction.components.data_cleaning_03 import DataCleaning

class FeatureExtraction:

    def __init__(self):
        self.data = DataCleaning().data_clining()
        self.log_path = 'log\logging_info.log'
        os.makedirs('artifact\\feature_extraction', exist_ok=True)

    def feature_extraction(self):
        try:

            # extract the year from the date column
            logger(self.log_path,logging.INFO, 'Extracting features from the data.')
            self.data['Month_of_Journey'] = pd.DatetimeIndex(self.data['Date_of_Journey']).month
            self.data['Day_of_Journey'] = pd.DatetimeIndex(self.data['Date_of_Journey']).day


            # Define function to categorize season
            logger(self.log_path,logging.INFO, 'Categorizing the season.')
            def categorize_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:
                    return 'Fall'

            # Apply function to create 'Season' column
            self.data['Season'] = self.data['Month_of_Journey'].apply(categorize_season)


            # Define function to categorize time of day
            logger(self.log_path,logging.INFO, 'Categorizing the time of day.')
            def categorize_time_of_day(dep_time):
                hour = int(dep_time.split(':')[0])
                if 6 <= hour < 12:
                    return 'Morning'
                elif 12 <= hour < 18:
                    return 'Afternoon'
                elif 18 <= hour < 21:
                    return 'Evening'
                else:
                    return 'Night'

            # Apply function to create 'Time_of_Day' column
            self.data['Time_of_Day'] = self.data['Dep_Time'].apply(categorize_time_of_day)


            # Define function to categorize flight duration
            logger(self.log_path,logging.INFO, 'Categorizing the flight duration.')
            def categorize_flight_duration(duration):
                if duration <= 300:
                    return 'Short'
                elif 300 < duration <= 600:
                    return 'Medium'
                else:
                    return 'Long'

            # Apply function to create 'Flight_Duration_Category' column
            self.data['Flight_Duration_Category'] = self.data['Duration(minute)'].apply(categorize_flight_duration)


            # Extract hours from Dep_Time and Arrival_Time columns
            logger(self.log_path,logging.INFO, 'Extracting hours from Dep_Time and Arrival_Time columns.')
            self.data['Dep_Hour'] = pd.to_datetime(self.data['Dep_Time'], format='%H:%M').dt.hour
            self.data['Arr_Hour'] = pd.to_datetime(self.data['Arrival_Time'], format='%H:%M').dt.hour


            # Extract minutes from Dep_Time and Arrival_Time columns
            logger(self.log_path,logging.INFO, 'Extracting minutes from Dep_Time and Arrival_Time columns.')
            self.data['Dep_Minute'] = pd.to_datetime(self.data['Dep_Time'], format='%H:%M').dt.minute
            self.data['Arr_Minute'] = pd.to_datetime(self.data['Arrival_Time'], format='%H:%M').dt.minute


            # Drop the original Dep_Time and Arrival_Time columns
            logger(self.log_path,logging.INFO, 'Dropping the original Dep_Time and Arrival_Time columns.')
            self.data.drop(columns=['Dep_Time', 'Arrival_Time'], inplace=True)




            # Fill missing values in 'Route' column with an empty string

            self.data['Route'] = self.data['Route'].fillna('')

            # Create a new column for the total stops based on the Route column
            logger(self.log_path,logging.INFO, 'Calculating the total stops.')
            self.data['Total_Stops'] = self.data['Route'].apply(lambda x: x.count('->') - 1 if 'non-stop' not in x else 0)







            # Export the data to a new CSV file
            logger(self.log_path,logging.INFO, 'Exporting the extracted features to a new CSV file .')

            export_to_csv(self.data, 'artifact\\feature_extraction\\feature_extracted_data.csv')

            # Return the extracted features
            return self.data
        
        except Exception as e:
            logger(self.log_path,logging.ERROR, CustomException(e,sys))
            raise CustomException(e,sys)
        







