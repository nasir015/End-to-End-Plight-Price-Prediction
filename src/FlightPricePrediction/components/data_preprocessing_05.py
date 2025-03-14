import pandas as pd
import os
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.custom_logging import logger
from utils.custom_exception import CustomException
from utils.common import *
from sklearn.preprocessing import LabelEncoder
from src.FlightPricePrediction.components.feature_extraction_04 import FeatureExtraction


class DataPreprocessing:

    def __init__(self):
        self.data = FeatureExtraction().feature_extraction()
        self.log_path = 'log\logging_info.log'
        os.makedirs('artifact\\data_preprocessing', exist_ok=True)
        os.makedirs('artifact\\final_data', exist_ok=True)

    def data_preprocessing(self):

        try:  

            # remove the rows with missing values
            logger(self.log_path,logging.INFO, 'Removing the rows with missing values')
            self.data.dropna(inplace=True)

            # drop the duplicate rows
            logger(self.log_path,logging.INFO, 'Removing the duplicate rows')
            self.data.drop_duplicates(inplace=True)

            # drop the columns which are not required
            logger(self.log_path,logging.INFO, 'Removing the columns which are not required')
            self.data.drop (['Date_of_Journey','Route'], axis=1, inplace=True)

            # identify the categorical columns
            logger(self.log_path,logging.INFO, 'Identifying the categorical columns')
            categorical_columns = [col for col in self.data.columns if self.data[col].dtype == 'object']

            # convert the categorical columns using level encoding
            logger(self.log_path,logging.INFO, 'Converting the categorical columns using level encoding')
            le = LabelEncoder()

            for col in categorical_columns:
                self.data[col] = le.fit_transform(self.data[col])
        
            # save the preprocessed data
            logger(self.log_path,logging.INFO, 'Saving the preprocessed data')
            self.data.to_csv('artifact\\data_preprocessing\\preprocessed_data.csv', index=False)

            #final dataset 
            self.data.to_csv('artifact\\final_data\\preprocessed_data.csv', index=False)

            return self.data

        except Exception as e:
            logger(self.log_path,logging.ERROR, CustomException(e,sys))
            raise CustomException(e,sys)






        
