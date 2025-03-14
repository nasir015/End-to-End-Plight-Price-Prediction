import os 
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.FlightPricePrediction.utils.custom_logging import logger
from src.FlightPricePrediction.utils.custom_exception import CustomException
from src.FlightPricePrediction.components.DB_operation_01 import dBOperation
from src.FlightPricePrediction.components.data_cleaning_03 import DataCleaning
from src.FlightPricePrediction.components.feature_extraction_04 import FeatureExtraction
from src.FlightPricePrediction.components.data_preprocessing_05 import DataPreprocessing
from src.FlightPricePrediction.components.data_ingestion_02 import DataIngestion
from src.FlightPricePrediction.components.model_build_06 import ModelBuild


log_path = 'log\logging_info.log'


Stage_name = "Database Operation"

try:
    logger(log_path, logging.INFO, f">>>>>>>>>>Starting {Stage_name}<<<<<<<<<<<")
    pipeline = dBOperation()
    pipeline.create_table_and_upload_data()
    logger(log_path, logging.INFO, f">>>>>>>>>>{Stage_name} completed successfully.<<<<<<<<<<<")
except CustomException as ce:
    logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
    raise CustomException(ce, sys)



Stage_name = "Data Ingestion"

try:
    logger(log_path, logging.INFO, f">>>>>>>>>>Starting {Stage_name}<<<<<<<<<<<")
    pipeline = DataIngestion()
    pipeline.export_data_to_csv()
    logger(log_path, logging.INFO, f">>>>>>>>>>{Stage_name} completed successfully.<<<<<<<<<<<")
except CustomException as ce:
    logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
    raise CustomException(ce, sys)


Stage_name = "Data Cleaning"

try:
    logger(log_path, logging.INFO, f">>>>>>>>>>Starting {Stage_name}<<<<<<<<<<<")
    pipeline = DataCleaning()
    pipeline.data_clining()
    logger(log_path, logging.INFO, f">>>>>>>>>>{Stage_name} completed successfully.<<<<<<<<<<<")
except CustomException as ce:
    logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
    raise CustomException(ce, sys)


Stage_name = "Feature Extraction"

try:
    logger(log_path, logging.INFO, f">>>>>>>>>>Starting {Stage_name}<<<<<<<<<<<")
    pipeline = FeatureExtraction()
    pipeline.feature_extraction()
    logger(log_path, logging.INFO, f">>>>>>>>>>{Stage_name} completed successfully.<<<<<<<<<<<")
except CustomException as ce:
    logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
    raise CustomException(ce, sys)


Stage_name = "Data Preprocessing"

try:
    logger(log_path, logging.INFO, f">>>>>>>>>>Starting {Stage_name}<<<<<<<<<<<")
    pipeline = DataPreprocessing()
    pipeline.data_preprocessing()
    logger(log_path, logging.INFO, f">>>>>>>>>>{Stage_name} completed successfully.<<<<<<<<<<<")
except CustomException as ce:
    logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
    raise CustomException(ce, sys)



Stage_name = "Model Building"

try:
    logger(log_path, logging.INFO, f">>>>>>>>>>Starting {Stage_name}<<<<<<<<<<<")
    pipeline = ModelBuild()
    pipeline.model_building()
    logger(log_path, logging.INFO, f">>>>>>>>>>{Stage_name} completed successfully.<<<<<<<<<<<")
except CustomException as ce:
    logger(log_path, logging.ERROR, f"CustomException occurred: {CustomException(ce, sys)}")
    raise CustomException(ce, sys)