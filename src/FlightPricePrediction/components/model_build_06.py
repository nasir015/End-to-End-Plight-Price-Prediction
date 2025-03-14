'''
import pandas as pd
import numpy as np
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import openpyxl
from utils.custom_logging import logger
from utils.custom_exception import CustomException
from src.FlightPricePrediction.components.data_preprocessing_05 import DataPreprocessing


class ModelBuild:

    def __init__(self):
        self.data = DataPreprocessing().data_preprocessing()
        self.log_path = 'log\logging_info.log'
        os.makedirs('artifact\\model_building', exist_ok=True)
        self.logger = logger


    def model_building(self):
        try:

            # Split the data into features and target
            self.logger(self.log_path,logging.INFO, 'Splitting the data into features and target')
            X = self.data.drop(columns=['Price'])
            y = self.data['Price']

            # Standardize the data
            logger(self.log_path,logging.INFO, 'Standardizing the data')
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split data into training and testing sets (80-20 split)
            logger(self.log_path,logging.INFO, 'Splitting the data into training and testing sets')
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Dictionary of regression models to test
            logger(self.log_path,logging.INFO, 'Creating a dictionary of regression models to test')
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Lasso Regression': Lasso(),
                'ElasticNet Regression': ElasticNet(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'SVR': SVR()

            }


            # Create an empty list to store evaluation metrics
            metrics_list = []

            # Loop through models and calculate evaluation metrics
            logger(self.log_path,logging.INFO, 'Looping through models and calculating evaluation metrics')
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                mae_train = mean_absolute_error(y_train, y_pred_train)
                mse_train = mean_squared_error(y_train, y_pred_train)
                rmse_train = np.sqrt(mse_train)
                r2_train = r2_score(y_train, y_pred_train)
                
                mae_test = mean_absolute_error(y_test, y_pred_test)
                mse_test = mean_squared_error(y_test, y_pred_test)
                rmse_test = np.sqrt(mse_test)
                r2_test = r2_score(y_test, y_pred_test)
                
                # Append results to the list
                metrics_list.append({
                    'Model': model_name,
                    'Train MAE': mae_train,
                    'Train MSE': mse_train,
                    'Train RMSE': rmse_train,
                    'Train R^2': r2_train,
                    'Test MAE': mae_test,
                    'Test MSE': mse_test,
                    'Test RMSE': rmse_test,
                    'Test R^2': r2_test
                })

            # Convert the list of results into a DataFrame
            logger(self.log_path,logging.INFO, 'Converting the list of results into a DataFrame')
            metrics = pd.DataFrame(metrics_list)
            metrics.to_csv('artifact\\regression_models\\metrics.csv', index=False)


            # Train the best model (Random Forest)
            logger(self.log_path,logging.INFO, 'Training the best model (Random Forest)')
            best_model = RandomForestRegressor()
            best_model.fit(X_train, y_train)

            # Save the trained model to a file
            logger(self.log_path,logging.INFO, 'Saving the trained model to a file')
            joblib.dump(best_model, 'artifact\\regression_models\\best_random_forest_model.pkl')

        except Exception as e:
            logger(self.log_path,logging.ERROR, CustomException(e,sys))
            raise CustomException(e,sys)
	


'''
import os
import logging
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from sklearn.linear_model import Ridge, Lasso, ElasticNet

import joblib
import numpy as np
from utils.custom_logging import logger
from utils.custom_exception import CustomException
from src.FlightPricePrediction.components.data_preprocessing_05 import DataPreprocessing
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import numpy as np
from utils.custom_logging import logger
from utils.custom_exception import CustomException
from src.FlightPricePrediction.components.data_preprocessing_05 import DataPreprocessing
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import numpy as np
from utils.custom_logging import logger
from utils.custom_exception import CustomException
from src.FlightPricePrediction.components.data_preprocessing_05 import DataPreprocessing
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

import optuna
import joblib
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from utils.custom_logging import logger
from utils.custom_exception import CustomException
from src.FlightPricePrediction.components.data_preprocessing_05 import DataPreprocessing

class ModelBuild:

    def __init__(self):
        self.data = DataPreprocessing().data_preprocessing()
        self.log_path = 'log\logging_info.log'
        os.makedirs('artifact\\model_building', exist_ok=True)
        self.logger = logger

    def model_building(self):
        try:
            # Split the data into features and target
            self.logger(self.log_path, logging.INFO, 'Splitting the data into features and target')
            X = self.data.drop(columns=['Price'])
            y = self.data['Price']

            # Standardize the data
            self.logger(self.log_path, logging.INFO, 'Standardizing the data')
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split data into training and testing sets (80-20 split)
            self.logger(self.log_path, logging.INFO, 'Splitting the data into training and testing sets')
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Function to optimize multiple models using Optuna
            def optimize_model(trial, model_name):
                if model_name == 'Ridge Regression':
                    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e2)
                    model = Ridge(alpha=alpha)

                elif model_name == 'Lasso Regression':
                    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e2)
                    model = Lasso(alpha=alpha)

                elif model_name == 'ElasticNet Regression':
                    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e2)
                    l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

                elif model_name == 'Random Forest':
                    n_estimators = trial.suggest_int('n_estimators', 100, 500)
                    max_depth = trial.suggest_int('max_depth', 10, 30)
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

                elif model_name == 'Gradient Boosting':
                    n_estimators = trial.suggest_int('n_estimators', 100, 500)
                    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
                    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

                elif model_name == 'SVR':
                    C = trial.suggest_loguniform('C', 1e-5, 1e2)
                    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
                    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
                    model = SVR(C=C, gamma=gamma, kernel=kernel)

                # Fit the model and calculate performance metrics
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)

                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                r2_test = r2_score(y_test, y_pred_test)

                return rmse_test

            # List of models to optimize
            model_names = ['Ridge Regression', 'Lasso Regression', 'ElasticNet Regression', 
                           'Random Forest', 'Gradient Boosting', 'SVR']

            # Initialize the study for hyperparameter optimization
            study_results = []

            for model_name in model_names:
                self.logger(self.log_path, logging.INFO, f"Starting optimization for {model_name}")
                study = optuna.create_study(direction='minimize')
                study.optimize(lambda trial: optimize_model(trial, model_name), n_trials=50)

                # Best trial
                best_trial = study.best_trial
                self.logger(self.log_path, logging.INFO, f"Best params for {model_name}: {best_trial.params}")
                study_results.append({
                    'Model': model_name,
                    'Best Params': best_trial.params,
                    'Test RMSE': best_trial.value
                })

            # Convert study results into a DataFrame
            study_results_df = pd.DataFrame(study_results)
            study_results_df.to_csv('artifact\\regression_models\\optuna_optimization_results_multiple_models.csv', index=False)

            # Find the best model based on RMSE
            best_model_info = study_results_df.loc[study_results_df['Test RMSE'].idxmin()]
            best_model_name = best_model_info['Model']
            best_model_params = best_model_info['Best Params']

            self.logger(self.log_path, logging.INFO, f"Training the best model: {best_model_name} with optimized parameters")

            # Train the best model
            if best_model_name == 'Ridge Regression':
                best_model = Ridge(alpha=best_model_params['alpha'])
            elif best_model_name == 'Lasso Regression':
                best_model = Lasso(alpha=best_model_params['alpha'])
            elif best_model_name == 'ElasticNet Regression':
                best_model = ElasticNet(alpha=best_model_params['alpha'], l1_ratio=best_model_params['l1_ratio'])
            elif best_model_name == 'Random Forest':
                best_model = RandomForestRegressor(n_estimators=best_model_params['n_estimators'], max_depth=best_model_params['max_depth'])
            elif best_model_name == 'Gradient Boosting':
                best_model = GradientBoostingRegressor(n_estimators=best_model_params['n_estimators'], learning_rate=best_model_params['learning_rate'])
            elif best_model_name == 'SVR':
                best_model = SVR(C=best_model_params['C'], gamma=best_model_params['gamma'], kernel=best_model_params['kernel'])

            best_model.fit(X_train, y_train)

            # Calculate the final model's train and test results
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            r2_train = r2_score(y_train, y_pred_train)

            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_test = r2_score(y_test, y_pred_test)

            # Log the final model results
            self.logger(self.log_path, logging.INFO, f"Final Model: {best_model_name}")
            self.logger(self.log_path, logging.INFO, f"Train RMSE: {rmse_train}, Train R^2: {r2_train}")
            self.logger(self.log_path, logging.INFO, f"Test RMSE: {rmse_test}, Test R^2: {r2_test}")

            # Save the trained best model
            joblib.dump(best_model, 'artifact\\regression_models\\best_model_tuned_final.pkl')

        except Exception as e:
            self.logger(self.log_path, logging.ERROR, CustomException(e, sys))
            raise CustomException(e, sys)
