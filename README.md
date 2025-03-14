
# Flight Price Prediction

This project builds a machine learning model to predict flight prices using various features such as airline, source, destination, total stops, flight duration, and other factors. The project utilizes a **Random Forest Regressor** model to predict the price of a flight based on historical data.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
- [Running the Application](#running-the-application)
  - [Training the Model](#training-the-model)
  - [Running the Flask App](#running-the-flask-app)
  - [Running with Docker](#running-with-docker)
- [Features](#features)
  - [Example Input Features](#example-input-features)
- [Project Algotithm](#project-algotithm)
- [Application Screenshot](#application-screenshot)

## Project Overview

The goal of this project is to predict flight prices accurately based on several input features, which include the airline, source and destination cities, number of stops, duration of the flight, and more. The model is built using **Random Forest** and is trained and tested on a dataset that contains features influencing flight prices.

### Key Results

- **Final Model**: Random Forest Regressor
- **Train RMSE**: 1207.05
- **Train R²**: 0.93
- **Test RMSE**: 1801.00
- **Test R²**: 0.84

These results indicate a strong model performance with a relatively small error on the test data and a high coefficient of determination (R²) for the training data.

## Technologies Used

- **Python**: Programming language used for the entire project.
- **scikit-learn**: For building machine learning models and data preprocessing.
- **Flask**: For creating the web application and serving the model.
- **Joblib**: For saving and loading the trained machine learning model.
- **Docker**: For containerizing the application and ensuring a consistent runtime environment.

## Installation

### Prerequisites

To run this project locally, ensure you have the following installed:

- Python (>= 3.7)
- pip (Python package manager)

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/nasir015/End-to-End-Plight-Price-Prediction.git
cd flight-price-prediction
```

### Install Dependencies

Make sure you have all the required Python libraries installed by running:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes dependencies like `Flask`, `joblib`, `scikit-learn`, and other necessary libraries.

## Running the Application

### Training the Model

1. To train the model, run the `train_model.py` script. This will preprocess the data, train the Random Forest model, and save it as `best_random_forest_model.pkl` in the `artifact\\regression_models\\` folder.

```bash
python train_model.py
```

### Running the Flask App

Once the model is trained, you can start the Flask web application with the following command:

```bash
python app.py
```

By default, the app will run on `http://127.0.0.1:5000/`. Open this in your browser to interact with the application.

### Running with Docker

To run the application inside a Docker container, use the following steps:

1. **Build the Docker image**:
   ```bash
   docker build -t flight-price-prediction .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 5000:5000 flight-price-prediction
   ```

Now, the application will be available at `http://localhost:5000/`.

## Features

- **Flight Price Prediction**: The app allows users to input various flight features and predicts the price based on the trained model.
- **Model Insights**: The app uses a Random Forest Regressor model and provides real-time predictions with a user-friendly interface.

### Example Input Features

| Feature                     | Description                                                        |
|-----------------------------|--------------------------------------------------------------------|
| **Airline**                  | The airline operating the flight.                                  |
| **Source**                   | The departure city.                                               |
| **Destination**              | The arrival city.                                                 |
| **Total Stops**              | The number of stops during the flight (Non-stop, 1 stop, 2 stops). |
| **Duration**                 | The total duration of the flight in minutes.                       |
| **Month of Journey**         | The month when the flight is scheduled.                            |
| **Day of Journey**           | The day when the flight is scheduled.                              |
| **Season**                   | The season in which the flight takes place (Spring, Summer, etc.). |
| **Time of Day**              | The time of day the flight takes place (Morning, Afternoon, etc.). |
| **Flight Duration Category** | Categorized flight duration (Long, Medium, or Short).             |
| **Dep Hour**                 | Hour of departure.                                                |
| **Arr Hour**                 | Hour of arrival.                                                  |
| **Dep Minute**               | Minute of departure.                                              |
| **Arr Minute**               | Minute of arrival.                                                |

## Project Algotithm
![Project-algorithm](https://github.com/nasir015/End-to-End-Plight-Price-Prediction/blob/main/image/Untitled%20Diagram.drawio.png)

## Application Screenshot
![app-ss](https://github.com/nasir015/End-to-End-Plight-Price-Prediction/blob/main/image/Screenshot_4.png)
