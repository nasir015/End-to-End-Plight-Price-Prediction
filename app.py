from flask import Flask, render_template, request
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the trained model using joblib
model = joblib.load('best_random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        departure_datetime = request.form['Departure_Date']
        arrival_datetime = request.form['Arrival_Date']

        # month of journey, day of journey 
        month_of_journey = datetime.strptime(departure_datetime, "%Y-%m-%dT%H:%M").month
        day_of_journey = datetime.strptime(departure_datetime, "%Y-%m-%dT%H:%M").day

        # Extract hours and minutes from the datetime inputs
        dep_time = datetime.strptime(departure_datetime, "%Y-%m-%dT%H:%M")
        arr_time = datetime.strptime(arrival_datetime, "%Y-%m-%dT%H:%M")
        
        dep_hour = dep_time.hour
        dep_minute = dep_time.minute
        arr_hour = arr_time.hour
        arr_minute = arr_time.minute
        
        # Get the duration input from the floating window (h:m format)
        duration = request.form['Duration']
        if duration:
            duration_parts = duration.split(":")
            duration_hours = int(duration_parts[0])
            duration_minutes = int(duration_parts[1])
        else:
            duration_hours = 0
            duration_minutes = 0

        # Define function to categorize time of day
        def categorize_time_of_day(dep_time):
            if 6 <= dep_time < 12:
                return 2  # Morning
            elif 12 <= dep_time < 18:
                return 0  # Afternoon
            elif 18 <= dep_time < 21:
                return 1  # Evening
            else:
                return 3  # Night

        # Categorize time of day based on dep_hour
        Time_of_Day = categorize_time_of_day(dep_hour)

        # Define function to categorize flight duration
        def categorize_flight_duration(duration):
            if duration <= 300:
                return 2  # Short
            elif 300 < duration <= 600:
                return 1  # Medium
            else:
                return 0  # Long

        # Categorize flight duration
        Flight_Duration_Category = categorize_flight_duration(duration_hours * 60 + duration_minutes)

        # Define function to categorize season
        def categorize_season(month):
            if month in [12, 1, 2]:
                return 4  # Winter
            elif month in [3, 4, 5]:
                return 0  # Spring
            elif month in [6, 7, 8]:
                return 1  # Summer
            else:
                return 3  # Fall

        # Categorize season based on month_of_journey
        Season = categorize_season(month_of_journey)

        # Get other features from the form
        features = [
            int(request.form['Airline']),
            int(request.form['Source']),
            int(request.form['Destination']),
            int(request.form['Total_Stops']),
            duration_hours * 60 + duration_minutes,  # Convert duration to minutes
            month_of_journey,
            day_of_journey,
            Season,
            Time_of_Day,
            Flight_Duration_Category,
            dep_hour,
            arr_hour,
            dep_minute,
            arr_minute
        ]
        
        # Convert input to numpy array and reshape for prediction
        input_array = np.array(features).reshape(1, -1)
        
        # Predict price
        predicted_price = model.predict(input_array)[0]
        
        # Render the result page with the predicted price
        return render_template('result.html', predicted_price=round(predicted_price, 2))
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
