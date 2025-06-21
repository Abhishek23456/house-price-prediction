from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

# Load the dataset to get locations
data = pd.read_csv("Cleaned_data.csv")
locations = sorted(data['location'].unique())

# Label encoding for location
le = LabelEncoder()
data['location'] = le.fit_transform(data['location'])

@app.route('/')
def index():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        bhk = request.form.get('bhk')
        bath = request.form.get('bath')
        sqft = request.form.get('total_sqft')

        # Validate if inputs are empty or invalid
        if not location or not bhk or not bath or not sqft:
            return "All fields are required."

        # Ensure that bhk and bath are integers and sqft is a float
        try:
            bhk = int(bhk)
            bath = int(bath)
            sqft = float(sqft.replace(" ", ""))  # Remove spaces and convert to float
        except ValueError:
            return "Please enter valid numbers for BHK, Bathrooms, and Square Feet."

        # Check if the location is valid
        if location not in locations:
            return "Location not found in the dataset. Please choose a valid location."

        # Encode the location using LabelEncoder
        location_encoded = le.transform([location])[0]

        # Prepare the input data
        input_data = pd.DataFrame([[location_encoded, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

        # Predict the price
        prediction = pipe.predict(input_data)[0] * 1e5  # Scale the result by 100,000
        return str(np.round(prediction, 2))  # Return prediction as a string

    except Exception as e:
        return f"Error: {str(e)}"  # Return any error message encountered

if __name__ == "__main__":
    app.run(debug=True, port=5001)
