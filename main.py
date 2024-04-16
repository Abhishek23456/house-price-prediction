from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load data and model
data = pd.read_csv("Cleaned_data.csv")

# Check if 'location' column exists in the dataset
if 'location' not in data.columns:
    raise ValueError("Error: 'location' column is missing from the dataset.")

pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

# Label encoding for 'location' column
label_encoder = LabelEncoder()
data['location_encoded'] = label_encoder.fit_transform(data['location'])

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        location = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        sqft = float(request.form.get('total_sqft'))
        print(location,bhk,bath,sqft)

        # Encode location
       # if location not in label_encoder.classes_:
        #    return "Error: Location not found in the training data."

        location_encoded = label_encoder.transform([location])[0]

        # Create input DataFrame
        input_data = pd.DataFrame({'location_encoded': [location_encoded], 
                                   'total_sqft': [sqft], 
                                   'bath': [bath], 
                                   'bhk': [bhk]})

        # Make prediction
        prediction = pipe.predict(input_data)[0] * 1e5
        return str(np.round(prediction, 2))

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
