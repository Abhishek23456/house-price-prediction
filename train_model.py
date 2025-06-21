import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pickle

# Load your dataset
data = pd.read_csv("Cleaned_data.csv")

# Separate features and target
X = data[['location', 'total_sqft', 'bath', 'bhk']]
y = data['price']

# Label encoding for the location feature
le = LabelEncoder()

# Use .loc to avoid the SettingWithCopyWarning
X.loc[:, 'location'] = le.fit_transform(X['location'])

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['total_sqft', 'bath', 'bhk']),
        ('cat', 'passthrough', ['location'])  # 'location' is already encoded
    ])

# Create a full pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('ridge', Ridge())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model and the label encoder
with open('RidgeModel.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the label encoder
with open('LabelEncoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)
