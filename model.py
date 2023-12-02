# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_model(data):
    # Assume 'crop_type' is the non-numeric target variable
    target_column = 'label'

    # Use LabelEncoder to convert non-numeric target variable to numeric
    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])

    # Features and target variables
    features = data.drop(target_column, axis=1)

    # Drop 'rainfall' and 'temperature' columns
    features = features.drop(['rainfall', 'temperature'], axis=1)

    target = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Create a machine learning pipeline with a RandomForestClassifier
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    # Train the model on the training data
    model.fit(X_train, y_train)

    return model, label_encoder, X_test, y_test
