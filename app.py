from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import pandas as pd
import numpy as np  # Add this import statement
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Declare feature_importances as a global variable
feature_importances = None
model = None
label_encoder = None
X_train = None
y_train = None

def train_model():
    global model, label_encoder, feature_importances, X_train, y_train, X_test, y_test

    # Load your datasets (replace 'your_data.csv' with your actual dataset names)
    data1 = pd.read_csv('/home/sir-derrick/Desktop/4.1/soil.csv')
    data2 = pd.read_csv('/home/sir-derrick/Desktop/4.1/soil1.csv')
    data = pd.read_csv('/home/sir-derrick/Desktop/4.1/sorted.csv')

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
    # Train the model on the entire dataset
    model.fit(features, target)

    # Feature names
    feature_names = features.columns

    # Get feature importances using permutation importance
    perm_importance = permutation_importance(model, features, target, n_repeats=30, random_state=42)

    # Set feature_importances as a global variable
    feature_importances = dict(zip(feature_names, perm_importance.importances_mean))
    
# Call train_model at the beginning of the script or outside the predict route
train_model()
# Convert numpy.int64 to standard Python int
def convert_np_int64_to_int(value):
    if isinstance(value, np.int64):
        return int(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()  # Convert NumPy array to list
    elif isinstance(value, dict):
        return {k: convert_np_int64_to_int(v) for k, v in value.items()}  # Recursively convert values in dictionaries
    return value

@app.route('/predict', methods=['POST'])
def predict():
    global model, label_encoder, feature_importances, X_train, y_train, X_test, y_test

    try:
        # Get data from the request
        data = request.json.get('data')

        if not data:
            raise ValueError("Invalid request format. 'data' key is missing.")

        # Convert data to DataFrame
        new_data = pd.DataFrame([data])  # <-- Remove the list wrapping
        # Convert numpy.int64 values to int
        new_data = new_data.applymap(convert_np_int64_to_int)

        # Convert non-numeric target variable to numeric
        new_data['label'] = label_encoder.transform(['rice'])  # Assuming 'rice' is the label for the first row
        # Make probability predictions for new data
        probability_predictions = model.predict_proba(new_data.drop('label', axis=1))

        # Get the classes and their corresponding probabilities
        classes = label_encoder.classes_
        probabilities = probability_predictions[0]  # Assuming there's only one row of new data

        # Create a dictionary with crop labels and their probabilities
        crop_probabilities = dict(zip(classes, probabilities))

        # Set a threshold for predicted probabilities
        threshold = 0.5

        # Filter crops with probabilities above the threshold
        potential_crops = {crop: prob for crop, prob in crop_probabilities.items() if prob >= threshold}

        # Initialize response dictionary
        response_data = {"success": True, "improvements": {}}

        # Find the top predicted crop and its probability
        top_crop = max(potential_crops, key=potential_crops.get, default=None)
        top_probability = potential_crops.get(top_crop, 0.0)

        # Find the second highest predicted crop and its probability
        sorted_crops = sorted(potential_crops, key=potential_crops.get, reverse=True)
        second_crop = sorted_crops[1] if len(sorted_crops) > 1 else None
        second_probability = potential_crops.get(second_crop, 0.0)

        # Check if the difference between top and second probabilities is below the threshold
        if potential_crops and top_probability - second_probability < 0.2:
            response_data["message"] = f"Consider improvements for {top_crop}."

            # Get feature importances using permutation importance
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

            # Set feature_importances as a global variable
            feature_importances = perm_importance.importances_mean

            # Feature names
            feature_names = X_test.columns

            # Check if feature_importances is not None before using it
            if feature_importances is not None:
                # Find the features that significantly contribute to the prediction
                significant_features = [(feature_names[i], feature_importances[i]) for i, importance in enumerate(feature_importances) if importance > 0]

                if significant_features:
                    response_data["improvements"][top_crop] = {"suggested_improvements": []}

                    for feature, importance in significant_features:
                        # Calculate the range of improvement (assuming normal distribution of feature values)
                        std_dev = X_train[feature].std()
                        improvement_range = std_dev * 0.5  # You can adjust the multiplier as needed

                        # Add specific recommendations for each feature to the response
                        response_data["improvements"][top_crop]["suggested_improvements"].append({
                            "feature": feature,
                            "current_value": convert_np_int64_to_int(new_data[feature].values[0]),
                            "improvement_range": improvement_range
                        })

                else:
                    response_data["improvements"][top_crop]["message"] = "No specific improvement recommendations."

                # Check for alternative crops with higher probability threshold
                alternative_crops = {crop: prob for crop, prob in crop_probabilities.items() if 0 < prob < 0.5}  # Exclude crops with prob >= 0.5

                if alternative_crops:  # Check if alternative_crops is not empty
                    # Sort alternative crops based on probabilities in descending order
                    alternative_crops = dict(sorted(alternative_crops.items(), key=lambda item: item[1], reverse=True))
                    response_data["alternative_crops"] = {}

                    for alt_crop, alt_prob in alternative_crops.items():
                        response_data["alternative_crops"][alt_crop] = {"probability": alt_prob, "suggested_improvements": []}

                        # Find the features that significantly contribute to the prediction for the alternative crop
                        alt_significant_features = None  # Initialize to None

                        # Get feature importances using permutation importance
                        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

                        # Set feature_importances as a local variable
                        alt_feature_importances = perm_importance.importances_mean

                        # Feature names
                        feature_names = X_test.columns

                        # Check if alt_feature_importances is not None before using it
                        if alt_feature_importances is not None:
                            # Find the features that significantly contribute to the prediction
                            alt_significant_features = [(feature_names[i], alt_feature_importances[i]) for i, importance in enumerate(alt_feature_importances) if importance > 0]

                        # Set alt_crop within the loop
                        alt_crop = alt_crop

                        if alt_significant_features:
                            for feature, importance in alt_significant_features:
                                # Calculate the average requirements from the dataset for each feature
                                avg_requirements = X_train[X_train.index.isin(y_train[y_train == label_encoder.transform([alt_crop])[0]].index)].mean()

                                # Add specific recommendations for each feature to the response
                                response_data["alternative_crops"][alt_crop]["suggested_improvements"].append({
                                    "feature": feature,
                                    "current_value": convert_np_int64_to_int(new_data[feature].values[0]),
                                    "target_value": avg_requirements[feature] + 0.5  # You can adjust the multiplier as needed
                                })

                        else:
                            response_data["alternative_crops"][alt_crop]["message"] = "No specific improvement recommendations."

                else:
                    response_data["message"] = "No alternative crops found with higher probability threshold."

            else:
                response_data["message"] = "No feature importances available."

        else:
            response_data["message"] = "Soil has suitable conditions for the above provided crops"

            # Check for alternative crops with higher probability threshold
            alternative_crops = {crop: prob for crop, prob in crop_probabilities.items() if 0 < prob < 0.5}  # Exclude crops with prob >= 0.5

            if alternative_crops:  # Check if alternative_crops is not empty
                # Sort alternative crops based on probabilities in descending order
                alternative_crops = dict(sorted(alternative_crops.items(), key=lambda item: item[1], reverse=True))
                response_data["alternative_crops"] = {}

                for alt_crop, alt_prob in alternative_crops.items():
                    response_data["alternative_crops"][alt_crop] = {"probability": alt_prob, "suggested_improvements": []}

                    # Find the features that significantly contribute to the prediction for the alternative crop
                    alt_significant_features = None  # Initialize to None

                    # Get feature importances using permutation importance
                    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

                    # Set feature_importances as a local variable
                    alt_feature_importances = perm_importance.importances_mean

                    # Feature names
                    feature_names = X_test.columns

                    # Check if alt_feature_importances is not None before using it
                    if alt_feature_importances is not None:
                        # Find the features that significantly contribute to the prediction
                        alt_significant_features = [(feature_names[i], alt_feature_importances[i]) for i, importance in enumerate(alt_feature_importances) if importance > 0]

                    # Set alt_crop within the loop
                    alt_crop = alt_crop

                    if alt_significant_features:
                        for feature, importance in alt_significant_features:
                            # Calculate the average requirements from the dataset for each feature
                            avg_requirements = X_train[X_train.index.isin(y_train[y_train == label_encoder.transform([alt_crop])[0]].index)].mean()

                            # Add specific recommendations for each feature to the response
                            response_data["alternative_crops"][alt_crop]["suggested_improvements"].append({
                                "feature": feature,
                                "current_value": convert_np_int64_to_int(new_data[feature].values[0]),
                                "target_value": avg_requirements[feature] + 0.5  # You can adjust the multiplier as needed
                            })

                    else:
                        response_data["alternative_crops"][alt_crop]["message"] = "No specific improvement recommendations."

                #print("No alternative crops.")

            else:
                response_data["message"] = "No more alternative crops found with higher probability threshold."
        # Convert NumPy int64 to int for jsonify compatibility
        response_data = {key: convert_np_int64_to_int(value) if key != 'improvements' else value for key, value in response_data.items()}

    except Exception as e:
        return jsonify({"error": str(e)})

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True)
