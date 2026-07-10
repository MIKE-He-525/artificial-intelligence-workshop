import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load and inspect the traffic dataset
def load_traffic_data(file_path):
    """Load the traffic dataset and define column names as per the workshop documentation"""
    columns = ['day_of_week', 'time_of_day', 'opponent_team', 'has_game', 'vehicle_count']
    data = pd.read_csv(file_path, names=columns, header=None)
    print("First 5 rows of the dataset:")
    print(data.head())
    print("\nBasic dataset information:")
    print(data.info())
    return data

# 2. Preprocess data (handle categorical and time features)
def preprocess_data(data):
    """
    Preprocess the dataset:
    - Convert time feature ("HH:MM") to total minutes
    - Encode categorical features (day_of_week, opponent_team, has_game)
    - Split into training and testing sets (80% train, 20% test)
    """
    # Convert time_of_day from "HH:MM" format to total minutes
    data['time_minutes'] = data['time_of_day'].apply(
        lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1])
    )
    # Drop the original time_of_day column (replaced by time_minutes)
    data = data.drop('time_of_day', axis=1)
    
    # Define categorical and numerical features
    categorical_features = ['day_of_week', 'opponent_team', 'has_game']
    numerical_features = ['time_minutes']
    
    # One-Hot Encoding for categorical features (avoid multicollinearity with drop='first')
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    # Column transformer to apply different preprocessing to different feature types
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', 'passthrough', numerical_features)  # No preprocessing for numerical features
        ])
    
    # Split features (X) and target variable (y) - target is vehicle_count
    X = data.drop('vehicle_count', axis=1)
    y = data['vehicle_count']
    
    # Split into training and testing sets (consistent with workshop's common split ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, shuffle=True
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

# 3. Build and train the Extremely Random Forest Regressor
def build_train_erf_regressor(preprocessor, X_train, y_train):
    """
    Build a pipeline combining preprocessing and Extremely Random Forest Regressor:
    - Preprocessing step (feature encoding)
    - Model training step (Extremely Random Forest Regressor)
    """
    # Initialize Extremely Random Forest Regressor with parameters from the workshop
    erf_regressor = ExtraTreesRegressor(
        n_estimators=100,  # Number of trees (consistent with workshop's classifier settings)
        max_depth=4,       # Maximum tree depth to avoid overfitting
        random_state=7,    # Random seed for reproducibility
        n_jobs=-1          # Use all CPU cores for faster training
    )
    
    # Create a pipeline to chain preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', erf_regressor)
    ])
    
    # Train the model on the training data
    pipeline.fit(X_train, y_train)
    print("\nExtremely Random Forest Regressor training completed!")
    return pipeline

# 4. Evaluate model performance
def evaluate_model(pipeline, X_test, y_test):
    """Evaluate model performance using MAE, MSE, and EVS (as in the workshop's evaluation logic)"""
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error (mentioned in workshop's expected result)
    mse = mean_squared_error(y_test, y_pred)   # Mean Squared Error
    evs = explained_variance_score(y_test, y_pred)  # Explained Variance Score (closer to 1 is better)
    
    # Print evaluation results
    print("\n" + "="*50)
    print("Model Evaluation Results")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Explained Variance Score (EVS): {evs:.2f}")
    print("="*50)
    return mae, y_pred

# 5. Predict traffic for custom input data (as per workshop example)
def predict_traffic(pipeline, input_data):
    """
    Predict traffic volume for custom input:
    - Input format: {'day_of_week': 'Saturday', 'time_of_day': '10:20', 'opponent_team': 'Atlanta', 'has_game': 'no'}
    - Returns rounded predicted vehicle count (since vehicle count is an integer)
    """
    # Convert input data to DataFrame (matching the format of training data)
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input data (same logic as training data: convert time to minutes)
    input_df['time_minutes'] = input_df['time_of_day'].apply(
        lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1])
    )
    input_df = input_df.drop('time_of_day', axis=1)
    
    # Make prediction
    predicted_vehicle = pipeline.predict(input_df)[0]
    # Round to integer (vehicle count cannot be a decimal)
    predicted_vehicle = round(predicted_vehicle)
    
    # Print prediction result (matching the workshop's example output format)
    print("\n" + "="*50)
    print("Traffic Volume Prediction Result")
    print("="*50)
    print(f"Test datapoint: {input_data}")
    print(f"Predicted traffic: {predicted_vehicle}")
    print("="*50)
    return predicted_vehicle

# 6. Main function to run the entire workflow
def main():
    # Path to the dataset (update this path if your file is in a different directory)
    file_path = 'traffic_data.txt'
    
    # Step 1: Load the traffic dataset
    data = load_traffic_data(file_path)
    
    # Step 2: Preprocess the data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)
    
    # Step 3: Build and train the model
    pipeline = build_train_erf_regressor(preprocessor, X_train, y_train)
    
    # Step 4: Evaluate the model
    mae, y_pred = evaluate_model(pipeline, X_test, y_test)
    
    # Step 5: Predict traffic for the example datapoint from the workshop
    test_input = {
        'day_of_week': 'Saturday',
        'time_of_day': '10:20',
        'opponent_team': 'Atlanta',
        'has_game': 'no'
    }
    predict_traffic(pipeline, test_input)

# Execute the main function
if __name__ == "__main__":
    main()
    