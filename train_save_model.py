"""
This script trains a RandomForestClassifier on the complete wine dataset and
saves the trained model to a file for later use in a web application.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import joblib

def train_and_save():
    """
    Trains a RandomForest model on the wine dataset and saves it.
    """
    # Load the wine dataset
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)

    # Initialize and train the RandomForestClassifier with default parameters
    # This model was identified as the best performer last week
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    print("Model training complete.")

    # Save the trained model to a file
    joblib.dump(model, 'wine_quality_model.joblib')
    print("Model saved successfully as 'wine_quality_model.joblib'")

if __name__ == '__main__':
    train_and_save()