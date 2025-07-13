"""
Model Evaluation and Hyperparameter Tuning.

This script trains and evaluates multiple machine learning models on the wine
dataset. It first evaluates the models with their default parameters. It then
performs hyperparameter tuning using both GridSearchCV and RandomizedSearchCV
to find optimal parameters. The script concludes by identifying the best
model with default parameters and the best model overall.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_wine
from sklearn.exceptions import NotFittedError

def model_evaluation_and_tuning():
    """
    Trains and evaluates multiple machine learning models with and without
    hyperparameter tuning.
    """
    # Load the wine dataset
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models and their hyperparameter grids
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(solver='liblinear', max_iter=1000),
            "params": {
                "penalty": ["l1", "l2"],
                "C": [0.1, 1, 10, 100]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
        }
    }

    results = []

    # First, evaluate models with default parameters
    print("--- Evaluating Models with Default Parameters ---")
    for model_name, config in models.items():
        model = config["model"]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results.append({
            "model": model_name,
            "tuner": "Default",
            "best_params": "Default",
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred)
        })
        print(f"Finished evaluating default {model_name}.")

    # Perform GridSearchCV
    print("\n--- Performing GridSearchCV ---")
    for model_name, config in models.items():
        gs = GridSearchCV(
            config["model"],
            config["params"],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results.append({
            "model": model_name,
            "tuner": "GridSearchCV",
            "best_params": gs.best_params_,
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred)
        })
        print(f"Finished GridSearchCV for {model_name}.")


    # Perform RandomizedSearchCV
    print("\n--- Performing RandomizedSearchCV ---")
    for model_name, config in models.items():
        # Check if the parameter distribution has enough combinations for n_iter
        param_combinations = 1
        for key in config["params"]:
            param_combinations *= len(config["params"][key])

        # Use a smaller n_iter if there are fewer combinations than the default 10
        n_iter = min(10, param_combinations)

        rs = RandomizedSearchCV(
            config["model"],
            config["params"],
            cv=5,
            scoring='accuracy',
            n_iter=n_iter,
            n_jobs=-1,
            random_state=42
        )
        try:
            rs.fit(X_train, y_train)
            y_pred = rs.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            results.append({
                "model": model_name,
                "tuner": "RandomizedSearchCV",
                "best_params": rs.best_params_,
                "accuracy": accuracy,
                "classification_report": classification_report(y_test, y_pred)
            })
            print(f"Finished RandomizedSearchCV for {model_name}.")
        except NotFittedError as e:
            print(f"Could not run RandomizedSearchCV for {model_name}: {e}")

    # --- Analysis and Reporting ---
    best_model_info = None
    best_accuracy = 0.0
    best_default_model_info = None
    best_default_accuracy = 0.0

    print("\n\n--- Full Model Evaluation Report ---")
    for result in results:
        print(f"\nModel: {result['model']} (Tuner: {result['tuner']})")
        print(f"Parameters: {result['best_params']}")
        print(f"Test Set Accuracy: {result['accuracy']:.4f}")
        print("Classification Report:")
        print(result['classification_report'])
        print("-" * 50)

        # Update best overall model
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_model_info = result

        # Update best model from the default set
        if result['tuner'] == 'Default' and result['accuracy'] > best_default_accuracy:
            best_default_accuracy = result['accuracy']
            best_default_model_info = result

    # Final summary
    if best_model_info:
        print("\n--- Final Analysis ---")
        if best_default_model_info:
            print(
                f"The best performing model without hyperparameter tuning is "
                f"{best_default_model_info['model']} with an accuracy of "
                f"{best_default_model_info['accuracy']:.4f}."
            )
        else:
            print("Could not determine the best model with default parameters.")

        print(
            f"\nThe best performing model overall is {best_model_info['model']} "
            f"with an accuracy of {best_model_info['accuracy']:.4f}, found "
            f"using the {best_model_info['tuner']} method."
        )

        if best_model_info['tuner'] != 'Default':
            print(
                "The best hyperparameters are: "
                f"`{best_model_info['best_params']}`."
            )
        else:
            print("This model used its default hyperparameters.")
    else:
        print("\n--- No results to analyze ---")

if __name__ == '__main__':
    model_evaluation_and_tuning()