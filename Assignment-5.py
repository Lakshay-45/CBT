import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import skew

# 1. Load Data
train_df = pd.read_csv('train-house.csv')
test_df = pd.read_csv('test-house.csv')

# Store test IDs for submission and combine data for preprocessing
test_ids = test_df['Id']
all_df = pd.concat([train_df.drop('SalePrice', axis=1), test_df], axis=0).drop('Id', axis=1)

# 2. Preprocessing and Feature Engineering
# Log-transform the target variable
y_log = np.log1p(train_df['SalePrice'])

# Handle missing values for features with 'NA' as a specific category
for col in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
            'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
    all_df[col] = all_df[col].fillna('None')

# Handle missing values for other categorical and numerical features
for col in all_df.columns:
    if all_df[col].dtype == 'object':
        all_df[col] = all_df[col].fillna(all_df[col].mode()[0])
    else:
        all_df[col] = all_df[col].fillna(all_df[col].median())

# Transform skewed numerical features
numeric_feats = all_df.dtypes[all_df.dtypes != "object"].index
skewed_feats = all_df[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75].index
all_df[skewed_feats] = np.log1p(all_df[skewed_feats])

# One-hot encode categorical features
all_df = pd.get_dummies(all_df, drop_first=True)

# Separate the combined data back into training and testing sets
X = all_df[:len(train_df)]
X_test = all_df[len(train_df):]


# 3. Model Evaluation (Calculating "Accuracy")
# We split our training data to create a validation set for evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Train a Ridge model on the training portion
model_for_eval = Ridge(alpha=10)
model_for_eval.fit(X_train, y_train)
y_pred_val = model_for_eval.predict(X_val)

# Calculate RMSE, a common metric for regression accuracy
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Model Performance (RMSE on validation set): {rmse:.4f}")
print("This score indicates the typical error in our model's price predictions on a log scale.")


# 4. Train Final Model and Predict for Submission
# Train a new model on the entire dataset for best performance
final_model = Ridge(alpha=10)
final_model.fit(X, y_log)

# Predict on the test data and reverse the log transformation
final_preds_log = final_model.predict(X_test)
final_preds = np.expm1(final_preds_log)

# Create the submission file
submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': final_preds})
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully.")