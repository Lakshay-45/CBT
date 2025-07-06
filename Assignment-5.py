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

# 2. Feature Engineering
# Create new features by combining existing ones
all_df['TotalSF'] = all_df['TotalBsmtSF'] + all_df['1stFlrSF'] + all_df['2ndFlrSF']
all_df['TotalBath'] = all_df['FullBath'] + (0.5 * all_df['HalfBath']) + \
                      all_df['BsmtFullBath'] + (0.5 * all_df['BsmtHalfBath'])
all_df['Age'] = all_df['YrSold'] - all_df['YearBuilt']
all_df['RemodAge'] = all_df['YrSold'] - all_df['YearRemodAdd']
all_df['OverallQual_sq'] = all_df['OverallQual']**2

# 3. Preprocessing
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

# Split original training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 4. Train Model
model = Ridge(alpha=10)
model.fit(X_train, y_train)

# 5. Evaluate Model on Validation Set
val_predictions = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"Model Performance (RMSE on validation set): {rmse:.4f}")

# 6. Predict for Submission
test_predictions_log = model.predict(X_test)
test_predictions = np.expm1(test_predictions_log)

# Create the submission file
submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': test_predictions})
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully.")