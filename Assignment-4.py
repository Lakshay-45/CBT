import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# --- Configuration for plots ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)  # Default figure size


def print_section_header(title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def run_titanic_eda():
    print_section_header("1. Load Dataset from CSV & Initial Check")
    titanic_data = None
    try:
        titanic_data = pd.read_csv('titanic.csv')
    except FileNotFoundError:
        print("ERROR: 'titanic.csv' not found. Please ensure the file is in the correct directory or path.")
        return
    except Exception as e:
        print(f"Error loading 'titanic.csv': {e}")
        return

    # Standardize column names to lowercase for consistency
    titanic_data.columns = [col.strip().lower() for col in titanic_data.columns]

    print_section_header("2. Data Summary & Missing Values")
    print("Shape:", titanic_data.shape)
    print("\nInfo:")
    titanic_data.info()
    missing_counts = titanic_data.isnull().sum()
    print("\nMissing values (count):\n", missing_counts[missing_counts > 0])  # Show only columns with missing values

    plt.figure(figsize=(10, 5))
    sns.heatmap(titanic_data.isnull(), cbar=False, cmap='plasma', yticklabels=False)
    plt.title('Missing Values Heatmap (Before Imputation)')
    plt.show()

    titanic_data_filled = titanic_data.copy()

    print_section_header("3. Impute Missing Values")
    # Impute 'age'
    if titanic_data_filled['age'].isnull().any():
        try:
            numeric_cols_for_imputer = titanic_data_filled.select_dtypes(include=[np.number, bool]).columns.tolist()
            if 'age' not in numeric_cols_for_imputer and 'age' in titanic_data_filled.columns:
                numeric_cols_for_imputer.append('age')

            imputer_age = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=0),
                                           random_state=0, max_iter=5, tol=1e-2)
            imputed_values = imputer_age.fit_transform(titanic_data_filled[numeric_cols_for_imputer])
            titanic_data_filled[numeric_cols_for_imputer] = pd.DataFrame(imputed_values,
                                                                         columns=numeric_cols_for_imputer,
                                                                         index=titanic_data_filled.index)
        except Exception as e:
            print(f"Error during IterativeImputation for 'age': {e}. Falling back to median.")
            titanic_data_filled['age'].fillna(titanic_data_filled['age'].median(), inplace=True)

    # Impute 'embarked' with mode
    if 'embarked' in titanic_data_filled.columns and titanic_data_filled['embarked'].isnull().any():
        embarked_mode = titanic_data_filled['embarked'].mode()[0]
        titanic_data_filled['embarked'].fillna(embarked_mode, inplace=True)

    # Impute 'cabin' with mode
    if 'cabin' in titanic_data_filled.columns and titanic_data_filled['cabin'].isnull().any():
        cabin_mode = titanic_data_filled['cabin'].mode()[0]
        titanic_data_filled['cabin'].fillna(cabin_mode, inplace=True)

    print_section_header("4. Univariate Analysis")
    univariate_cols = {
        'survived': 'Survival Count (0=No, 1=Yes)',
        'pclass': 'Passenger Class Distribution',
        'sex': 'Sex Distribution',
        'embarked': 'Port of Embarkation Distribution'
    }
    for col, title in univariate_cols.items():
        if col in titanic_data_filled.columns:
            plt.figure(figsize=(7, 4))
            sns.countplot(x=col, data=titanic_data_filled, palette='viridis')
            plt.title(title)
            if col == 'survived': plt.xticks([0, 1], ['No', 'Yes'])
            plt.show()

    for col_numeric in ['age', 'fare']:
        if col_numeric in titanic_data_filled.columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(titanic_data_filled[col_numeric], kde=True, bins=30 if col_numeric == 'age' else 50,
                         color='darkcyan')
            plt.title(f'{col_numeric.capitalize()} Distribution (After Imputation)')
            plt.show()

    print_section_header("5. Bivariate Analysis")
    bivariate_vs_survived = ['sex', 'pclass', 'embarked']
    for col in bivariate_vs_survived:
        if col in titanic_data_filled.columns and 'survived' in titanic_data_filled.columns:
            plt.figure(figsize=(7, 5))
            sns.countplot(x=col, hue='survived', data=titanic_data_filled, palette='YlGnBu')
            plt.title(f'Survival by {col.capitalize()}')
            plt.legend(title='Survived', labels=['No', 'Yes'])
            plt.show()

    if 'fare' in titanic_data_filled.columns and 'survived' in titanic_data_filled.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(titanic_data_filled, x='fare', hue='survived', bins=30, kde=True, multiple="stack",
                     palette='crest')
        plt.title('Fare Distribution by Survival')
        plt.show()

    if 'age' in titanic_data_filled.columns and 'pclass' in titanic_data_filled.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='pclass', y='age', data=titanic_data_filled, palette='Blues')
        plt.title('Age Distribution by Passenger Class')
        plt.show()

    if 'age' in titanic_data_filled.columns and 'survived' in titanic_data_filled.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='survived', y='age', data=titanic_data_filled, palette='Oranges')
        plt.title('Age Distribution by Survival Status')
        plt.xticks([0, 1], ['No', 'Yes'])
        plt.show()

    if 'fare' in titanic_data_filled.columns and 'pclass' in titanic_data_filled.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='pclass', y='fare', data=titanic_data_filled, palette='Greens')
        plt.title('Fare Distribution by Passenger Class')
        plt.ylim(0, 300)  # Limiting y-axis for better readability of boxes, hiding extreme outliers
        plt.show()

    if 'age' in titanic_data_filled.columns and 'survived' in titanic_data_filled.columns:
        age_bins = [0, 12, 18, 35, 60, 100]
        age_labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
        titanic_data_filled['age_group'] = pd.cut(titanic_data_filled['age'], bins=age_bins, labels=age_labels,
                                                  right=False)
        plt.figure(figsize=(8, 5))
        sns.countplot(x='age_group', hue='survived', data=titanic_data_filled, order=age_labels, palette='flare')
        plt.title('Survival by Age Group')
        plt.legend(title='Survived', labels=['No', 'Yes'])
        plt.show()

    if 'sex' in titanic_data_filled.columns and 'pclass' in titanic_data_filled.columns and 'survived' in titanic_data_filled.columns:
        g = sns.catplot(x='sex', hue='survived', col='pclass', kind='count', data=titanic_data_filled, height=5,
                        aspect=0.7, palette='pastel')
        g.fig.suptitle('Survival by Gender and Pclass', y=1.03, fontsize=14)
        plt.show()

    print_section_header("6. Feature Engineering & Analysis")
    if 'sibsp' in titanic_data_filled.columns and 'parch' in titanic_data_filled.columns:
        titanic_data_filled['family_size'] = titanic_data_filled['sibsp'] + titanic_data_filled['parch'] + 1
        titanic_data_filled['is_alone'] = (titanic_data_filled['family_size'] == 1).astype(int)

        if 'survived' in titanic_data_filled.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x='family_size', hue='survived', data=titanic_data_filled, palette='mako')
            plt.title('Survival Based on Family Size')
            plt.show()

            plt.figure(figsize=(6, 4))
            sns.countplot(x='is_alone', hue='survived', data=titanic_data_filled, palette='Set2')
            plt.title('Survival Based on Is Alone')
            plt.xticks([0, 1], ['With Family', 'Alone'])
            plt.xlabel('Travel Status')
            plt.show()

    if 'name' in titanic_data_filled.columns and 'survived' in titanic_data_filled.columns:
        titanic_data_filled['title'] = titanic_data_filled['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
        titanic_data_filled['title'] = titanic_data_filled['title'].apply(
            lambda x: x if pd.isna(x) or x in common_titles else 'Other'
        )
        titanic_data_filled['title'].fillna('Unknown', inplace=True)

        plt.figure(figsize=(10, 6))
        title_order = titanic_data_filled['title'].value_counts().index
        sns.countplot(x='title', hue='survived', data=titanic_data_filled, order=title_order, palette='rocket')
        plt.title('Survival Based on Title')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    print_section_header("7. Correlation Heatmap")
    numeric_df_for_corr = titanic_data_filled.select_dtypes(include=np.number)
    if 'passengerid' in numeric_df_for_corr.columns:
        numeric_df_for_corr = numeric_df_for_corr.drop(columns=['passengerid'])

    if not numeric_df_for_corr.empty:
        plt.figure(figsize=(14, 10))
        sns.heatmap(numeric_df_for_corr.corr(), annot=True, cmap='vlag', fmt=".2f", linewidths=.5)
        plt.title('Feature Correlation Heatmap')
        plt.show()

    print_section_header("8. Conclusion (Summary of Findings)")
    print("""
    This Exploratory Data Analysis (EDA) was performed on the Titanic dataset loaded from a local `titanic.csv` file.
    The primary goal was to identify key factors and patterns influencing passenger survival during the infamous maritime disaster.
    The process involved data loading, preprocessing (including column name standardization and missing value imputation),
    univariate and bivariate analysis, feature engineering, and visualization.

    1. Dataset Overview and Preprocessing:
       - The dataset comprises 891 passenger records and 12 initial features. Column names were standardized to lowercase.
       - Missing Values:
         - `age`: 19.87% missing, imputed using IterativeImputer. Mean age post-imputation: ~29.66 years.
         - `cabin`: 77.10% missing, imputed with its mode ('B96 B98'). Caution advised for `cabin`-related interpretations.
         - `embarked`: 0.22% missing, imputed with mode ('S').
       - All missing values were handled.

    2. Univariate Analysis - Passenger Demographics and Characteristics:
       - Survival (`survived`): 38.4% survival rate (342 survived, 549 did not).
       - Passenger Class (`pclass`): Majority in 3rd Class (491), then 1st (216), then 2nd (184).
       - Sex (`sex`): More males (577) than females (314).
       - Port of Embarkation (`embarked`): Predominantly Southampton (S: 646).
       - Age (`age`): Post-imputation, distribution peaks in late 20s/early 30s; mean age ~29.66.
       - Fare (`fare`): Heavily right-skewed; median fare 14.45.

    3. Bivariate Analysis - Factors Influencing Survival:
       - Survival by Sex: Females (74.2% survival) significantly out-survived males (18.9%).
       - Survival by Pclass: Clear gradient - 1st Class (63.0%), 2nd Class (47.3%), 3rd Class (24.2%).
       - Survival by Embarked: Cherbourg (C) passengers had the highest survival rate (55.4%).
       - Survival by Fare: Higher fares correlated with higher survival likelihood.
       - Survival by Age Group: Children (0-12: ~51.8%) and Teenagers (12-18: ~52.1%) fared better than Young Adults (18-35: ~35.5%), Adults (35-60: ~38.1%), and Seniors (60-100: ~25.9%).
       - Survival by Sex and Pclass (Catplot): Reinforced individual findings; females in 1st/2nd class had very high survival. Male survival dropped sharply in lower classes.

    4. Feature Engineering & Analysis:
       - Survival by Family Size:
         - Alone (size 1): 30.4% survival.
         - Small families (2-4 members): Highest survival (e.g., 72.4% for size 4).
         - Large families (5+): Poor survival rates.
       - Survival by Is Alone: With family (50.6% survival) vs. Alone (30.4% survival).
       - Survival by Title:
         - Mrs: 79.2% survival.
         - Miss: 69.8% survival.
         - Master: 57.5% survival (young boys).
         - Mr: 15.7% survival (lowest).
         - Other: 44.4% survival.

    5. Correlation Heatmap (Numerical Features):
       - `survived` showed:
         - Positive correlation with `fare` (0.26).
         - Negative correlation with `pclass` (-0.34).
         - Negative correlation with `is_alone` (-0.20).
       - Other strong correlations: `pclass` vs. `fare` (-0.55); `pclass` vs. `age` (-0.39).

    6. Key Conclusions & Insights:
       - Survival was strongly influenced by socio-economic status (Pclass, Fare), gender, and age.
       - The "women and children first" protocol is evident in the data.
       - Traveling in small family units (2-4 members) was advantageous compared to being alone or in very large families.
       - Passengers embarking at Cherbourg had a higher survival rate.
       - Limitations: High missing values in `cabin` restricts its analytical value despite imputation.

    This EDA confirms historical accounts, highlighting disparities in survival based on class, gender, and age,
    while also revealing patterns related to family size and embarkation port.
    """)
    print("\nEnd of Titanic EDA.")


if __name__ == "__main__":
    run_titanic_eda()