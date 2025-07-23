"""
This script performs a full Customer Lifetime Value (CLV) analysis on the
Online Retail dataset. It covers data download from both years (2009-2011),
cleaning, feature engineering, model fitting (BG/NBD and Gamma-Gamma),
and CLV prediction.
"""

import pandas as pd
import lifetimes
import requests
import zipfile
import io
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- 1. Data Loading and Preparation ---

def download_and_load_data():
    """
    Downloads, extracts, and loads both sheets from the Online Retail dataset,
    then combines them into a single DataFrame.
    """
    print("Downloading and loading data...")
    url = 'https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip'
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise an exception for bad status codes
        z = zipfile.ZipFile(io.BytesIO(r.content))

        # The zip file contains an Excel file
        file_name = z.namelist()[0]
        print(f"Reading from file: {file_name}")

        # Read all sheets from the Excel file into a dictionary of DataFrames
        all_sheets = pd.read_excel(z.open(file_name), sheet_name=None)

        # Combine the DataFrames from all sheets into a single DataFrame
        df_list = []
        for sheet_name, sheet_df in all_sheets.items():
            print(f"  - Loading sheet: '{sheet_name}' with {len(sheet_df)} rows")
            df_list.append(sheet_df)

        combined_df = pd.concat(df_list, ignore_index=True)

        print(f"Data from all sheets combined successfully. Total shape: {combined_df.shape}")
        return combined_df
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return None


def clean_data(df):
    """
    Cleans the raw transactional data.
    """
    print("Cleaning data...")
    # Drop rows with missing CustomerID
    df.dropna(subset=['Customer ID'], inplace=True)
    df['Customer ID'] = df['Customer ID'].astype(int)

    # Remove returns (Invoice starts with 'C') and transactions with negative quantity
    df = df[~df['Invoice'].astype(str).str.contains('C', na=False)]
    df = df[df['Quantity'] > 0]

    # Remove transactions with zero unit price
    df = df[df['Price'] > 0]

    # Create the TotalPrice column
    df['TotalPrice'] = df['Quantity'] * df['Price']

    print(f"Data cleaned. New shape: {df.shape}")
    return df

def cap_outliers(df, column):
    """
    Caps outliers in a specified column at the 1% and 99% quantiles.
    """
    lower_bound = df[column].quantile(0.01)
    upper_bound = df[column].quantile(0.99)
    df.loc[df[column] < lower_bound, column] = lower_bound
    df.loc[df[column] > upper_bound, column] = upper_bound
    return df

# --- 2. Feature Engineering (RFM) ---

def create_rfm_summary(df):
    """
    Creates the RFM (Recency, Frequency, Monetary, Tenure) summary data.
    """
    print("Creating RFM summary data...")
    # The 'lifetimes' library expects specific column names
    rfm = lifetimes.utils.summary_data_from_transaction_data(
        df,
        customer_id_col='Customer ID',
        datetime_col='InvoiceDate',
        monetary_value_col='TotalPrice',
        observation_period_end=df['InvoiceDate'].max()
    )
    print(f"RFM summary created. Shape: {rfm.shape}")
    return rfm

# --- 3. Modeling ---

def fit_bgnbd_model(rfm_summary):
    """
    Fits the BG/NBD model to predict purchase frequency.
    """
    print("Fitting BG/NBD model...")
    bgf = lifetimes.BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm_summary['frequency'], rfm_summary['recency'], rfm_summary['T'])
    print("BG/NBD model fitted successfully.")
    return bgf

def check_gamma_gamma_assumption(rfm_summary):
    """
    Checks the assumption that monetary value and frequency are not correlated.
    """
    print("Checking Gamma-Gamma model assumption...")
    # We only check this for returning customers, as they are the ones used in the model
    returning_customers_summary = rfm_summary[rfm_summary['frequency'] > 0]
    correlation = returning_customers_summary[['monetary_value', 'frequency']].corr().iloc[0, 1]
    print(f"Correlation between monetary value and frequency: {correlation:.4f}")
    if abs(correlation) > 0.1:
        print("Warning: Assumption may be violated. Proceed with caution.")
    else:
        print("Assumption holds. It is safe to proceed with the Gamma-Gamma model.")
    return

def fit_gamma_gamma_model(rfm_summary):
    """
    Fits the Gamma-Gamma model to predict average purchase value.
    """
    print("Fitting Gamma-Gamma model...")
    # Filter for customers with at least one repeat purchase
    returning_customers = rfm_summary[rfm_summary['frequency'] > 0]
    ggf = lifetimes.GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])
    print("Gamma-Gamma model fitted successfully.")
    return ggf

# --- Main Execution ---

if __name__ == '__main__':
    # Step 1: Load and clean data from both sheets
    retail_df = download_and_load_data()

    if retail_df is not None:
        retail_df_clean = clean_data(retail_df)

        # Handle outliers in TotalPrice for more stable monetary value calculation
        retail_df_clean = cap_outliers(retail_df_clean, 'TotalPrice')

        # Step 2: Create RFM summary
        rfm_summary = create_rfm_summary(retail_df_clean)

        # Step 3: Fit models
        bgf_model = fit_bgnbd_model(rfm_summary)

        check_gamma_gamma_assumption(rfm_summary)
        ggf_model = fit_gamma_gamma_model(rfm_summary)

        # Step 4: Predict CLV
        print("\n--- Calculating Customer Lifetime Value (CLV) ---")

        # Predict CLV for the next 12 months, with a monthly discount rate of 1%
        clv_prediction = ggf_model.customer_lifetime_value(
            bgf_model,
            rfm_summary['frequency'],
            rfm_summary['recency'],
            rfm_summary['T'],
            rfm_summary['monetary_value'],
            time=12,  # months
            discount_rate=0.01  # monthly
        )

        clv_df = pd.DataFrame(clv_prediction).reset_index()
        clv_df.rename(columns={'clv': 'predicted_clv_12_months'}, inplace=True)

        # --- Display Results ---
        print("\nTop 10 Customers by Predicted 12-Month CLV:")
        top_10_clv = clv_df.sort_values(by='predicted_clv_12_months', ascending=False).head(10)
        print(top_10_clv.to_string(index=False))

        plt.figure(figsize=(10, 6))
        sns.histplot(clv_df['predicted_clv_12_months'], bins=50, kde=True)
        plt.title('Distribution of Predicted 12-Month CLV (2009-2011 Data)')
        plt.xlabel('Predicted CLV ($)')
        plt.ylabel('Number of Customers')
        plt.xlim(0, clv_df['predicted_clv_12_months'].quantile(0.99)) # zoom in on 99% of data
        print("\nA plot showing the CLV distribution has been generated.")
        plt.show()