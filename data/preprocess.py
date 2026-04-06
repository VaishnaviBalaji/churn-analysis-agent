import pandas as pd
import sys


def preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows")

    # Binary columns: Yes/No → 1/0
    yes_no_cols = {
        'Partner':          'has_partner',
        'Dependents':       'has_dependents',
        'PhoneService':     'has_phone',
        'MultipleLines':    'multiple_lines',
        'OnlineSecurity':   'has_online_security',
        'TechSupport':      'has_tech_support',
        'OnlineBackup':     'has_online_backup',
        'DeviceProtection': 'has_device_protection',
        'StreamingTV':      'has_streaming_tv',
        'StreamingMovies':  'has_streaming_movies',
        'PaperlessBilling': 'paperless_billing',
    }

    for old_col, new_col in yes_no_cols.items():
        df[new_col] = (df[old_col] == 'Yes').astype(int)

    # Special cases
    df['gender_male'] = (df['gender'] == 'Male').astype(int)
    df['is_senior'] = df['SeniorCitizen'].astype(int)

    # Rename passthrough columns
    df['contract_type'] = df['Contract']
    df['payment_method'] = df['PaymentMethod']
    df['internet_service'] = df['InternetService']
    df['monthly_charges'] = df['MonthlyCharges']

    # Derived feature
    bundle_cols = [
        'has_online_security', 'has_online_backup', 'has_device_protection',
        'has_tech_support', 'has_streaming_tv', 'has_streaming_movies'
    ]
    df['bundle_depth'] = df[bundle_cols].sum(axis=1)

    # Keep only model features
    feature_cols = [
        'tenure', 'gender_male', 'is_senior', 'has_partner', 'has_dependents',
        'contract_type', 'paperless_billing', 'payment_method', 'monthly_charges',
        'has_phone', 'multiple_lines', 'internet_service', 'has_online_security',
        'has_tech_support', 'has_online_backup', 'has_device_protection',
        'has_streaming_tv', 'has_streaming_movies', 'bundle_depth'
    ]

    df = df[feature_cols]
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/customers.csv"
    preprocess(input_path, output_path)
