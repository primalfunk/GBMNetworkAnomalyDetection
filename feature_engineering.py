import pandas as pd

def engineer_datetime_features(df):
    """
    Convert timestamp into separate year, month, day, and hour features.
    
    Parameters:
        df (DataFrame): DataFrame with the raw features including timestamp
    
    Returns:
        DataFrame: Modified DataFrame with engineered features
    """
    
    # Make sure 'timestamp' is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create new features from 'timestamp'
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    
    # Drop original 'timestamp' column
    df.drop(['timestamp'], axis=1, inplace=True)
    
    return df