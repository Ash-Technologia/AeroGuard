"""
Data Preprocessing Module
Handles missing values, outliers, and feature engineering for AQI forecasting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

class AQIPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        
    def handle_missing_values(self, df):
        """
        Handle missing values using multiple strategies:
        1. Forward fill for short gaps (< 3 hours)
        2. KNN imputation for longer gaps
        3. Interpolation for smoothness
        """
        df = df.copy()
        
        # Sort by sensor and time
        df = df.sort_values(['sensor_id', 'timestamp']).reset_index(drop=True)
        
        print(f"Missing values before preprocessing: {df['aqi'].isna().sum()}")
        
        # Strategy 1: Forward fill for very short gaps
        df['aqi'] = df.groupby('sensor_id')['aqi'].fillna(method='ffill', limit=2)
        df['pm25'] = df.groupby('sensor_id')['pm25'].fillna(method='ffill', limit=2)
        
        # Strategy 2: Linear interpolation
        df['aqi'] = df.groupby('sensor_id')['aqi'].transform(
            lambda x: x.interpolate(method='linear', limit=6)
        )
        df['pm25'] = df.groupby('sensor_id')['pm25'].transform(
            lambda x: x.interpolate(method='linear', limit=6)
        )
        
        # Strategy 3: KNN imputation for remaining gaps
        numeric_cols = ['aqi', 'pm25', 'temperature', 'humidity', 'wind_speed']
        if df[numeric_cols].isna().sum().sum() > 0:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        print(f"Missing values after preprocessing: {df['aqi'].isna().sum()}")
        
        return df
    
    def remove_outliers(self, df, column='aqi', method='iqr', threshold=3):
        """Remove or cap outliers using IQR or Z-score method"""
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df = df[z_scores < threshold]
        
        return df
    
    def create_temporal_features(self, df):
        """Create time-based features for modeling"""
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour (0-23)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week (0-6)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                    bins=[0, 6, 12, 18, 24],
                                    labels=['night', 'morning', 'afternoon', 'evening'],
                                    include_lowest=True)
        
        # Rush hour indicator
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 10) | 
                              (df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
        
        return df
    
    def create_lag_features(self, df, target_col='aqi', lags=[1, 3, 6, 12, 24]):
        """Create lag features for time series forecasting"""
        df = df.copy()
        df = df.sort_values(['sensor_id', 'timestamp'])
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby('sensor_id')[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, target_col='aqi', windows=[3, 6, 12, 24]):
        """Create rolling statistics features"""
        df = df.copy()
        df = df.sort_values(['sensor_id', 'timestamp'])
        
        for window in windows:
            # Rolling mean
            df[f'{target_col}_rolling_mean_{window}h'] = (
                df.groupby('sensor_id')[target_col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )
            
            # Rolling std
            df[f'{target_col}_rolling_std_{window}h'] = (
                df.groupby('sensor_id')[target_col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).std())
            )
            
            # Rolling min/max
            df[f'{target_col}_rolling_min_{window}h'] = (
                df.groupby('sensor_id')[target_col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).min())
            )
            
            df[f'{target_col}_rolling_max_{window}h'] = (
                df.groupby('sensor_id')[target_col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).max())
            )
        
        return df
    
    def create_weather_interaction_features(self, df):
        """Create weather-pollution interaction features"""
        df = df.copy()
        
        # Wind chill effect
        if 'wind_speed' in df.columns and 'temperature' in df.columns:
            df['wind_chill_index'] = df['temperature'] - (df['wind_speed'] * 0.5)
        
        # Humidity-temperature interaction
        if 'humidity' in df.columns and 'temperature' in df.columns:
            df['heat_index'] = df['temperature'] + (0.5 * df['humidity'])
        
        # Pollution dispersal index (higher wind = better dispersal)
        if 'wind_speed' in df.columns:
            df['dispersal_index'] = 100 / (1 + df['wind_speed'])
        
        # Stagnation index (low wind + high humidity = pollution trapping)
        if 'wind_speed' in df.columns and 'humidity' in df.columns:
            df['stagnation_index'] = df['humidity'] / (df['wind_speed'] + 1)
        
        return df
    
    def full_preprocessing_pipeline(self, df, create_features=True):
        """
        Complete preprocessing pipeline
        """
        print("Starting preprocessing pipeline...")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Remove outliers
        df = self.remove_outliers(df, column='aqi')
        df = self.remove_outliers(df, column='pm25')
        
        if create_features:
            # Step 3: Create temporal features
            df = self.create_temporal_features(df)
            
            # Step 4: Create lag features
            df = self.create_lag_features(df, target_col='aqi')
            df = self.create_lag_features(df, target_col='pm25')
            
            # Step 5: Create rolling features
            df = self.create_rolling_features(df, target_col='aqi')
            
            # Step 6: Weather interaction features
            df = self.create_weather_interaction_features(df)
        
        # Drop rows with NaN created by lag/rolling features
        initial_len = len(df)
        df = df.dropna()
        print(f"Dropped {initial_len - len(df)} rows due to feature engineering NaNs")
        
        print("Preprocessing complete!")
        print(f"Final dataset shape: {df.shape}")
        
        return df

def prepare_train_test_split(df, test_size=0.2):
    """
    Time-based train-test split for time series
    """
    df = df.sort_values('timestamp')
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train set: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Test set: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    return train_df, test_df

if __name__ == '__main__':
    # Test preprocessing
    import os
    
    data_path = '../data/aqi_data.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        preprocessor = AQIPreprocessor()
        df_processed = preprocessor.full_preprocessing_pipeline(df)
        
        print("\nProcessed columns:")
        print(df_processed.columns.tolist())
    else:
        print("Please run data_generator.py first!")
