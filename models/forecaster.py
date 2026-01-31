"""
AQI Forecasting Models
Implements LSTM, Prophet, SARIMA, and XGBoost for AQI prediction
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Statistical Models
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Machine Learning
from xgboost import XGBRegressor

# Sklearn utilities
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

class LSTMForecaster:
    def __init__(self, lookback=24, forecast_horizon=6):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data, target):
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - self.lookback - self.forecast_horizon + 1):
            X.append(data[i:(i + self.lookback)])
            y.append(target[i + self.lookback:i + self.lookback + self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(128, activation='relu', return_sequences=True, 
                 input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, train_df, feature_cols, target_col='aqi'):
        """Train LSTM model"""
        # Prepare data
        features = train_df[feature_cols].values
        target = train_df[target_col].values
        
        # Scale data
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, target)
        
        if len(X) == 0:
            raise ValueError("Not enough data for sequence creation")
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Train
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        return history
    
    def predict(self, test_df, feature_cols):
        """Make predictions"""
        features = test_df[feature_cols].values
        features_scaled = self.scaler.transform(features)
        
        predictions = []
        
        # Rolling prediction
        for i in range(len(features_scaled) - self.lookback):
            seq = features_scaled[i:i + self.lookback].reshape(1, self.lookback, -1)
            pred = self.model.predict(seq, verbose=0)
            predictions.append(pred[0])
        
        return np.array(predictions)

class ProphetForecaster:
    def __init__(self, forecast_horizon=6):
        self.forecast_horizon = forecast_horizon
        self.model = None
        
    def train(self, train_df, target_col='aqi'):
        """Train Prophet model"""
        # Prepare data for Prophet
        prophet_df = train_df[['timestamp', target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Initialize and fit model
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Add additional regressors if available
        if 'temperature' in train_df.columns:
            prophet_df['temperature'] = train_df['temperature'].values
            self.model.add_regressor('temperature')
        
        if 'humidity' in train_df.columns:
            prophet_df['humidity'] = train_df['humidity'].values
            self.model.add_regressor('humidity')
        
        if 'wind_speed' in train_df.columns:
            prophet_df['wind_speed'] = train_df['wind_speed'].values
            self.model.add_regressor('wind_speed')
        
        self.model.fit(prophet_df)
        
        return self.model
    
    def predict(self, test_df):
        """Make predictions"""
        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=self.forecast_horizon,
            freq='H'
        )
        
        # Add regressors
        if 'temperature' in test_df.columns:
            future['temperature'] = test_df['temperature'].iloc[:len(future)].values
        if 'humidity' in test_df.columns:
            future['humidity'] = test_df['humidity'].iloc[:len(future)].values
        if 'wind_speed' in test_df.columns:
            future['wind_speed'] = test_df['wind_speed'].iloc[:len(future)].values
        
        # Predict
        forecast = self.model.predict(future)
        
        return forecast['yhat'].values[-self.forecast_horizon:]

class SARIMAForecaster:
    def __init__(self, forecast_horizon=6, order=(1,1,1), seasonal_order=(1,1,1,24)):
        self.forecast_horizon = forecast_horizon
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        
    def train(self, train_df, target_col='aqi'):
        """Train SARIMA model"""
        # Prepare data
        ts_data = train_df.set_index('timestamp')[target_col]
        ts_data = ts_data.asfreq('H')
        
        # Fit model
        self.model = SARIMAX(
            ts_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.model_fit = self.model.fit(disp=False, maxiter=100)
        
        return self.model_fit
    
    def predict(self, steps=None):
        """Make predictions"""
        if steps is None:
            steps = self.forecast_horizon
            
        forecast = self.model_fit.forecast(steps=steps)
        
        return forecast.values

class XGBoostForecaster:
    def __init__(self, forecast_horizon=6):
        self.forecast_horizon = forecast_horizon
        self.models = []
        
    def train(self, train_df, feature_cols, target_col='aqi'):
        """Train XGBoost model for multi-step forecasting"""
        X_train = train_df[feature_cols].values
        
        # Train separate model for each forecast step
        for h in range(1, self.forecast_horizon + 1):
            # Create target: AQI h hours ahead
            y_train = train_df[target_col].shift(-h).values[:-h]
            X_train_h = X_train[:-h]
            
            model = XGBRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train_h, y_train)
            self.models.append(model)
        
        return self.models
    
    def predict(self, test_df, feature_cols):
        """Make predictions"""
        X_test = test_df[feature_cols].values
        
        predictions = []
        
        for i in range(len(X_test)):
            pred_horizons = []
            for model in self.models:
                pred = model.predict(X_test[i].reshape(1, -1))
                pred_horizons.append(pred[0])
            predictions.append(pred_horizons)
        
        return np.array(predictions)

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate forecasting metrics"""
        # Flatten arrays for multi-step forecasting
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Remove NaN values
        mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
        y_true_flat = y_true_flat[mask]
        y_pred_flat = y_pred_flat[mask]
        
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    @staticmethod
    def compare_models(results_dict):
        """Compare multiple model results"""
        comparison_df = pd.DataFrame(results_dict).T
        comparison_df = comparison_df.round(3)
        
        # Rank models
        comparison_df['Rank_RMSE'] = comparison_df['RMSE'].rank()
        comparison_df['Rank_MAE'] = comparison_df['MAE'].rank()
        comparison_df['Rank_R2'] = comparison_df['R2'].rank(ascending=False)
        
        comparison_df['Average_Rank'] = (
            comparison_df['Rank_RMSE'] + 
            comparison_df['Rank_MAE'] + 
            comparison_df['Rank_R2']
        ) / 3
        
        comparison_df = comparison_df.sort_values('Average_Rank')
        
        return comparison_df

def get_important_features(feature_cols):
    """Select most important features for modeling"""
    # Core temporal features
    temporal = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    
    # Lag features
    lag_features = [col for col in feature_cols if 'lag' in col]
    
    # Rolling features
    rolling_features = [col for col in feature_cols if 'rolling_mean' in col or 'rolling_std' in col]
    
    # Weather features
    weather = ['temperature', 'humidity', 'wind_speed']
    
    # Weather interactions
    weather_interaction = [col for col in feature_cols if 'index' in col]
    
    selected = []
    for feat_list in [temporal, lag_features[:5], rolling_features[:4], weather, weather_interaction]:
        selected.extend([f for f in feat_list if f in feature_cols])
    
    return list(set(selected))

if __name__ == '__main__':
    print("Forecasting models module loaded successfully!")
    print("Available models: LSTM, Prophet, SARIMA, XGBoost")
