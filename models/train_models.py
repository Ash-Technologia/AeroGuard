"""
Model Training Pipeline
Train and evaluate all forecasting models
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import AQIPreprocessor, prepare_train_test_split
from models.forecaster import (
    LSTMForecaster, ProphetForecaster, SARIMAForecaster, 
    XGBoostForecaster, ModelEvaluator, get_important_features
)

def load_and_preprocess_data(data_path='../data/aqi_data.csv'):
    """Load and preprocess data"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing data...")
    preprocessor = AQIPreprocessor()
    df_processed = preprocessor.full_preprocessing_pipeline(df)
    
    return df_processed

def train_all_models(train_df, test_df, forecast_horizon=6):
    """Train all models and compare performance"""
    print("\n" + "="*80)
    print("TRAINING ALL MODELS")
    print("="*80)
    
    results = {}
    trained_models = {}
    
    # Select important features
    feature_cols = get_important_features(train_df.columns.tolist())
    print(f"\nUsing {len(feature_cols)} features for modeling")
    
    # Filter sensor for consistent evaluation
    # Use first sensor only for model comparison
    sensor_id = train_df['sensor_id'].iloc[0]
    train_sensor = train_df[train_df['sensor_id'] == sensor_id].copy()
    test_sensor = test_df[test_df['sensor_id'] == sensor_id].copy()
    
    print(f"Training on sensor: {sensor_id}")
    print(f"Train samples: {len(train_sensor)}, Test samples: {len(test_sensor)}")
    
    # -------------------------------------------------------------------------
    # 1. LSTM Model
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Training LSTM Model...")
    print("-"*80)
    try:
        lstm_model = LSTMForecaster(lookback=24, forecast_horizon=forecast_horizon)
        lstm_model.train(train_sensor, feature_cols, target_col='aqi')
        
        # Predict
        lstm_predictions = lstm_model.predict(test_sensor, feature_cols)
        
        # Align predictions with actuals
        y_true = test_sensor['aqi'].values[24:24+len(lstm_predictions)]
        y_pred = lstm_predictions[:len(y_true), 0]  # First hour predictions
        
        # Calculate metrics
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        results['LSTM'] = metrics
        trained_models['LSTM'] = lstm_model
        
        print("✓ LSTM training complete")
        print(f"  RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, R²: {metrics['R2']:.3f}")
        
    except Exception as e:
        print(f"✗ LSTM training failed: {str(e)}")
        results['LSTM'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'MAPE': np.nan}
    
    # -------------------------------------------------------------------------
    # 2. Prophet Model
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Training Prophet Model...")
    print("-"*80)
    try:
        prophet_model = ProphetForecaster(forecast_horizon=forecast_horizon)
        prophet_model.train(train_sensor, target_col='aqi')
        
        # Predict
        prophet_predictions = []
        for i in range(len(test_sensor) - forecast_horizon):
            pred = prophet_model.predict(test_sensor.iloc[i:i+forecast_horizon])
            prophet_predictions.append(pred[0])
        
        prophet_predictions = np.array(prophet_predictions)
        
        # Align with actuals
        y_true = test_sensor['aqi'].values[:len(prophet_predictions)]
        y_pred = prophet_predictions
        
        # Calculate metrics
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        results['Prophet'] = metrics
        trained_models['Prophet'] = prophet_model
        
        print("✓ Prophet training complete")
        print(f"  RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, R²: {metrics['R2']:.3f}")
        
    except Exception as e:
        print(f"✗ Prophet training failed: {str(e)}")
        results['Prophet'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'MAPE': np.nan}
    
    # -------------------------------------------------------------------------
    # 3. SARIMA Model
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Training SARIMA Model...")
    print("-"*80)
    try:
        sarima_model = SARIMAForecaster(
            forecast_horizon=forecast_horizon,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 24)
        )
        sarima_model.train(train_sensor, target_col='aqi')
        
        # Predict
        sarima_predictions = []
        for i in range(len(test_sensor) - forecast_horizon):
            pred = sarima_model.predict(steps=1)
            sarima_predictions.append(pred[0])
        
        sarima_predictions = np.array(sarima_predictions)
        
        # Align with actuals
        y_true = test_sensor['aqi'].values[:len(sarima_predictions)]
        y_pred = sarima_predictions
        
        # Calculate metrics
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        results['SARIMA'] = metrics
        trained_models['SARIMA'] = sarima_model
        
        print("✓ SARIMA training complete")
        print(f"  RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, R²: {metrics['R2']:.3f}")
        
    except Exception as e:
        print(f"✗ SARIMA training failed: {str(e)}")
        results['SARIMA'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'MAPE': np.nan}
    
    # -------------------------------------------------------------------------
    # 4. XGBoost Model
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("Training XGBoost Model...")
    print("-"*80)
    try:
        xgb_model = XGBoostForecaster(forecast_horizon=forecast_horizon)
        xgb_model.train(train_sensor, feature_cols, target_col='aqi')
        
        # Predict
        xgb_predictions = xgb_model.predict(test_sensor, feature_cols)
        
        # Align with actuals
        y_true = test_sensor['aqi'].values[:len(xgb_predictions)]
        y_pred = xgb_predictions[:, 0]  # First hour predictions
        
        # Calculate metrics
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        results['XGBoost'] = metrics
        trained_models['XGBoost'] = xgb_model
        
        print("✓ XGBoost training complete")
        print(f"  RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, R²: {metrics['R2']:.3f}")
        
    except Exception as e:
        print(f"✗ XGBoost training failed: {str(e)}")
        results['XGBoost'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'MAPE': np.nan}
    
    return results, trained_models

def save_models(trained_models, output_dir='../models/saved_models'):
    """Save trained models"""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in trained_models.items():
        filepath = os.path.join(output_dir, f'{name.lower()}_model.pkl')
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved {name} model to {filepath}")
        except Exception as e:
            print(f"✗ Failed to save {name} model: {str(e)}")

def main():
    """Main training pipeline"""
    print("="*80)
    print("AEROGUARD - MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Load data
    data_path = '../data/aqi_data.csv'
    
    if not os.path.exists(data_path):
        print(f"\n✗ Data file not found: {data_path}")
        print("Please run 'python utils/data_generator.py' first!")
        return
    
    df = load_and_preprocess_data(data_path)
    
    # Train-test split
    print("\nSplitting data...")
    train_df, test_df = prepare_train_test_split(df, test_size=0.2)
    
    # Train all models
    results, trained_models = train_all_models(train_df, test_df, forecast_horizon=6)
    
    # Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison_df = ModelEvaluator.compare_models(results)
    print(comparison_df.to_string())
    
    # Determine best model
    best_model = comparison_df.index[0]
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Average Rank: {comparison_df.loc[best_model, 'Average_Rank']:.2f}")
    
    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    save_models(trained_models)
    
    # Save comparison results
    comparison_df.to_csv('../models/model_comparison.csv')
    print("\n✓ Model comparison saved to ../models/model_comparison.csv")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review model comparison results")
    print("  2. Run 'streamlit run app.py' to launch the web app")
    print("="*80)

if __name__ == '__main__':
    # Create models directory
    os.makedirs('../models/saved_models', exist_ok=True)
    
    main()
