"""
Synthetic Air Quality Data Generator
Generates realistic AQI data with temporal patterns, spatial variation, and noise
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

class AQIDataGenerator:
    def __init__(self, num_locations=10, days=30):
        self.num_locations = num_locations
        self.days = days
        self.hours = days * 24
        
    def generate_sensor_locations(self):
        """Generate sensor locations across a city (Mumbai coordinates)"""
        # Mumbai approximate bounds
        lat_min, lat_max = 18.90, 19.30
        lon_min, lon_max = 72.80, 73.00
        
        locations = []
        location_types = ['Residential', 'Industrial', 'Traffic Junction', 
                         'Green Park', 'Commercial', 'Coastal']
        
        for i in range(self.num_locations):
            lat = np.random.uniform(lat_min, lat_max)
            lon = np.random.uniform(lon_min, lon_max)
            loc_type = np.random.choice(location_types)
            
            locations.append({
                'sensor_id': f'SENSOR_{i+1:03d}',
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'location_type': loc_type,
                'area_name': f'Area_{i+1}'
            })
        
        return pd.DataFrame(locations)
    
    def generate_base_pattern(self, location_type):
        """Generate base AQI pattern based on location type"""
        hours = np.arange(self.hours)
        
        # Base level by location type
        base_levels = {
            'Residential': 60,
            'Industrial': 120,
            'Traffic Junction': 100,
            'Green Park': 40,
            'Commercial': 80,
            'Coastal': 50
        }
        base = base_levels.get(location_type, 70)
        
        # Daily pattern (morning and evening peaks)
        daily_pattern = 20 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
        
        # Weekly pattern (weekday vs weekend)
        weekly_pattern = 10 * np.sin(2 * np.pi * hours / (24*7))
        
        # Seasonal trend (slow drift)
        seasonal_trend = 15 * np.sin(2 * np.pi * hours / (24*30))
        
        # Combine patterns
        aqi = base + daily_pattern + weekly_pattern + seasonal_trend
        
        return aqi
    
    def add_weather_influence(self, aqi, hours):
        """Add weather-related variations"""
        # Temperature influence (higher temp = more pollution dispersal issues)
        temp_base = 28
        temp_variation = 5 * np.sin(2 * np.pi * hours / 24 - np.pi/4)
        temperature = temp_base + temp_variation + np.random.normal(0, 2, len(hours))
        
        # Humidity (monsoon season effects)
        humidity = 60 + 20 * np.sin(2 * np.pi * hours / (24*7)) + np.random.normal(0, 5, len(hours))
        humidity = np.clip(humidity, 30, 95)
        
        # Wind speed (dispersal factor)
        wind_speed = 10 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 2, len(hours))
        wind_speed = np.clip(wind_speed, 2, 25)
        
        # Weather influence on AQI
        # Low wind = higher pollution accumulation
        wind_factor = (15 - wind_speed) / 15 * 20
        
        # High humidity can trap pollutants
        humidity_factor = (humidity - 50) / 50 * 10
        
        aqi_adjusted = aqi + wind_factor + humidity_factor
        
        return aqi_adjusted, temperature, humidity, wind_speed
    
    def add_noise_and_events(self, aqi):
        """Add realistic noise and pollution events"""
        # Random noise
        noise = np.random.normal(0, 5, len(aqi))
        
        # Random pollution spikes (traffic jams, industrial events)
        num_events = int(len(aqi) * 0.05)  # 5% of time
        event_indices = np.random.choice(len(aqi), num_events, replace=False)
        events = np.zeros(len(aqi))
        events[event_indices] = np.random.uniform(20, 50, num_events)
        
        aqi_final = aqi + noise + events
        aqi_final = np.clip(aqi_final, 0, 500)  # AQI range
        
        return aqi_final
    
    def introduce_missing_data(self, df, missing_rate=0.05):
        """Introduce realistic missing data patterns"""
        # Random missing values
        mask = np.random.random(len(df)) < missing_rate
        df.loc[mask, 'pm25'] = np.nan
        df.loc[mask, 'aqi'] = np.nan
        
        # Sensor failure periods (consecutive missing)
        if len(df) > 100:
            failure_start = np.random.randint(0, len(df) - 24)
            df.loc[failure_start:failure_start+6, ['pm25', 'aqi']] = np.nan
        
        return df
    
    def generate_full_dataset(self):
        """Generate complete AQI dataset"""
        # Generate sensor locations
        sensor_df = self.generate_sensor_locations()
        
        # Generate time series for each sensor
        all_data = []
        
        start_time = datetime.now() - timedelta(days=self.days)
        timestamps = [start_time + timedelta(hours=i) for i in range(self.hours)]
        
        for _, sensor in sensor_df.iterrows():
            # Base pattern
            aqi = self.generate_base_pattern(sensor['location_type'])
            
            # Weather influence
            hours_array = np.arange(self.hours)
            aqi, temp, humidity, wind = self.add_weather_influence(aqi, hours_array)
            
            # Add noise and events
            aqi = self.add_noise_and_events(aqi)
            
            # Convert AQI to PM2.5 (approximate relationship)
            pm25 = aqi * 0.7 + np.random.normal(0, 3, len(aqi))
            pm25 = np.clip(pm25, 0, 350)
            
            # Create dataframe for this sensor
            sensor_data = pd.DataFrame({
                'timestamp': timestamps,
                'sensor_id': sensor['sensor_id'],
                'latitude': sensor['latitude'],
                'longitude': sensor['longitude'],
                'location_type': sensor['location_type'],
                'area_name': sensor['area_name'],
                'aqi': aqi.round(1),
                'pm25': pm25.round(1),
                'temperature': temp.round(1),
                'humidity': humidity.round(1),
                'wind_speed': wind.round(1)
            })
            
            # Introduce missing data
            sensor_data = self.introduce_missing_data(sensor_data)
            
            all_data.append(sensor_data)
        
        # Combine all sensor data
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sort_values(['timestamp', 'sensor_id']).reset_index(drop=True)
        
        return final_df, sensor_df
    
    def save_data(self, output_dir='data'):
        """Generate and save datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating synthetic AQI data...")
        aqi_data, sensor_locations = self.generate_full_dataset()
        
        # Save files
        aqi_path = os.path.join(output_dir, 'aqi_data.csv')
        sensor_path = os.path.join(output_dir, 'sensor_locations.csv')
        
        aqi_data.to_csv(aqi_path, index=False)
        sensor_locations.to_csv(sensor_path, index=False)
        
        print(f"✓ Generated {len(aqi_data):,} records across {len(sensor_locations)} sensors")
        print(f"✓ Saved to {aqi_path}")
        print(f"✓ Saved sensor locations to {sensor_path}")
        print(f"\nData Summary:")
        print(f"  - Date range: {aqi_data['timestamp'].min()} to {aqi_data['timestamp'].max()}")
        print(f"  - AQI range: {aqi_data['aqi'].min():.1f} to {aqi_data['aqi'].max():.1f}")
        print(f"  - Missing values: {aqi_data['aqi'].isna().sum()} ({aqi_data['aqi'].isna().sum()/len(aqi_data)*100:.2f}%)")
        
        return aqi_data, sensor_locations

if __name__ == '__main__':
    generator = AQIDataGenerator(num_locations=10, days=30)
    aqi_data, sensor_locations = generator.save_data()
