"""
AI-Powered Explainability Module
Uses Claude API to generate human-readable explanations for AQI forecasts
"""

import numpy as np
import pandas as pd

class AQIExplainer:
    def __init__(self, use_ai=False):
        """
        Initialize explainer
        
        Args:
            use_ai: If True, use Claude API for AI-generated explanations
                   If False, use rule-based explanations
        """
        self.use_ai = use_ai
        
    def analyze_trends(self, historical_aqi, forecast_aqi):
        """Analyze AQI trends"""
        # Current vs forecast
        current = historical_aqi[-1]
        forecast_avg = np.mean(forecast_aqi)
        forecast_max = np.max(forecast_aqi)
        forecast_min = np.min(forecast_aqi)
        
        # Trend direction
        if forecast_avg > current * 1.1:
            trend = "rising"
        elif forecast_avg < current * 0.9:
            trend = "falling"
        else:
            trend = "stable"
        
        # Volatility
        forecast_std = np.std(forecast_aqi)
        if forecast_std > 20:
            volatility = "high"
        elif forecast_std > 10:
            volatility = "moderate"
        else:
            volatility = "low"
        
        return {
            'current_aqi': current,
            'forecast_avg': forecast_avg,
            'forecast_max': forecast_max,
            'forecast_min': forecast_min,
            'trend': trend,
            'volatility': volatility,
            'change_percent': ((forecast_avg - current) / current) * 100
        }
    
    def identify_contributing_factors(self, weather_data, time_data):
        """Identify factors contributing to AQI levels"""
        factors = []
        
        # Wind speed analysis
        avg_wind = weather_data.get('wind_speed', 10)
        if avg_wind < 5:
            factors.append({
                'factor': 'Low wind speed',
                'impact': 'negative',
                'description': f'Wind speed of {avg_wind:.1f} km/h is preventing pollutant dispersal'
            })
        elif avg_wind > 15:
            factors.append({
                'factor': 'High wind speed',
                'impact': 'positive',
                'description': f'Wind speed of {avg_wind:.1f} km/h is helping disperse pollutants'
            })
        
        # Humidity analysis
        avg_humidity = weather_data.get('humidity', 60)
        if avg_humidity > 75:
            factors.append({
                'factor': 'High humidity',
                'impact': 'negative',
                'description': f'Humidity at {avg_humidity:.1f}% is trapping pollutants near the ground'
            })
        
        # Temperature analysis
        avg_temp = weather_data.get('temperature', 25)
        if avg_temp > 35:
            factors.append({
                'factor': 'High temperature',
                'impact': 'negative',
                'description': f'Temperature of {avg_temp:.1f}°C may increase ground-level ozone formation'
            })
        
        # Time-based factors
        hour = time_data.get('hour', 12)
        is_rush_hour = time_data.get('is_rush_hour', False)
        
        if is_rush_hour or (7 <= hour <= 10) or (17 <= hour <= 20):
            factors.append({
                'factor': 'Rush hour traffic',
                'impact': 'negative',
                'description': 'Increased vehicular emissions during peak traffic hours'
            })
        
        # Weekend vs weekday
        is_weekend = time_data.get('is_weekend', False)
        if is_weekend:
            factors.append({
                'factor': 'Weekend pattern',
                'impact': 'positive',
                'description': 'Reduced industrial and commercial activity on weekends'
            })
        
        return factors
    
    def generate_rule_based_explanation(self, trends, factors, forecast_aqi):
        """Generate rule-based natural language explanation"""
        current = trends['current_aqi']
        forecast_avg = trends['forecast_avg']
        trend = trends['trend']
        
        # Opening statement
        if trend == "rising":
            opening = f"AQI is expected to rise from {current:.0f} to an average of {forecast_avg:.0f} over the next 6 hours."
        elif trend == "falling":
            opening = f"AQI is expected to improve from {current:.0f} to an average of {forecast_avg:.0f} over the next 6 hours."
        else:
            opening = f"AQI is expected to remain relatively stable around {forecast_avg:.0f} over the next 6 hours."
        
        # Contributing factors
        if factors:
            factor_text = "\n\nKey contributing factors:\n"
            for factor in factors:
                impact_symbol = "⚠️" if factor['impact'] == 'negative' else "✓"
                factor_text += f"{impact_symbol} {factor['description']}\n"
        else:
            factor_text = "\n\nNo significant unusual factors detected."
        
        # Persistence analysis
        if trends['volatility'] == 'high':
            persistence = "\n\nAir quality is expected to be highly variable, with significant fluctuations throughout the forecast period."
        elif trends['volatility'] == 'moderate':
            persistence = "\n\nAir quality will show some variation but should remain relatively predictable."
        else:
            persistence = "\n\nAir quality is expected to remain steady with minimal fluctuations."
        
        # Peak warning
        peak_hour = np.argmax(forecast_aqi) + 1
        peak_aqi = np.max(forecast_aqi)
        
        if peak_aqi > forecast_avg * 1.2:
            peak_warning = f"\n\n⚠️ Peak AQI of {peak_aqi:.0f} expected around hour {peak_hour}. Plan accordingly."
        else:
            peak_warning = ""
        
        # Recommendations
        if forecast_avg > 150:
            recommendation = "\n\n🏥 Recommendation: Limit outdoor activities and use air purification where available."
        elif forecast_avg > 100:
            recommendation = "\n\n⚠️ Recommendation: Sensitive groups should consider reducing prolonged outdoor exertion."
        else:
            recommendation = "\n\n✓ Air quality should be acceptable for most outdoor activities."
        
        explanation = opening + factor_text + persistence + peak_warning + recommendation
        
        return explanation
    
    def generate_ai_explanation(self, trends, factors, forecast_aqi, weather_data, time_data):
        """
        Generate AI-powered explanation using Claude API
        
        This is a placeholder - in production, this would call the Anthropic API
        """
        # Prepare context
        context = f"""
        Current AQI: {trends['current_aqi']:.1f}
        Forecast Average: {trends['forecast_avg']:.1f}
        Trend: {trends['trend']}
        Forecast Range: {trends['forecast_min']:.1f} to {trends['forecast_max']:.1f}
        
        Weather Conditions:
        - Wind Speed: {weather_data.get('wind_speed', 'N/A')} km/h
        - Humidity: {weather_data.get('humidity', 'N/A')}%
        - Temperature: {weather_data.get('temperature', 'N/A')}°C
        
        Time Context:
        - Hour: {time_data.get('hour', 'N/A')}
        - Rush Hour: {time_data.get('is_rush_hour', False)}
        - Weekend: {time_data.get('is_weekend', False)}
        
        Contributing Factors:
        {chr(10).join([f"- {f['description']}" for f in factors])}
        """
        
        # In production, this would call:
        # response = call_claude_api(context, prompt)
        
        # For now, fallback to rule-based
        return self.generate_rule_based_explanation(trends, factors, forecast_aqi)
    
    def explain_forecast(self, historical_aqi, forecast_aqi, weather_data, time_data):
        """
        Generate comprehensive explanation for AQI forecast
        
        Args:
            historical_aqi: Array of past AQI values (last 24 hours)
            forecast_aqi: Array of forecasted AQI values (next 6 hours)
            weather_data: Dict with weather conditions
            time_data: Dict with temporal information
        
        Returns:
            Dict with explanation text and metadata
        """
        # Analyze trends
        trends = self.analyze_trends(historical_aqi, forecast_aqi)
        
        # Identify contributing factors
        factors = self.identify_contributing_factors(weather_data, time_data)
        
        # Generate explanation
        if self.use_ai:
            explanation_text = self.generate_ai_explanation(
                trends, factors, forecast_aqi, weather_data, time_data
            )
        else:
            explanation_text = self.generate_rule_based_explanation(
                trends, factors, forecast_aqi
            )
        
        return {
            'explanation': explanation_text,
            'trends': trends,
            'contributing_factors': factors,
            'forecast_summary': {
                'min': trends['forecast_min'],
                'max': trends['forecast_max'],
                'avg': trends['forecast_avg'],
                'current': trends['current_aqi']
            }
        }
    
    def get_simple_summary(self, forecast_aqi):
        """Get a simple one-line summary"""
        avg = np.mean(forecast_aqi)
        
        if avg < 50:
            return "☀️ Excellent air quality expected - great day for outdoor activities!"
        elif avg < 100:
            return "🌤️ Good air quality expected - safe for most outdoor activities."
        elif avg < 150:
            return "⚠️ Moderate air quality - sensitive groups should take precautions."
        elif avg < 200:
            return "🚫 Unhealthy air quality - limit outdoor exposure."
        else:
            return "🚨 Hazardous air quality - stay indoors and seek clean air!"

def demo_explainer():
    """Demonstrate AQI explanation generation"""
    explainer = AQIExplainer(use_ai=False)
    
    # Example data
    historical_aqi = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 
                               130, 135, 140, 145, 150, 155, 160, 165, 170, 175,
                               180, 185, 190, 195])
    
    forecast_aqi = np.array([200, 205, 210, 200, 195, 190])
    
    weather_data = {
        'wind_speed': 3.5,
        'humidity': 82,
        'temperature': 36
    }
    
    time_data = {
        'hour': 8,
        'is_rush_hour': True,
        'is_weekend': False
    }
    
    # Generate explanation
    result = explainer.explain_forecast(
        historical_aqi, forecast_aqi, weather_data, time_data
    )
    
    print("="*80)
    print("AQI FORECAST EXPLANATION")
    print("="*80)
    print(result['explanation'])
    print("\n" + "="*80)
    
    # Simple summary
    summary = explainer.get_simple_summary(forecast_aqi)
    print(f"\nQuick Summary: {summary}")

if __name__ == '__main__':
    demo_explainer()
