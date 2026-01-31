"""
Health Risk Classification Module
Translates AQI forecasts into personalized health recommendations
"""

import pandas as pd
import numpy as np

class HealthRiskClassifier:
    def __init__(self):
        # WHO/EPA AQI Standard Categories
        self.base_thresholds = {
            'Good': (0, 50),
            'Moderate': (51, 100),
            'Unhealthy for Sensitive': (101, 150),
            'Unhealthy': (151, 200),
            'Very Unhealthy': (201, 300),
            'Hazardous': (301, 500)
        }
        
        # Persona-specific threshold adjustments
        self.persona_adjustments = {
            'general_public': {
                'multiplier': 1.0,
                'description': 'Healthy adults with no pre-existing conditions'
            },
            'children_elderly': {
                'multiplier': 0.8,  # More sensitive - lower thresholds
                'description': 'Children under 12 and adults over 65'
            },
            'outdoor_workers': {
                'multiplier': 0.7,  # Most sensitive - highest exposure
                'description': 'Athletes, construction workers, outdoor professionals'
            }
        }
        
        # Health recommendations by category and persona
        self.recommendations = {
            'Good': {
                'general_public': {
                    'risk_level': 'Low',
                    'color': '#00E400',
                    'message': 'Air quality is excellent. Enjoy outdoor activities!',
                    'actions': [
                        "Air quality is ideal for outdoor activities.",
                        "Open windows to ventilate your home with fresh air.",
                        "Great time for walking, running, or cycling outdoors.",
                        "No specific health precautions are needed today."
                    ]
                },
                'children_elderly': {
                    'risk_level': 'Low',
                    'color': '#00E400',
                    'message': 'Air quality is excellent. Safe for all outdoor activities!',
                    'actions': [
                        "It's a perfect day for outdoor play and parks.",
                        "Enjoy fresh air; outdoor exercise is highly beneficial.",
                        "Ventilate children's rooms and living spaces.",
                        "No restrictions on activity levels today."
                    ]
                },
                'outdoor_workers': {
                    'risk_level': 'Low',
                    'color': '#00E400',
                    'message': 'Air quality is excellent. Safe for intense outdoor work!',
                    'actions': [
                        "Conditions are safe for full-day outdoor work.",
                        "No special respiratory protection equipment needed.",
                        "Good visibility and air quality for construction sites.",
                        "Take normal breaks as per schedule."
                    ]
                }
            },
            'Moderate': {
                'general_public': {
                    'risk_level': 'Low',
                    'color': '#FFFF00',
                    'message': 'Air quality is acceptable for most people.',
                    'actions': [
                        '✓ Generally safe for outdoor activities',
                        '⚠ Sensitive individuals may experience minor symptoms',
                        '• Monitor air quality if you have respiratory conditions'
                    ]
                },
                'children_elderly': {
                    'risk_level': 'Moderate',
                    'color': '#FF7E00',
                    'message': 'Air quality may cause minor discomfort for sensitive groups.',
                    'actions': [
                        '⚠ Limit prolonged outdoor exertion',
                        '⚠ Watch for coughing or breathing difficulty',
                        '• Consider indoor activities during peak hours',
                        '• Keep rescue medications accessible'
                    ]
                },
                'outdoor_workers': {
                    'risk_level': 'Moderate',
                    'color': '#FF7E00',
                    'message': 'Air quality requires caution for intense outdoor work.',
                    'actions': [
                        '⚠ Take frequent breaks in clean air zones',
                        '⚠ Reduce intensity of physical exertion',
                        '• Consider N95 masks for prolonged exposure',
                        '• Monitor for respiratory symptoms'
                    ]
                }
            },
            'Unhealthy for Sensitive': {
                'general_public': {
                    'risk_level': 'Moderate',
                    'color': '#FF7E00',
                    'message': 'Air quality is unhealthy for sensitive groups.',
                    'actions': [
                        '⚠ Limit prolonged outdoor exertion',
                        '⚠ Sensitive groups should reduce outdoor activity',
                        '• Close windows during peak pollution hours',
                        '• Use air purifiers indoors if available'
                    ]
                },
                'children_elderly': {
                    'risk_level': 'High',
                    'color': '#FF0000',
                    'message': 'Air quality is unhealthy. Stay indoors if possible.',
                    'actions': [
                        '⛔ Avoid outdoor activities',
                        '⛔ Keep windows closed',
                        '⚠ Use N95 masks if going outside is necessary',
                        '⚠ Monitor health symptoms closely',
                        '• Have rescue medications ready',
                        '• Use air purifiers indoors'
                    ]
                },
                'outdoor_workers': {
                    'risk_level': 'High',
                    'color': '#FF0000',
                    'message': 'Air quality is unhealthy. Protective measures required.',
                    'actions': [
                        '⛔ Mandatory N95/N99 respiratory protection',
                        '⛔ Minimize outdoor work duration',
                        '⚠ Take frequent breaks in clean environments',
                        '⚠ Reduce work intensity significantly',
                        '• Seek medical attention if breathing issues occur'
                    ]
                }
            },
            'Unhealthy': {
                'general_public': {
                    'risk_level': 'High',
                    'color': '#FF0000',
                    'message': 'Air quality is unhealthy for everyone.',
                    'actions': [
                        '⛔ Avoid outdoor activities',
                        '⛔ Keep windows and doors closed',
                        '⚠ Wear N95 masks if you must go outside',
                        '• Use air purifiers on high settings',
                        '• Monitor health symptoms'
                    ]
                },
                'children_elderly': {
                    'risk_level': 'Hazardous',
                    'color': '#8F3F97',
                    'message': 'Air quality is hazardous. Stay indoors!',
                    'actions': [
                        '⛔ STAY INDOORS - Do not go outside',
                        '⛔ Seal windows and doors',
                        '⛔ Run air purifiers continuously',
                        '⚠ Seek immediate medical help if experiencing symptoms',
                        '⚠ Have emergency medications ready',
                        '• Consider relocating to cleaner air area if possible'
                    ]
                },
                'outdoor_workers': {
                    'risk_level': 'Hazardous',
                    'color': '#8F3F97',
                    'message': 'Air quality is hazardous. Outdoor work should stop!',
                    'actions': [
                        '⛔ CEASE OUTDOOR WORK if possible',
                        '⛔ Mandatory high-grade respiratory protection (N99/P100)',
                        '⛔ Emergency protocols should be in effect',
                        '⚠ Minimize exposure time absolutely',
                        '⚠ Immediate medical evaluation if symptoms appear',
                        '• Work should only proceed if absolutely critical'
                    ]
                }
            },
            'Very Unhealthy': {
                'general_public': {
                    'risk_level': 'Hazardous',
                    'color': '#8F3F97',
                    'message': 'Health alert! Everyone should avoid outdoor exposure.',
                    'actions': [
                        '⛔ STAY INDOORS at all times',
                        '⛔ Seal all windows and doors',
                        '⛔ Run air purifiers continuously',
                        '⚠ Wear N95+ masks even for brief outdoor exposure',
                        '⚠ Seek medical attention if experiencing any symptoms',
                        '• Consider evacuation if air quality persists'
                    ]
                },
                'children_elderly': {
                    'risk_level': 'Hazardous',
                    'color': '#7E0023',
                    'message': 'EMERGENCY! Extremely hazardous air quality!',
                    'actions': [
                        '🚨 EMERGENCY - Shelter in place',
                        '⛔ Absolutely no outdoor exposure',
                        '⛔ Seal all openings completely',
                        '⛔ Create clean air room with purifiers',
                        '⚠ Seek immediate medical evacuation if possible',
                        '⚠ Have emergency services on standby',
                        '• Evacuation strongly recommended'
                    ]
                },
                'outdoor_workers': {
                    'risk_level': 'Hazardous',
                    'color': '#7E0023',
                    'message': 'EMERGENCY! All outdoor work must stop immediately!',
                    'actions': [
                        '🚨 EMERGENCY - All outdoor operations HALT',
                        '⛔ No outdoor work under any circumstances',
                        '⛔ Workers must shelter indoors immediately',
                        '⚠ Emergency medical protocols in effect',
                        '⚠ Full respiratory protection even for indoor work near openings',
                        '• Site evacuation should be considered'
                    ]
                }
            },
            'Hazardous': {
                'general_public': {
                    'risk_level': 'Hazardous',
                    'color': '#7E0023',
                    'message': 'HEALTH EMERGENCY! Serious health effects for everyone!',
                    'actions': [
                        '🚨 HEALTH EMERGENCY in effect',
                        '⛔ Remain indoors at all times',
                        '⛔ Complete sealing of living spaces',
                        '⛔ Multiple air purifiers running',
                        '⚠ Emergency evacuation recommended',
                        '⚠ Seek immediate medical help for any symptoms',
                        '• Follow official emergency protocols'
                    ]
                },
                'children_elderly': {
                    'risk_level': 'Hazardous',
                    'color': '#7E0023',
                    'message': 'CRITICAL EMERGENCY! Immediate evacuation recommended!',
                    'actions': [
                        '🚨 CRITICAL EMERGENCY - EVACUATE',
                        '⛔ Immediate relocation to safe area required',
                        '⛔ Emergency medical services should be contacted',
                        '⛔ If evacuation impossible: sealed room with air purifiers',
                        '⚠ Medical monitoring essential',
                        '⚠ Have evacuation plan ready',
                        '• Follow all official emergency directives'
                    ]
                },
                'outdoor_workers': {
                    'risk_level': 'Hazardous',
                    'color': '#7E0023',
                    'message': 'CRITICAL EMERGENCY! Complete site shutdown required!',
                    'actions': [
                        '🚨 CRITICAL EMERGENCY - SITE SHUTDOWN',
                        '⛔ All operations cease immediately',
                        '⛔ Complete evacuation of all personnel',
                        '⛔ Emergency services on site',
                        '⚠ No personnel exposure under any circumstances',
                        '⚠ Medical screening for all exposed workers',
                        '• Site remains closed until air quality improves'
                    ]
                }
            }
        }
    
    def get_adjusted_thresholds(self, persona='general_public'):
        """Get persona-adjusted AQI thresholds"""
        multiplier = self.persona_adjustments[persona]['multiplier']
        
        adjusted = {}
        for category, (low, high) in self.base_thresholds.items():
            adjusted[category] = (
                int(low * multiplier),
                int(high * multiplier)
            )
        
        return adjusted
    
    def classify_aqi(self, aqi_value, persona='general_public'):
        """Classify AQI value into risk category"""
        thresholds = self.get_adjusted_thresholds(persona)
        
        for category, (low, high) in thresholds.items():
            if low <= aqi_value <= high:
                return category
        
        # If AQI exceeds all thresholds
        if aqi_value > 500:
            return 'Hazardous'
        
        return 'Good'
    
    def get_health_advice(self, aqi_value, persona='general_public'):
        """Get complete health advice for given AQI and persona"""
        category = self.classify_aqi(aqi_value, persona)
        advice = self.recommendations[category][persona].copy()
        
        advice['aqi'] = round(aqi_value, 1)
        advice['category'] = category
        advice['persona'] = persona
        advice['persona_description'] = self.persona_adjustments[persona]['description']
        
        return advice
    
    def get_forecast_risks(self, aqi_forecast, persona='general_public'):
        """Get risk assessment for entire forecast period"""
        risks = []
        
        for i, aqi in enumerate(aqi_forecast):
            advice = self.get_health_advice(aqi, persona)
            advice['hour_ahead'] = i + 1
            risks.append(advice)
        
        return pd.DataFrame(risks)
    
    def get_risk_summary(self, aqi_forecast, persona='general_public'):
        """Get summary of risks over forecast period"""
        risks_df = self.get_forecast_risks(aqi_forecast, persona)
        
        # Find worst period
        worst_idx = risks_df['aqi'].idxmax()
        worst_period = risks_df.iloc[worst_idx]
        
        # Risk level distribution
        risk_counts = risks_df['risk_level'].value_counts().to_dict()
        
        summary = {
            'max_aqi': risks_df['aqi'].max(),
            'min_aqi': risks_df['aqi'].min(),
            'avg_aqi': risks_df['aqi'].mean(),
            'worst_hour': worst_period['hour_ahead'],
            'worst_category': worst_period['category'],
            'worst_risk_level': worst_period['risk_level'],
            'risk_distribution': risk_counts,
            'should_avoid_outdoor': any(risks_df['risk_level'].isin(['High', 'Hazardous']))
        }
        
        return summary

def demo_health_classifier():
    """Demonstrate health risk classification"""
    classifier = HealthRiskClassifier()
    
    # Example AQI values
    test_aqis = [45, 85, 125, 175, 225, 325]
    personas = ['general_public', 'children_elderly', 'outdoor_workers']
    
    print("="*80)
    print("HEALTH RISK CLASSIFICATION DEMONSTRATION")
    print("="*80)
    
    for aqi in test_aqis:
        print(f"\n{'='*80}")
        print(f"AQI VALUE: {aqi}")
        print('='*80)
        
        for persona in personas:
            advice = classifier.get_health_advice(aqi, persona)
            print(f"\n{persona.upper().replace('_', ' ')}:")
            print(f"  Category: {advice['category']}")
            print(f"  Risk Level: {advice['risk_level']}")
            print(f"  Message: {advice['message']}")
            print(f"  Actions:")
            for action in advice['actions']:
                print(f"    {action}")

if __name__ == '__main__':
    demo_health_classifier()
