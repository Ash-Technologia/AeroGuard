import random

class AeroGuardBot:
    def __init__(self):
        self.context = {}
        
    def update_context(self, context_data):
        """Update context with latest app data (AQI, Location, etc.)"""
        self.context.update(context_data)

    def get_response(self, user_input):
        user_input = user_input.lower()
        
        # dynamic data from context
        current_aqi = self.context.get('aqi', 'unknown')
        location = self.context.get('location', 'your location')
        risk = self.context.get('risk', 'unknown')
        
        # GREETINGS
        if any(word in user_input for word in ["hi", "hello", "hey", "greetings"]):
            return f"Hello! I'm AeroBot. I can help you with the air quality in {location} (Current AQI: {current_aqi})."
        
        # CURRENT STATUS
        if any(word in user_input for word in ["current", "now", "today", "status"]):
            if current_aqi != 'unknown':
                return f"Currently, the AQI in {location} is **{current_aqi}**. The risk level is **{risk}**."
            else:
                return "I don't have the live data right now. Please check the dashboard."

        # AQI EXPLANATIONS
        if "aqi" in user_input and "what" in user_input:
            return "AQI (Air Quality Index) is a scale from 0 to 500. **0-50** is Good 🟢, **50-100** is Moderate 🟡, and anything above **100** starts getting unhealthy 🟠🔴."
            
        if any(word in user_input for word in ["aqi", "pollution", "air quality"]):
            return f"The Air Quality Index (AQI) tracks pollutants like PM2.5 and PM10. In {location}, the level is {current_aqi}."

        # HEALTH ADVICE
        if any(word in user_input for word in ["health", "safe", "mask", "run", "walk", "exercise"]):
            if isinstance(current_aqi, (int, float)):
                if current_aqi < 50:
                    return "The air is clean! It's a great day for outdoor activities. 🏃‍♂️🚴‍♀️"
                elif current_aqi < 100:
                    return "It's acceptable, but sensitive individuals should monitor their breathing. 😷"
                elif current_aqi < 150:
                    return "I'd recommend wearing a mask if you're sensitive. Limit heavy exertion outdoors."
                else:
                    return "⚠️ **Health Alert:** Please wear an N95 mask and avoid outdoor exercise. Use an air purifier indoors."
            return "Check the 'Safety & Precautions' page for detailed advice based on your profile!"

        # MODEL INFO
        if any(word in user_input for word in ["model", "ai", "forecast", "predict", "work"]):
            return "I use a hybrid AI system combining **LSTM** (Deep Learning) for complex patterns and **Prophet** for seasonal trends. This allows me to forecast AQI up to 12 hours ahead with high accuracy. 🧠"

        # APP INFO
        if any(word in user_input for word in ["app", "hackathon", "made", "creator"]):
            return "I am part of AeroGuard, a hyper-local air quality forecasting system built for the PS02 Hackathon. My goal is to keep you safe from invisible pollutants! 🌍"
            
        # DEFAULT
        return "I'm not sure about that. Try asking: 'What is the AQI?', 'Is it safe to run?', or 'How does the AI work?'"
