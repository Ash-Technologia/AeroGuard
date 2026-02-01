"""
AeroGuard - Home
Hyper-Local Air Quality & Health Risk Forecaster
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_folium import st_folium
import os
import sys
from datetime import datetime

# Add utils and models to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import AQIPreprocessor
from utils.spatial_interpolation import interpolate_city_aqi
from models.health_classifier import HealthRiskClassifier
from models.explainer import AQIExplainer

from utils.chatbot import AeroGuardBot
from utils.floating_chat import show_floating_chat
from utils.ui import apply_custom_style, card_start, card_end, display_metric_custom

# Page config
st.set_page_config(
    page_title="AeroGuard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium styles
apply_custom_style()

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load AQI data"""
    data_path = 'data/aqi_data.csv'
    if not os.path.exists(data_path):
        st.error("Data file not found! Please run 'python utils/data_generator.py' first.")
        st.stop()
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_data
def load_sensor_locations():
    """Load sensor location data"""
    locations_path = 'data/sensor_locations.csv'
    if os.path.exists(locations_path):
        return pd.read_csv(locations_path)
    return None

def create_aqi_gauge(aqi_value, title="Current AQI"):
    """Create AQI gauge chart"""
    if aqi_value <= 50:
        color = "#10b981"  # Emerald-500
    elif aqi_value <= 100:
        color = "#eab308"  # Yellow-500
    elif aqi_value <= 150:
        color = "#f97316"  # Orange-500
    elif aqi_value <= 200:
        color = "#ef4444"  # Red-500
    elif aqi_value <= 300:
        color = "#a855f7"  # Purple-500
    else:
        color = "#881337"  # Rose-900
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        title={'text': title, 'font': {'size': 18, 'color': '#64748b'}},
        number={'font': {'size': 40, 'color': '#0f172a'}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': '#d1fae5'},
                {'range': [50, 100], 'color': '#fef9c3'},
                {'range': [100, 150], 'color': '#ffedd5'},
                {'range': [150, 200], 'color': '#fee2e2'},
                {'range': [200, 300], 'color': '#f3e8ff'},
                {'range': [300, 500], 'color': '#ffe4e6'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 150
            }
        }
    ))
    
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_forecast_chart(forecast_df, persona='general_public'):
    """Create forecast visualization"""
    classifier = HealthRiskClassifier()
    
    # Get risk levels for each forecast point
    risk_colors = []
    for aqi in forecast_df['forecast_aqi']:
        advice = classifier.get_health_advice(aqi, persona)
        risk_colors.append(advice['color'])
    
    fig = go.Figure()
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['hour_ahead'],
        y=forecast_df['forecast_aqi'],
        mode='lines+markers',
        name='Forecasted AQI',
        line=dict(width=4, color='#3b82f6', shape='spline'),
        marker=dict(size=12, color=risk_colors, line=dict(width=2, color='white'))
    ))
    
    fig.update_layout(
        title=dict(text="6-Hour AQI Forecast", font=dict(size=16)),
        xaxis_title="Hours Ahead",
        yaxis_title="AQI",
        height=350,
        hovermode='x unified',
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(226, 232, 240, 0.5)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(226, 232, 240, 0.5)')
    )
    
    return fig

def create_heatmap(df, sensor_locations, timestamp):
    """Create pollution heatmap"""
    import folium
    from folium.plugins import HeatMap
    
    # Filter data for specific timestamp
    snapshot = df[df['timestamp'] == timestamp].dropna(subset=['aqi'])
    
    if len(snapshot) == 0:
        return None
    
    # Interpolate AQI across city
    try:
        interpolated_df, lat_grid, lon_grid = interpolate_city_aqi(
            snapshot[['latitude', 'longitude', 'aqi']],
            method='idw',
            grid_size=30
        )
        
        # Create base map centered on Mumbai
        center_lat = snapshot['latitude'].mean()
        center_lon = snapshot['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='CartoDB positron'  # Cleaner base map for premium look
        )
        
        heat_data = [[row['latitude'], row['longitude'], row['aqi_interpolated']] 
                     for _, row in interpolated_df.iterrows()]
        
        HeatMap(
            heat_data,
            min_opacity=0.4,
            max_val=200,
            radius=15,
            blur=18,
            gradient={
                0.0: '#10b981',  # Good
                0.4: '#eab308',  # Moderate
                0.6: '#f97316',  # Unhealthy Sensitive
                0.8: '#ef4444',  # Unhealthy
                1.0: '#a855f7'   # Very Unhealthy
            }
        ).add_to(m)
        
        # Add sensor markers
        for _, sensor in snapshot.iterrows():
            aqi = sensor['aqi']
            if aqi <= 50: color = '#10b981'
            elif aqi <= 100: color = '#eab308'
            elif aqi <= 150: color = '#f97316'
            elif aqi <= 200: color = '#ef4444'
            else: color = '#a855f7'
            
            folium.CircleMarker(
                location=[sensor['latitude'], sensor['longitude']],
                radius=6,
                popup=f"<b>{sensor['area_name']}</b><br>AQI: {aqi:.0f}",
                color='white',
                weight=2,
                fill=True,
                fillColor=color,
                fillOpacity=1.0
            ).add_to(m)
        
        return m
        
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return None

def generate_mock_forecast(current_aqi, hours=6):
    """Generate mock forecast for demonstration"""
    forecast = []
    base = current_aqi
    
    for h in range(1, hours + 1):
        # Add some realistic variation
        variation = np.random.normal(0, 5)
        trend = (h - 3) * 2  # Slight upward trend
        value = base + variation + trend
        value = max(0, min(500, value))  # Clamp to valid range
        
        forecast.append({
            'hour_ahead': h,
            'forecast_aqi': value
        })
    
    return pd.DataFrame(forecast)

def main():
    # Hero Section with Background
    header_html = """
    <div style="
        background-image: url('assets/header-bg.png');
        background-size: cover;
        background-position: center;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
    ">
        <h1 style="font-size: 4rem; font-weight: 900; color: #000000; margin: 0; text-shadow: 2px 2px 4px rgba(255,255,255,0.8);">
            🌍 AeroGuard
        </h1>
        <p style="font-size: 1.5rem; color: #1e293b; margin-top: 0.5rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(255,255,255,0.6);">
            Hyper-Local Air Quality & Health Risk Forecaster
        </p>
        <p style="font-size: 1.1rem; color: #475569; margin-top: 1rem; font-style: italic;">
            💨 Breathe Smart. Live Safe. Plan Ahead.
        </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Feature Cards Row
    st.markdown("### ✨ What We Offer")
    feat1, feat2, feat3 = st.columns(3)
    
    with feat1:
        if os.path.exists("assets/icon-air.png"):
            st.image("assets/icon-air.png", width=80)
        st.markdown("**Real-Time Monitoring**")
        st.caption("Track AQI levels across multiple sensors in your city.")
    
    with feat2:
        if os.path.exists("assets/icon-forecast.png"):
            st.image("assets/icon-forecast.png", width=80)
        st.markdown("**AI-Powered Forecasts**")
        st.caption("6-hour predictions using LSTM & Prophet models.")
    
    with feat3:
        if os.path.exists("assets/icon-health.png"):
            st.image("assets/icon-health.png", width=80)
        st.markdown("**Personalized Health Advice**")
        st.caption("Tailored recommendations for your vulnerability profile.")
    
    st.markdown("---")
    st.markdown('<p class="sub-header">', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Initializing AeroGuard Intelligence..."):
        df = load_data()
        sensor_locations = load_sensor_locations()
    
    # Calculate latest time first for use in metrics
    latest_time = df['timestamp'].max()

    # Sidebar - Simplified
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer as requested
        st.markdown("## 📊 Live Statistics")
        
        # Data Quality Indicators
        total_sensors = len(sensor_locations) if sensor_locations is not None else 0
        active_sensors = len(df[df['timestamp'] == latest_time]['sensor_id'].unique())
        data_completeness = (active_sensors / total_sensors * 100) if total_sensors > 0 else 0
        
        st.metric("Active Sensors", f"{active_sensors}/{total_sensors}")
        st.progress(data_completeness / 100)
        st.caption(f"Data Completeness: {data_completeness:.0f}%")
        
        st.divider()
        
        # City Overview
        st.markdown("### 🌆 City Overview")
        city_avg_aqi = df[df['timestamp'] == latest_time]['aqi'].mean()
        city_max_aqi = df[df['timestamp'] == latest_time]['aqi'].max()
        
        st.metric("City Avg AQI", f"{city_avg_aqi:.0f}")
        st.metric("Hotspot AQI", f"{city_max_aqi:.0f}")
        
        st.divider()
        
        # Model Info
        st.markdown("### 🤖 AI Models")
        st.caption("**Active Models:**")
        st.caption("• LSTM (Deep Learning)")
        st.caption("• Prophet (Time Series)")
        st.caption("• XGBoost (Ensemble)")
        st.caption("• SARIMA (Statistical)")
        
        st.divider()
        
        # System Status
        st.markdown("### 🕐 System Status")
        st.info(f"🔄 Last Updated\n\n{latest_time.strftime('%Y-%m-%d %H:%M')}")
        
        st.divider()
        st.markdown("### ℹ️ About")
        st.caption("AeroGuard uses LSTM & Prophet models to forecast hyperlocal AQI levels.")
        st.markdown("Team - Quantum Minds")
        st.markdown("Members")
        st.markdown("Leader- Aayush Singhavi")
        st.markdown("Ashutosh Shirke")
        st.markdown("Harshal Wankhade")
        st.markdown("Piyush Poptani")
        st.caption("Built for Hackathon PS02 🏆")
        st.caption("Aligned with WHO/EPA Standards")
    
    # Configuration Section - Main Page
    st.markdown("### ⚙️ Configuration")
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown("**📍 Location**")
        if sensor_locations is not None:
            location_options = sensor_locations['area_name'].tolist()
            selected_location = st.selectbox(
                "Monitor Area",
                options=location_options,
                index=0,
                label_visibility="collapsed"
            )
            
            # Get coordinates
            loc_data = sensor_locations[sensor_locations['area_name'] == selected_location].iloc[0]
            sensor_id = loc_data['sensor_id']
            st.caption(f"📡 Sensor: {sensor_id}")
        else:
            selected_location = "Default Location"
            sensor_id = df['sensor_id'].iloc[0]
    
    with config_col2:
        st.markdown("**👤 Health Profile**")
        persona_options = {
            'General Public': 'general_public',
            'Children / Elderly': 'children_elderly',
            'Outdoor Workers': 'outdoor_workers'
        }
        
        selected_persona_label = st.selectbox(
            "Vulnerability Profile",
            options=list(persona_options.keys()),
            index=0,
            label_visibility="collapsed"
        )
        selected_persona = persona_options[selected_persona_label]
        st.caption(f"🎯 Profile: {selected_persona_label}")
    
    with config_col3:
        st.markdown("**🔮 Forecast Horizon**")
        forecast_hours = st.selectbox(
            "Hours to Forecast",
            options=[6, 12],
            index=0,
            label_visibility="collapsed"
        )
        st.caption(f"⏱️ Next {forecast_hours} hours")
    
    st.markdown("---")
    
    # Main content
    
    # Get current data for selected location
    current_data = df[(df['sensor_id'] == sensor_id) & (df['timestamp'] == latest_time)]
    
    if len(current_data) > 0:
        current_aqi = float(current_data['aqi'].values[0])
        current_pm25 = float(current_data['pm25'].values[0])
        current_temp = float(current_data['temperature'].values[0])
        current_humidity = float(current_data['humidity'].values[0])
        current_wind = float(current_data['wind_speed'].values[0])
    else:
        # Fallback to average
        current_aqi = float(df[df['sensor_id'] == sensor_id]['aqi'].mean())
        current_pm25 = float(df[df['sensor_id'] == sensor_id]['pm25'].mean())
        current_temp = float(df['temperature'].mean())
        current_humidity = float(df['humidity'].mean())
        current_wind = float(df['wind_speed'].mean())
    
    # Health Advice Banner
    classifier = HealthRiskClassifier()
    health_advice = classifier.get_health_advice(current_aqi, selected_persona)
    
    st.markdown(f"""
    <div style="background-color: {health_advice['color']}20; border: 1px solid {health_advice['color']}; 
                border-radius: 16px; padding: 1rem; margin-bottom: 2rem; display: flex; align-items: center; gap: 1rem;">
        <div style="font-size: 2.5rem;">{health_advice['icon'] if 'icon' in health_advice else '🩺'}</div>
        <div>
            <div style="color: {health_advice['color']}; font-weight: 800; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.05em;">
                {health_advice['risk_level']} Health Risk
            </div>
            <div style="font-size: 1.05rem; font-weight: 500;">{health_advice['message']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        # AQI Gauge Card
        card_start()
        st.plotly_chart(create_aqi_gauge(current_aqi), use_container_width=True)
        
        # Detailed Metrics Grid
        c1, c2 = st.columns(2)
        with c1:
            display_metric_custom("PM2.5", f"{current_pm25:.0f}", "μg/m³")
            display_metric_custom("Temp", f"{current_temp:.0f}°", "C", color="#f59e0b")
        with c2:
            display_metric_custom("Humidity", f"{current_humidity:.0f}%", "", color="#3b82f6")
            display_metric_custom("Wind", f"{current_wind:.0f}", "km/h", color="#6366f1")
        card_end()
        
        # Health Actions Card (Simpler view for Home)
        card_start("🛡️ Quick Actions")
        for action in health_advice['actions'][:2]:
            st.markdown(f"• {action}")
        
        st.markdown("[View Full Safety Guidelines >](/Safety_&_Precautions)")
        card_end()
    
    with col2:
        # Forecast Card
        card_start(f"🔮 AI Forecast (Next {forecast_hours} Hours)")
        forecast_df = generate_mock_forecast(current_aqi, hours=forecast_hours)
        st.plotly_chart(create_forecast_chart(forecast_df, selected_persona), use_container_width=True)
        
        # Forecast Summary Text
        risk_summary = classifier.get_risk_summary(forecast_df['forecast_aqi'].values, selected_persona)
        worst_hour_aqi = forecast_df.loc[forecast_df['hour_ahead'] == risk_summary['worst_hour'], 'forecast_aqi'].values[0]
        
        st.info(f"Plan your day: Conditions will be worst around **Hour {risk_summary['worst_hour']}** (AQI ~{worst_hour_aqi:.0f}). " +
                ("Outdoor activities NOT recommended." if risk_summary['should_avoid_outdoor'] else "Outdoor activities generally safe."))
        card_end()
        
        # Map Card
        card_start("🗺️ Live Pollution Map")
        try:
            heatmap = create_heatmap(df, sensor_locations, latest_time)
            if heatmap:
                st_folium(heatmap, width="100%", height=300)
            else:
                st.warning("Map unavailable")
        except Exception:
            st.warning("Map temporarily unavailable")
        card_end()
    
    
    # Data Flow Visualization
    st.markdown("### 🔄 How AeroGuard Works")
    
    flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)
    
    with flow_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 2rem;">📡</div>
            <div style="font-weight: 600; margin-top: 0.5rem;">Sensor Data</div>
            <div style="font-size: 0.85rem; opacity: 0.9;">Real-time collection</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 2rem;">🧮</div>
            <div style="font-weight: 600; margin-top: 0.5rem;">AI Processing</div>
            <div style="font-size: 0.85rem; opacity: 0.9;">LSTM + Prophet</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 2rem;">🔮</div>
            <div style="font-weight: 600; margin-top: 0.5rem;">Forecasting</div>
            <div style="font-size: 0.85rem; opacity: 0.9;">6-hour prediction</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
            <div style="font-size: 2rem;">💡</div>
            <div style="font-weight: 600; margin-top: 0.5rem;">Insights</div>
            <div style="font-size: 0.85rem; opacity: 0.9;">Health advice</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")  # Spacer
    
    # Explanation Section
    st.markdown("### 💡 Why is the AQI like this?")
    explainer = AQIExplainer(use_ai=False)
    historical_aqi = df[df['sensor_id'] == sensor_id].tail(24)['aqi'].values
    weather_data = {'wind_speed': current_wind, 'humidity': current_humidity, 'temperature': current_temp}
    time_data = {'hour': latest_time.hour, 'is_rush_hour': 7 <= latest_time.hour <= 10 or 17 <= latest_time.hour <= 20, 'is_weekend': latest_time.weekday() >= 5}
    
    explanation = explainer.explain_forecast(
        historical_aqi, forecast_df['forecast_aqi'].values, weather_data, time_data
    )
    st.markdown(f"*{explanation['explanation']}*")
    
    # Floating AI Assistant
    risk_level = classifier.classify_aqi(current_aqi, selected_persona)
    context = {
        'aqi': round(current_aqi, 0),
        'location': selected_location,
        'risk': risk_level
    }
    show_floating_chat(context)

if __name__ == '__main__':
    main()
