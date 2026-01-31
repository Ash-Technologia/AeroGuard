import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui import apply_custom_style, card_start, card_end
from utils.floating_chat import show_floating_chat

st.set_page_config(
    page_title="Pollution Causes - AeroGuard",
    page_icon="🏭",
    layout="wide"
)

apply_custom_style()

st.markdown('<h1 class="main-header">🏭 Pollution Sources</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Understanding the Roots of Air Quality Deterioration</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col1:
    card_start("📉 Major Contributors Breakdown")
    
    # Mock Data for Pollution Sources
    data = {
        'Source': ['Vehicular Emissions', 'Industrial Waste', 'Construction Dust', 'Biomass Burning', 'Road Dust', 'Other'],
        'Percentage': [35, 25, 15, 12, 8, 5]
    }
    df = pd.DataFrame(data)
    
    fig = px.pie(df, values='Percentage', names='Source', 
                 color='Source',
                 color_discrete_sequence=px.colors.sequential.RdBu,
                 hole=0.4)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=True,
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    card_end()

with col2:
    card_start("🚗 Vehicular Emissions")
    st.info("The largest contributor in urban areas.")
    st.markdown("""
    - **NO2 (Nitrogen Dioxide)**: From diesel engines.
    - **CO (Carbon Monoxide)**: Incomplete combustion.
    - **PM2.5**: Tire wear and exhaust particles.
    """)
    card_end()
    
    card_start("🏗️ Construction & Dust")
    st.warning("Rising due to rapid infrastructure development.")
    st.markdown("""
    - **PM10**: Coarse suspended particles.
    - **Cement Dust**: Hazardous to respiratory health.
    - **Mitigation**: Sprinkling water, covering debris.
    """)
    card_end()

st.divider()

card_start("🌱 What Can We Do?")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 🚘 Transport")
    st.write("Carpool, use public transport, or switch to Electric Vehicles.")

with c2:
    st.markdown("### ⚡ Energy")
    st.write("Switch to solar power and use energy-efficient appliances.")

with c3:
    st.markdown("### ♻️ Lifestyle")
    st.write("Reduce waste burning, plant trees, and support green policies.")
card_end()

# Floating Chat
show_floating_chat({'location': 'Education Center', 'aqi': 'N/A'})
