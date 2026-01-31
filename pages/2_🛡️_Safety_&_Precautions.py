import streamlit as st
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui import apply_custom_style, card_start, card_end
from utils.floating_chat import show_floating_chat

st.set_page_config(
    page_title="Safety & Precautions - AeroGuard",
    page_icon="🛡️",
    layout="wide"
)

apply_custom_style()

# Header
st.markdown('<h1 class="main-header">🛡️ Safety Hub</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Personalized Health Guidelines & Emergency Protocols</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Content Logic
# -----------------------------------------------------------------------------

persona_advice = {
    "General Public": {
        "Good (0-50)": ["Enjoy outdoor activities.", "Open windows for ventilation."],
        "Moderate (51-100)": ["Sensitive individuals should reduce prolonged outdoor exertion."],
        "Unhealthy (101-150)": ["Reduce prolonged outdoor exertion.", "Wear a mask if coughing.", "Use air purifiers indoors."],
        "Very Unhealthy (201-300)": ["Avoid all outdoor activities.", "Wear N95 mask outdoors.", "Run air purifier on high."],
    },
    "Children & Elderly": {
        "Good (0-50)": ["Safe for outdoor play.", "Ideal for walks."],
        "Moderate (51-100)": ["limit time near busy roads.", "Monitor for breathing issues."],
        "Unhealthy (101-150)": ["Avoid outdoor play.", "Keep asthma medication handy.", "Stay indoors with filtration."],
        "Very Unhealthy (201-300)": ["STRICTLY NO OUTDOOR EXPOSURE.", "Create a clean room at home."],
    },
    "Outdoor Workers": {
        "Good (0-50)": ["Safe to work without protection."],
        "Moderate (51-100)": ["Take regular breaks in clean air areas."],
        "Unhealthy (101-150)": ["Wear N95/P100 mask.", "Reduce heavy exertion.", "Take frequent breaks."],
        "Very Unhealthy (201-300)": ["Stop heavy outdoor work.", "Mandatory respirator use.", "Rotate shifts to minimize exposure."],
    }
}

# -----------------------------------------------------------------------------
# UI Layout
# -----------------------------------------------------------------------------

# Persona Tabs
tabs = st.tabs(list(persona_advice.keys()))

for i, (persona, advice_dict) in enumerate(persona_advice.items()):
    with tabs[i]:
        st.markdown(f"### 🎯 Guidelines for: **{persona}**")
        st.markdown("---")
        
        # Create better visual hierarchy
        col_left, col_right = st.columns([1, 1], gap="large")
        
        with col_left:
            st.markdown("#### 🟢 Safe Conditions")
            
            # Good Air Quality
            st.markdown("""
            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                        padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #10b981;">
                <div style="font-weight: 700; color: #065f46; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    ✅ AQI 0-50: Good
                </div>
            """, unsafe_allow_html=True)
            for advice in advice_dict["Good (0-50)"]:
                st.markdown(f"• {advice}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Moderate Air Quality
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%); 
                        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #eab308;">
                <div style="font-weight: 700; color: #854d0e; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    ⚠️ AQI 51-100: Moderate
                </div>
            """, unsafe_allow_html=True)
            for advice in advice_dict["Moderate (51-100)"]:
                st.markdown(f"• {advice}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_right:
            st.markdown("#### 🔴 Hazardous Conditions")
            
            # Unhealthy Air Quality
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%); 
                        padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #f97316;">
                <div style="font-weight: 700; color: #9a3412; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    🚨 AQI 101-200: Unhealthy
                </div>
            """, unsafe_allow_html=True)
            for advice in advice_dict["Unhealthy (101-150)"]:
                st.markdown(f"• {advice}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Very Unhealthy Air Quality
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #ef4444;">
                <div style="font-weight: 700; color: #991b1b; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    ☠️ AQI 201+: Very Unhealthy
                </div>
            """, unsafe_allow_html=True)
            for advice in advice_dict["Very Unhealthy (201-300)"]:
                st.markdown(f"**• {advice}**")
            st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# Emergency Section
col1, col2 = st.columns([2, 1])

# Emergency Section
col1, col2 = st.columns([1.5, 1])

with col1:
    card_start("🎒 Emergency Kit Checklist")
    st.markdown("Ensure you have these items ready during high pollution days:")
    st.checkbox("N95 / KNN95 Masks (Family Pack)")
    st.checkbox("Portable Air Purifier")
    st.checkbox("Inhalers / Asthma Medication")
    st.checkbox("Saline Nasal Drops")
    st.checkbox("Emergency Contacts List")
    
    if os.path.exists("assets/safety-infographic.png"):
        st.write("") # Spacer
        # Infographic centered and larger
        st.markdown("---")
        _, c_img, _ = st.columns([1, 4, 1])
        with c_img:
            st.image("assets/safety-infographic.png", caption="Balanced Safety & Prevention Protocol", use_column_width=True)
    
    card_end()

with col2:
    card_start("📞 Emergency Contacts")
    st.markdown("""
    <div style="background-color: #fee2e2; padding: 1rem; border-radius: 10px; border: 1px solid #fecaca;">
        <div style="font-weight: bold; color: #991b1b; margin-bottom: 0.5rem;">🚑 Medical Emergency</div>
        <div style="font-size: 1.5rem; font-weight: 800; color: #ef4444;">102 / 108</div>
    </div>
    <div style="margin-top: 1rem;">
        <b>Pollution Control Board:</b><br>1800-123-456
    </div>
    <div style="margin-top: 1rem;">
        <b>Local Lung Specialist:</b><br>Dr. Aero Smith (+91 98765 43210)
    </div>
    """, unsafe_allow_html=True)
    card_end()
    
    card_start("🏥 Nearest Safe Zones")
    st.markdown("• **City Hospital** (HEPA Filtered)")
    st.markdown("• **Central Mall** (Central AC)")
    st.markdown("• **Public Library**")
    card_end()
    
    # Floating Chat
    show_floating_chat({'location': 'Safety Page', 'aqi': 'N/A'})
