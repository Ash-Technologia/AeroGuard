
import streamlit as st

def apply_custom_style():
    """Apply premium CSS styles to the Streamlit app"""
    st.markdown("""
        <style>
        /* Import premium fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1a202c;
        }
        

        .stApp {
            background-color: #f8fafc;
            background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
            background-size: 24px 24px;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
            border-right: 1px solid #e0f2fe;
            box-shadow: 6px 0 24px rgba(59, 130, 246, 0.05);
        }
        
        section[data-testid="stSidebar"] .block-container {
            padding-top: 3rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        
        section[data-testid="stSidebar"] h1 {
            color: #0c4a6e;
            font-size: 1.5rem;
        }

        /* Typography */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            letter-spacing: -0.03em;
            color: #0f172a;
        }
        
        .main-header {
            font-size: 4rem;
            font-weight: 900;
            color: #000000;
            text-align: center;
            margin-bottom: 0.5rem;
            margin-top: 1rem;
        }
        
        .sub-header {
            text-align: center;
            font-size: 1.5rem;
            color: #64748b;
            font-weight: 400;
            margin-bottom: 4rem;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background-color: transparent;
            border-bottom: none;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3.5rem;
            white-space: pre-wrap;
            border-radius: 12px;
            font-weight: 600;
            color: #64748b;
            background-color: white;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: all 0.2s;
            padding: 0 1.5rem;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #3b82f6;
            color: white;
            border: none;
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
        }

        /* Cards & Containers */
        .premium-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.04);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 2.5rem;
        }
        
        .premium-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 20px 40px -5px rgba(0, 0, 0, 0.08);
        }

        /* Metrics */
        .metric-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
            margin: 0.5rem 0;
        }
        
        /* Custom Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 14px;
            font-weight: 600;
            letter-spacing: 0.02em;
            transition: all 0.2s;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            width: 100%;
        }

        
        .stButton > button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        /* Alerts/Badges */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
        }
        
        </style>
    """, unsafe_allow_html=True)

def card_start(title=None):
    """Start a premium card container"""
    if title:
        st.markdown(f"""
        <div class="premium-card">
            <h3 style="margin-top: 0; margin-bottom: 1rem; font-size: 1.25rem;">{title}</h3>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)

def card_end():
    """End a premium card container"""
    st.markdown('</div>', unsafe_allow_html=True)

def display_metric_custom(label, value, delta=None, color=None):
    """Display a custom styled metric"""
    delta_html = ""
    if delta:
        delta_color = "#10b981" if not delta.startswith("-") else "#ef4444"
        delta_sign = "+" if not delta.startswith("-") and not delta.startswith("+") else ""
        delta_html = f'<div style="color: {delta_color}; font-size: 0.875rem; font-weight: 600; margin-top: 4px;">{delta_sign}{delta}</div>'
    
    style_color = f'color: {color};' if color else ''
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="{style_color}">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)
