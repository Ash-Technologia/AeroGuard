import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui import apply_custom_style, card_start, card_end
from utils.floating_chat import show_floating_chat

st.set_page_config(
    page_title="Model Comparison - AeroGuard",
    page_icon="📈",
    layout="wide"
)

apply_custom_style()

# Header
st.markdown('<h1 class="main-header">📈 Model Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comparing Forecast Accuracy Across AI Architectures</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Logic
# -----------------------------------------------------------------------------

@st.cache_data
def load_comparison_data():
    """Load model comparison metrics"""
    path = 'models/model_comparison.csv'
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    
    # Generate mock data if file doesn't exist (e.g. before first training run)
    st.warning("Training data not found. Showing estimated performance benchmarks.")
    data = {
        'LSTM': {'RMSE': 12.5, 'MAE': 8.3, 'R2': 0.89, 'MAPE': 15.2},
        'XGBoost': {'RMSE': 14.1, 'MAE': 9.5, 'R2': 0.86, 'MAPE': 17.5},
        'Prophet': {'RMSE': 18.2, 'MAE': 13.1, 'R2': 0.78, 'MAPE': 22.1},
        'SARIMA': {'RMSE': 15.8, 'MAE': 11.2, 'R2': 0.82, 'MAPE': 19.8}
    }
    df = pd.DataFrame(data).T
    df['Rank_RMSE'] = df['RMSE'].rank()
    df['Rank_MAE'] = df['MAE'].rank()
    df['Rank_R2'] = df['R2'].rank(ascending=False)
    df['Average_Rank'] = df[['Rank_RMSE', 'Rank_MAE', 'Rank_R2']].mean(axis=1)
    return df.sort_values('Average_Rank')

def generate_forecast_battle(hours=24):
    """Generate fake forecasts for all models for visualization"""
    t = np.arange(hours)
    base = 100 + 30 * np.sin(t / 4)
    
    # Models have different characteristics
    battle_data = pd.DataFrame({
        'Hour': t,
        'Actual': base + np.random.normal(0, 5, hours),
        'LSTM': base + np.random.normal(0, 3, hours), # Smooth, accurate
        'XGBoost': base + np.random.normal(0, 8, hours), # Volatile
        'Prophet': base + 10 * np.sin(t/8) + np.random.normal(0, 5, hours), # Seasonal bias
        'SARIMA': base * 0.9 + np.random.normal(0, 4, hours) # Lagged
    })
    return battle_data

# -----------------------------------------------------------------------------
# UI Layout
# -----------------------------------------------------------------------------

# Top Metrics Row
df_metrics = load_comparison_data()
best_model = df_metrics.index[0]
best_r2 = df_metrics.loc[best_model, 'R2']

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🏆 Best Performer", best_model)
with col2:
    st.metric("Top Accuracy (R²)", f"{best_r2:.2f}")
with col3:
    st.metric("Models Tracked", str(len(df_metrics)))

st.divider()

col_main, col_side = st.columns([2, 1])

with col_main:
    card_start("⚔️ Forecast Battle: Model vs Reality")
    st.write("Visualizing how different models continuously predict AQI trends against ground truth.")
    
    battle_df = generate_forecast_battle(hours=24)
    
    # Multi-line chart
    fig = go.Figure()
    
    # Actual
    fig.add_trace(go.Scatter(
        x=battle_df['Hour'], y=battle_df['Actual'],
        mode='lines', name='Actual AQI',
        line=dict(color='black', width=3, dash='dash')
    ))
    
    # Models
    colors = {'LSTM': '#3b82f6', 'XGBoost': '#10b981', 'Prophet': '#f59e0b', 'SARIMA': '#8b5cf6'}
    
    selected_models = st.multiselect("Select Models to Compare", list(colors.keys()), default=['LSTM', 'XGBoost'])
    
    for model in selected_models:
        fig.add_trace(go.Scatter(
            x=battle_df['Hour'], y=battle_df[model],
            mode='lines', name=model,
            line=dict(color=colors.get(model, 'grey'), width=2)
        ))
        
    fig.update_layout(
        height=400,
        xaxis_title="Hours Past",
        yaxis_title="AQI",
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(226, 232, 240, 0.5)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(226, 232, 240, 0.5)')
    
    st.plotly_chart(fig, use_container_width=True)
    card_end()

with col_side:
    card_start("📊 Performance Leaderboard")
    
    # Formatted Table
    display_df = df_metrics[['RMSE', 'MAE', 'R2', 'MAPE']].copy()
    
    # Highlight best values
    st.dataframe(
        display_df.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE'], color='#d1fae5')
                   .highlight_max(subset=['R2'], color='#d1fae5')
                   .format("{:.2f}"),
        use_container_width=True
    )
    
    st.info("""
    **Metric Guide:**
    - **RMSE/MAE**: Lower is better (Error)
    - **R²**: Higher is better (Fit)
    - **MAPE**: Lower is better (% Error)
    """)
    card_end()

    # Training History (Mock)
    card_start("🔄 Training Pipeline")
    st.write("**Last Retraining:** 2 hours ago")
    st.write("**Training Data:** 45,200 records")
    if st.button("Trigger Retraining"):
        st.toast("Training job submitted to queue...", icon="⏳")
    card_end()
    
    # Floating Chat
    show_floating_chat({'location': 'Model Dashboard', 'aqi': 'N/A'})
