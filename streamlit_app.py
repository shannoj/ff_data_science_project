import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from Cleaning import position_cleaning, handle_categoricals
from model import predict_category_xg, bootstrap_predictions
from predict_future import create_qb_prediction_features
import joblib

st.set_page_config(
    page_title="NFL QB Yards Predictor",
    layout="wide"
)

st.title("NFL Quarterback Yards Predictor")
st.markdown("Predicting passing yards using machine learning and game context")

@st.cache_data
def load_data():
    PlayerStats2025 = pd.read_csv("enhanced_stats_2025_3.csv", low_memory=False)
    QbStats2025 = position_cleaning(PlayerStats2025, 'QB')
    return PlayerStats2025, QbStats2025

@st.cache_resource
def load_model():
    """Load pre-trained model instead of training from scratch"""
    model = joblib.load('qb_yards_model.pkl')
    features = joblib.load('model_features.pkl')
    metrics = joblib.load('model_metrics.pkl')
    return model, features, metrics['r2'], metrics['rmse']

with st.spinner("Loading model..."):
    PlayerStats2025, QbStats2025 = load_data()
    yards_model, model_features, r2, rmse = load_model()

# Get list of QBs from the original data, filtered for QBs only
qb_list = sorted([
    str(name) for name in PlayerStats2025[PlayerStats2025['position'] == 'QB']['player_name'].unique() 
    if pd.notna(name)
])
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Game Setup")
    
    selected_qb = st.selectbox("Select Quarterback", qb_list, index=qb_list.index('P.Mahomes') if 'P.Mahomes' in qb_list else 0)

    qb_team = PlayerStats2025[PlayerStats2025['player_name'] == selected_qb]['team'].iloc[0]
    
    col1a, col1b, col1c = st.columns(3)
    
    with col1a:
        teams = sorted(QbStats2025['team'].unique())
        opponent = st.selectbox("Opponent", [t for t in teams if t != qb_team])
    
    with col1b:
        is_home = st.radio("Location", ["Home", "Away"], horizontal=True)
        is_home_val = 1 if is_home == "Home" else 0
    
    with col1c:
        week = st.number_input("Week", min_value=1, max_value=18, value=18)

with col2:
    st.subheader("Advanced Options")
    st.subheader("Be wary of results if the player did not play in any games within the average window selected")
    num_games = st.slider("Recent games to average", 3, 12, 5)
    show_confidence = st.checkbox("Show confidence intervals", value=True)

st.markdown("---")
game_date = st.date_input("Game Date", value=datetime(2025, 1, 5))
game_time = st.time_input("Game Time", value=datetime.strptime("13:00", "%H:%M").time())
game_datetime = datetime.combine(game_date, game_time)

if st.button("Generate Prediction", type="primary"):
    with st.spinner("Generating prediction..."):
        
        qb_predictions = create_qb_prediction_features(
            selected_qb, 
            qb_team, 
            opponent, 
            is_home_val, 
            week, 
            PlayerStats2025, 
            num_recent_games=num_games,
            game_date=game_datetime
        )
        
        features_encoded = handle_categoricals(qb_predictions)
        X_pred = features_encoded[model_features]
        pred_yards = yards_model.predict(X_pred)[0]
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric(
                label=f"{selected_qb} Projected Yards",
                value=f"{pred_yards:.0f}",
                delta=f"±{rmse:.0f} (model RMSE)"
            )
        
            lower_rmse = pred_yards - 1.28 * rmse
            upper_rmse = pred_yards + 1.28 * rmse
            
            with col_result2:
                st.metric("80% CI Lower", f"{lower_rmse:.0f}")
            with col_result3:
                st.metric("80% CI Upper", f"{upper_rmse:.0f}")
            
            st.info(f"There's an 80% chance {selected_qb} throws between **{lower_rmse:.0f}** and **{upper_rmse:.0f}** yards")
        
        with st.expander("View Key Features"):
            feature_df = pd.DataFrame({
                'Feature': ['Passer Rating', 'Passing EPA', 'Home/Away', 'Opponent', 'Weather'],
                'Value': [
                    f"{qb_predictions['passer_rating'].iloc[0]:.0f}" if 'passer_rating' in qb_predictions else 'N/A',
                    f"{qb_predictions['passing_epa'].iloc[0]:.2%}" if 'passing_epa' in qb_predictions else 'N/A',
                    is_home,
                    opponent,
                    f"{qb_predictions['temp'].iloc[0]:.0f}°F" if 'temp' in qb_predictions else 'N/A'
                ]
            })
            st.dataframe(feature_df, hide_index=True)

st.markdown("---")
st.markdown("""
**About this model:**
- Trained on NFL games from 2025 season
- Uses XGBoost with 42 engineered features
- Incorporates recent performance, opponent strength, weather, and game context
- Model performance: R²=0.685
""")