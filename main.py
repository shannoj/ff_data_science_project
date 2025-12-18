import pandas as pd
from Cleaning import position_cleaning, handle_categoricals, add_year_suffix
from model import predict_category
from predict_future import create_qb_prediction_features
from future_games_features import get_current_week
import joblib
from weather import update_stadium_forecasts

#update_stadium_forecasts()

PlayerStats2025 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/enhanced_stats_2025.csv", low_memory=False)

QbStats2025 = position_cleaning(PlayerStats2025, 'QB')

QbStats2025_encoded = handle_categoricals(QbStats2025)

r2_yards, rmse_yards, yards_model = predict_category('passing_yards', QbStats2025_encoded)
r2_tds, rmse_tds, tds_model = predict_category('passing_tds', QbStats2025_encoded)

yards_model = joblib.load('qb_yards_model_2025.pkl')
tds_model = joblib.load('qb_tds_model_2025.pkl')
model_features = joblib.load('model_features_2025.pkl')

darnold_qb_predictions = create_qb_prediction_features('S.Darnold', 'SEA', 'LAR', 1, 
                                16, PlayerStats2025, num_recent_games=5)

Stafford_qb_predictions = create_qb_prediction_features('M.Stafford', 'LAR', 'SEA', 0, 
                                16, PlayerStats2025, num_recent_games=5)

features_encoded = handle_categoricals(darnold_qb_predictions)

features_encoded_stafford = handle_categoricals(Stafford_qb_predictions)

X_pred = features_encoded[model_features]

X_pred_staff = features_encoded_stafford[model_features]
    
pred_yards = yards_model.predict(X_pred)[0]
pred_tds = tds_model.predict(X_pred)[0]

pred_yards_staff = yards_model.predict(X_pred_staff)[0]
pred_tds_staff = tds_model.predict(X_pred_staff)[0]

print(pred_yards)
print(pred_tds)

print(pred_yards_staff)
print(pred_tds_staff)


