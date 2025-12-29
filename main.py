import pandas as pd
from Cleaning import position_cleaning, handle_categoricals
from model import predict_category, predict_category_xg
from predict_future import create_qb_prediction_features
from future_games_features import get_current_week
import joblib
from weather import update_stadium_forecasts

#update_stadium_forecasts()

PlayerStats2025 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/enhanced_stats_2025_2.csv", low_memory=False)

QbStats2025 = position_cleaning(PlayerStats2025, 'QB')

QbStats2025_encoded = handle_categoricals(QbStats2025)

r2_yards_xg, rmse_yards_xg, yards_model_xg, model_features_1 = predict_category_xg('passing_yards', QbStats2025_encoded)

model_features = model_features_1

qb_predictions = create_qb_prediction_features('J.Allen', 'BUF', 'PHI', 1, 17, PlayerStats2025, num_recent_games=3)

features_encoded = handle_categoricals(qb_predictions)

X_pred = features_encoded[model_features]

pred_yards_xg = yards_model_xg.predict(X_pred)[0]

print(pred_yards_xg)
