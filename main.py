import pandas as pd
from Cleaning import position_cleaning, handle_categoricals
from model import predict_category_xg
from predict_future import create_qb_prediction_features
import joblib
from model import bootstrap_predictions
from datetime import datetime

PlayerStats2025 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/enhanced_stats_2025_3.csv", low_memory=False)

QbStats2025 = position_cleaning(PlayerStats2025, 'QB')

QbStats2025_encoded = handle_categoricals(QbStats2025)

r2_yards_xg, rmse_yards_xg, yards_model_xg, model_features_1, X_train, y_train = predict_category_xg('passing_yards', QbStats2025_encoded)

model_features = model_features_1

game_datetime = datetime(2026, 1, 4, 13, 0) 

qb_predictions = create_qb_prediction_features('B.Young', 'CAR', 'TB', 0, 18, PlayerStats2025, num_recent_games=8, game_date=game_datetime)

features_encoded = handle_categoricals(qb_predictions)

X_pred = features_encoded[model_features]

pred_yards_xg = yards_model_xg.predict(X_pred)[0]

lower, median, upper = bootstrap_predictions(yards_model_xg, X_train, y_train, X_pred, n_iterations=100)

print(pred_yards_xg)

print(lower)

print(median)

print(upper)