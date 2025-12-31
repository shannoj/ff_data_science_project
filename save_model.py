# save_model.py
import pandas as pd
import joblib
from Cleaning import position_cleaning, handle_categoricals
from model import predict_category_xg

# Load and prepare data
PlayerStats2025 = pd.read_csv("enhanced_stats_2025_3.csv", low_memory=False)
QbStats2025 = position_cleaning(PlayerStats2025, 'QB')
QbStats2025_encoded = handle_categoricals(QbStats2025)

QbStats2025.to_csv('QB_stats_2025_3.csv', index=False)

# Train model
r2, rmse, model, features, X_train, y_train = predict_category_xg('passing_yards', QbStats2025_encoded)

# Save everything
joblib.dump(model, 'qb_yards_model.pkl')
joblib.dump(features, 'model_features.pkl')
joblib.dump({'r2': r2, 'rmse': rmse}, 'model_metrics.pkl')

print(f"Model saved. R²={r2:.3f}, RMSE={rmse:.1f}")