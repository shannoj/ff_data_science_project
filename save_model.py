# save_model.py
import pandas as pd
import joblib
from Cleaning import position_cleaning, handle_categoricals
from model import predict_category_xg
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare data
PlayerStats2025 = pd.read_csv("enhanced_stats_2025_3.csv", low_memory=False)
QbStats2025 = position_cleaning(PlayerStats2025, 'QB')
QbStats2025_encoded = handle_categoricals(QbStats2025)

QbStats2025.to_csv('QB_stats_2025_3.csv', index=False)

# Train model

hyperparameter = np.arange(1, 50, 0.5) 

r2_list = []
r2_train_list = []

for param in hyperparameter:
    r2, r2_train, rmse, model, features, X_train, y_train = predict_category_xg('passing_yards', QbStats2025_encoded, 0.02, 3, param)
    r2_list.append(1-r2)
    r2_train_list.append(1-r2_train)


plt.plot(hyperparameter, r2_list, label='test error')
plt.plot(hyperparameter, r2_train_list, label="train error")
plt.legend()
plt.show()


# # Save everything
# joblib.dump(model, 'qb_yards_model.pkl')
# joblib.dump(features, 'model_features.pkl')
# joblib.dump({'r2': r2, 'rmse': rmse}, 'model_metrics.pkl')

# print(f"Model saved. R²={r2:.3f}, RMSE={rmse:.1f}")