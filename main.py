import pandas as pd
from Cleaning import QB_cleaning
from Cleaning import add_year_suffix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from features import PassingPoints
from features import RushingPoints
from features import TDPoints

PlayerStats2021 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2021.csv")
PlayerStats2022 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2022.csv")
PlayerStats2023 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2023.csv")
PlayerStats2024 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2024.csv")

PlayerStats2021 = PlayerStats2021.drop_duplicates()
PlayerStats2022 = PlayerStats2022.drop_duplicates()
PlayerStats2023 = PlayerStats2023.drop_duplicates()
PlayerStats2024 = PlayerStats2024.drop_duplicates()

QbStats2021 = TDPoints(RushingPoints(PassingPoints(QB_cleaning(PlayerStats2021))))
QbStats2022 = TDPoints(RushingPoints(PassingPoints(QB_cleaning(PlayerStats2022, is_2022=True))))
QbStats2023 = TDPoints(RushingPoints(PassingPoints(QB_cleaning(PlayerStats2023))))
QbStats2024 = TDPoints(RushingPoints(PassingPoints(QB_cleaning(PlayerStats2024, is_2022=True))))

QbStats2021 = add_year_suffix(QbStats2021, 2021)
QbStats2022 = add_year_suffix(QbStats2022, 2022)
QbStats2023 = add_year_suffix(QbStats2023, 2023)
QbStats2024 = add_year_suffix(QbStats2024, 2024)

QbStatsTotal = pd.merge(QbStats2021, QbStats2022, on='player_id', how='inner')
QbStatsTotal = pd.merge(QbStatsTotal, QbStats2023, on='player_id', how='inner')
QbStatsTotal = pd.merge(QbStatsTotal, QbStats2024, on='player_id', how='inner')

QbStatsML = QbStatsTotal.dropna(subset=['fantasy_points_ppr_2024'])

QbStatsML2 = QbStatsML[~(QbStatsML == 0).all(axis=1)]

y = QbStatsML2['fantasy_points_ppr_2024']

X = QbStatsML2[[col for col in QbStatsML.columns if 
               any(year in col for year in ['2021', '2022', '2023']) and
               'fantasy_points_ppr' not in col and
               col != 'player_id']]

print(X.head(10))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

print(y_train)

model = CatBoostRegressor(verbose=0, random_seed=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# feature_importances = model.get_feature_importance()
# feature_names = X.columns

# plt.figure(figsize=(12, 6))
# plt.barh(feature_names, feature_importances)
# plt.xlabel("Importance")
# plt.title("Feature Importance from CatBoost")
# plt.tight_layout()
# plt.show()