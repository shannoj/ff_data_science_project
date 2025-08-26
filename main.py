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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

PlayerStats2021 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2021.csv")
PlayerStats2022 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2022.csv")
PlayerStats2023 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2023.csv")
PlayerStats2024 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2024.csv")


QbStats2021 = (QB_cleaning(PlayerStats2021))
QbStats2022 = (QB_cleaning(PlayerStats2022, is_2022=True))
QbStats2023 = (QB_cleaning(PlayerStats2023))
QbStats2024 = (QB_cleaning(PlayerStats2024, is_2022=True))


QbStats2021 = add_year_suffix(QbStats2021, 2021)
QbStats2022 = add_year_suffix(QbStats2022, 2022)
QbStats2023 = add_year_suffix(QbStats2023, 2023)
QbStats2024 = add_year_suffix(QbStats2024, 2024)

QbStats2021['player_week_year'] = (
    QbStats2021['player_id'].astype(str) + '_' +
    QbStats2021['week_2021'].astype(str) + '_' +
    '2021'
)

QbStats2022['player_week_year'] = (
    QbStats2022['player_id'].astype(str) + '_' +
    QbStats2022['week_2022'].astype(str) + '_' +
    '2022'
)

QbStats2023['player_week_year'] = (
    QbStats2023['player_id'].astype(str) + '_' +
    QbStats2023['week_2023'].astype(str) + '_' +
    '2023'
)

QbStats2024['player_week_year'] = (
    QbStats2024['player_id'].astype(str) + '_' +
    QbStats2024['week_2024'].astype(str) + '_' +
    '2024'
)

drop_cols = ['player_id', 'week'] 


print(QbStats2021.head())
print(QbStats2022.head())
print(QbStats2023.head())
print(QbStats2024.head())
print(QbStats2024.columns)

QbStats2021.drop(columns=drop_cols, inplace=True, errors='ignore')
QbStats2022.drop(columns=drop_cols, inplace=True, errors='ignore')
QbStats2023.drop(columns=drop_cols, inplace=True, errors='ignore')
QbStats2024.drop(columns=drop_cols, inplace=True, errors='ignore')

QbStatsTotal = pd.concat([QbStats2021, QbStats2022, QbStats2023, QbStats2024], axis=0)

QbStatsTotal = QbStatsTotal.fillna(0)

print(QbStatsTotal.head())
print(QbStatsTotal.shape)

y = QbStatsTotal['fantasy_points_ppr_2024']

X = QbStatsTotal

print(X.columns)

X = X.drop(['fantasy_points_ppr_2024','fantasy_points_ppr_2023','fantasy_points_ppr_2022','fantasy_points_ppr_2021','player_week_year'], axis=1)

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1
)

model = CatBoostRegressor(
    iterations=50,
    learning_rate=0.05,
    depth=4,
    loss_function='RMSE',
    verbose= False  # Print every 100 iterations
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")


y_pred_series = pd.Series(y_pred)
y_pred_sorted = y_pred_series.sort_values(ascending=False)
print(y_pred_sorted)