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

PlayerStats2021 = PlayerStats2021.drop_duplicates()
PlayerStats2022 = PlayerStats2022.drop_duplicates()
PlayerStats2023 = PlayerStats2023.drop_duplicates()
PlayerStats2024 = PlayerStats2024.drop_duplicates()

QbStats2021 = QB_cleaning(PlayerStats2021)
QbStats2022 = TDPoints(RushingPoints(PassingPoints(QB_cleaning(PlayerStats2022, is_2022=True))))
QbStats2023 = TDPoints(RushingPoints(PassingPoints(QB_cleaning(PlayerStats2023))))
QbStats2024 = TDPoints(RushingPoints(PassingPoints(QB_cleaning(PlayerStats2024, is_2022=True))))

QbStats2021 = add_year_suffix(QbStats2021, 2021)
QbStats2022 = add_year_suffix(QbStats2022, 2022)
QbStats2023 = add_year_suffix(QbStats2023, 2023)
QbStats2024 = add_year_suffix(QbStats2024, 2024)

QbStats2021 = QbStats2021.drop_duplicates(subset='player_id')
QbStats2022 = QbStats2022.drop_duplicates(subset='player_id')
QbStats2023 = QbStats2023.drop_duplicates(subset='player_id')
QbStats2024 = QbStats2024.drop_duplicates(subset='player_id')

QbStatsTotal = pd.merge(QbStats2021, QbStats2022, on='player_id', how='inner')
QbStatsTotal = pd.merge(QbStatsTotal, QbStats2023, on='player_id', how='inner')
QbStatsTotal = pd.merge(QbStatsTotal, QbStats2024, on='player_id', how='inner')

QbStatsML = QbStatsTotal.dropna(subset=['fantasy_points_ppr_2024'])

QbStatsML2 = QbStatsML[~(QbStatsML == 0).all(axis=1)]

print(QbStatsTotal.head())

y = QbStatsML2['fantasy_points_ppr_2024']

X = QbStatsML2[[col for col in QbStatsML.columns if 
               any(year in col for year in ['2021', '2022', '2023', '2024']) and
               col != 'player_id']]

X = X.drop('fantasy_points_ppr_2024', axis=1)

X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.20, random_state=1)

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
