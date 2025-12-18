import pandas as pd
from Cleaning import position_cleaning, handle_categoricals, add_year_suffix
from model import predict_category

PlayerStats2024 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2024_with_home_away.csv")

QbStats2024 = (position_cleaning(PlayerStats2024, 'QB'))

QbStats2024 = add_year_suffix(QbStats2024, 2024)

QbStats2024_encoded = handle_categoricals(QbStats2024)

r2_passing_yard, rmse_passing_yards, passing_yards_model = predict_category('passing_yards_2024', QbStats2024_encoded)

r2_pass_td, rmse_pass_td, passing_td_model = predict_category('passing_tds_2024', QbStats2024_encoded)

print(r2_pass_td)

PlayerStats2024 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2024.csv")

QbStats2024 = (position_cleaning(PlayerStats2024, 'QB'))

QbStats2024 = add_year_suffix(QbStats2024, 2024)

QbStats2024_encoded = handle_categoricals(QbStats2024)

r2_passing_yard, rmse_passing_yards, passing_yards_model = predict_category('passing_yards_2024', QbStats2024_encoded)

r2_pass_td, rmse_pass_td, passing_td_model = predict_category('passing_tds_2024', QbStats2024_encoded)

print(r2_pass_td)