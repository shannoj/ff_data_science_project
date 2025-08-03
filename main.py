import pandas as pd
from QB_cleaning import QB_cleaning

PlayerStats2021 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2021.csv")
PlayerStats2022 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2022.csv")
PlayerStats2023 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2023.csv")
PlayerStats2024 = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2024.csv")

QbStats2021 = QB_cleaning(PlayerStats2021)
QbStats2022 = QB_cleaning(PlayerStats2022, is_2022=True)
QbStats2023 = QB_cleaning(PlayerStats2023)
QbStats2024 = QB_cleaning(PlayerStats2024, is_2022=True)

QbStatsTotal = pd.merge(QbStats2021, QbStats2022, QbStats2023, QbStats2024, on='player_id', how='inner')

print(QbStatsTotal.head())