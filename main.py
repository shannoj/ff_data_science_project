import pandas as pd
from QB_cleaning import QB_cleaning

PlayerStats = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2021.csv")

QbStats2021 = QB_cleaning(PlayerStats)

print(QbStats2021.head())
print(QbStats2021.columns)