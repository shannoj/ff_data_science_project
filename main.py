import pandas as pd

PlayerStats = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2021.csv")

print(PlayerStats.head())
print(PlayerStats.columns)

print(PlayerStats.position.head())

position = 'position'
Quarterbacks = 'QB'

mask = PlayerStats[position] == Quarterbacks

PlayerStats_QB = PlayerStats[mask]

drop_empty_cols = [col for col in PlayerStats_QB.columns if (PlayerStats_QB[col] == 0).all()]

PlayerStats_QB_cleaned = PlayerStats_QB.drop(columns=drop_empty_cols)

Qb_Stats = PlayerStats_QB_cleaned.drop(columns=['receptions', 'targets','receiving_yards','receiving_air_yards','receiving_yards_after_catch','receiving_first_downs','receiving_epa','receiving_2pt_conversions', 'target_share','air_yards_share'])

print(Qb_Stats.columns)


