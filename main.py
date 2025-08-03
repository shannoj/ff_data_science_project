import pandas as pd

PlayerStats = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/player_stats_2021.csv")

print(PlayerStats.head())
print(PlayerStats.columns)

print(PlayerStats.position.head())

position = 'position'
Quarterbacks = 'QB'

mask = PlayerStats[position] == Quarterbacks

PlayerStats_QB = PlayerStats[mask]

print("\nFiltered DataFrame (keeping 'QB'):")
print(PlayerStats_QB)
