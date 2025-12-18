import pandas as pd
import nflreadpy as nfl

# Load your player stats
player_stats = pd.read_csv('player_stats_2024.csv')

player_stats = player_stats[player_stats['week'].between(1, 18)]

# Get 2024 schedule data
schedule_2024_polars = nfl.load_schedules([2024])

print(schedule_2024_polars.columns)

schedule_2024 = schedule_2024_polars.to_pandas()

schedule_2024 = schedule_2024[schedule_2024['week'].between(1, 18)]

# Create a mapping of team + week to home/away
# Each game has home_team and away_team
home_games = schedule_2024[['week', 'home_team', 'roof', 'surface', 'temp', 'wind']].copy()
home_games['is_home'] = 1
home_games.rename(columns={'home_team': 'recent_team'}, inplace=True)

away_games = schedule_2024[['week', 'away_team', 'roof', 'surface', 'temp', 'wind']].copy()
away_games['is_home'] = 0
away_games.rename(columns={'away_team': 'recent_team'}, inplace=True)

# Combine home and away
home_away_mapping = pd.concat([home_games, away_games], ignore_index=True)

# Merge with your player stats
# Assuming your CSV has 'week' and 'recent_team' columns
player_stats_enhanced = player_stats.merge(
    home_away_mapping,
    on=['week', 'recent_team'],
    how='left'
)

# Verify the merge
print(f"Original rows: {len(player_stats)}")
print(f"Enhanced rows: {len(player_stats_enhanced)}")
print(f"Missing home/away data: {player_stats_enhanced['is_home'].isna().sum()}")

# Check the new column
print("\nHome/Away distribution:")
print(player_stats_enhanced['is_home'].value_counts())

# Save the enhanced dataframe
player_stats_enhanced.to_csv('player_stats_2024_with_home_away.csv', index=False)