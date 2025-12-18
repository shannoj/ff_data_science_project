import pandas as pd
import nflreadpy as nfl

player_stats = pd.read_csv('player_stats_2024.csv')

player_stats = player_stats[player_stats['week'].between(1, 18)]

team_stats_polars = nfl.load_team_stats([2024])

team_stats_2024 = team_stats_polars.to_pandas()

team_stats_2024 = team_stats_2024[team_stats_2024['week'].between(1, 18)]

def_stats_2024 = team_stats_2024[['week', 'team', 'def_tackles_solo', 'def_tackles_with_assist', 'def_tackle_assists', 'def_tackles_for_loss', 'def_tackles_for_loss_yards', 'def_fumbles_forced', 'def_sacks', 'def_sack_yards', 'def_qb_hits', 'def_interceptions', 'def_interception_yards', 'def_pass_defended', 'def_tds', 'def_fumbles', 'def_safeties']].copy()

def_stats_2024 = def_stats_2024.rename(columns={'team': 'opponent'})

schedule_2024_polars = nfl.load_schedules([2024])

schedule_2024 = schedule_2024_polars.to_pandas()

schedule_2024 = schedule_2024[schedule_2024['week'].between(1, 18)]

home_games = schedule_2024[['week', 'home_team', 'roof', 'surface', 'temp', 'wind']].copy()
home_games['is_home'] = 1
home_games.rename(columns={'home_team': 'recent_team'}, inplace=True)

away_games = schedule_2024[['week', 'away_team', 'roof', 'surface', 'temp', 'wind']].copy()
away_games['is_home'] = 0
away_games.rename(columns={'away_team': 'recent_team'}, inplace=True)

home_away_mapping = pd.concat([home_games, away_games], ignore_index=True)

player_stats_enhanced = player_stats.merge(
    home_away_mapping,
    on=['week', 'recent_team'],
    how='left'
)

player_stats_enhanced = player_stats_enhanced.rename(columns={'opponent_team': 'opponent'})

player_stats_final = player_stats_enhanced.merge(
    def_stats_2024,
    on=['week', 'opponent'],
    how='left'
)

print(player_stats_final.shape)

# player_stats_final.to_csv('enhanced_stats.csv', index=False)