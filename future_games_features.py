import pandas as pd
import nflreadpy as nfl

def get_current_week():
    
    current_week = nfl.get_current_week()
    predict_week = current_week + 1

    # Load schedule for future week
    schedule_polars = nfl.load_schedules([2025])
    schedule = schedule_polars.to_pandas()

    # Get week 16 games
    week_games = schedule[schedule['week'] == predict_week]

    return week_games

player_stats = nfl.load_player_stats([2025]).to_pandas()

player_stats = player_stats[player_stats['week'].between(1, 15)]

print(player_stats.columns)

team_stats_polars = nfl.load_team_stats([2025])

team_stats_2025 = team_stats_polars.to_pandas()

team_stats_2025 = team_stats_2025[team_stats_2025['week'].between(1, 15)]

def_stats_2025 = team_stats_2025[['week', 'team', 'def_tackles_solo', 'def_tackles_with_assist', 'def_tackle_assists', 'def_tackles_for_loss', 'def_tackles_for_loss_yards', 'def_fumbles_forced', 'def_sacks', 'def_sack_yards', 'def_qb_hits', 'def_interceptions', 'def_interception_yards', 'def_pass_defended', 'def_tds', 'def_fumbles', 'def_safeties']].copy()

def_stats_2025 = def_stats_2025.rename(columns={'team': 'opponent'})

schedule_2025_polars = nfl.load_schedules([2025])

schedule_2025 = schedule_2025_polars.to_pandas()

schedule_2025 = schedule_2025[schedule_2025['week'].between(1, 15)]

home_games = schedule_2025[['week', 'home_team', 'roof', 'surface', 'temp', 'wind']].copy()
home_games['is_home'] = 1
home_games.rename(columns={'home_team': 'team'}, inplace=True)

away_games = schedule_2025[['week', 'away_team', 'roof', 'surface', 'temp', 'wind']].copy()
away_games['is_home'] = 0
away_games.rename(columns={'away_team': 'team'}, inplace=True)

home_away_mapping = pd.concat([home_games, away_games], ignore_index=True)

player_stats_enhanced = player_stats.merge(
    home_away_mapping,
    on=['week', 'team'],
    how='left'
)

player_stats_enhanced = player_stats_enhanced.rename(columns={'opponent_team': 'opponent'})

player_stats_final = player_stats_enhanced.merge(
    def_stats_2025,
    on=['week', 'opponent'],
    how='left'
)

#player_stats_final.to_csv('enhanced_stats_2025.csv', index=False)


