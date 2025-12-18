import pandas as pd

def create_qb_prediction_features(player_name, team, opponent, is_home, 
                                   week, historical_data, num_recent_games=5):
    """
    Create prediction features that match the training data structure
    """
    # Get player's recent games (before prediction week)
    player_history = historical_data[
        (historical_data['player_name'] == player_name) &
        (historical_data['week'] < week)
    ].sort_values('week', ascending=False).head(num_recent_games)
    
    if player_history.empty:
        print(f"No history for {player_name}")
        return None
    
    # START WITH THE MOST RECENT GAME AS A TEMPLATE
    # This ensures we have ALL columns the model expects
    template = player_history.iloc[0:1].copy()
    
    # Update game-specific information
    template['week'] = week
    template['team'] = team
    template['opponent'] = opponent
    template['is_home'] = is_home
    
    stat_cols = [
        'completions', 'attempts', 'passing_yards', 'passing_tds',
        'passing_interceptions', 'sacks_suffered', 'sack_yards_lost',
        'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
        'passing_yards_after_catch', 'passing_first_downs', 'passing_epa',
        'passing_cpoe', 'passing_2pt_conversions', 'pacr',
        'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
        'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa',
        'rushing_2pt_conversions'
    ]
    
    for col in stat_cols:
        if col in player_history.columns:
            template[col] = player_history[col].mean()
    
    opponent_games = historical_data[
        (historical_data['opponent'] == opponent) &
        (historical_data['week'] < week)
    ]
    
    if not opponent_games.empty:
        def_cols = [col for col in opponent_games.columns if 'opponent_def_' in col]
        for col in def_cols:
            template[col] = opponent_games[col].mean()
    
    if is_home == 0:
        opponent_home_games = historical_data[
            (historical_data['team'] == opponent) &
            (historical_data['is_home'] == 1) &
            (historical_data['week'] < week)
        ]
        
        if not opponent_home_games.empty:
            if 'temp' in opponent_home_games.columns:
                template['temp'] = opponent_home_games['temp'].mean()
            if 'wind' in opponent_home_games.columns:
                template['wind'] = opponent_home_games['wind'].mean()
            if 'roof' in opponent_home_games.columns:
                mode_roof = opponent_home_games['roof'].mode()
                if len(mode_roof) > 0:
                    template['roof'] = mode_roof[0]
            if 'surface' in opponent_home_games.columns:
                mode_surface = opponent_home_games['surface'].mode()
                if len(mode_surface) > 0:
                    template['surface'] = mode_surface[0]
    
    template = template.drop(columns=['passing_yards', 'passing_tds'], errors='ignore')
    
    return template