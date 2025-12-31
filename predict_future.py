from weather_csv import get_game_weather_forecast


def create_qb_prediction_features(player_name, team, opponent, is_home, 
                                   week, historical_data, num_recent_games=5, game_date=None):

    current_player = historical_data[
        (historical_data['player_name'] == player_name) &
        (historical_data['team'] == team)
    ]

    player_id = current_player['player_id'].iloc[0]


    player_history = historical_data[
        (historical_data['player_id'] == player_id) &
        (historical_data['week'] < week)
    ].sort_values('week', ascending=False).head(num_recent_games)
    
    if player_history.empty:
        print(f"No history for {player_name}")
        return None
    
    template = player_history.iloc[0:1].copy()
    
    template['week'] = week
    template['team'] = team
    template['opponent'] = opponent
    template['is_home'] = is_home
    
    stat_cols = [
        'passing_epa',
        'passing_cpoe', 'pacr', 'comp_percent', 'avg_time_to_throw','aggressiveness', 'passer_rating','expected_completion_percentage','completion_percentage_above_expectation',
    ]
    
    for col in stat_cols:
        if col in player_history.columns:
            template[col] = player_history[col].mean()
    
    opponent_games = historical_data[
        (historical_data['team'] == opponent) &
        (historical_data['week'] < week)]

    opponent_def_history = historical_data[
        (historical_data['opponent'] == opponent) &  # Games against this opponent
        (historical_data['week'] < week)
    ].sort_values('week')

    if not opponent_def_history.empty:

        # Get the MOST RECENT defense stats (last week's values)
        latest_def = opponent_def_history.iloc[-1]
        
        # Defense stats from the merge
        template['cumulative_avg'] = latest_def.get('cumulative_avg', 250)
        template['def_recent_avg_allowed'] = latest_def.get('def_recent_avg_allowed', 250)
        template['def_5game_avg_allowed'] = latest_def.get('def_5game_avg_allowed', 250)
        template['def_trend'] = latest_def.get('def_trend', 0)
        template['def_consistency'] = latest_def.get('def_consistency', 50)
    
    if not opponent_games.empty:
        def_cols = [col for col in opponent_games.columns if col.startswith('def_') and col.endswith('_y')]
        for col in def_cols:
            template[col] = opponent_games[col].mean()

    if game_date is None:

        from datetime import datetime, timedelta
        today = datetime.now()
        days_until_sunday = (6 - today.weekday()) % 7
        if days_until_sunday == 0:
            days_until_sunday = 7
        game_date = today + timedelta(days=days_until_sunday)
        game_date = game_date.replace(hour=13, minute=0, second=0)
    
    weather = get_game_weather_forecast(team, is_home, opponent, game_date)
    
    template['temp'] = weather['temp'] if weather['temp'] is not None else 70  
    template['wind'] = weather['wind'] if weather['wind'] is not None else 5 

    
    if is_home == 0:
        opponent_home_games = historical_data[
            (historical_data['team'] == opponent) &
            (historical_data['is_home'] == 1) &
            (historical_data['week'] < week)
        ]
        
        if not opponent_home_games.empty:
            if 'roof' in opponent_home_games.columns:
                mode_roof = opponent_home_games['roof'].mode()

                if len(mode_roof) > 0:
                    template['roof'] = mode_roof[0]
            if 'surface' in opponent_home_games.columns:
                mode_surface = opponent_home_games['surface'].mode()

                if len(mode_surface) > 0:
                    template['surface'] = mode_surface[0]
    
    template = template.drop(columns=['passing_yards'], errors='ignore')
    
    return template