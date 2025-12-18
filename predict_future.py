import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os


def get_game_weather_forecast(team, is_home, opponent):
    """
    Get weather forecast from your NFL stadiums database
    """
    load_dotenv()
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD'),
        'database': 'nfl_stadiums'
    }
    
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Get forecast for the stadium where game is being played
        stadium_team = team if is_home == 1 else opponent
        
        query = """
            SELECT team, temperature, wind_speed, forecast, timestamp
            FROM stadium_forecasts
            WHERE team = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        
        cursor.execute(query, (stadium_team,))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            # Parse wind speed (remove ' mph' if present)
            wind = result['wind_speed'].replace(' mph', '') if result['wind_speed'] else '0'
            
            return {
                'temp': result['temperature'],
                'wind': float(wind),
                'forecast': result['forecast']
            }
        else:
            print(f"No weather data found for {stadium_team}")
            return None
            
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None



def create_qb_prediction_features(player_name, team, opponent, is_home, 
                                   week, historical_data, num_recent_games=5):
    # Get player's recent games (before prediction week)
    player_history = historical_data[
        (historical_data['player_name'] == player_name) &
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
        def_cols = [col for col in opponent_games.columns if 'opponent_def' in col]
        for col in def_cols:
            template[col] = opponent_games[col].mean()

    weather = get_game_weather_forecast(team, is_home, opponent)

    template['temp'] = weather['temp']
    template['wind'] = weather['wind']

    
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
    
    template = template.drop(columns=['passing_yards', 'passing_tds'], errors='ignore')
    
    return template