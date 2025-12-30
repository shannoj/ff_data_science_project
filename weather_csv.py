# weather_csv.py

import requests
import pandas as pd
from datetime import datetime, timedelta

def load_stadium_data():
    """Load stadium locations from CSV"""
    try:
        stadiums = pd.read_csv('stadium_locations.csv')
        return stadiums
    except FileNotFoundError:
        print("Warning: stadium_locations.csv not found")
        return pd.DataFrame()

def get_weather_forecast(lat, lon, forecast_date=None):
    """
    Get weather forecast for a specific date
    
    Args:
        lat: Stadium latitude
        lon: Stadium longitude
        forecast_date: datetime object for game date (default: current)
    
    Returns:
        tuple: (temperature, wind_speed, forecast_description)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    # Current weather (default)
    if forecast_date is None:
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,wind_speed_10m,weather_code',
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph'
        }
    else:
        # Future forecast
        if not isinstance(forecast_date, datetime):
            forecast_date = pd.to_datetime(forecast_date)
        
        # Check if date is too far in future (>7 days)
        today = datetime.now().date()
        if forecast_date.date() > today + timedelta(days=7):
            print(f"Warning: {forecast_date.date()} is >7 days out, using daily forecast")
            # Use daily forecast
            params = {
                'latitude': lat,
                'longitude': lon,
                'daily': 'temperature_2m_max,temperature_2m_min,wind_speed_10m_max,weather_code',
                'temperature_unit': 'fahrenheit',
                'wind_speed_unit': 'mph',
                'forecast_days': 16
            }
        else:
            # Use hourly forecast (more accurate)
            params = {
                'latitude': lat,
                'longitude': lon,
                'daily': 'temperature_2m_max,temperature_2m_min,wind_speed_10m_max,weather_code',
                'temperature_unit': 'fahrenheit',
                'wind_speed_unit': 'mph',
                'forecast_days': 7
            }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Weather codes
        weather_codes = {
            0: 'Clear', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
            45: 'Foggy', 61: 'Rain', 71: 'Snow', 95: 'Thunderstorm'
        }
        
        if 'current' in data:
            # Current weather
            current = data['current']
            temperature = int(current['temperature_2m'])
            wind_speed = int(current['wind_speed_10m'])
            weather_code = current.get('weather_code', 0)
        else:
            # Daily forecast
            daily = data['daily']
            
            # Find matching date or use first day
            if forecast_date:
                date_str = forecast_date.strftime('%Y-%m-%d')
                if date_str in daily['time']:
                    idx = daily['time'].index(date_str)
                else:
                    idx = 0  # Use first available day
            else:
                idx = 0
            
            temp_max = int(daily['temperature_2m_max'][idx])
            temp_min = int(daily['temperature_2m_min'][idx])
            temperature = (temp_max + temp_min) // 2
            wind_speed = int(daily['wind_speed_10m_max'][idx])
            weather_code = daily['weather_code'][idx]
        
        forecast = weather_codes.get(weather_code, 'Unknown')
        return temperature, wind_speed, forecast
        
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None, None, None


def get_game_weather_forecast(team, is_home, opponent, game_datetime=None):
    """
    Get weather forecast for a specific game
    
    Args:
        team: Team abbreviation (e.g., 'SEA')
        is_home: 1 if home game, 0 if away
        opponent: Opponent team abbreviation
        game_datetime: datetime object for game (default: next Sunday at 1 PM)
    
    Returns:
        dict: {'temp': int, 'wind': int, 'conditions': str}
    """
    # Load stadium data
    stadiums = load_stadium_data()
    
    if stadiums.empty:
        print("Warning: No stadium data available, using defaults")
        return {'temp': 70, 'wind': 5, 'conditions': 'Unknown'}
    
    # Determine which stadium
    stadium_team = team if is_home == 1 else opponent
    
    # Get stadium location
    stadium = stadiums[stadiums['team'] == stadium_team]
    
    if stadium.empty:
        print(f"Warning: Stadium not found for {stadium_team}, using defaults")
        return {'temp': 70, 'wind': 5, 'conditions': 'Unknown'}
    
    lat = stadium.iloc[0]['latitude']
    lon = stadium.iloc[0]['longitude']
    
    # Get weather
    temp, wind, conditions = get_weather_forecast(lat, lon, game_datetime)
    
    if temp is None:
        # Fallback to defaults
        return {'temp': 70, 'wind': 5, 'conditions': 'Unknown'}
    
    return {'temp': temp, 'wind': wind, 'conditions': conditions}


# For backward compatibility
def update_stadium_forecasts():
    """
    Deprecated - used to update database
    Now just fetches current weather for all stadiums
    """
    stadiums = load_stadium_data()
    
    if stadiums.empty:
        print("No stadium data available")
        return
    
    print(f"Fetching current weather for {len(stadiums)} stadiums...")
    
    results = []
    for _, row in stadiums.iterrows():
        team = row['team']
        lat = row['latitude']
        lon = row['longitude']
        
        temp, wind, forecast = get_weather_forecast(lat, lon)
        
        if temp is not None:
            results.append({
                'team': team,
                'temperature': temp,
                'wind_speed': wind,
                'forecast': forecast,
                'timestamp': datetime.now()
            })
            print(f"{team}: {temp}°F, {wind} mph, {forecast}")
    
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv('current_weather.csv', index=False)
        print(f"\n✓ Saved current weather to current_weather.csv")


if __name__ == "__main__":
    # Test
    update_stadium_forecasts()