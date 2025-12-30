import requests
import mysql.connector
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Database configuration
load_dotenv()
db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD'),
        'database': 'nfl_stadiums'
}

def get_weather_forecast(lat, lon, forecast_date=None):
    
    url = "https://api.open-meteo.com/v1/forecast"
    if forecast_date is None:
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,wind_speed_10m,weather_code',
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'timezone': 'America/New_York'
        }

    else:
        date_str = forecast_date.strftime('%Y-%m-%d')
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'temperature_2m,wind_speed_10m,weather_code',
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'timezone': 'auto',
            'start_date': date_str,
            'end_date': date_str
        }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Weather code to description mapping
        weather_codes = {
            0: 'Clear sky',
            1: 'Mainly clear',
            2: 'Partly cloudy',
            3: 'Overcast',
            45: 'Foggy',
            48: 'Depositing rime fog',
            51: 'Light drizzle',
            53: 'Moderate drizzle',
            55: 'Dense drizzle',
            61: 'Slight rain',
            63: 'Moderate rain',
            65: 'Heavy rain',
            71: 'Slight snow',
            73: 'Moderate snow',
            75: 'Heavy snow',
            80: 'Slight rain showers',
            81: 'Moderate rain showers',
            82: 'Violent rain showers',
            95: 'Thunderstorm',
            96: 'Thunderstorm with hail'
        }
        if forecast_date is None:

            current = data['current']
            temperature = int(current['temperature_2m'])
            wind_speed = f"{current['wind_speed_10m']} mph"
            weather_code = current['weather_code']
            forecast = weather_codes.get(weather_code, 'Unknown')
        
        else:
            game_hour = forecast_date.hour if forecast_date.hour else 13  # 1 PM default
            
            hourly = data['hourly']
            times = hourly['time']
            
            target_time = f"{date_str}T{game_hour:02d}:00"
            
            if target_time in times:
                idx = times.index(target_time)
            else:
                idx = 0
            
            temperature = int(hourly['temperature_2m'][idx])
            wind_speed = int(hourly['wind_speed_10m'][idx])
            weather_code = hourly['weather_code'][idx]
            forecast = weather_codes.get(weather_code, 'Unknown')       
        
        return temperature, wind_speed, forecast
    
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None, None, None
    
def get_game_weather_forecast(team, is_home, opponent, game_datetime=None):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Determine which stadium to get weather for
        stadium_team = team if is_home == 1 else opponent
        
        # Get stadium location
        query = "SELECT latitude, longitude FROM stadium_locations WHERE team = %s"
        cursor.execute(query, (stadium_team,))
        result = cursor.fetchone()
        
        if not result:
            print(f"Stadium not found for {stadium_team}")
            return {'temp': None, 'wind': None}
        
        lat, lon = result
        
        # If no game datetime provided, use next Sunday at 1 PM
        if game_datetime is None:
            today = datetime.now()
            days_until_sunday = (6 - today.weekday()) % 7
            if days_until_sunday == 0:
                days_until_sunday = 7  # Next Sunday, not today
            game_datetime = today + timedelta(days=days_until_sunday)
            game_datetime = game_datetime.replace(hour=13, minute=0, second=0, microsecond=0)
        
        # Get weather forecast
        temp, wind, forecast_desc = get_weather_forecast(lat, lon, game_datetime)
        
        cursor.close()
        conn.close()
        
        return {'temp': temp, 'wind': wind}
    
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return {'temp': None, 'wind': None}
    

def update_stadium_forecasts():
    try:
        # Connect to database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Get all stadium locations
        cursor.execute("SELECT team, latitude, longitude FROM stadium_locations")
        stadiums = cursor.fetchall()
        
        print(f"Fetching weather for {len(stadiums)} stadiums...")
        
        for team, lat, lon in stadiums:
            print(f"Getting weather for {team}...", end=' ')
            
            temperature, wind_speed, forecast = get_weather_forecast(lat, lon)
            
            if temperature is not None:
                # Insert forecast into database
                insert_query = """
                    INSERT INTO stadium_forecasts (team, temperature, wind_speed, forecast)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(insert_query, (team, temperature, wind_speed, forecast))
                print(f"✓ {temperature}°F, {wind_speed}, {forecast}")
            else:
                print("Failed")
            
            # Be nice to the API - small delay between requests
            time.sleep(0.5)
        
        conn.commit()
        print(f"\n✓ Updated forecasts for all stadiums at {datetime.now()}")
        
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def view_latest_forecasts():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        query = """
            SELECT team, temperature, wind_speed, forecast, timestamp
            FROM stadium_forecasts
            ORDER BY timestamp DESC
            LIMIT 32
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        print("\n" + "="*80)
        print("LATEST NFL STADIUM WEATHER FORECASTS")
        print("="*80)
        
        for team, temp, wind, forecast, timestamp in results:
            print(f"{team:6} | {temp:3}°F | {wind:12} | {forecast:20} | {timestamp}")
        
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def update_stadium_forecasts_for_week(week_number, year=2025):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # You'll need a table with game schedules
        # For now, let's just update current weather for all stadiums
        cursor.execute("SELECT team, latitude, longitude FROM stadium_locations")
        stadiums = cursor.fetchall()
        
        print(f"Fetching weather forecasts for Week {week_number}...")
        
        # This is simplified - you'd want to get actual game dates/times from schedule
        # For now, assume games are on Sunday at 1 PM
        today = datetime.now()
        days_until_sunday = (6 - today.weekday()) % 7
        game_date = today + timedelta(days=days_until_sunday)
        game_datetime = game_date.replace(hour=13, minute=0, second=0)
        
        for team, lat, lon in stadiums:
            print(f"Getting forecast for {team}...", end=' ')
            
            temperature, wind_speed, forecast = get_weather_forecast(lat, lon, game_datetime)
            
            if temperature is not None:
                # Update or insert forecast
                insert_query = """
                    INSERT INTO stadium_forecasts 
                    (team, temperature, wind_speed, forecast, game_date, week)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    temperature = VALUES(temperature),
                    wind_speed = VALUES(wind_speed),
                    forecast = VALUES(forecast),
                    timestamp = CURRENT_TIMESTAMP
                """
                cursor.execute(insert_query, (
                    team, temperature, wind_speed, forecast, 
                    game_date.date(), week_number
                ))
                print(f"✓ {temperature}°F, {wind_speed} mph, {forecast}")
            else:
                print("Failed")
            
            time.sleep(0.5)
        
        conn.commit()
        print(f"\n✓ Updated forecasts for Week {week_number}")
        
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    # Update forecasts
    update_stadium_forecasts()
    
    # View results
    view_latest_forecasts()