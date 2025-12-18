import requests
import mysql.connector
from datetime import datetime
import time

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',  # or your MySQL username
    'password': 'Shannon12!',  # Replace with your actual password
    'database': 'nfl_stadiums'
}

def get_weather_forecast(lat, lon):
    """
    Fetch weather forecast using Open-Meteo API (free, no API key needed)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat,
        'longitude': lon,
        'current': 'temperature_2m,wind_speed_10m,weather_code',
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'timezone': 'America/New_York'
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
        
        current = data['current']
        temperature = int(current['temperature_2m'])
        wind_speed = f"{current['wind_speed_10m']} mph"
        weather_code = current['weather_code']
        forecast = weather_codes.get(weather_code, 'Unknown')
        
        return temperature, wind_speed, forecast
    
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None, None, None

def update_stadium_forecasts():
    """
    Fetch weather for all stadiums and update database
    """
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
                print("✗ Failed")
            
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
    """
    Display the latest forecasts from database
    """
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

if __name__ == "__main__":
    # Update forecasts
    update_stadium_forecasts()
    
    # View results
    view_latest_forecasts()