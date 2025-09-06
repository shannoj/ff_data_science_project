import requests

import mysql.connector

conn = mysql.connector.connect(
    host="localhost",       # or your db server
    user="root",            # your mysql username
    password="", # your mysql password
    database="nfl_stadiums"
)

cursor = conn.cursor()

cursor.execute("SELECT team, latitude, longitude FROM stadium_locations")
stadiums = cursor.fetchall()

headers = {'User-Agent': '(MyWeatherApp, shannman6@gmail.com)'}

def get_weather_data(lat, lon, headers):
    # Step 1: Get the grid point data from the /points endpoint.
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    response = requests.get(points_url, headers=headers)
    response.raise_for_status()
    grid_data = response.json()
    
    # Step 2: Extract the forecast URL from the grid point data.
    forecast_url = grid_data['properties']['forecast']
    
    # Step 3: Get the actual forecast data from the forecast URL.
    forecast_response = requests.get(forecast_url, headers=headers)
    forecast_response.raise_for_status()
    forecast_data = forecast_response.json()

    # The detailed data is in a "periods" list within the response's 'properties'.
    # This example fetches data for the first forecast period (usually the current one).
    first_period = forecast_data['properties']['periods'][0]
    
    # Return a dictionary with the desired data points.
    return {
        "temperature": first_period['temperature'],
        "short_forecast": first_period['shortForecast'],
        "wind_speed": first_period['windSpeed'],
    }

for team, lat, lon in stadiums:
    try:
        weather_data = get_weather_data(lat, lon, headers)
        print(f"=== {team} Stadium Forecast ===")
        print(f"Temperature: {weather_data['temperature']}°F")
        print(f"Wind Speed: {weather_data['wind_speed']}")
        print(f"Forecast: {weather_data['short_forecast']}\n")

        insert_query = """
        INSERT INTO stadium_forecasts (team, temperature, wind_speed, forecast)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
        team,
        weather_data['temperature'],
        weather_data['wind_speed'],
        weather_data['short_forecast']
        ))
        conn.commit()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather for {team}: {e}")



