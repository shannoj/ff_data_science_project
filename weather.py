import requests
import sqlite3

sql_script_path = "/Users/jamesshannon/repos/ff_data_science_project/stadium_locations.sql"
db_path = "/Users/jamesshannon/repos/ff_data_science_project/stadium_locations.db"

# Create database from the SQL script file
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Read and clean the SQL script to make it SQLite compatible
with open(sql_script_path, 'r') as f:
    sql_content = f.read()

# Clean up common MySQL/PostgreSQL syntax that doesn't work in SQLite
sql_lines = []
for line in sql_content.split('\n'):
    line = line.strip()
    
    # Skip these types of statements that SQLite doesn't need/support
    if (line.upper().startswith('CREATE DATABASE') or 
        line.upper().startswith('USE ') or
        line.upper().startswith('SET ') or
        line.upper().startswith('DROP DATABASE') or
        line == '' or 
        line.startswith('--')):
        continue
    
    # Convert AUTO_INCREMENT to AUTOINCREMENT for SQLite
    line = line.replace('AUTO_INCREMENT', 'AUTOINCREMENT')
    
    sql_lines.append(line)

cleaned_sql = '\n'.join(sql_lines)

try:
    # Execute the cleaned SQL script
    conn.executescript(cleaned_sql)
    print("Successfully loaded stadium data from SQL script")
except sqlite3.Error as e:
    print(f"Error executing SQL script: {e}")
    print("First few lines of the cleaned SQL:")
    print('\n'.join(cleaned_sql.split('\n')[:10]))
    exit(1)

# The stadium_forecasts table is already created from your SQL file
# No need to create it again

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

        # Fixed: Use ? placeholders for SQLite instead of %s (which is for MySQL/PostgreSQL)
        insert_query = """
        INSERT INTO stadium_forecasts (team, temperature, wind_speed, forecast)
        VALUES (?, ?, ?, ?)
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
    except Exception as e:
        print(f"Error processing data for {team}: {e}")

# Close the database connection
conn.close()