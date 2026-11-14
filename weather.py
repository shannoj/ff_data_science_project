"""Kickoff weather forecasts from Open-Meteo for upcoming outdoor games."""
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

STADIUMS_CSV = Path(__file__).parent / "stadium_locations.csv"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS = 16  # Open-Meteo's forecast horizon


@lru_cache(maxsize=1)
def _stadiums():
    return pd.read_csv(STADIUMS_CSV).set_index("team")


@lru_cache(maxsize=256)
def kickoff_forecast(stadium_team, gameday, gametime):
    """(temp °F, wind mph) at kickoff, or (None, None) if unavailable.

    `gameday` is 'YYYY-MM-DD' and `gametime` is 'HH:MM' Eastern, as in the nflverse schedule.
    """
    if stadium_team not in _stadiums().index:
        return None, None
    days_out = (date.fromisoformat(gameday) - date.today()).days
    if not 0 <= days_out < FORECAST_DAYS:
        return None, None

    lat, lon = _stadiums().loc[stadium_team, ["latitude", "longitude"]]
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "America/New_York",
        "start_date": gameday, "end_date": gameday,
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=10)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        hour = datetime.strptime(gametime or "13:00", "%H:%M").hour
        idx = hourly["time"].index(f"{gameday}T{hour:02d}:00")
        return float(hourly["temperature_2m"][idx]), float(hourly["wind_speed_10m"][idx])
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Weather lookup failed for {stadium_team} on {gameday}: {e}")
        return None, None


def add_forecasts(rows):
    """Fill temp/wind for upcoming outdoor, non-neutral-site games."""
    rows = rows.copy()
    outdoor = (rows["roof_type"] == 0.0) & (rows["location"] != "Neutral")
    for i in rows.index[outdoor]:
        r = rows.loc[i]
        temp, wind = kickoff_forecast(r["stadium_team"], r["gameday"], r["gametime"])
        if temp is not None:
            rows.loc[i, ["temp", "wind"]] = [temp, wind]
    return rows
