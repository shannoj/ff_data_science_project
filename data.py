"""Load NFL data from nflverse via nflreadpy."""
import nflreadpy as nfl
import pandas as pd

FIRST_SEASON = 2020


def current_season():
    return nfl.get_current_season()


def seasons_to_load(through=None):
    return list(range(FIRST_SEASON, (through or current_season()) + 1))


def load_player_stats(seasons):
    df = nfl.load_player_stats(seasons).to_pandas()
    return df[df["season_type"].isin(["REG", "POST"])].reset_index(drop=True)


def load_schedules(seasons):
    return nfl.load_schedules(seasons).to_pandas()


def load_all(through=None):
    """Player stats and schedules for every season from FIRST_SEASON to now."""
    seasons = seasons_to_load(through)
    return load_player_stats(seasons), load_schedules(seasons)


def next_week(schedule, season=None):
    """(season, week) of the earliest week that still has unplayed games, or None."""
    season = season or int(schedule["season"].max())
    remaining = schedule[(schedule["season"] == season) & schedule["home_score"].isna()]
    if remaining.empty:
        return None
    return season, int(remaining["week"].min())
