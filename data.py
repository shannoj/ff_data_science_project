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


SCHEDULES_CSV = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


def load_schedules(seasons):
    """Schedules via nflreadpy (parquet), falling back to the CSV copy in the same release.

    nflverse re-uploads these files during updates, and the parquet has gone
    missing for hours at a time while the CSV stayed up.
    """
    try:
        return nfl.load_schedules(seasons).to_pandas()
    except ConnectionError as e:
        print(f"Schedules parquet unavailable ({e}); using the CSV copy.")
        df = pd.read_csv(SCHEDULES_CSV, encoding_errors="replace", low_memory=False)
        return df[df["season"].isin(seasons)].reset_index(drop=True)


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
