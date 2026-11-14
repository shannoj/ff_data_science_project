"""Load NFL data from nflverse via nflreadpy."""
from typing import NamedTuple

import nflreadpy as nfl
import pandas as pd

FIRST_SEASON = 2020

# Play-by-play has ~370 columns; keep only what the features use.
PBP_COLUMNS = [
    "game_id", "season", "week", "posteam", "defteam", "play_type", "pass", "rush",
    "qb_dropback", "qb_scramble", "qb_hit", "sack", "down", "wp", "half_seconds_remaining",
    "pass_oe", "passer_id",
]


class NFLData(NamedTuple):
    player_stats: pd.DataFrame
    schedule: pd.DataFrame
    pbp: pd.DataFrame
    injuries: pd.DataFrame


def current_season():
    return nfl.get_current_season()


def seasons_to_load(through=None):
    return list(range(FIRST_SEASON, (through or current_season()) + 1))


def load_player_stats(seasons):
    df = nfl.load_player_stats(seasons).to_pandas()
    return df[df["season_type"].isin(["REG", "POST"])].reset_index(drop=True)


def load_schedules(seasons):
    return nfl.load_schedules(seasons).to_pandas()


def load_pbp(seasons):
    # One season at a time so the full-width table never sits in memory all at once.
    return pd.concat([nfl.load_pbp([s]).select(PBP_COLUMNS).to_pandas() for s in seasons], ignore_index=True)


def load_injuries(seasons):
    available = [s for s in seasons if s >= 2009]
    return nfl.load_injuries(available).to_pandas()


def load_all(through=None):
    """Everything the features need, for every season from FIRST_SEASON to now."""
    seasons = seasons_to_load(through)
    return NFLData(load_player_stats(seasons), load_schedules(seasons), load_pbp(seasons), load_injuries(seasons))


def next_week(schedule, season=None):
    """(season, week) of the earliest week that still has unplayed games, or None."""
    season = season or int(schedule["season"].max())
    remaining = schedule[(schedule["season"] == season) & schedule["home_score"].isna()]
    if remaining.empty:
        return None
    return season, int(remaining["week"].min())
