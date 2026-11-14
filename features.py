"""Pre-game features for QB passing-yard projections.

Every feature for a game is built only from information available before
kickoff (earlier games, betting lines, venue, weather). Rows the model trains
on are therefore built the same way as the rows it predicts.
"""
import numpy as np
import pandas as pd

TARGET = "passing_yards"

# A QB game counts toward "form" (and as a training row) only with this many
# attempts, so mop-up cameos don't drag averages around.
MIN_ATTEMPTS = 10

QB_FEATURES = [
    "qb_ypg_3", "qb_ypg_8", "qb_ypg_ewm", "qb_ypg_season",
    "qb_att_8", "qb_ypa_8", "qb_air_ypa_8", "qb_epa_db_8", "qb_cpoe_8",
    "qb_sack_rate_8", "qb_int_rate_8", "qb_games_prior",
]
DEF_FEATURES = [
    "def_ypg_5", "def_ypg_ewm", "def_ypa_10", "def_epa_db_10",
    "def_sack_rate_10", "def_int_rate_10",
]
GAME_FEATURES = [
    "is_home", "spread", "total_line", "implied_total", "roof_type", "turf",
    "temp", "wind", "div_game", "rest", "week", "postseason",
]
FEATURES = QB_FEATURES + DEF_FEATURES + GAME_FEATURES

_PASS_COLS = ["passing_yards", "attempts", "completions", "passing_air_yards",
              "passing_epa", "sacks_suffered", "passing_interceptions"]


# ---------------------------------------------------------------- helpers

def _prior(df, key, cols, window, how):
    """Rolling sum/mean of `cols` over each key's previous `window` rows (current row excluded)."""
    prev = df.groupby(key)[cols].shift(1)
    prev[key] = df[key]
    rolled = prev.groupby(key)[cols].rolling(window, min_periods=1)
    rolled = rolled.sum() if how == "sum" else rolled.mean()
    return rolled.reset_index(level=0, drop=True).reindex(df.index)


def _prior_ewm(df, key, col, halflife=4):
    prev = df.groupby(key)[col].shift(1)
    return prev.groupby(df[key]).transform(lambda s: s.ewm(halflife=halflife, ignore_na=True).mean())


def _ratio(num, den):
    return num / den.replace(0, np.nan)


# ---------------------------------------------------------------- game context

def team_games(schedule):
    """Two rows per game (one per team) with everything known before kickoff."""
    s = schedule
    base = ["game_id", "season", "week", "game_type", "gameday", "gametime",
            "location", "roof", "surface", "temp", "wind", "div_game", "total_line"]

    def side(home):
        me, them = ("home", "away") if home else ("away", "home")
        d = s[base].copy()
        d["team"] = s[f"{me}_team"]
        d["opponent"] = s[f"{them}_team"]
        d["is_home"] = float(home)
        d["spread"] = s["spread_line"] if home else -s["spread_line"]  # + = this team favored
        d["rest"] = s[f"{me}_rest"]
        d["sched_qb_id"] = s[f"{me}_qb_id"]
        d["sched_qb_name"] = s[f"{me}_qb_name"]
        d["stadium_team"] = s["home_team"]
        d["completed"] = s["home_score"].notna()
        return d

    tg = pd.concat([side(True), side(False)], ignore_index=True)
    tg.loc[tg["location"] == "Neutral", "is_home"] = 0.0
    tg["implied_total"] = (tg["total_line"] + tg["spread"]) / 2
    tg["postseason"] = (tg["game_type"] != "REG").astype(float)

    # Retractable roofs are listed as open/closed after the game but blank
    # beforehand, so treat them as one category in training and prediction.
    tg["roof_type"] = tg["roof"].map({"outdoors": 0.0, "dome": 1.0}).fillna(0.5)
    surface = tg["surface"].fillna("").str.strip()
    tg["turf"] = np.where(surface == "", np.nan, (surface != "grass").astype(float))

    # Weather only matters outdoors. Domes get fixed values; retractables are unknown.
    tg.loc[tg["roof_type"] == 1.0, ["temp", "wind"]] = [70.0, 0.0]
    tg.loc[tg["roof_type"] == 0.5, ["temp", "wind"]] = np.nan
    return tg


# ---------------------------------------------------------------- QB + defense history

def qb_history(player_stats):
    q = player_stats[(player_stats["position"] == "QB")
                     & (player_stats["attempts"] >= MIN_ATTEMPTS)].copy()
    q["dropbacks"] = q["attempts"] + q["sacks_suffered"]
    return q[["player_id", "player_display_name", "team", "game_id", "season", "week",
              "passing_cpoe", "dropbacks"] + _PASS_COLS]


def defense_history(player_stats):
    """Passing allowed by each defense in each game (all passers it faced)."""
    d = (player_stats.groupby(["game_id", "season", "week", "opponent_team"], as_index=False)[_PASS_COLS]
         .sum().rename(columns={"opponent_team": "def_team"}))
    d["dropbacks"] = d["attempts"] + d["sacks_suffered"]
    return d


def last_starters(qb_hist):
    """Each team's QB with the most attempts in its most recent game."""
    latest = qb_hist.sort_values(["season", "week", "attempts"]).groupby("team").tail(1)
    return latest.set_index("team")[["player_id", "player_display_name"]]


def _add_qb_form(rows):
    rows = rows.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    k = "player_id"
    s8 = _prior(rows, k, ["passing_yards", "attempts", "passing_air_yards", "passing_epa",
                          "dropbacks", "sacks_suffered", "passing_interceptions"], 8, "sum")
    m8 = _prior(rows, k, ["passing_yards", "attempts", "passing_cpoe"], 8, "mean")
    rows["qb_ypg_3"] = _prior(rows, k, ["passing_yards"], 3, "mean")["passing_yards"]
    rows["qb_ypg_8"] = m8["passing_yards"]
    rows["qb_ypg_ewm"] = _prior_ewm(rows, k, "passing_yards")
    rows["qb_att_8"] = m8["attempts"]
    rows["qb_cpoe_8"] = m8["passing_cpoe"]
    rows["qb_ypa_8"] = _ratio(s8["passing_yards"], s8["attempts"])
    rows["qb_air_ypa_8"] = _ratio(s8["passing_air_yards"], s8["attempts"])
    rows["qb_epa_db_8"] = _ratio(s8["passing_epa"], s8["dropbacks"])
    rows["qb_sack_rate_8"] = _ratio(s8["sacks_suffered"], s8["dropbacks"])
    rows["qb_int_rate_8"] = _ratio(s8["passing_interceptions"], s8["attempts"])
    rows["qb_games_prior"] = rows.groupby(k).cumcount().clip(upper=50)

    season_prev = rows.groupby([k, "season"])["passing_yards"].shift(1)
    rows["qb_ypg_season"] = (season_prev.groupby([rows[k], rows["season"]])
                             .transform(lambda s: s.expanding().mean()))
    return rows


def _defense_form(def_rows):
    d = def_rows.sort_values(["def_team", "season", "week"]).reset_index(drop=True)
    k = "def_team"
    s10 = _prior(d, k, ["passing_yards", "attempts", "passing_epa", "dropbacks",
                        "sacks_suffered", "passing_interceptions"], 10, "sum")
    d["def_ypg_5"] = _prior(d, k, ["passing_yards"], 5, "mean")["passing_yards"]
    d["def_ypg_ewm"] = _prior_ewm(d, k, "passing_yards")
    d["def_ypa_10"] = _ratio(s10["passing_yards"], s10["attempts"])
    d["def_epa_db_10"] = _ratio(s10["passing_epa"], s10["dropbacks"])
    d["def_sack_rate_10"] = _ratio(s10["sacks_suffered"], s10["dropbacks"])
    d["def_int_rate_10"] = _ratio(s10["passing_interceptions"], s10["attempts"])
    return d[["game_id", "def_team"] + DEF_FEATURES]


# ---------------------------------------------------------------- main entry point

def build_dataset(player_stats, schedule, predict_week=None):
    """One row per QB-game with pre-game features.

    Completed games carry the actual `passing_yards`. If `predict_week`
    (season, week) is given, rows for that week's unplayed games are added
    with the projected starter for each team and no target.
    """
    tg = team_games(schedule)
    qb = qb_history(player_stats)
    ctx = tg.drop(columns=["season", "week"])
    rows = qb.merge(ctx, on=["game_id", "team"], how="inner")

    upcoming = pd.DataFrame()
    if predict_week is not None:
        season, week = predict_week
        upcoming = tg[(tg["season"] == season) & (tg["week"] == week) & ~tg["completed"]].copy()
        fallback = last_starters(qb)
        names = qb.drop_duplicates("player_id", keep="last").set_index("player_id")["player_display_name"]
        upcoming["player_id"] = upcoming["sched_qb_id"].fillna(upcoming["team"].map(fallback["player_id"]))
        upcoming["player_display_name"] = (upcoming["player_id"].map(names)
                                           .fillna(upcoming["sched_qb_name"]))
        upcoming = upcoming[upcoming["player_id"].notna()]
        rows = pd.concat([rows, upcoming], ignore_index=True)

    rows = _add_qb_form(rows)

    defense = defense_history(player_stats)
    if not upcoming.empty:
        up_def = upcoming[["game_id", "season", "week", "opponent"]].rename(columns={"opponent": "def_team"})
        defense = pd.concat([defense, up_def.drop_duplicates()], ignore_index=True)
    rows = rows.merge(_defense_form(defense), left_on=["game_id", "opponent"],
                      right_on=["game_id", "def_team"], how="left")
    return rows.drop(columns="def_team")
