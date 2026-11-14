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
BASE_FEATURES = QB_FEATURES + DEF_FEATURES + GAME_FEATURES

# Feature groups built from play-by-play and injury reports. Each was tested
# by adding it to the base set in a 2023-2026 backtest (see FEATURES below).
FEATURE_GROUPS = {
    "pass_tendency": ["team_neutral_pass_8", "team_proe_8", "team_plays_8"],
    "pressure": ["qb_pressure_rate_8", "def_pressure_rate_10"],
    "scramble": ["qb_scramble_rate_8"],
    "injuries": ["missing_target_share", "questionable_target_share", "ol_out", "qb_questionable"],
}
ALL_FEATURES = BASE_FEATURES + [f for g in FEATURE_GROUPS.values() for f in g]
# Backtest result (2023-2026, 2,057 games, research/feature_ablation.py): none of
# the groups changed MAE by more than 0.06 yds, even on injury-heavy subsets,
# so the model keeps the base set. Vegas lines already price in this information.
FEATURES = BASE_FEATURES

OUT_STATUSES = {"Out", "Doubtful"}
OL_POSITIONS = {"T", "G", "C", "OL", "OT", "OG"}
TARGET_SHARE_GAMES = 6  # recent team games used for each receiver's share of targets

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


# ---------------------------------------------------------------- play-by-play

def _pressured(pbp):
    return ((pbp["qb_hit"] == 1) | (pbp["sack"] == 1)).astype(float)


def pbp_qb_games(pbp):
    """Per QB-game dropbacks (incl. sacks and scrambles), scrambles and pressures."""
    db = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_id"].notna()].assign(pressured=_pressured)
    return (db.groupby(["passer_id", "game_id"], as_index=False)
            .agg(pbp_dropbacks=("qb_dropback", "size"), scrambles=("qb_scramble", "sum"),
                 pressures=("pressured", "sum"))
            .rename(columns={"passer_id": "player_id"}))


def pbp_defense_games(pbp):
    db = pbp[(pbp["qb_dropback"] == 1) & pbp["defteam"].notna()].assign(pressured=_pressured)
    return (db.groupby(["defteam", "game_id"], as_index=False)
            .agg(def_pbp_dropbacks=("qb_dropback", "size"), def_pressures=("pressured", "sum"))
            .rename(columns={"defteam": "def_team"}))


def pbp_team_games(pbp):
    """Per team-game offensive plays, plus pass rate and pass rate over expected in neutral situations.

    Neutral = 1st/2nd down, win probability 20-80%, outside the last two minutes
    of each half, so the rate reflects play-calling rather than game script.
    """
    plays = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()]
    total = plays.groupby(["posteam", "game_id"], as_index=False).agg(off_plays=("play_type", "size"))
    neutral = plays[plays["down"].isin([1, 2]) & plays["wp"].between(0.2, 0.8)
                    & (plays["half_seconds_remaining"] > 120)]
    neu = neutral.groupby(["posteam", "game_id"], as_index=False).agg(
        neutral_plays=("pass", "size"), neutral_passes=("pass", "sum"),
        proe_sum=("pass_oe", "sum"), proe_n=("pass_oe", "count"))
    return total.merge(neu, on=["posteam", "game_id"], how="left").rename(columns={"posteam": "team"})


def _team_form(tg, team_pbp):
    t = tg[["team", "game_id", "season", "week"]].merge(team_pbp, on=["team", "game_id"], how="left")
    t = t.sort_values(["team", "season", "week"]).reset_index(drop=True)
    s8 = _prior(t, "team", ["neutral_passes", "neutral_plays", "proe_sum", "proe_n"], 8, "sum")
    t["team_neutral_pass_8"] = _ratio(s8["neutral_passes"], s8["neutral_plays"])
    t["team_proe_8"] = _ratio(s8["proe_sum"], s8["proe_n"])
    t["team_plays_8"] = _prior(t, "team", ["off_plays"], 8, "mean")["off_plays"]
    return t[["team", "game_id"] + FEATURE_GROUPS["pass_tendency"]]


# ---------------------------------------------------------------- injury reports

def recent_target_shares(player_stats, tg):
    """Each player's share of his team's targets over the team's previous TARGET_SHARE_GAMES games."""
    targets = player_stats[player_stats["targets"] > 0][["team", "game_id", "player_id", "targets"]]
    out = []
    for team, games in tg[["team", "game_id", "season", "week"]].groupby("team"):
        order = games.sort_values(["season", "week"])["game_id"]
        grid = (targets[targets["team"] == team]
                .pivot_table(index="game_id", columns="player_id", values="targets", aggfunc="sum")
                .reindex(order).fillna(0))
        prior = grid.shift(1).rolling(TARGET_SHARE_GAMES, min_periods=1).sum()
        share = prior.div(prior.sum(axis=1).replace(0, np.nan), axis=0)
        long = share.stack().rename("target_share").reset_index()
        long = long[long["target_share"] > 0]
        long["team"] = team
        out.append(long)
    return pd.concat(out, ignore_index=True)


def injury_features(injuries, player_stats, tg):
    """Per team-game: target share of receivers listed Out/Doubtful or Questionable, and OL out.

    Built from the injury report published before each game, never from who
    actually played, so history matches what is known before kickoff.
    """
    inj = injuries[["season", "week", "team", "gsis_id", "position", "report_status"]].rename(
        columns={"gsis_id": "player_id"})
    inj = inj.merge(tg[["team", "season", "week", "game_id"]], on=["team", "season", "week"], how="inner")
    shares = recent_target_shares(player_stats, tg)
    inj = inj.merge(shares, on=["team", "game_id", "player_id"], how="left")
    inj["target_share"] = inj["target_share"].fillna(0.0)
    out = inj["report_status"].isin(OUT_STATUSES)
    q = inj["report_status"] == "Questionable"
    agg = inj.assign(
        missing=np.where(out, inj["target_share"], 0.0),
        questionable=np.where(q, inj["target_share"], 0.0),
        ol=(out & inj["position"].isin(OL_POSITIONS)).astype(float),
    ).groupby(["team", "game_id"], as_index=False).agg(
        missing_target_share=("missing", "sum"), questionable_target_share=("questionable", "sum"),
        ol_out=("ol", "sum"))
    qb_q = inj.loc[q, ["player_id", "game_id"]].drop_duplicates().assign(qb_questionable=1.0)
    return agg, qb_q


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
    p8 = _prior(rows, k, ["pbp_dropbacks", "scrambles", "pressures"], 8, "sum")
    rows["qb_scramble_rate_8"] = _ratio(p8["scrambles"], p8["pbp_dropbacks"])
    rows["qb_pressure_rate_8"] = _ratio(p8["pressures"], p8["pbp_dropbacks"])

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
    p10 = _prior(d, k, ["def_pbp_dropbacks", "def_pressures"], 10, "sum")
    d["def_pressure_rate_10"] = _ratio(p10["def_pressures"], p10["def_pbp_dropbacks"])
    return d[["game_id", "def_team"] + DEF_FEATURES + ["def_pressure_rate_10"]]


# ---------------------------------------------------------------- main entry point

def build_dataset(data, predict_week=None):
    """One row per QB-game with pre-game features, from a data.NFLData bundle.

    Completed games carry the actual `passing_yards`. If `predict_week`
    (season, week) is given, rows for that week's unplayed games are added
    with the projected starter for each team and no target.
    """
    player_stats, schedule = data.player_stats, data.schedule
    tg = team_games(schedule)
    qb = qb_history(player_stats).merge(pbp_qb_games(data.pbp), on=["player_id", "game_id"], how="left")
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

    defense = defense_history(player_stats).merge(pbp_defense_games(data.pbp), on=["def_team", "game_id"], how="left")
    if not upcoming.empty:
        up_def = upcoming[["game_id", "season", "week", "opponent"]].rename(columns={"opponent": "def_team"})
        defense = pd.concat([defense, up_def.drop_duplicates()], ignore_index=True)
    rows = rows.merge(_defense_form(defense), left_on=["game_id", "opponent"],
                      right_on=["game_id", "def_team"], how="left").drop(columns="def_team")

    rows = rows.merge(_team_form(tg, pbp_team_games(data.pbp)), on=["team", "game_id"], how="left")

    inj_team, qb_q = injury_features(data.injuries, player_stats, tg)
    rows = rows.merge(inj_team, on=["team", "game_id"], how="left")
    rows = rows.merge(qb_q, on=["player_id", "game_id"], how="left")
    inj_cols = FEATURE_GROUPS["injuries"]
    rows[inj_cols] = rows[inj_cols].fillna(0.0)  # not on the report = healthy
    return rows
