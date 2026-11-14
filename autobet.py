"""Paper-trading experiment: does the model beat sportsbook passing-yard props?

Runs daily (GitHub Action). For games starting within the next LOOKAHEAD_HOURS
it pulls QB passing-yard lines from The Odds API, records the line and the
model's probability for every projected starter, and places a hypothetical
STAKE bet whenever the better side's expected value is at least MIN_EV. Each
QB-game is logged once, using the line available at that time; nothing is
ever edited afterwards, so the log is an honest out-of-sample record.

Every line is logged, including ones with no bet, so other strategies and
controls (always over, always under, any positive EV) can be judged on the
same data. See `evaluate`.

    ODDS_API_KEY=... python autobet.py
"""
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import binomtest

import model as qb_model
from bets import expected_value, implied_prob, payout_multiple
from data import load_all, next_week
from pipeline import project_week

LOG_PATH = Path(__file__).parent / "experiment" / "auto_bets.csv"
API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKET = "player_pass_yds"
BOOKS = ["draftkings", "fanduel", "betmgm", "caesars"]  # first one with both sides wins
LOOKAHEAD_HOURS = 36
STAKE = 100.0
MIN_EV = 0.03  # the pre-registered betting rule: bet only with >= +3% expected return

TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

LOG_COLUMNS = [
    "logged_at", "season", "week", "game_id", "gameday", "commence_time", "player_id", "qb",
    "team", "opponent", "book", "line", "over_odds", "under_odds", "projection", "p_over",
    "ev_over", "ev_under", "side", "odds", "stake", "model_trained_through",
]


def _norm(name):
    """'Michael Penix Jr.' -> 'michael penix'."""
    name = re.sub(r"[^a-z ]", "", str(name).lower())
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return " ".join(name.split())


def load_log():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH, dtype={"game_id": str, "player_id": str})
    return pd.DataFrame(columns=LOG_COLUMNS)


# ---------------------------------------------------------------- Odds API

class OddsAPI:
    def __init__(self, key):
        self.key = key
        self.remaining = None

    def _get(self, url, **params):
        r = requests.get(url, params={"apiKey": self.key, **params}, timeout=20)
        r.raise_for_status()
        self.remaining = r.headers.get("x-requests-remaining", self.remaining)
        return r.json()

    def events(self):
        return self._get(f"{API}/events")  # free: costs no credits

    def pass_yds_props(self, event_id):
        return self._get(f"{API}/events/{event_id}/odds", regions="us", markets=MARKET,
                         oddsFormat="american", bookmakers=",".join(BOOKS))


def best_book_lines(event_odds):
    """{normalized QB name: (book, line, over_odds, under_odds)} from the first preferred book quoting both sides."""
    by_book = {b["key"]: b for b in event_odds.get("bookmakers", [])}
    found = {}
    for key in BOOKS:
        book = by_book.get(key)
        if not book:
            continue
        sides = {}
        for market in book.get("markets", []):
            if market["key"] != MARKET:
                continue
            for o in market["outcomes"]:
                sides.setdefault(_norm(o["description"]), {})[o["name"].lower()] = (o["point"], o["price"])
        for qb, s in sides.items():
            if qb in found or "over" not in s or "under" not in s or s["over"][0] != s["under"][0]:
                continue
            found[qb] = (key, float(s["over"][0]), int(s["over"][1]), int(s["under"][1]))
    return found


# ---------------------------------------------------------------- daily run

def decide(meta, projection, line, over_odds, under_odds):
    p_over = qb_model.prob_over(meta, projection, line)
    ev_over = expected_value(p_over, over_odds)
    ev_under = expected_value(1 - p_over, under_odds)
    side, odds = "none", np.nan
    if max(ev_over, ev_under) >= MIN_EV:
        side, odds = ("over", over_odds) if ev_over >= ev_under else ("under", under_odds)
    return p_over, ev_over, ev_under, side, odds


def run(api_key, now=None):
    now = now or datetime.now(timezone.utc)
    player_stats, schedule = load_all()
    target = next_week(schedule)
    if target is None:
        print("No unplayed games left this season.")
        return
    model, meta = qb_model.load()
    proj = project_week(player_stats, schedule, model, meta, *target)

    log = load_log()
    logged = set(zip(log["game_id"], log["player_id"]))
    api = OddsAPI(api_key)
    new_rows = []

    for ev in api.events():
        start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if not now < start <= now + timedelta(hours=LOOKAHEAD_HOURS):
            continue
        home, away = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
        game = proj[proj["team"].isin([home, away]) & proj["opponent"].isin([home, away])]
        todo = game[[(g, p) not in logged for g, p in zip(game["game_id"], game["player_id"])]]
        if todo.empty:
            continue
        lines = best_book_lines(api.pass_yds_props(ev["id"]))
        for _, r in todo.iterrows():
            quote = lines.get(_norm(r["player_display_name"]))
            if quote is None:
                print(f"  no line yet: {r['player_display_name']} ({r['team']})")
                continue
            book, line, over_odds, under_odds = quote
            p_over, ev_over, ev_under, side, odds = decide(meta, r["projection"], line, over_odds, under_odds)
            new_rows.append({
                "logged_at": now.isoformat(timespec="seconds"), "season": r["season"], "week": r["week"],
                "game_id": r["game_id"], "gameday": r["gameday"], "commence_time": ev["commence_time"],
                "player_id": r["player_id"], "qb": r["player_display_name"], "team": r["team"],
                "opponent": r["opponent"], "book": book, "line": line, "over_odds": over_odds,
                "under_odds": under_odds, "projection": round(float(r["projection"]), 1),
                "p_over": round(p_over, 4), "ev_over": round(ev_over, 4), "ev_under": round(ev_under, 4),
                "side": side, "odds": odds, "stake": STAKE if side != "none" else 0.0,
                "model_trained_through": "{season}-wk{week}".format(**meta["trained_through"]),
            })
            print(f"  {r['player_display_name']:22s} line {line:6.1f}  proj {r['projection']:6.1f}  "
                  f"P(over) {p_over:.0%}  -> {side}")

    if new_rows:
        LOG_PATH.parent.mkdir(exist_ok=True)
        new = pd.DataFrame(new_rows, columns=LOG_COLUMNS)
        log = new if log.empty else pd.concat([log, new], ignore_index=True)
        log.to_csv(LOG_PATH, index=False)
    print(f"Logged {len(new_rows)} lines ({sum(r['side'] != 'none' for r in new_rows)} bets). "
          f"Odds API credits remaining: {api.remaining}")


# ---------------------------------------------------------------- evaluation

def grade_lines(log, player_stats, schedule):
    """Add actual yards and a `settled` flag (game final and QB recorded stats)."""
    done = set(schedule.loc[schedule["home_score"].notna(), "game_id"])
    actual = player_stats.set_index(["player_id", "game_id"])["passing_yards"]
    out = log.copy()
    out["actual"] = [actual.get(k) for k in zip(out["player_id"], out["game_id"])]
    out["settled"] = out["game_id"].isin(done) & out["actual"].notna()
    return out


def _strategy(rows, sides, odds):
    """Results of betting `sides` (Series of over/under) at `odds` on `rows`; pushes excluded from W/L."""
    over_hit = rows["actual"] > rows["line"]
    push = rows["actual"] == rows["line"]
    win = (over_hit == (sides == "over")) & ~push
    profit = np.where(push, 0.0, np.where(win, STAKE * odds.map(payout_multiple), -STAKE))
    n, wins = int((~push).sum()), int(win.sum())
    breakeven = float(odds.map(implied_prob).mean()) if len(odds) else np.nan
    staked = STAKE * len(rows)
    return {
        "bets": len(rows), "wins": wins, "losses": n - wins, "pushes": int(push.sum()),
        "win_rate": wins / n if n else np.nan, "breakeven": breakeven,
        "profit": float(profit.sum()), "roi": float(profit.sum() / staked) if staked else np.nan,
        # Chance of doing at least this well with no real edge (coin flips at break-even odds)
        "p_luck": binomtest(wins, n, breakeven, alternative="greater").pvalue if n else np.nan,
    }


def evaluate(graded):
    """Strategy comparison, model-vs-line accuracy, and calibration on settled lines."""
    s = graded[graded["settled"]].copy()
    if s.empty:
        return {"strategies": pd.DataFrame(), "accuracy": {}, "calibration": pd.DataFrame(), "settled": s}
    model_side = np.where(s["ev_over"] >= s["ev_under"], "over", "under")
    model_odds = pd.Series(np.where(model_side == "over", s["over_odds"], s["under_odds"]), index=s.index)
    best_ev = s[["ev_over", "ev_under"]].max(axis=1)

    rows = {}
    rule = s[s["side"] != "none"]
    rows[f"Model rule (EV ≥ {MIN_EV:.0%}) — the real test"] = _strategy(rule, rule["side"], rule["odds"])
    pos = best_ev > 0
    rows["Any positive EV"] = _strategy(s[pos], pd.Series(model_side, index=s.index)[pos], model_odds[pos])
    rows["Model's side on every line"] = _strategy(s, pd.Series(model_side, index=s.index), model_odds)
    rows["Control: always over"] = _strategy(s, pd.Series("over", index=s.index), s["over_odds"])
    rows["Control: always under"] = _strategy(s, pd.Series("under", index=s.index), s["under_odds"])
    strategies = pd.DataFrame(rows).T

    accuracy = {
        "n": len(s),
        "model_mae": float((s["actual"] - s["projection"]).abs().mean()),
        "line_mae": float((s["actual"] - s["line"]).abs().mean()),
        "model_closer_pct": float(((s["actual"] - s["projection"]).abs() < (s["actual"] - s["line"]).abs()).mean()),
    }

    p_side = np.where(model_side == "over", s["p_over"], 1 - s["p_over"])
    hit = np.where(model_side == "over", s["actual"] > s["line"], s["actual"] < s["line"])
    cal = pd.DataFrame({"p": p_side, "hit": hit})
    cal["bucket"] = pd.cut(cal["p"], [0.5, 0.53, 0.56, 0.6, 0.65, 1.0], include_lowest=True)
    calibration = (cal.groupby("bucket", observed=True)
                   .agg(lines=("hit", "size"), model_said=("p", "mean"), actually_hit=("hit", "mean"))
                   .reset_index())
    calibration["bucket"] = calibration["bucket"].astype(str)
    return {"strategies": strategies, "accuracy": accuracy, "calibration": calibration, "settled": s}


MIN_BETS_FOR_VERDICT = 100


def verdict(res):
    """(level, message) summarizing whether to trust the model. level: info/success/warning/error."""
    strat, s = res["strategies"], res["settled"]
    n = int((s["side"] != "none").sum()) if not s.empty else 0
    if strat.empty or n < MIN_BETS_FOR_VERDICT:
        return "info", (f"**Too early to tell.** {n} settled bets so far (verdict starts at {MIN_BETS_FOR_VERDICT}). "
                        "Single-game yards are noisy, so luck dominates until there are a few hundred bets; "
                        "with 100 bets the win rate is only known to about ±10%.")
    rule, acc = strat.iloc[0], res["accuracy"]
    if rule["roi"] > 0 and rule["p_luck"] < 0.05 and acc["model_mae"] < acc["line_mae"]:
        return "success", (f"**Promising.** The model rule is up {rule['roi']:+.1%} over {n} bets, there's only a "
                           f"{rule['p_luck']:.1%} chance of a record this good by luck, and the model's projections "
                           "have been closer to actual yards than the sportsbook lines.")
    if rule["roi"] > 0:
        return "warning", (f"**Profitable but not proven.** Up {rule['roi']:+.1%} over {n} bets, but there's a "
                           f"{rule['p_luck']:.0%} chance of doing this well with no real edge. Keep collecting.")
    return "error", f"**No edge so far.** The model rule is {rule['roi']:+.1%} over {n} bets. Don't bet real money on it."


if __name__ == "__main__":
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ODDS_API_KEY not set; skipping. Get a free key at https://the-odds-api.com")
    else:
        run(key)
