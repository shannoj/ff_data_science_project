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

Bets are placed at the best price across books (line shopping); each row also
keeps a single-book quote so the value of shopping can be measured. A second
job (`--closing`) snapshots the lines shortly before kickoff to measure
closing line value (CLV), the fastest reliable signal of a real edge.

    ODDS_API_KEY=... python autobet.py             # daily: log lines + paper bets
    ODDS_API_KEY=... python autobet.py --closing   # near kickoff: closing lines
"""
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import binomtest

import kalshi
import model as qb_model
from bets import expected_value, implied_prob, payout_multiple
from data import load_all, next_week
from pipeline import project_week

LOG_PATH = Path(__file__).parent / "experiment" / "auto_bets.csv"
API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKET = "player_pass_yds"
BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "betrivers", "fanatics"]  # first = primary
LOOKAHEAD_HOURS = 36
STAKE = 100.0
MIN_EV = 0.03  # the pre-registered betting rule: bet only with >= +3% expected return
KELLY_CAP = 3.0  # shadow Kelly strategy: no bet larger than 3x the average stake

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
    "team", "opponent",
    # Quote used for the model's side after line shopping (best expected value across books)
    "book", "line", "over_odds", "under_odds", "projection", "p_over", "ev_over", "ev_under",
    "side", "odds", "stake", "model_trained_through",
    # Single-book reference (first of BOOKS quoting the QB), to measure what shopping adds
    "primary_book", "primary_line", "primary_over_odds", "primary_under_odds", "primary_p_over",
    "consensus_line", "n_books", "quotes",
]
CLOSE_PATH = Path(__file__).parent / "experiment" / "closing_lines.csv"
CLOSE_COLUMNS = ["captured_at", "game_id", "player_id", "qb", "commence_time", "minutes_before",
                 "consensus_line", "n_books", "quotes"]
CLOSE_WINDOW_MINUTES = 70  # closing snapshot is taken by the first run within this long before kickoff


def _norm(name):
    """'Michael Penix Jr.' -> 'michael penix'."""
    name = re.sub(r"[^a-z ]", "", str(name).lower())
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return " ".join(name.split())


def _read(path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, dtype={"game_id": str, "player_id": str, "quotes": str})


def load_log():
    log = _read(LOG_PATH, LOG_COLUMNS)
    # Rows logged before line shopping existed used a single book: backfill the reference columns.
    for ref, src in [("primary_book", "book"), ("primary_line", "line"), ("primary_over_odds", "over_odds"),
                     ("primary_under_odds", "under_odds"), ("primary_p_over", "p_over"), ("consensus_line", "line")]:
        if ref not in log.columns:
            log[ref] = log[src]
    if "n_books" not in log.columns:
        log["n_books"] = 1
    if "quotes" not in log.columns:
        log["quotes"] = [_quotes_str([(b, l, o, u)]) for b, l, o, u in
                         zip(log["book"], log["line"], log["over_odds"], log["under_odds"])]
    return log[LOG_COLUMNS]


def load_closing():
    return _read(CLOSE_PATH, CLOSE_COLUMNS)


def _quotes_str(quotes):
    return "|".join(f"{b}:{l:g}:{int(o)}:{int(u)}" for b, l, o, u in quotes)


def parse_quotes(text):
    """'draftkings:235.5:-110:-110|fanduel:...' -> {book: (line, over_odds, under_odds)}."""
    out = {}
    for part in str(text).split("|"):
        if part.count(":") == 3:
            b, l, o, u = part.split(":")
            out[b] = (float(l), int(o), int(u))
    return out


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
        # Up to 10 bookmakers cost the same single credit as one.
        return self._get(f"{API}/events/{event_id}/odds", regions="us", markets=MARKET,
                         oddsFormat="american", bookmakers=",".join(BOOKS))


def all_book_quotes(event_odds):
    """{normalized QB name: [(book, line, over_odds, under_odds), ...]} in BOOKS preference order."""
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
            if "over" in s and "under" in s and s["over"][0] == s["under"][0]:
                found.setdefault(qb, []).append((key, float(s["over"][0]), int(s["over"][1]), int(s["under"][1])))
    return found


def _match(events, home_away_rows, now, horizon):
    """Yield (event, rows for that game) for events starting in (now, now + horizon]."""
    for ev in events:
        start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if not now < start <= now + horizon:
            continue
        teams = [TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])]
        rows = home_away_rows[home_away_rows["team"].isin(teams) & home_away_rows["opponent"].isin(teams)]
        if not rows.empty:
            yield ev, rows


# ---------------------------------------------------------------- daily run

def decide(meta, projection, line, over_odds, under_odds):
    """Model call at a single quote."""
    p_over = qb_model.prob_over(meta, projection, line)
    ev_over = expected_value(p_over, over_odds)
    ev_under = expected_value(1 - p_over, under_odds)
    side, odds = "none", np.nan
    if max(ev_over, ev_under) >= MIN_EV:
        side, odds = ("over", over_odds) if ev_over >= ev_under else ("under", under_odds)
    return p_over, ev_over, ev_under, side, odds


def shop(meta, projection, quotes):
    """Pick the quote with the best expected value for whichever side the model prefers.

    Returns (chosen quote, decide() result at that quote).
    """
    best = {}
    for q in quotes:
        p, ev_o, ev_u, _, _ = decide(meta, projection, q[1], q[2], q[3])
        for side, ev in (("over", ev_o), ("under", ev_u)):
            if side not in best or ev > best[side][0]:
                best[side] = (ev, q)
    side = "over" if best["over"][0] >= best["under"][0] else "under"
    q = best[side][1]
    return q, decide(meta, projection, q[1], q[2], q[3])


def run(api_key, now=None):
    """Daily job: sportsbook lines + paper bets (needs api_key), then Kalshi (free)."""
    now = now or datetime.now(timezone.utc)
    player_stats, schedule = load_all()
    target = next_week(schedule)
    if target is None:
        print("No unplayed games left this season.")
        return
    model, meta = qb_model.load()
    proj = project_week(player_stats, schedule, model, meta, *target)

    if api_key:
        run_sportsbooks(api_key, proj, meta, now)
    else:
        print("ODDS_API_KEY not set; skipping sportsbook lines. Get a free key at https://the-odds-api.com")
    try:
        kalshi.log_trades(proj, meta, now, LOOKAHEAD_HOURS)
    except requests.RequestException as e:  # never let Kalshi break the sportsbook log
        print(f"Kalshi lookup failed: {e}")


def run_sportsbooks(api_key, proj, meta, now):
    """Log sportsbook prop lines (best price across books) and paper bets."""

    log = load_log()
    logged = set(zip(log["game_id"], log["player_id"]))
    proj = proj[[(g, p) not in logged for g, p in zip(proj["game_id"], proj["player_id"])]]
    api = OddsAPI(api_key)
    new_rows = []

    for ev, todo in _match(api.events(), proj, now, timedelta(hours=LOOKAHEAD_HOURS)):
        quotes_by_qb = all_book_quotes(api.pass_yds_props(ev["id"]))
        for _, r in todo.iterrows():
            quotes = quotes_by_qb.get(_norm(r["player_display_name"]))
            if not quotes:
                print(f"  no line yet: {r['player_display_name']} ({r['team']})")
                continue
            (book, line, over_odds, under_odds), (p_over, ev_over, ev_under, side, odds) = shop(meta, r["projection"], quotes)
            primary = quotes[0]
            new_rows.append({
                "logged_at": now.isoformat(timespec="seconds"), "season": r["season"], "week": r["week"],
                "game_id": r["game_id"], "gameday": r["gameday"], "commence_time": ev["commence_time"],
                "player_id": r["player_id"], "qb": r["player_display_name"], "team": r["team"],
                "opponent": r["opponent"], "book": book, "line": line, "over_odds": over_odds,
                "under_odds": under_odds, "projection": round(float(r["projection"]), 1),
                "p_over": round(p_over, 4), "ev_over": round(ev_over, 4), "ev_under": round(ev_under, 4),
                "side": side, "odds": odds, "stake": STAKE if side != "none" else 0.0,
                "model_trained_through": "{season}-wk{week}".format(**meta["trained_through"]),
                "primary_book": primary[0], "primary_line": primary[1], "primary_over_odds": primary[2],
                "primary_under_odds": primary[3],
                "primary_p_over": round(qb_model.prob_over(meta, r["projection"], primary[1]), 4),
                "consensus_line": float(np.median([q[1] for q in quotes])), "n_books": len(quotes),
                "quotes": _quotes_str(quotes),
            })
            print(f"  {r['player_display_name']:22s} {book:10s} line {line:6.1f}  proj {r['projection']:6.1f}  "
                  f"P(over) {p_over:.0%}  -> {side}  ({len(quotes)} books)")

    if new_rows:
        LOG_PATH.parent.mkdir(exist_ok=True)
        new = pd.DataFrame(new_rows, columns=LOG_COLUMNS)
        log = new if log.empty else pd.concat([log, new], ignore_index=True)
        log.to_csv(LOG_PATH, index=False)
    print(f"Logged {len(new_rows)} lines ({sum(r['side'] != 'none' for r in new_rows)} bets). "
          f"Odds API credits remaining: {api.remaining}")


def run_closing(api_key, now=None):
    """Snapshot lines shortly before kickoff for every logged QB-game (once each)."""
    now = now or datetime.now(timezone.utc)
    log, closing = load_log(), load_closing()
    done = set(zip(closing["game_id"], closing["player_id"]))
    kickoff = pd.to_datetime(log["commence_time"], utc=True)
    window = timedelta(minutes=CLOSE_WINDOW_MINUTES)
    due = log[(kickoff > now) & (kickoff <= now + window)
              & np.array([(g, p) not in done for g, p in zip(log["game_id"], log["player_id"])], dtype=bool)]
    if due.empty:
        print("No logged games kicking off soon; nothing to do (no API credits used).")
        return
    api = OddsAPI(api_key)
    rows = []
    for ev, todo in _match(api.events(), due, now, window):
        quotes_by_qb = all_book_quotes(api.pass_yds_props(ev["id"]))
        start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        for _, r in todo.iterrows():
            quotes = quotes_by_qb.get(_norm(r["qb"]))
            if not quotes:
                print(f"  no closing line: {r['qb']}")
                continue
            rows.append({"captured_at": now.isoformat(timespec="seconds"), "game_id": r["game_id"],
                         "player_id": r["player_id"], "qb": r["qb"], "commence_time": ev["commence_time"],
                         "minutes_before": round((start - now).total_seconds() / 60),
                         "consensus_line": float(np.median([q[1] for q in quotes])), "n_books": len(quotes),
                         "quotes": _quotes_str(quotes)})
            print(f"  close {r['qb']:22s} {rows[-1]['consensus_line']:6.1f} (opened {r['line']:g} at {r['book']})")
    if rows:
        CLOSE_PATH.parent.mkdir(exist_ok=True)
        new = pd.DataFrame(rows, columns=CLOSE_COLUMNS)
        closing = new if closing.empty else pd.concat([closing, new], ignore_index=True)
        closing.to_csv(CLOSE_PATH, index=False)
    print(f"Captured {len(rows)} closing lines. Odds API credits remaining: {api.remaining}")


# ---------------------------------------------------------------- evaluation

def grade_lines(log, player_stats, schedule):
    """Add actual yards and a `settled` flag (game final and QB recorded stats)."""
    # Final AND stats published: scores post right away, player stats usually overnight, so a
    # finished game with no stats yet stays pending instead of looking like the QB sat out.
    done = set(schedule.loc[schedule["home_score"].notna(), "game_id"]) & set(player_stats["game_id"])
    actual = player_stats.set_index(["player_id", "game_id"])["passing_yards"]
    out = log.copy()
    out["actual"] = [actual.get(k) for k in zip(out["player_id"], out["game_id"])]
    out["settled"] = out["game_id"].isin(done) & out["actual"].notna()
    return out


def _ev(p, odds):
    return p * odds.map(payout_multiple) - (1 - p)


def kelly_stakes(p, odds, cap=KELLY_CAP):
    """Stakes proportional to the Kelly fraction (EV / payout), scaled so they average STAKE.

    Scaling uses only the model's probabilities and the odds, never outcomes, so the
    total money staked matches the flat rule and profit is directly comparable.
    Each bet is capped at about `cap` times the average so one wild projection can't dominate.
    """
    b = odds.map(payout_multiple)
    f = ((p * b - (1 - p)) / b).clip(lower=0)
    if len(f) == 0 or f.sum() == 0:
        return pd.Series(STAKE, index=p.index)
    stakes = (f / f.mean()).clip(upper=cap)
    return STAKE * stakes / stakes.mean()


def _strategy(rows, line, sides, odds, stakes=None):
    """Results of betting `sides` at `line`/`odds` on `rows`; pushes excluded from W/L.

    `stakes` defaults to a flat STAKE per bet.
    """
    stakes = pd.Series(STAKE, index=rows.index) if stakes is None else stakes
    over_hit = rows["actual"] > line
    push = rows["actual"] == line
    win = (over_hit == (sides == "over")) & ~push
    profit = np.where(push, 0.0, np.where(win, stakes * odds.map(payout_multiple), -stakes))
    n, wins = int((~push).sum()), int(win.sum())
    breakeven = float(odds.map(implied_prob).mean()) if len(odds) else np.nan
    staked = float(stakes.sum())
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

    # One-book view: what the model would have done without line shopping.
    line1, over1, under1 = s["primary_line"], s["primary_over_odds"], s["primary_under_odds"]
    ev_o1, ev_u1 = _ev(s["primary_p_over"], over1), _ev(1 - s["primary_p_over"], under1)
    side1 = pd.Series(np.where(ev_o1 >= ev_u1, "over", "under"), index=s.index)
    odds1 = pd.Series(np.where(side1 == "over", over1, under1), index=s.index)
    best1 = np.maximum(ev_o1, ev_u1)

    rows = {}
    rule = s[s["side"] != "none"]
    rows[f"Model rule (EV ≥ {MIN_EV:.0%}), best price — the real test"] = _strategy(
        rule, rule["line"], rule["side"], rule["odds"])
    p_rule = np.where(rule["side"] == "over", rule["p_over"], 1 - rule["p_over"])
    kelly = _strategy(rule, rule["line"], rule["side"], rule["odds"],
                      kelly_stakes(pd.Series(p_rule, index=rule.index), rule["odds"]))
    kelly["p_luck"] = np.nan  # same bets as the rule: the win/loss luck test is identical
    rows["Shadow: model rule, Kelly-sized (same total stake)"] = kelly
    r1 = best1 >= MIN_EV
    rows["Model rule, one book (no line shopping)"] = _strategy(s[r1], line1[r1], side1[r1], odds1[r1])
    pos = best1 > 0
    rows["Any positive EV (one book)"] = _strategy(s[pos], line1[pos], side1[pos], odds1[pos])
    rows["Model's side on every line (one book)"] = _strategy(s, line1, side1, odds1)
    rows["Control: always over"] = _strategy(s, line1, pd.Series("over", index=s.index), over1)
    rows["Control: always under"] = _strategy(s, line1, pd.Series("under", index=s.index), under1)
    strategies = pd.DataFrame(rows).T

    market = s["consensus_line"]
    accuracy = {
        "n": len(s),
        "model_mae": float((s["actual"] - s["projection"]).abs().mean()),
        "line_mae": float((s["actual"] - market).abs().mean()),
        "model_closer_pct": float(((s["actual"] - s["projection"]).abs() < (s["actual"] - market).abs()).mean()),
        **probability_scores(s),
    }

    model_side = np.where(s["ev_over"] >= s["ev_under"], "over", "under")
    p_side = np.where(model_side == "over", s["p_over"], 1 - s["p_over"])
    hit = np.where(model_side == "over", s["actual"] > s["line"], s["actual"] < s["line"])
    cal = pd.DataFrame({"p": p_side, "hit": hit})
    cal["bucket"] = pd.cut(cal["p"], [0.5, 0.53, 0.56, 0.6, 0.65, 1.0], include_lowest=True)
    calibration = (cal.groupby("bucket", observed=True)
                   .agg(lines=("hit", "size"), model_said=("p", "mean"), actually_hit=("hit", "mean"))
                   .reset_index())
    calibration["bucket"] = calibration["bucket"].astype(str)
    return {"strategies": strategies, "accuracy": accuracy, "calibration": calibration, "settled": s}


def no_vig_p_over(over_odds, under_odds):
    """Market's P(over) with the bookmaker's margin removed."""
    po, pu = over_odds.map(implied_prob), under_odds.map(implied_prob)
    return po / (po + pu)


def probability_scores(s):
    """Brier score and log loss of the model's P(over) vs. the market's, at the primary book.

    Both reward confident right answers and punish confident wrong ones (log loss harshly).
    Lower is better. `prob_skill` > 0 means the model's probabilities beat the market's.
    """
    s = s[s["actual"] != s["primary_line"]]  # pushes have no over/under outcome
    if s.empty:
        return {"prob_n": 0}
    y = (s["actual"] > s["primary_line"]).astype(float).to_numpy()
    model = s["primary_p_over"].astype(float).clip(0.01, 0.99).to_numpy()
    market = no_vig_p_over(s["primary_over_odds"], s["primary_under_odds"]).to_numpy()

    def brier(p):
        return float(np.mean((p - y) ** 2))

    def log_loss(p):
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return {
        "prob_n": int(len(y)),
        "model_brier": brier(model), "market_brier": brier(market),
        "model_log_loss": log_loss(model), "market_log_loss": log_loss(market),
        "prob_skill": 1 - brier(model) / brier(market),
    }


def closing_value(log, closing):
    """Per-bet closing line value: yards the line moved toward the bet after it was placed.

    Positive means the market moved your way (you got a better number than the close),
    the most reliable early sign of a real edge. Uses the same book's closing line when
    available, else the closing consensus.
    """
    bets = log[log["side"] != "none"]
    if bets.empty or closing.empty:
        return pd.DataFrame()
    c = closing[["game_id", "player_id", "consensus_line", "quotes", "minutes_before"]].rename(
        columns={"consensus_line": "close_consensus", "quotes": "close_quotes"})
    m = bets.merge(c, on=["game_id", "player_id"], how="inner")
    same = [parse_quotes(q).get(b) for q, b in zip(m["close_quotes"], m["book"])]
    m["close_line"] = [s[0] if s else cc for s, cc in zip(same, m["close_consensus"])]
    m["same_book_close"] = [s is not None for s in same]
    m["clv_yards"] = np.where(m["side"] == "over", m["close_line"] - m["line"], m["line"] - m["close_line"])
    return m


def clv_summary(clv):
    if clv.empty:
        return {"n": 0}
    x = clv["clv_yards"]
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "ci95": float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
        "beat_pct": float((x > 0).mean()),
        "same_pct": float((x == 0).mean()),
        "worse_pct": float((x < 0).mean()),
    }


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
    import sys
    key = os.environ.get("ODDS_API_KEY")
    if "--closing" in sys.argv:
        if key:
            run_closing(key)
        else:
            print("ODDS_API_KEY not set; skipping closing lines.")
    else:
        run(key)
