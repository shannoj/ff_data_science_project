"""Paper-trade Kalshi QB passing-yard markets with the model.

Kalshi lists a ladder of yes/no contracts per QB ("Drew Lock: 250+ passing
yards"). Each contract pays $1 if it resolves the way you bought it. For every
projected starter in a game starting soon, this prices every rung with the
model, subtracts Kalshi's trading fee, and takes the single best yes/no trade
if its expected return is at least MIN_EV. It also records how many contracts
were actually on offer at that price, because thin order books are the main
practical limit on exchanges.

Uses Kalshi's public market-data API (no account or key needed).
"""
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import model as qb_model

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNFLPASSYDS"
LOG_PATH = Path(__file__).parent / "experiment" / "kalshi_bets.csv"
STAKE = 100.0
MIN_EV = 0.03
# Only trade contracts priced 25-75 cents. Picking the best of ~20 contracts per QB otherwise
# favors cheap long shots, where the model's tail probabilities are least reliable.
MIN_PRICE, MAX_PRICE = 0.25, 0.75
FEE_RATE = 0.07  # Kalshi taker fee per contract: FEE_RATE * price * (1 - price)
KALSHI_TEAM = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS"}  # Kalshi abbreviation -> nflverse
EASTERN = ZoneInfo("America/New_York")

COLUMNS = [
    "logged_at", "season", "week", "game_id", "kickoff", "player_id", "qb", "team", "opponent",
    "projection", "ticker", "strike", "side", "price", "fee", "cost", "model_p", "ev",
    "size_available", "fillable", "stake", "volume", "n_rungs", "ladder", "model_trained_through",
]


def _norm(name):
    name = re.sub(r"[^a-z ]", "", str(name).lower())
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return " ".join(name.split())


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def fee(price):
    return FEE_RATE * price * (1 - price)


def load_log():
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(LOG_PATH, dtype={"game_id": str, "player_id": str, "ladder": str})


# ---------------------------------------------------------------- market data

def fetch_markets():
    markets, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{API}/markets", params=params, timeout=20)
        r.raise_for_status()
        body = r.json()
        markets += body.get("markets", [])
        cursor = body.get("cursor")
        if not cursor:
            return markets


def fetch_event_teams(event_tickers):
    """{event_ticker: frozenset of nflverse team codes} from each event's 'SEA vs LV (Nov 15)' subtitle."""
    out = {}
    for ev in event_tickers:
        r = requests.get(f"{API}/events/{ev}", timeout=20)
        if r.status_code != 200:
            continue
        sub = r.json().get("event", {}).get("sub_title", "")
        m = re.match(r"\s*([A-Z]{2,3}) vs ([A-Z]{2,3})", sub)
        if m:
            out[ev] = frozenset(KALSHI_TEAM.get(t, t) for t in m.groups())
    return out


def ladders(markets, event_teams):
    """{(teams, normalized QB name): [rung, ...]} where a rung is one 'X+ yards' contract."""
    out = {}
    for m in markets:
        teams = event_teams.get(m["event_ticker"])
        name = (m.get("yes_sub_title") or "").split(":")[0]
        if teams is None or not name:
            continue
        out.setdefault((teams, _norm(name)), []).append({
            "ticker": m["ticker"],
            "strike": _num(m.get("floor_strike")),  # 249.5 means "250+ yards"
            "yes_bid": _num(m.get("yes_bid_dollars")), "yes_ask": _num(m.get("yes_ask_dollars")),
            "no_ask": _num(m.get("no_ask_dollars")),
            "yes_bid_size": _num(m.get("yes_bid_size_fp")), "yes_ask_size": _num(m.get("yes_ask_size_fp")),
            "volume": _num(m.get("volume_fp")),
        })
    for rungs in out.values():
        rungs.sort(key=lambda r: r["strike"])
    return out


def _ladder_str(rungs):
    return "|".join(f"{r['strike']:g}:{r['yes_bid']:.2f}:{r['yes_ask']:.2f}:{r['yes_bid_size']:g}:{r['yes_ask_size']:g}"
                    for r in rungs)


# ---------------------------------------------------------------- pricing

def best_trade(meta, projection, rungs):
    """Best yes/no trade across the ladder by expected return per dollar (after fees), or None."""
    best = None
    for r in rungs:
        if np.isnan(r["strike"]):
            continue
        p_over = qb_model.prob_over(meta, projection, r["strike"])
        # Buying NO costs no_ask; the contracts on offer are whoever is bidding for YES.
        for side, price, p, size in (("yes", r["yes_ask"], p_over, r["yes_ask_size"]),
                                     ("no", r["no_ask"], 1 - p_over, r["yes_bid_size"])):
            if not MIN_PRICE <= price <= MAX_PRICE:
                continue
            cost = price + fee(price)
            ev = p / cost - 1
            if best is None or ev > best["ev"]:
                best = {"ticker": r["ticker"], "strike": r["strike"], "side": side, "price": price,
                        "fee": round(fee(price), 4), "cost": round(cost, 4), "model_p": round(p, 4),
                        "ev": round(ev, 4), "size_available": size, "volume": r["volume"]}
    return best


def kickoff_utc(gameday, gametime):
    local = datetime.strptime(f"{gameday} {gametime or '13:00'}", "%Y-%m-%d %H:%M").replace(tzinfo=EASTERN)
    return local.astimezone(timezone.utc)


def log_trades(proj, meta, now=None, lookahead_hours=36):
    """Log one Kalshi paper trade (or a no-trade row) per projected QB in games starting soon."""
    now = now or datetime.now(timezone.utc)
    log = load_log()
    logged = set(zip(log["game_id"], log["player_id"]))
    kick = [kickoff_utc(d, t) for d, t in zip(proj["gameday"], proj["gametime"])]
    soon = [now < k <= now + timedelta(hours=lookahead_hours) for k in kick]
    todo = proj[np.array(soon, dtype=bool)
                & np.array([(g, p) not in logged for g, p in zip(proj["game_id"], proj["player_id"])], dtype=bool)]
    if todo.empty:
        print("Kalshi: no new games in the window.")
        return

    markets = fetch_markets()
    books = ladders(markets, fetch_event_teams({m["event_ticker"] for m in markets}))
    rows = []
    for _, r in todo.iterrows():
        rungs = books.get((frozenset([r["team"], r["opponent"]]), _norm(r["player_display_name"])))
        if not rungs:
            print(f"  Kalshi: no market for {r['player_display_name']} ({r['team']})")
            continue
        t = best_trade(meta, r["projection"], rungs)
        if t is None:
            continue
        bet = t["ev"] >= MIN_EV
        rows.append({
            "logged_at": now.isoformat(timespec="seconds"), "season": r["season"], "week": r["week"],
            "game_id": r["game_id"], "kickoff": kickoff_utc(r["gameday"], r["gametime"]).isoformat(),
            "player_id": r["player_id"], "qb": r["player_display_name"], "team": r["team"],
            "opponent": r["opponent"], "projection": round(float(r["projection"]), 1),
            **t, "side": t["side"] if bet else "none",
            "fillable": bool(bet and t["size_available"] * t["price"] >= STAKE),
            "stake": STAKE if bet else 0.0, "n_rungs": len(rungs), "ladder": _ladder_str(rungs),
            "model_trained_through": "{season}-wk{week}".format(**meta["trained_through"]),
        })
        print(f"  Kalshi {r['player_display_name']:22s} best: {t['side']} {t['strike'] + 0.5:g}+ @ "
              f"${t['price']:.2f} (model {t['model_p']:.0%}, EV {t['ev']:+.1%}, "
              f"{t['size_available']:g} on offer) -> {'BET' if bet else 'pass'}")
    if rows:
        LOG_PATH.parent.mkdir(exist_ok=True)
        new = pd.DataFrame(rows, columns=COLUMNS)
        log = new if log.empty else pd.concat([log, new], ignore_index=True)
        log.to_csv(LOG_PATH, index=False)
    print(f"Kalshi: logged {len(rows)} QBs ({sum(r['side'] != 'none' for r in rows)} trades).")


# ---------------------------------------------------------------- grading

def grade(log, player_stats, schedule):
    """Add actual yards, result (win/loss/void/pending) and profit for each trade."""
    if log.empty:
        return log.assign(actual=pd.Series(dtype=float), result=pd.Series(dtype=str), profit=pd.Series(dtype=float))
    done = set(schedule.loc[schedule["home_score"].notna(), "game_id"])
    actual = player_stats.set_index(["player_id", "game_id"])["passing_yards"]
    out = log.copy()
    out["actual"] = [actual.get(k) for k in zip(out["player_id"], out["game_id"])]
    over = out["actual"] > out["strike"]
    won = np.where(out["side"] == "yes", over, ~over)
    out["result"] = np.select(
        [~out["game_id"].isin(done), out["actual"].isna(), won], ["pending", "void", "win"], default="loss")
    out["profit"] = np.select([out["result"] == "win", out["result"] == "loss"],
                              [out["stake"] * (1 / out["cost"] - 1), -out["stake"]], default=0.0)
    out.loc[out["result"] == "pending", "profit"] = np.nan
    return out


def summarize(graded):
    trades = graded[graded["side"] != "none"]
    settled = trades[trades["result"].isin(["win", "loss"])]
    staked = settled["stake"].sum()
    return {
        "qbs_priced": int(len(graded)),
        "trades": int(len(trades)),
        "fillable_pct": float(trades["fillable"].astype(bool).mean()) if len(trades) else np.nan,
        "wins": int((settled["result"] == "win").sum()),
        "losses": int((settled["result"] == "loss").sum()),
        "pending": int((trades["result"] == "pending").sum()),
        "profit": float(settled["profit"].sum()),
        "roi": float(settled["profit"].sum() / staked) if staked else np.nan,
        "fillable_profit": float(settled.loc[settled["fillable"].astype(bool), "profit"].sum()),
    }
