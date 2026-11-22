"""Live in-game tracking of the experiment's lines, from ESPN's public scoreboard.

ESPN's site API updates passing yards during games and needs no key. For each
logged QB this gives current yards vs. the line, the game clock, and a live
chance the bet hits:

    final = current + projection * f + error * sqrt(f)

where f is the share of the game left and `error` is drawn from the model's
held-out residuals, so at kickoff it equals the pre-game probability and at
the final whistle it's settled. It ignores game script (a team up 21 runs the
ball), so treat it as a rough guide.
"""
import re

import numpy as np
import pandas as pd
import requests

import model as qb_model
from bets import payout_multiple

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_TEAM = {"WSH": "WAS", "LAR": "LA"}  # ESPN abbreviation -> nflverse


def _norm(name):
    name = re.sub(r"[^a-z ]", "", str(name).lower())
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return " ".join(name.split())


def _get(path, **params):
    r = requests.get(f"{ESPN}/{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def games():
    """This week's games: {frozenset(teams): status dict with ESPN event id}."""
    out = {}
    for e in _get("scoreboard").get("events", []):
        comp, st = e["competitions"][0], e["status"]
        teams = {ESPN_TEAM.get(t["team"]["abbreviation"], t["team"]["abbreviation"]): t.get("score")
                 for t in comp["competitors"]}
        out[frozenset(teams)] = {
            "event_id": e["id"], "state": st["type"]["state"],  # pre / in / post
            "detail": st["type"]["shortDetail"], "period": int(st.get("period") or 0),
            "clock": float(st.get("clock") or 0), "scores": teams,
        }
    return out


def passing(event_id):
    """{(team, normalized name): yards} for everyone who threw a pass so far."""
    out = {}
    for team in _get("summary", event=event_id).get("boxscore", {}).get("players", []):
        abbr = ESPN_TEAM.get(team["team"]["abbreviation"], team["team"]["abbreviation"])
        for s in team["statistics"]:
            if s["name"] != "passing":
                continue
            yds = s["labels"].index("YDS")
            for a in s["athletes"]:
                try:
                    out[(abbr, _norm(a["athlete"]["displayName"]))] = (a["athlete"]["displayName"], float(a["stats"][yds]))
                except (ValueError, IndexError):
                    pass
    return out


def fraction_left(g):
    if g["state"] == "pre":
        return 1.0
    if g["state"] == "post" or g["period"] > 4:  # final, or overtime: treat as nearly done
        return 0.0
    return max(0.0, ((4 - g["period"]) * 15 * 60 + g["clock"]) / 3600)


def live_p_over(meta, projection, line, current, f):
    if f <= 0:
        return float(current > line)
    q = np.asarray(meta["residual_quantiles"]) * np.sqrt(f)
    return float(np.mean(current + projection * f + q > line))


def status(log, meta):
    """One row per logged QB in this week's games that have started or are about to.

    Columns: qb, team, game, detail, state, line, side, odds, stake, projection,
    yards, pace, live_p (chance the model's side wins; for no-bet lines, P(over)),
    result (win / loss / push / "" while live), profit (settled bets only).
    """
    g_by_teams = games()
    rows, box_cache = [], {}
    for _, r in log.iterrows():
        g = g_by_teams.get(frozenset({r["team"], r["opponent"]}))
        if g is None:
            continue
        yards, shown_name = 0.0, r["qb"]
        if g["state"] != "pre":
            if g["event_id"] not in box_cache:
                try:
                    box_cache[g["event_id"]] = passing(g["event_id"])
                except requests.RequestException:
                    box_cache[g["event_id"]] = {}
            hit = box_cache[g["event_id"]].get((r["team"], _norm(r["qb"])))
            if hit:
                shown_name, yards = hit
            else:  # the logged starter hasn't thrown (benched, injured, or a different starter)
                others = [n for (t, _), (n, _) in box_cache[g["event_id"]].items() if t == r["team"]]
                if others:
                    shown_name = f"{r['qb']} (out; {', '.join(others)} passing)"
        f = fraction_left(g)
        p_over = live_p_over(meta, r["projection"], r["line"], yards, f)
        bet = r["side"] in ("over", "under")
        live_p = p_over if r["side"] != "under" else 1 - p_over
        result, profit = "", np.nan
        if g["state"] == "post":
            if yards == r["line"]:
                result, profit = "push", 0.0
            elif bet:
                won = (yards > r["line"]) == (r["side"] == "over")
                result = "win" if won else "loss"
                profit = r["stake"] * payout_multiple(r["odds"]) if won else -r["stake"]
        scores = " – ".join(f"{t} {s}" for t, s in g["scores"].items())
        elapsed = 1 - f
        rows.append({
            "qb": shown_name, "team": r["team"], "game": scores, "detail": g["detail"], "state": g["state"],
            "line": r["line"], "side": r["side"], "odds": r["odds"], "stake": r["stake"],
            "projection": r["projection"], "yards": yards,
            "pace": yards / elapsed if g["state"] == "in" and elapsed > 0.05 else np.nan,
            "live_p": live_p, "result": result, "profit": profit,
        })
    return pd.DataFrame(rows)


def week_lines(log, season, week):
    return log[(log["season"] == season) & (log["week"] == week)]


def current_week(log):
    if log.empty:
        return None
    last = log.sort_values("logged_at").iloc[-1]
    return int(last["season"]), int(last["week"])


def summarize(st_rows):
    """Sweat summary for bets: settled record/profit plus live bets' expected profit."""
    bets = st_rows[st_rows["side"].isin(["over", "under"])]
    settled = bets[bets["result"] != ""]
    live = bets[bets["result"] == ""]
    exp = sum(p * s * payout_multiple(o) - (1 - p) * s for p, s, o in zip(live["live_p"], live["stake"], live["odds"]))
    return {
        "wins": int((settled["result"] == "win").sum()), "losses": int((settled["result"] == "loss").sum()),
        "settled_profit": float(settled["profit"].sum()), "live_bets": len(live),
        "live_expected": float(exp), "in_progress": int((live["state"] == "in").sum()),
    }
