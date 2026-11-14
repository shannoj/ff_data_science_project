"""Weekly auto-bet experiment report, posted as a GitHub issue by the Tuesday run.

    python weekly_report.py report.md   # writes the report and prints its title

Writes nothing (and prints nothing) if there is no completed week with logged
lines to report on, so the workflow can skip creating an issue.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import autobet
import model as qb_model
from bets import payout_multiple
from data import load_all, next_week

MENTION = "@shannoj"  # GitHub user to notify (sends the email)
LOG_URL = "https://github.com/shannoj/ff_data_science_project/blob/main/experiment/auto_bets.csv"


def _table(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("—" if (isinstance(v, float) and np.isnan(v)) else (fmt[c](v) if c in fmt else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _bet_results(rows):
    won = (rows["actual"] > rows["line"]) == (rows["side"] == "over")
    push = rows["actual"] == rows["line"]
    settled = rows["settled"]
    result = np.select([settled & push, settled & won, settled, rows["game_done"]],
                       ["push", "win", "loss", "void"], default="pending")  # done but no stats = void
    profit = np.where(result == "win", rows["stake"] * rows["odds"].map(payout_multiple),
                      np.where(result == "loss", -rows["stake"], 0.0))
    return rows.assign(result=result, profit=profit)


def report_week(schedule):
    """(season, week) of the most recently completed week."""
    nxt = next_week(schedule)
    if nxt is None:
        season = int(schedule["season"].max())
        return season, int(schedule.loc[schedule["season"] == season, "week"].max())
    season, week = nxt
    return (season, week - 1) if week > 1 else None


def build_report():
    player_stats, schedule = load_all()
    target = report_week(schedule)
    log = autobet.load_log()
    if target is None or log.empty:
        return None
    season, week = target
    graded = autobet.grade_lines(log, player_stats, schedule)
    done = set(schedule.loc[schedule["home_score"].notna(), "game_id"])
    graded["game_done"] = graded["game_id"].isin(done)
    wk = graded[(graded["season"] == season) & (graded["week"] == week)]
    if wk.empty:
        return None

    res = autobet.evaluate(graded)
    level, message = autobet.verdict(res)
    _, meta = qb_model.load()

    bets = _bet_results(wk[wk["side"] != "none"].copy())
    w, l, p = (bets["result"] == "win").sum(), (bets["result"] == "loss").sum(), (bets["result"] == "push").sum()
    staked = bets.loc[bets["result"].isin(["win", "loss", "push"]), "stake"].sum()
    profit = bets["profit"].sum()
    title = f"Auto-bet report: {season} week {week} — {w}-{l}" + (f"-{p}" if p else "") + f", ${profit:+,.0f}"

    icon = {"info": "⏳", "success": "✅", "warning": "⚠️", "error": "❌"}[level]
    lines = [f"{MENTION} here's how the model's paper bets did in **{season} week {week}**.", "",
             f"## {icon} Verdict", message, ""]

    lines += [f"## Week {week}",
              f"**{w}-{l}-{p}**, profit **${profit:+,.2f}** on ${staked:,.0f} staked"
              + (f" (ROI {profit / staked:+.1%})" if staked else "") + ".", ""]
    if not bets.empty:
        bets = bets.sort_values("profit", ascending=False)
        lines.append(_table(pd.DataFrame({
            "QB": bets["qb"], "Bet": bets["side"].str.title() + " " + bets["line"].map("{:g}".format),
            "Odds": bets["odds"].map(lambda o: f"{o:+.0f}"), "Projection": bets["projection"].map("{:.0f}".format),
            "Model chance": np.where(bets["side"] == "over", bets["p_over"], 1 - bets["p_over"]),
            "Actual": bets["actual"], "Result": bets["result"], "Profit": bets["profit"],
        }), {"Model chance": "{:.0%}".format, "Actual": "{:.0f}".format, "Profit": "${:+.0f}".format}))
        lines.append("")
    no_bet = (wk["side"] == "none").sum()
    starters = 2 * schedule[(schedule["season"] == season) & (schedule["week"] == week)].shape[0]
    lines += [f"{no_bet} more lines were logged without a bet (under +{autobet.MIN_EV:.0%} expected value). "
              f"Lines were captured for {len(wk)} of {starters} starters.", ""]
    if len(wk) < 0.8 * starters:
        lines += ["> ⚠️ Fewer lines than expected were logged this week. Check the "
                  "[Actions runs](https://github.com/shannoj/ff_data_science_project/actions) and Odds API credits.", ""]

    strat = res["strategies"]
    if not strat.empty:
        lines += ["## Season to date", ""]
        s = strat.reset_index(names="Strategy")
        s["Record"] = s["wins"].astype(int).astype(str) + "-" + s["losses"].astype(int).astype(str) + "-" + s["pushes"].astype(int).astype(str)
        lines.append(_table(s[["Strategy", "Record", "win_rate", "breakeven", "profit", "roi", "p_luck"]].rename(columns={
            "win_rate": "Win rate", "breakeven": "Break-even", "profit": "Profit", "roi": "ROI", "p_luck": "Chance it's luck"}),
            {"Win rate": "{:.1%}".format, "Break-even": "{:.1%}".format, "Profit": "${:+,.0f}".format,
             "ROI": "{:+.1%}".format, "Chance it's luck": "{:.0%}".format}))
        acc = res["accuracy"]
        lines += ["", f"**Model vs. sportsbook line** ({acc['n']} settled lines): the model's projection missed by "
                  f"{acc['model_mae']:.1f} yds on average vs. {acc['line_mae']:.1f} for the line, and was closer "
                  f"{acc['model_closer_pct']:.0%} of the time.", "",
                  "**Calibration** (when the model says a side hits X%, how often it actually did):", ""]
        lines.append(_table(res["calibration"].rename(columns={
            "bucket": "Model chance", "lines": "Lines", "model_said": "Model said", "actually_hit": "Actually hit"}),
            {"Model said": "{:.0%}".format, "Actually hit": "{:.0%}".format}))
        lines.append("")

    tt = meta["trained_through"]
    lines += ["---", f"Model retrained through {tt['season']} week {tt['week']} "
              f"(test MAE {meta['test_metrics']['mae']:.1f} yds). [Full bet log]({LOG_URL}). "
              "Older reports are closed automatically."]
    return title, "\n".join(lines)


if __name__ == "__main__":
    out = build_report()
    if out is not None:
        title, body = out
        Path(sys.argv[1] if len(sys.argv) > 1 else "report.md").write_text(body)
        print(title)
