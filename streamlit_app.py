from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import autobet
import kalshi
import model as qb_model
from bets import payout_multiple, expected_value, get_store, grade, implied_prob, new_bet, summarize
from data import load_all, next_week
from pipeline import PREDICTIONS_DIR, project_week

st.set_page_config(page_title="NFL QB Passing Yards", layout="wide")


@st.cache_resource
def get_model():
    return qb_model.load()


@st.cache_data(ttl=3 * 3600, show_spinner="Pulling the latest NFL data...")
def get_data():
    return load_all()


@st.cache_data(ttl=3 * 3600, show_spinner="Projecting this week's games...")
def get_projections(season, week):
    player_stats, schedule = get_data()
    model, meta = get_model()
    return project_week(player_stats, schedule, model, meta, season, week)


@st.cache_data(ttl=3 * 3600)
def get_track_record():
    """Saved weekly predictions joined to what actually happened."""
    files = sorted(Path(PREDICTIONS_DIR).glob("*.csv"))
    if not files:
        return pd.DataFrame()
    preds = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    player_stats, _ = get_data()
    actual = player_stats[["player_id", "game_id", "passing_yards"]]
    return preds.merge(actual, on=["player_id", "game_id"], how="inner")


model, meta = get_model()
player_stats, schedule = get_data()
target = next_week(schedule)

st.title("NFL Quarterback Passing Yards")
st.caption("Weekly projections from an XGBoost model on nflverse data, Vegas lines and kickoff weather.")

tab_week, tab_exp, tab_bets, tab_record, tab_model = st.tabs(
    ["This week", "Auto-bet experiment", "My bets", "Track record", "About the model"])

with tab_week:
    if target is None:
        st.info("No unplayed games left this season.")
    else:
        season, week = target
        proj = get_projections(season, week)
        st.subheader(f"{season} Week {week}")

        table = pd.DataFrame({
            "QB": proj["player_display_name"],
            "Team": proj["team"],
            "Matchup": proj.apply(lambda r: ("vs " if r["is_home"] else "@ ") + r["opponent"], axis=1),
            "Kickoff": pd.to_datetime(proj["gameday"] + " " + proj["gametime"]).dt.strftime("%a %b %d %-I:%M %p"),
            "Spread": proj["spread"],
            "Team total": proj["implied_total"],
            "Projection": proj["projection"].round(0),
            "80% range": proj.apply(lambda r: f"{r['low']:.0f}–{r['high']:.0f}", axis=1),
        })
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={
                "Spread": st.column_config.NumberColumn(format="%+.1f", help="Positive = team favored"),
                "Team total": st.column_config.NumberColumn(format="%.1f", help="Vegas implied points"),
                "Projection": st.column_config.ProgressColumn(
                    format="%d yds", min_value=0, max_value=float(proj["high"].max())),
            },
        )
        st.caption("Starters come from the nflverse schedule and fall back to each team's most recent starter. "
                   "Weather is the Open-Meteo forecast at kickoff for outdoor stadiums.")

        st.markdown("#### Why this projection?")
        qb = st.selectbox("Quarterback", proj["player_display_name"])
        r = proj[proj["player_display_name"] == qb].iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Recent form**")
            st.metric("Yards/game, last 3", f"{r['qb_ypg_3']:.0f}" if pd.notna(r["qb_ypg_3"]) else "—")
            st.metric("Yards/game, last 8", f"{r['qb_ypg_8']:.0f}" if pd.notna(r["qb_ypg_8"]) else "—")
            st.metric("EPA/dropback, last 8", f"{r['qb_epa_db_8']:+.2f}" if pd.notna(r["qb_epa_db_8"]) else "—")
        with c2:
            st.markdown(f"**{r['opponent']} pass defense**")
            st.metric("Yards allowed/game, last 5", f"{r['def_ypg_5']:.0f}" if pd.notna(r["def_ypg_5"]) else "—")
            st.metric("Yards/attempt allowed, last 10", f"{r['def_ypa_10']:.2f}" if pd.notna(r["def_ypa_10"]) else "—")
            st.metric("Sack rate, last 10", f"{r['def_sack_rate_10']:.1%}" if pd.notna(r["def_sack_rate_10"]) else "—")
        with c3:
            st.markdown("**Game context**")
            st.metric("Vegas team total", f"{r['implied_total']:.1f}" if pd.notna(r["implied_total"]) else "—")
            venue = {0.0: "Outdoors", 0.5: "Retractable roof", 1.0: "Dome"}.get(r["roof_type"], "—")
            weather = f"{r['temp']:.0f}°F, {r['wind']:.0f} mph wind" if pd.notna(r["temp"]) and r["roof_type"] == 0 else venue
            st.metric("Conditions", weather)
            st.metric("Projection", f"{r['projection']:.0f} yds", help=f"80% range {r['low']:.0f}–{r['high']:.0f}")
        if r["qb_games_prior"] < 3:
            st.warning(f"{qb} has only {int(r['qb_games_prior'])} prior starts in the data, so this projection is shaky.")

with tab_record:
    record = get_track_record()
    if record.empty:
        st.info("No graded predictions yet. Each run of `python pipeline.py` saves that week's "
                "projections, and they're graded here once the games are played.")
    else:
        record["error"] = record["passing_yards"] - record["projection"]
        record["in_range"] = record["passing_yards"].between(record["low"], record["high"])
        weekly = record.groupby(["season", "week"]).agg(
            games=("error", "size"),
            mae=("error", lambda e: e.abs().mean()),
            bias=("error", "mean"),
            in_range=("in_range", "mean"),
        ).reset_index()
        c1, c2, c3 = st.columns(3)
        c1.metric("Graded projections", len(record))
        c2.metric("Average miss (MAE)", f"{record['error'].abs().mean():.1f} yds")
        c3.metric("Inside 80% range", f"{record['in_range'].mean():.0%}")
        st.dataframe(weekly, hide_index=True, width="stretch", column_config={
            "mae": st.column_config.NumberColumn("MAE", format="%.1f"),
            "bias": st.column_config.NumberColumn("Bias (actual − proj)", format="%+.1f"),
            "in_range": st.column_config.NumberColumn("In 80% range", format="percent"),
        })
        st.dataframe(record[["season", "week", "player_display_name", "team", "opponent",
                             "projection", "low", "high", "passing_yards", "error"]].round(1),
                     hide_index=True, width="stretch")

with tab_model:
    t, b = meta["test_metrics"], meta["baseline_metrics"]
    tt = meta["trained_through"]
    st.markdown(f"The model is the **{meta.get('model_type', 'XGBoost')}**. The live version is trained on "
                f"**every completed game from 2020 through {tt['season']} week {tt['week']}** and retrained each Tuesday.")
    st.markdown(f"To score it honestly, a separate copy is trained on {meta['split']['train']}, tuned on "
                f"{meta['split']['valid']}, and tested on {meta['split']['test']}, which it never saw:")
    st.markdown(f"**Test results, {meta['split']['test']} season ({t['n']} QB games)**")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{t['mae']:.1f} yds", f"{t['mae'] - b['mae']:+.1f} vs 8-game average", delta_color="inverse")
    c2.metric("RMSE", f"{t['rmse']:.1f} yds", f"{t['rmse'] - b['rmse']:+.1f} vs 8-game average", delta_color="inverse")
    c3.metric("R²", f"{t['r2']:.3f}", f"{t['r2'] - b['r2']:+.3f} vs 8-game average")
    st.markdown("Single-game passing yards are very noisy, so even good models have a typical miss of 50+ yards. "
                "Every feature is computed only from information available before kickoff.")
    st.markdown("**What the tree model leans on** (XGBoost part)")
    imp = pd.Series(meta["importance"]).head(15).sort_values()
    st.bar_chart(imp, horizontal=True)

with tab_bets:
    store = get_store(st.secrets)
    st.caption(f"Hypothetical bets on QB passing-yard props, saved to {store.label}. "
               "Enter the line and odds from your sportsbook; bets are graded automatically once games finish.")
    if not store.persistent_on_cloud:
        st.caption("On Streamlit Cloud a local file is wiped whenever the app restarts. "
                   "Add a GitHub token to the app's secrets to keep bets permanently (see `bets.py`).")

    if target is not None:
        proj = get_projections(*target)
        st.markdown("#### Log a bet")
        bet_qb = st.selectbox("Quarterback", proj["player_display_name"], key="bet_qb")
        r = proj[proj["player_display_name"] == bet_qb].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        line = c1.number_input("Line (yards)", min_value=0.0, max_value=600.0,
                               value=float(round(r["projection"]) + 0.5), step=1.0, key=f"line_{bet_qb}")
        over_odds = c2.number_input("Over odds", value=-110, step=5, key="over_odds")
        under_odds = c3.number_input("Under odds", value=-110, step=5, key="under_odds")
        stake = c4.number_input("Stake ($)", min_value=1.0, value=10.0, step=5.0, key="stake")

        valid_odds = all(o <= -100 or o >= 100 for o in (over_odds, under_odds))
        if not valid_odds:
            st.error("American odds must be ≤ -100 or ≥ +100.")
        else:
            p_over = qb_model.prob_over(meta, r["projection"], line)
            sides = {
                "over": (p_over, over_odds, expected_value(p_over, over_odds)),
                "under": (1 - p_over, under_odds, expected_value(1 - p_over, under_odds)),
            }
            best = max(sides, key=lambda s: sides[s][2])
            m1, m2, m3 = st.columns(3)
            m1.metric("Model projection", f"{r['projection']:.0f} yds")
            for col, s in ((m2, "over"), (m3, "under")):
                p, o, ev = sides[s]
                col.metric(f"{s.title()} {line:g}: model chance", f"{p:.0%}",
                           f"{ev:+.1%} EV (break-even {implied_prob(o):.0%})",
                           delta_color="normal" if ev > 0 else "inverse" if ev < 0 else "off")
            if sides[best][2] > 0:
                st.success(f"Model leans **{best} {line:g}** (+{sides[best][2]:.1%} expected return per $1).")
            else:
                st.info("Neither side has positive expected value by the model at these odds.")

            side = st.radio("Side", ["over", "under"], index=["over", "under"].index(best), horizontal=True)
            notes = st.text_input("Notes (optional)", key="bet_notes")
            if st.button("Log bet", type="primary"):
                p, o, _ = sides[side]
                log = store.load()
                bet = new_bet(r, line, side, o, stake, p, notes)
                log = pd.DataFrame([bet]) if log.empty else pd.concat([log, pd.DataFrame([bet])], ignore_index=True)
                store.save(log, f"Log bet: {bet['qb']} {side} {line:g} ({o:+d})")
                st.success(f"Logged ${stake:.0f} on {bet['qb']} {side} {line:g} at {o:+d}.")

    st.markdown("#### Results")
    log = store.load()
    if log.empty:
        st.info("No bets logged yet.")
    else:
        graded = grade(log, player_stats, schedule)
        s = summarize(graded)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Record (W-L-P)", f"{s['wins']}-{s['losses']}-{s['pushes']}")
        c2.metric("Profit", f"${s['profit']:+,.2f}")
        c3.metric("ROI", f"{s['roi']:+.1%}", help="Profit ÷ amount staked on settled bets")
        c4.metric("Pending", s["pending"])

        settled = graded[graded["result"].isin(["win", "loss", "push"])].sort_values(["gameday", "placed_at"])
        if len(settled) >= 2:
            st.line_chart(settled.assign(**{"Cumulative profit ($)": settled["profit"].cumsum()})
                          .reset_index(drop=True)["Cumulative profit ($)"])
            with_model = settled["model_prob"] > 0.5
            split = pd.DataFrame({
                "Bets": ["Model agreed (>50%)", "Model disagreed"],
                "Count": [with_model.sum(), (~with_model).sum()],
                "Profit": [settled.loc[with_model, "profit"].sum(), settled.loc[~with_model, "profit"].sum()],
            })
            st.dataframe(split, hide_index=True, column_config={"Profit": st.column_config.NumberColumn(format="$%+.2f")})

        show = graded.sort_values("placed_at", ascending=False)
        st.dataframe(
            show[["season", "week", "qb", "opponent", "side", "line", "odds", "stake",
                  "projection", "model_prob", "actual", "result", "profit"]],
            hide_index=True, width="stretch",
            column_config={
                "model_prob": st.column_config.NumberColumn("Model chance", format="percent"),
                "profit": st.column_config.NumberColumn(format="$%+.2f"),
                "stake": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        with st.expander("Remove bets"):
            labels = {b: f"{q} {sd} {ln:g} (wk {wk}, ${stk:.0f})" for b, q, sd, ln, wk, stk in
                      zip(log["bet_id"], log["qb"], log["side"], log["line"], log["week"], log["stake"])}
            drop = st.multiselect("Select bets to remove", list(labels), format_func=labels.get)
            if drop and st.button("Remove selected"):
                store.save(log[~log["bet_id"].isin(drop)], f"Remove {len(drop)} bet(s)")
                st.rerun()

with tab_exp:
    st.markdown("Every day an automated job pulls sportsbook passing-yard lines for games starting in the next "
                f"{autobet.LOOKAHEAD_HOURS} hours and records the model's call on each one. It places a hypothetical "
                f"**${autobet.STAKE:.0f} bet whenever the model sees at least +{autobet.MIN_EV:.0%} expected value**. "
                "Lines are logged before kickoff and never edited, so this is a true out-of-sample test.")
    exp_log = autobet.load_log()
    if exp_log.empty:
        st.info("No lines logged yet. The experiment starts once an Odds API key is added to the GitHub repo "
                "secrets as `ODDS_API_KEY`.")
    else:
        graded = autobet.grade_lines(exp_log, player_stats, schedule)
        res = autobet.evaluate(graded)
        bets_placed = graded[graded["side"] != "none"]
        settled_bets = bets_placed[bets_placed["settled"]]
        strat = res["strategies"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lines logged", len(graded))
        c2.metric("Bets placed", len(bets_placed), f"{(~bets_placed['settled']).sum()} pending", delta_color="off")
        if not strat.empty:
            rule = strat.iloc[0]
            c3.metric("Profit (model rule)", f"${rule['profit']:+,.0f}")
            c4.metric("ROI (model rule)", f"{rule['roi']:+.1%}" if rule["bets"] else "—")

        level, message = autobet.verdict(res)
        getattr(st, level)(message)

        st.markdown("#### Closing line value (CLV)")
        st.caption("How far the line moved toward each bet between placing it and kickoff. Sharp bettors "
                   "consistently get better numbers than the close; this shows an edge after ~100 bets, "
                   "long before profit does. Uses all bets, settled or not.")
        clv = autobet.closing_value(exp_log, autobet.load_closing())
        cs = autobet.clv_summary(clv)
        if cs["n"] == 0:
            st.info("No closing lines captured yet. They're recorded automatically shortly before each kickoff.")
        else:
            v1, v2, v3 = st.columns(3)
            v1.metric("Average CLV", f"{cs['mean']:+.1f} yds",
                      f"± {cs['ci95']:.1f} (95%)" if cs["n"] > 1 else None, delta_color="off")
            v2.metric("Beat the closing line", f"{cs['beat_pct']:.0%}",
                      f"{cs['same_pct']:.0%} same, {cs['worse_pct']:.0%} worse", delta_color="off")
            v3.metric("Bets with a closing line", cs["n"])
            if cs["n"] >= 50 and cs["mean"] - cs["ci95"] > 0:
                st.success("Bets are consistently beating the closing line: a strong sign of a real edge.")
            elif cs["n"] >= 50 and cs["mean"] + cs["ci95"] < 0:
                st.error("The market consistently moves against these bets: the books know something the model doesn't.")


        if not strat.empty:
            st.markdown("#### Strategies on the same lines")
            st.caption("The model rule is the real test. Controls show what blind betting would have done; "
                       "\"Chance it's luck\" is the probability of at least this win rate with no edge at all.")
            st.dataframe(
                strat.reset_index(names="Strategy")[["Strategy", "bets", "wins", "losses", "pushes",
                                                     "win_rate", "breakeven", "profit", "roi", "p_luck"]],
                hide_index=True, width="stretch",
                column_config={
                    "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
                    "breakeven": st.column_config.NumberColumn("Break-even", format="percent"),
                    "profit": st.column_config.NumberColumn("Profit", format="$%+.0f"),
                    "roi": st.column_config.NumberColumn("ROI", format="percent"),
                    "p_luck": st.column_config.NumberColumn("Chance it's luck", format="percent"),
                },
            )

            acc = res["accuracy"]
            st.markdown("#### Model vs. the sportsbook line")
            a1, a2, a3 = st.columns(3)
            a1.metric("Model avg miss", f"{acc['model_mae']:.1f} yds")
            a2.metric("Line avg miss", f"{acc['line_mae']:.1f} yds")
            a3.metric("Model closer to actual", f"{acc['model_closer_pct']:.0%}", help=f"Of {acc['n']} settled lines")
            st.caption("If the line keeps missing by less than the model, the books know something the model doesn't.")

            if acc.get("prob_n"):
                st.markdown("#### Probability accuracy: model vs. market")
                st.caption("Brier score and log loss punish confident wrong calls and reward confident right ones "
                           "(lower is better). The market's probability is the sportsbook's odds with the vig removed. "
                           "Positive skill means the model's probabilities beat the book's.")
                b1, b2, b3 = st.columns(3)
                b1.metric("Brier: model / market", f"{acc['model_brier']:.3f} / {acc['market_brier']:.3f}")
                b2.metric("Log loss: model / market", f"{acc['model_log_loss']:.3f} / {acc['market_log_loss']:.3f}")
                b3.metric("Skill vs. market", f"{acc['prob_skill']:+.1%}", help=f"Of {acc['prob_n']} settled lines")

            st.markdown("#### Calibration")
            st.caption("When the model says a side hits X% of the time, does it? Points well below the "
                       "\"model said\" values mean the model is overconfident.")
            st.dataframe(res["calibration"], hide_index=True, column_config={
                "model_said": st.column_config.NumberColumn("Model said", format="percent"),
                "actually_hit": st.column_config.NumberColumn("Actually hit", format="percent"),
            })

            if len(settled_bets) >= 2:
                sb = settled_bets.sort_values("commence_time").copy()
                won = (sb["actual"] > sb["line"]) == (sb["side"] == "over")
                push = sb["actual"] == sb["line"]
                sb["profit"] = np.where(push, 0.0, np.where(won, sb["stake"] * sb["odds"].map(payout_multiple), -sb["stake"]))
                st.markdown("#### Cumulative profit, model rule")
                st.line_chart(sb["profit"].cumsum().reset_index(drop=True).rename("Profit ($)"))

        st.markdown("#### Bet log")
        log_view = graded.sort_values("commence_time", ascending=False)
        st.dataframe(
            log_view[["week", "qb", "opponent", "book", "line", "over_odds", "under_odds", "n_books",
                      "consensus_line", "projection", "p_over", "side", "odds", "actual"]],
            hide_index=True, width="stretch",
            column_config={"p_over": st.column_config.NumberColumn("Model P(over)", format="percent")},
        )

with tab_exp:
    st.markdown("---")
    st.markdown("### Kalshi (prediction market)")
    st.caption(f"The same model, paper-trading Kalshi's QB passing-yard contracts: the best yes/no contract priced "
               f"{kalshi.MIN_PRICE:.0%}-{kalshi.MAX_PRICE:.0%}, after Kalshi's fee, if expected return is at least "
               f"+{kalshi.MIN_EV:.0%}. Exchanges don't limit winners, but order books are thin: \"fillable\" means "
               f"${kalshi.STAKE:.0f} was actually on offer at that price.")
    k_log = kalshi.load_log()
    if k_log.empty:
        st.info("No Kalshi markets logged yet. The daily job records them for games starting within 36 hours.")
    else:
        kg = kalshi.grade(k_log, player_stats, schedule)
        ks = kalshi.summarize(kg)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Trades", ks["trades"], f"{ks['pending']} pending", delta_color="off")
        k2.metric("Record", f"{ks['wins']}-{ks['losses']}")
        k3.metric("Profit", f"${ks['profit']:+,.0f}", f"ROI {ks['roi']:+.1%}" if ks["wins"] + ks["losses"] else None,
                  delta_color="off")
        k4.metric("Fillable at $100", f"{ks['fillable_pct']:.0%}" if ks["trades"] else "—",
                  help="Share of trades where the full stake was available at the quoted price")
        trades = kg[kg["side"] != "none"].sort_values("kickoff", ascending=False)
        st.dataframe(
            trades.assign(contract=trades["side"].str.upper() + " " + (trades["strike"] + 0.5).map("{:g}+".format))[
                ["week", "qb", "opponent", "contract", "price", "model_p", "ev", "size_available", "fillable",
                 "actual", "result", "profit"]],
            hide_index=True, width="stretch",
            column_config={
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "model_p": st.column_config.NumberColumn("Model chance", format="percent"),
                "ev": st.column_config.NumberColumn("Expected return", format="percent"),
                "size_available": st.column_config.NumberColumn("Contracts on offer", format="%d"),
                "profit": st.column_config.NumberColumn(format="$%+.0f"),
            },
        )

