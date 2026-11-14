from pathlib import Path

import pandas as pd
import streamlit as st

import model as qb_model
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

tab_week, tab_record, tab_model = st.tabs(["This week", "Track record", "About the model"])

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
    st.markdown(f"Trained through **{tt['season']} week {tt['week']}** on {meta['n_trees']} trees "
                f"(train {meta['split']['train']}, early-stopped on {meta['split']['valid']}).")
    st.markdown(f"**Out-of-sample test, {meta['split']['test']} season ({t['n']} QB games)**")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{t['mae']:.1f} yds", f"{t['mae'] - b['mae']:+.1f} vs 8-game average", delta_color="inverse")
    c2.metric("RMSE", f"{t['rmse']:.1f} yds", f"{t['rmse'] - b['rmse']:+.1f} vs 8-game average", delta_color="inverse")
    c3.metric("R²", f"{t['r2']:.3f}", f"{t['r2'] - b['r2']:+.3f} vs 8-game average")
    st.markdown("Single-game passing yards are very noisy, so even good models have a typical miss of 50+ yards. "
                "Every feature is computed only from information available before kickoff.")
    st.markdown("**What the model leans on**")
    imp = pd.Series(meta["importance"]).head(15).sort_values()
    st.bar_chart(imp, horizontal=True)
