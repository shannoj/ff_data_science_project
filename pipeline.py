"""Weekly pipeline: pull the latest data, retrain, and project the next week.

    python pipeline.py              # retrain + project next week (run weekly, e.g. Tuesday)
    python pipeline.py --no-train   # project with the saved model
    python pipeline.py --week 12    # project a specific week of the current season
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import model as qb_model
from data import load_all, next_week
from features import build_dataset
from weather import add_forecasts

PREDICTIONS_DIR = Path(__file__).parent / "predictions"

OUTPUT_COLS = [
    "season", "week", "game_id", "gameday", "gametime", "team", "opponent", "is_home",
    "player_id", "player_display_name", "spread", "total_line", "implied_total",
    "temp", "wind", "projection", "low", "high",
]


def project_week(data, model, meta, season, week, weather=True):
    """Projections for every unplayed game in (season, week). `data` is a data.NFLData."""
    dataset = build_dataset(data, predict_week=(season, week))
    upcoming = dataset[~dataset["completed"].astype(bool)
                       & (dataset["season"] == season) & (dataset["week"] == week)]
    if weather:
        upcoming = add_forecasts(upcoming)
    out = qb_model.predict(model, meta, upcoming)
    return out.sort_values("projection", ascending=False).reset_index(drop=True)


def save_predictions(proj, meta, season, week):
    """Write predictions/<season>_week<NN>.csv. Rows already saved for games
    that have since kicked off are kept, so the file is a true pre-game record."""
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    path = PREDICTIONS_DIR / f"{season}_week{week:02d}.csv"
    out = proj[OUTPUT_COLS].copy()
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["model_trained_through"] = "{season}-wk{week}".format(**meta["trained_through"])
    if path.exists():
        old = pd.read_csv(path)
        kept = old[~old["game_id"].isin(out["game_id"])]
        if not kept.empty:
            out = pd.concat([kept, out], ignore_index=True)
    out.to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-train", action="store_true", help="use the saved model")
    ap.add_argument("--week", type=int, help="week of the current season to project")
    args = ap.parse_args()

    print("Loading nflverse data...")
    data = load_all()
    schedule = data.schedule

    if args.no_train:
        model, meta = qb_model.load()
    else:
        print("Training...")
        model, meta = qb_model.train(build_dataset(data))
        qb_model.save(model, meta)
        t, b = meta["test_metrics"], meta["baseline_metrics"]
        print(f"  {meta['n_trees']} trees, trained through {meta['trained_through']}")
        print(f"  Test {meta['split']['test']}: MAE {t['mae']:.1f}  RMSE {t['rmse']:.1f}  R² {t['r2']:.3f}  (n={t['n']})")
        print(f"  Baseline (8-game avg): MAE {b['mae']:.1f}  RMSE {b['rmse']:.1f}  R² {b['r2']:.3f}")

    target = (int(schedule["season"].max()), args.week) if args.week else next_week(schedule)
    if target is None:
        print("No unplayed games left this season.")
        return
    season, week = target
    proj = project_week(data, model, meta, season, week)
    path = save_predictions(proj, meta, season, week)

    print(f"\n{season} week {week} projections -> {path}")
    show = proj[["player_display_name", "team", "opponent", "spread", "total_line", "projection", "low", "high"]]
    print(show.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
