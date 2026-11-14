"""Train, evaluate, save and load the QB passing-yards model."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import Booster, XGBRegressor

from features import FEATURES, TARGET

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "qb_yards_model.json"
META_PATH = ARTIFACT_DIR / "qb_yards_meta.json"

# Shallow trees, a slow learning rate, and early stopping on a held-out season.
# The tree count comes from early stopping, so there is nothing to sweep by hand.
PARAMS = dict(
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)
MAX_TREES = 2000
EARLY_STOPPING = 100
INTERVAL = (0.10, 0.90)  # 80% projection range


def _metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "n": int(len(y)),
    }


def labeled_rows(dataset):
    return dataset[dataset["completed"].astype(bool) & dataset[TARGET].notna()]


def train(dataset):
    """Evaluate on a time-based split, then refit on every completed game.

    Train: all seasons before the validation season.
    Valid: the previous season, used for early stopping.
    Test:  the current season so far, never seen while choosing the tree count.
    """
    data = labeled_rows(dataset)
    seasons = sorted(data["season"].unique())
    test_season = seasons[-1]
    if (data["season"] == test_season).sum() < 100:  # too early in the season to test on
        test_season = seasons[-2]
    valid_season = test_season - 1

    tr = data[data["season"] < valid_season]
    va = data[data["season"] == valid_season]
    te = data[data["season"] >= test_season]

    model = XGBRegressor(n_estimators=MAX_TREES, early_stopping_rounds=EARLY_STOPPING, **PARAMS)
    model.fit(tr[FEATURES], tr[TARGET], eval_set=[(va[FEATURES], va[TARGET])], verbose=False)
    n_trees = model.best_iteration + 1

    test_pred = model.predict(te[FEATURES])
    baseline = te["qb_ypg_8"].fillna(tr[TARGET].mean())
    holdout = np.concatenate([va[TARGET] - model.predict(va[FEATURES]), te[TARGET] - test_pred])

    # Final model: all completed games, a few more trees for the extra data.
    final = XGBRegressor(n_estimators=int(n_trees * 1.1), **PARAMS)
    final.fit(data[FEATURES], data[TARGET], verbose=False)

    last = data.sort_values(["season", "week"]).iloc[-1]
    importance = dict(zip(FEATURES, final.feature_importances_.astype(float)))
    meta = {
        "features": FEATURES,
        "params": PARAMS,
        "n_trees": int(final.n_estimators),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_through": {"season": int(last["season"]), "week": int(last["week"])},
        "split": {
            "train": f"{int(tr['season'].min())}-{int(tr['season'].max())}",
            "valid": int(valid_season),
            "test": f"{int(test_season)}+",
        },
        "test_metrics": _metrics(te[TARGET], test_pred),
        "baseline_metrics": _metrics(te[TARGET], baseline),
        "interval": {"low": float(np.quantile(holdout, INTERVAL[0])),
                     "high": float(np.quantile(holdout, INTERVAL[1]))},
        "importance": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    }
    return final, meta


def _booster(model):
    return model.get_booster() if isinstance(model, XGBRegressor) else model


def save(model, meta):
    # Save the raw booster: XGBRegressor.save_model breaks under scikit-learn 1.8.
    ARTIFACT_DIR.mkdir(exist_ok=True)
    _booster(model).save_model(MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2))


def load():
    booster = Booster()
    booster.load_model(MODEL_PATH)
    return booster, json.loads(META_PATH.read_text())


def predict(model, meta, rows):
    """Add `projection`, `low` and `high` (80% range) columns."""
    out = rows.copy()
    out["projection"] = _booster(model).inplace_predict(out[meta["features"]].astype(float))
    out["low"] = (out["projection"] + meta["interval"]["low"]).clip(lower=0)
    out["high"] = out["projection"] + meta["interval"]["high"]
    return out
