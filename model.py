"""Train, evaluate, save and load the QB passing-yards model.

The model is the average of two parts that make different kinds of errors:
an XGBoost tree model and a ridge (regularized linear) regression. In a
2023-2026 backtest the average beat either one alone in every season.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import Booster, XGBRegressor

from features import FEATURES, TARGET

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "qb_yards_model.json"
META_PATH = ARTIFACT_DIR / "qb_yards_meta.json"
RIDGE_PATH = ARTIFACT_DIR / "qb_yards_ridge.json"

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
RIDGE_ALPHAS = np.logspace(-2, 4, 30)


class LinearModel:
    """Ridge regression with median imputation, missing-value flags and standardization.

    Kept as plain numbers (saved to JSON) so it loads anywhere without pickles.
    """

    def __init__(self, features, medians, flag_cols, means, scales, coef, intercept, alpha):
        self.features, self.medians, self.flag_cols = features, medians, flag_cols
        self.means, self.scales = np.asarray(means), np.asarray(scales)
        self.coef, self.intercept, self.alpha = np.asarray(coef), float(intercept), float(alpha)

    def _design(self, X):
        X = X[self.features].astype(float)
        flags = X[self.flag_cols].isna().astype(float).to_numpy()
        filled = X.fillna(pd.Series(self.medians)).to_numpy()
        return (np.hstack([filled, flags]) - self.means) / self.scales

    @classmethod
    def fit(cls, X, y, features):
        X = X[features].astype(float)
        medians = X.median().fillna(0).to_dict()
        flag_cols = [c for c in features if X[c].isna().any()]
        raw = np.hstack([X.fillna(pd.Series(medians)).to_numpy(), X[flag_cols].isna().astype(float).to_numpy()])
        means, scales = raw.mean(axis=0), raw.std(axis=0)
        scales[scales == 0] = 1.0
        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit((raw - means) / scales, y)
        return cls(features, medians, flag_cols, means, scales, ridge.coef_, ridge.intercept_, ridge.alpha_)

    def predict(self, X):
        return self._design(X) @ self.coef + self.intercept

    def to_dict(self):
        return {"features": self.features, "medians": self.medians, "flag_cols": self.flag_cols,
                "means": self.means.tolist(), "scales": self.scales.tolist(), "coef": self.coef.tolist(),
                "intercept": self.intercept, "alpha": self.alpha}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class Ensemble:
    """Average of the XGBoost booster and the linear model."""

    def __init__(self, xgb, linear):
        self.xgb, self.linear = xgb, linear

    def predict(self, X, features):
        tree = _booster(self.xgb).inplace_predict(X[features].astype(float))
        return tree if self.linear is None else (tree + self.linear.predict(X)) / 2


def _metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "n": int(len(y)),
    }


def labeled_rows(dataset):
    """Completed games the QB started (relief appearances are form history only)."""
    keep = dataset["completed"].astype(bool) & dataset[TARGET].notna()
    if "is_start" in dataset.columns:
        keep &= dataset["is_start"].astype(bool)
    return dataset[keep]


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

    # The linear part has no early stopping, so it can train on train + valid.
    trva = data[data["season"] < test_season]
    linear = LinearModel.fit(trva, trva[TARGET], FEATURES)
    # Valid-season errors use a linear fit that hasn't seen that season.
    linear_tr = LinearModel.fit(tr, tr[TARGET], FEATURES)

    test_xgb = model.predict(te[FEATURES])
    test_pred = (test_xgb + linear.predict(te)) / 2
    valid_pred = (model.predict(va[FEATURES]) + linear_tr.predict(va)) / 2
    baseline = te["qb_ypg_8"].fillna(tr[TARGET].mean())
    holdout = np.concatenate([va[TARGET] - valid_pred, te[TARGET] - test_pred])

    # Final model: all completed games (a few more trees for the extra data).
    xgb_final = XGBRegressor(n_estimators=int(n_trees * 1.1), **PARAMS)
    xgb_final.fit(data[FEATURES], data[TARGET], verbose=False)
    final = Ensemble(xgb_final, LinearModel.fit(data, data[TARGET], FEATURES))

    last = data.sort_values(["season", "week"]).iloc[-1]
    importance = dict(zip(FEATURES, xgb_final.feature_importances_.astype(float)))
    meta = {
        "model_type": "average of XGBoost and ridge regression",
        "features": FEATURES,
        "params": PARAMS,
        "n_trees": int(xgb_final.n_estimators),
        "ridge_alpha": final.linear.alpha,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_through": {"season": int(last["season"]), "week": int(last["week"])},
        "split": {
            "train": f"{int(tr['season'].min())}-{int(tr['season'].max())}",
            "valid": int(valid_season),
            "test": f"{int(test_season)}+",
        },
        "test_metrics": _metrics(te[TARGET], test_pred),
        "test_metrics_xgb_only": _metrics(te[TARGET], test_xgb),
        "baseline_metrics": _metrics(te[TARGET], baseline),
        "interval": {"low": float(np.quantile(holdout, INTERVAL[0])),
                     "high": float(np.quantile(holdout, INTERVAL[1]))},
        # Percentiles of (actual - projection) on held-out seasons; used for over/under odds.
        "residual_quantiles": [round(float(q), 2) for q in np.quantile(holdout, np.linspace(0.005, 0.995, 100))],
        "importance": dict(sorted(importance.items(), key=lambda kv: -kv[1])),  # XGBoost part
    }
    return final, meta


def _booster(model):
    return model.get_booster() if isinstance(model, XGBRegressor) else model


def save(model, meta):
    # Save the raw booster: XGBRegressor.save_model breaks under scikit-learn 1.8.
    ARTIFACT_DIR.mkdir(exist_ok=True)
    _booster(model.xgb).save_model(MODEL_PATH)
    RIDGE_PATH.write_text(json.dumps(model.linear.to_dict(), indent=1))
    META_PATH.write_text(json.dumps(meta, indent=2))


def load():
    booster = Booster()
    booster.load_model(MODEL_PATH)
    linear = LinearModel.from_dict(json.loads(RIDGE_PATH.read_text())) if RIDGE_PATH.exists() else None
    return Ensemble(booster, linear), json.loads(META_PATH.read_text())


def predict(model, meta, rows):
    """Add `projection`, `low` and `high` (80% range) columns."""
    out = rows.copy()
    if not isinstance(model, Ensemble):
        model = Ensemble(model, None)
    out["projection"] = model.predict(out, meta["features"])
    out["low"] = (out["projection"] + meta["interval"]["low"]).clip(lower=0)
    out["high"] = out["projection"] + meta["interval"]["high"]
    return out


def prob_over(meta, projection, line):
    """Model probability that actual yards finish above `line`.

    Uses the empirical distribution of held-out errors, so it reflects how far
    off the model really tends to be rather than assuming a bell curve.
    """
    q = meta.get("residual_quantiles")
    if q is None:  # older artifacts: fall back to a normal approximation
        from math import erf, sqrt
        sigma = (meta["interval"]["high"] - meta["interval"]["low"]) / 2.563
        return 0.5 * (1 - erf((line - projection) / (sigma * sqrt(2))))
    return float(np.mean(np.asarray(q) > line - projection))
