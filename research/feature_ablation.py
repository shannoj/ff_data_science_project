"""Backtest: does adding each play-by-play / injury feature group improve the model?  Run from repo root."""
import sys, warnings; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from data import load_all
from features import build_dataset, BASE_FEATURES, FEATURE_GROUPS, TARGET
from model import PARAMS, LinearModel, labeled_rows

d = labeled_rows(build_dataset(load_all())).reset_index(drop=True)
sets = {"base": BASE_FEATURES}
for g, cols in FEATURE_GROUPS.items():
    sets[f"base + {g}"] = BASE_FEATURES + cols
sets["base + all four"] = BASE_FEATURES + [c for cols in FEATURE_GROUPS.values() for c in cols]

def ens_errors(feats, test):
    tr, va, te = d[d.season < test - 1], d[d.season == test - 1], d[d.season == test]
    x = XGBRegressor(n_estimators=2000, early_stopping_rounds=100, **PARAMS).fit(tr[feats], tr[TARGET], eval_set=[(va[feats], va[TARGET])], verbose=False)
    trva = pd.concat([tr, va]); lin = LinearModel.fit(trva, trva[TARGET], feats)
    return np.abs(te[TARGET].values - (x.predict(te[feats]) + lin.predict(te)) / 2)

res = {name: {t: ens_errors(f, t) for t in (2023, 2024, 2025, 2026)} for name, f in sets.items()}
base_all = np.concatenate(list(res["base"].values()))
rows = []
for name, by in res.items():
    allerr = np.concatenate(list(by.values())); diff = allerr - base_all
    rows.append({"features": name, **{t: e.mean() for t, e in by.items()}, "all": allerr.mean(),
                 "vs base (95% CI)": f"{diff.mean():+.2f} ± {1.96 * diff.std(ddof=1) / np.sqrt(len(diff)):.2f}"})
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).set_index("features").round(2).to_string()); print("n =", len(base_all))

print("\nSubsets (base vs base + injuries):")
te_all = pd.concat([d[d.season == t] for t in (2023, 2024, 2025, 2026)])
inj_all = np.concatenate(list(res["base + injuries"].values()))
for label, mask in [("receivers out >= 20% of targets", te_all.missing_target_share >= 0.2),
                    ("receivers out >= 35% of targets", te_all.missing_target_share >= 0.35),
                    ("QB listed Questionable", te_all.qb_questionable == 1),
                    ("2+ offensive linemen out", te_all.ol_out >= 2)]:
    m = mask.values; diff = inj_all[m] - base_all[m]
    print(f"  {label:34s} n={m.sum():4d}  base MAE {base_all[m].mean():5.1f}  +injuries {inj_all[m].mean():5.1f}  diff {diff.mean():+.2f} ± {1.96*diff.std(ddof=1)/np.sqrt(m.sum()):.2f}")
