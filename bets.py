"""Hypothetical bet log: storage, grading and profit/loss.

Bets are kept in a CSV. On Streamlit Cloud the local disk is wiped on every
restart, so when a GitHub token is configured the CSV lives on a separate
branch of the repo (default `bet-log`) and is read/written through the GitHub
API. Writing there does not redeploy the app. Without a token, bets are saved
to a local `bets.csv`.
"""
import base64
import io
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

LOCAL_PATH = Path(__file__).parent / "bets.csv"
FILE_NAME = "bets.csv"

COLUMNS = [
    "bet_id", "placed_at", "season", "week", "game_id", "gameday", "player_id", "qb",
    "team", "opponent", "line", "side", "odds", "stake", "projection", "model_prob", "notes",
]


# ---------------------------------------------------------------- odds math

def implied_prob(odds):
    """Break-even win probability for American odds."""
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def payout_multiple(odds):
    """Profit per $1 staked on a win."""
    return 100 / -odds if odds < 0 else odds / 100


def expected_value(prob, odds):
    """Expected profit per $1 staked."""
    return prob * payout_multiple(odds) - (1 - prob)


# ---------------------------------------------------------------- storage

class LocalStore:
    label = "local file (bets.csv)"
    persistent_on_cloud = False

    def __init__(self, path=LOCAL_PATH):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_csv(self.path, dtype={"bet_id": str, "game_id": str, "player_id": str})

    def save(self, df, message=None):
        df[COLUMNS].to_csv(self.path, index=False)


class GitHubStore:
    """Read/write bets.csv on a branch of a GitHub repo via the contents API."""
    persistent_on_cloud = True

    def __init__(self, token, repo, branch="bet-log"):
        self.repo, self.branch = repo, branch
        self.api = f"https://api.github.com/repos/{repo}"
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        self.label = f"GitHub ({repo}, branch `{branch}`)"
        self._sha = None

    def _get(self, url, **kw):
        return requests.get(url, headers=self.headers, timeout=15, **kw)

    def _ensure_branch(self):
        if self._get(f"{self.api}/branches/{self.branch}").status_code == 200:
            return
        repo = self._get(self.api).json()
        base = self._get(f"{self.api}/git/ref/heads/{repo['default_branch']}").json()["object"]["sha"]
        r = requests.post(f"{self.api}/git/refs", headers=self.headers, timeout=15,
                          json={"ref": f"refs/heads/{self.branch}", "sha": base})
        r.raise_for_status()

    def load(self):
        r = self._get(f"{self.api}/contents/{FILE_NAME}", params={"ref": self.branch})
        if r.status_code == 404:
            self._sha = None
            return pd.DataFrame(columns=COLUMNS)
        r.raise_for_status()
        body = r.json()
        self._sha = body["sha"]
        text = base64.b64decode(body["content"]).decode()
        return pd.read_csv(io.StringIO(text), dtype={"bet_id": str, "game_id": str, "player_id": str})

    def save(self, df, message="Update bet log"):
        self._ensure_branch()
        if self._sha is None:
            self.load()  # pick up the current sha if the file appeared since
        payload = {
            "message": message,
            "content": base64.b64encode(df[COLUMNS].to_csv(index=False).encode()).decode(),
            "branch": self.branch,
        }
        if self._sha:
            payload["sha"] = self._sha
        r = requests.put(f"{self.api}/contents/{FILE_NAME}", headers=self.headers, json=payload, timeout=15)
        r.raise_for_status()
        self._sha = r.json()["content"]["sha"]


def get_store(secrets=None):
    """GitHubStore if `secrets` has a [bets] github_token, else LocalStore."""
    cfg = {}
    try:
        cfg = dict(secrets["bets"]) if secrets is not None and "bets" in secrets else {}
    except Exception:  # no secrets file at all
        cfg = {}
    if cfg.get("github_token") and cfg.get("github_repo"):
        return GitHubStore(cfg["github_token"], cfg["github_repo"], cfg.get("branch", "bet-log"))
    return LocalStore()


# ---------------------------------------------------------------- log + grade

def new_bet(row, line, side, odds, stake, model_prob, notes=""):
    """A bet record from a projection row (see pipeline.project_week)."""
    return {
        "bet_id": uuid.uuid4().hex[:10],
        "placed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": int(row["season"]), "week": int(row["week"]),
        "game_id": row["game_id"], "gameday": row["gameday"],
        "player_id": row["player_id"], "qb": row["player_display_name"],
        "team": row["team"], "opponent": row["opponent"],
        "line": float(line), "side": side, "odds": int(odds), "stake": float(stake),
        "projection": round(float(row["projection"]), 1),
        "model_prob": round(float(model_prob), 4), "notes": notes,
    }


def grade(bets, player_stats, schedule):
    """Add `actual`, `result` (win/loss/push/void/pending) and `profit` columns.

    A bet is void (stake returned, like a sportsbook) if the game finished but
    the QB recorded no passing stats.
    """
    if bets.empty:
        return bets.assign(actual=pd.Series(dtype=float), result=pd.Series(dtype=str),
                           profit=pd.Series(dtype=float))
    # Final AND stats published: scores post right away, player stats usually overnight, so a
    # finished game with no stats yet stays pending instead of looking like the QB sat out.
    done = set(schedule.loc[schedule["home_score"].notna(), "game_id"]) & set(player_stats["game_id"])
    actual = player_stats.set_index(["player_id", "game_id"])["passing_yards"]
    out = bets.copy()
    keys = list(zip(out["player_id"], out["game_id"]))
    out["actual"] = [actual.get(k) for k in keys]

    def result(r):
        if r["game_id"] not in done:
            return "pending"
        if pd.isna(r["actual"]):
            return "void"
        if r["actual"] == r["line"]:
            return "push"
        over_hit = r["actual"] > r["line"]
        return "win" if over_hit == (r["side"] == "over") else "loss"

    out["result"] = out.apply(result, axis=1)
    win_profit = out["stake"] * out["odds"].map(payout_multiple)
    out["profit"] = 0.0
    out.loc[out["result"] == "win", "profit"] = win_profit
    out.loc[out["result"] == "loss", "profit"] = -out["stake"]
    out.loc[out["result"] == "pending", "profit"] = float("nan")
    return out


def summarize(graded):
    settled = graded[graded["result"].isin(["win", "loss", "push"])]
    staked = settled["stake"].sum()
    profit = settled["profit"].sum()
    return {
        "wins": int((settled["result"] == "win").sum()),
        "losses": int((settled["result"] == "loss").sum()),
        "pushes": int((settled["result"] == "push").sum()),
        "pending": int((graded["result"] == "pending").sum()),
        "staked": float(staked),
        "profit": float(profit),
        "roi": float(profit / staked) if staked else 0.0,
    }
