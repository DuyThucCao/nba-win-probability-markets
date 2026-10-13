"""Turn the cached season pages into one clean game-level table.

Each archive page lists two rows per game (visitor first, then home).
This module pairs the rows, builds calendar dates, normalizes team names
across relocations, and converts closing moneylines into market-implied
win probabilities with the bookmaker margin (vig) removed.
"""
from io import StringIO
from pathlib import Path

import json

import numpy as np
import pandas as pd

from schedule import all_schedules

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "games.csv"
DATE_AUDIT = ROOT / "results" / "date_audit.json"

# Keep one identity per franchise so ratings carry across relocations.
TEAM_ALIASES = {"NewJersey": "Brooklyn", "Seattle": "OklahomaCity"}


def normalize_team(name: str) -> str:
    name = str(name).replace(" ", "")
    return TEAM_ALIASES.get(name, name)


def moneyline_to_prob(ml: pd.Series) -> pd.Series:
    """Raw implied probability of an American moneyline (includes vig)."""
    ml = pd.to_numeric(ml, errors="coerce")
    return pd.Series(
        np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100)), index=ml.index
    )


def to_number(value) -> float:
    if isinstance(value, str) and value.strip().lower() in {"pk", "pick"}:
        return 0.0
    return pd.to_numeric(value, errors="coerce")


def signed_home_spread(away_cell, home_cell) -> float:
    """Home spread (negative when home is favored) from one pair of cells."""
    away_value, home_value = to_number(away_cell), to_number(home_cell)
    if home_value < away_value:
        return -home_value
    return away_value


def parse_season(path: Path) -> pd.DataFrame:
    season = path.stem.replace("nba_odds_", "")
    start_year = int(season[:4])
    table = pd.read_html(StringIO(path.read_text(errors="ignore")))[0]
    table.columns = list(table.iloc[0])
    table = table.iloc[1:].reset_index(drop=True)

    games, year, prev_month = [], start_year, None
    for i in range(0, len(table) - 1, 2):
        away, home = table.iloc[i], table.iloc[i + 1]
        mmdd = int(away["Date"])
        month, day = divmod(mmdd, 100)
        # Rows are chronological, so a big drop in month means a new year.
        if prev_month is not None and month < prev_month - 6:
            year += 1
        prev_month = month

        # The favorite's cell holds the spread, the other holds the total.
        home_spread_open = signed_home_spread(away["Open"], home["Open"])
        home_spread_close = signed_home_spread(away["Close"], home["Close"])

        games.append(
            {
                "season": season,
                "date": pd.Timestamp(year=year, month=month, day=day),
                "away_team": normalize_team(away["Team"]),
                "home_team": normalize_team(home["Team"]),
                "away_pts": int(away["Final"]),
                "home_pts": int(home["Final"]),
                "away_ml": pd.to_numeric(away["ML"], errors="coerce"),
                "home_ml": pd.to_numeric(home["ML"], errors="coerce"),
                "home_spread_open": home_spread_open,
                "home_spread_close": home_spread_close,
                "neutral": away["VH"] == "N" or home["VH"] == "N",
            }
        )
    return pd.DataFrame(games)


def apply_official_dates(games: pd.DataFrame) -> pd.DataFrame:
    """Replace archive dates with Basketball-Reference schedule dates.

    Games are matched on season, both teams, and both final scores, which
    does not depend on the date being right. Neutral-site games may list
    the teams in either order, so a swapped match is tried as a fallback.
    """
    sched = all_schedules(sorted(games["season"].unique()))
    keys = ["season", "away_team", "home_team", "away_pts", "home_pts"]
    games = games.reset_index(drop=True).assign(row=lambda d: np.arange(len(d)))

    def match(left, right):
        m = left.merge(right.rename(columns={"date": "official_date"}), on=keys, how="inner")
        m["gap"] = (m["official_date"] - m["date"]).dt.days.abs()
        return m.sort_values("gap").drop_duplicates("row")

    direct = match(games, sched)
    swapped_sched = sched.rename(columns={"away_team": "home_team", "home_team": "away_team",
                                          "away_pts": "home_pts", "home_pts": "away_pts"})
    rest = games[~games["row"].isin(direct["row"])]
    swapped = match(rest, swapped_sched)
    matched = pd.concat([direct, swapped])
    matched = matched[matched["gap"] <= 3]          # never move a game by more than 3 days

    # Same teams on the same date but a different final score: the archive
    # mistyped the score, so take the official score (and keep the odds).
    leftover = games[~games["row"].isin(matched["row"])]
    pair_keys = ["season", "date", "away_team", "home_team"]
    fixed = leftover.drop(columns=["away_pts", "home_pts"]).merge(sched, on=pair_keys, how="inner")
    fixed = fixed.drop_duplicates("row")
    score_fix = fixed.set_index("row")[["away_pts", "home_pts"]]
    if len(score_fix):
        games.loc[games["row"].isin(score_fix.index), ["away_pts", "home_pts"]] = (
            score_fix.loc[games.loc[games["row"].isin(score_fix.index), "row"]].to_numpy())
        matched = pd.concat([matched, fixed.assign(official_date=fixed["date"], gap=0)])

    official = matched.set_index("row")["official_date"]
    games["date_source"] = np.where(games["row"].isin(official.index), "basketball-reference", "archive")
    moved = games["row"].isin(official.index) & (games["row"].map(official) != games["date"])
    audit = {
        "games": int(len(games)),
        "matched_to_basketball_reference": int(games["row"].isin(official.index).sum()),
        "dates_corrected": int(moved.sum()),
        "scores_corrected": int(len(score_fix)),
        "unmatched_kept_archive_date": int((games["date_source"] == "archive").sum()),
        "corrected_examples": games.loc[moved].head(10).assign(
            official=lambda d: d["row"].map(official).dt.date.astype(str),
            archive=lambda d: d["date"].dt.date.astype(str),
        )[["season", "away_team", "home_team", "archive", "official"]].to_dict(orient="records"),
    }
    games.loc[games["row"].isin(official.index), "date"] = games["row"].map(official)
    return games.drop(columns="row"), audit


def check_schedule(games: pd.DataFrame) -> None:
    """Fail loudly if any team appears in two games on the same date."""
    long = pd.concat([games[["date", "home_team"]].set_axis(["date", "team"], axis=1),
                      games[["date", "away_team"]].set_axis(["date", "team"], axis=1)])
    dupes = long[long.duplicated(keep=False)]
    if len(dupes):
        raise ValueError(f"Team plays twice on one date:\n{dupes.head()}")


def build_games(official_dates: bool = True) -> pd.DataFrame:
    frames = [parse_season(p) for p in sorted(RAW_DIR.glob("nba_odds_*.html"))]
    games = pd.concat(frames, ignore_index=True)
    if official_dates:
        games, audit = apply_official_dates(games)
        DATE_AUDIT.parent.mkdir(parents=True, exist_ok=True)
        DATE_AUDIT.write_text(json.dumps(audit, indent=2))
        check_schedule(games)
    games = games.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
    games["game_id"] = np.arange(len(games))
    games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)

    raw_home = moneyline_to_prob(games["home_ml"])
    raw_away = moneyline_to_prob(games["away_ml"])
    games["overround"] = raw_home + raw_away
    games["p_market"] = raw_home / games["overround"]
    # Flag lines that cannot be real two-way prices (typos in the archive).
    games["market_ok"] = games["overround"].between(1.0, 1.12)
    return games


if __name__ == "__main__":
    games = build_games()
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(PROCESSED, index=False)
    print(games.shape, games["market_ok"].mean().round(4))
    print(games.groupby("season").size())
