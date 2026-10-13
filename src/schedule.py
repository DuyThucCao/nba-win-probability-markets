"""Official game dates from Basketball-Reference season schedules.

The odds archive occasionally records a game on the wrong calendar day
(for example Minnesota at the Lakers on 2020-12-27 appears as 12-28).
Wrong dates corrupt rest-day features, so every archive game is matched
to Basketball-Reference by season, teams, and final score, and the
schedule date replaces the archive date. Pages are cached in data/raw/bref/.
"""
from io import StringIO
from pathlib import Path
import re
import ssl
import time
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BREF_DIR = ROOT / "data" / "raw" / "bref"
BASE = "https://www.basketball-reference.com"

BREF_TO_ARCHIVE = {
    "Atlanta Hawks": "Atlanta", "Boston Celtics": "Boston", "Brooklyn Nets": "Brooklyn",
    "New Jersey Nets": "Brooklyn", "Charlotte Bobcats": "Charlotte", "Charlotte Hornets": "Charlotte",
    "Chicago Bulls": "Chicago", "Cleveland Cavaliers": "Cleveland", "Dallas Mavericks": "Dallas",
    "Denver Nuggets": "Denver", "Detroit Pistons": "Detroit", "Golden State Warriors": "GoldenState",
    "Houston Rockets": "Houston", "Indiana Pacers": "Indiana", "Los Angeles Clippers": "LAClippers",
    "LA Clippers": "LAClippers", "Los Angeles Lakers": "LALakers", "Memphis Grizzlies": "Memphis",
    "Miami Heat": "Miami", "Milwaukee Bucks": "Milwaukee", "Minnesota Timberwolves": "Minnesota",
    "New Orleans Hornets": "NewOrleans", "New Orleans Pelicans": "NewOrleans",
    "New York Knicks": "NewYork", "Oklahoma City Thunder": "OklahomaCity",
    "Seattle SuperSonics": "OklahomaCity", "Orlando Magic": "Orlando",
    "Philadelphia 76ers": "Philadelphia", "Phoenix Suns": "Phoenix",
    "Portland Trail Blazers": "Portland", "Sacramento Kings": "Sacramento",
    "San Antonio Spurs": "SanAntonio", "Toronto Raptors": "Toronto", "Utah Jazz": "Utah",
    "Washington Wizards": "Washington",
}


def _get(url: str) -> str:
    try:
        ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    except FileNotFoundError:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        return resp.read().decode("utf-8", "ignore")


def _cached(name: str, url: str, pause: float) -> str:
    BREF_DIR.mkdir(parents=True, exist_ok=True)
    path = BREF_DIR / name
    if not path.exists():
        path.write_text(_get(url))
        time.sleep(pause)  # Basketball-Reference allows about 20 requests per minute
    return path.read_text()


def season_schedule(end_year: int, pause: float = 3.5) -> pd.DataFrame:
    index = _cached(f"NBA_{end_year}_games.html", f"{BASE}/leagues/NBA_{end_year}_games.html", pause)
    # Month pages are named like "december" or, in seasons that span two
    # calendar years in the same month (2019-20 bubble), "october-2019".
    months = sorted(set(re.findall(rf"/leagues/NBA_{end_year}_games-([a-z]+(?:-\d{{4}})?)\.html", index)))
    frames = []
    for month in months:
        page = _cached(f"NBA_{end_year}_games-{month}.html",
                       f"{BASE}/leagues/NBA_{end_year}_games-{month}.html", pause)
        table = pd.read_html(StringIO(page), attrs={"id": "schedule"})[0]
        table = table[table["Date"] != "Playoffs"]
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(table["Date"], format="%a, %b %d, %Y"),
            "away_team": table["Visitor/Neutral"].map(BREF_TO_ARCHIVE),
            "home_team": table["Home/Neutral"].map(BREF_TO_ARCHIVE),
            "away_pts": pd.to_numeric(table.iloc[:, 3], errors="coerce"),
            "home_pts": pd.to_numeric(table.iloc[:, 5], errors="coerce"),
        }))
    sched = pd.concat(frames, ignore_index=True).dropna()
    sched[["away_pts", "home_pts"]] = sched[["away_pts", "home_pts"]].astype(int)
    if sched[["away_team", "home_team"]].isna().any().any():
        raise ValueError(f"Unmapped team name in {end_year} schedule")
    return sched


def all_schedules(seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        end_year = int(season[:4]) + 1
        frames.append(season_schedule(end_year).assign(season=season))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    from ingest import SEASONS
    s = all_schedules(SEASONS)
    print(s.groupby("season").size())
