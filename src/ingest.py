"""Download and cache the season-level NBA scores and odds archive.

Source: sportsbookreviewsonline.com NBA scores and odds archive
(2007-08 through part of 2022-23). Each season page is saved once to
data/raw/ so later runs work offline and never hit the site again.
"""
from pathlib import Path
import ssl
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BASE_URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{season}"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2007, 2023)]


def download_season(season: str, pause: float = 2.0) -> Path:
    """Fetch one season page unless it is already cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"nba_odds_{season}.html"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    try:
        ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    except FileNotFoundError:
        ctx = ssl.create_default_context()
    request = urllib.request.Request(
        BASE_URL.format(season=season), headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, context=ctx, timeout=60) as response:
        path.write_bytes(response.read())
    time.sleep(pause)  # be polite to the host
    return path


def download_all() -> list[Path]:
    return [download_season(season) for season in SEASONS]


if __name__ == "__main__":
    for p in download_all():
        print(p.name, p.stat().st_size)
