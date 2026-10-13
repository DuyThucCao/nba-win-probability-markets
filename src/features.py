"""Pre-game features built only from information available before tip-off.

* Elo ratings with home-court advantage, a margin-of-victory multiplier,
  and partial reversion to the mean between seasons.
* Rolling form: average point margin over the last 10 games.
* Schedule: rest days and back-to-back flags.
* Season-to-date win rate, shrunk toward .500 early in the season.
"""
import numpy as np
import pandas as pd

ELO_MEAN = 1505.0


def elo_win_prob(diff: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def add_elo(games: pd.DataFrame, k: float = 20.0, hfa: float = 100.0,
            carryover: float = 0.75) -> pd.DataFrame:
    """Walk through games in date order and record pre-game Elo ratings."""
    ratings: dict[str, float] = {}
    last_season: dict[str, str] = {}
    pre_home, pre_away, probs = [], [], []

    for row in games.itertuples(index=False):
        for team in (row.home_team, row.away_team):
            rating = ratings.get(team, ELO_MEAN)
            if last_season.get(team) not in (None, row.season):
                rating = carryover * rating + (1 - carryover) * ELO_MEAN
            ratings[team], last_season[team] = rating, row.season

        r_home, r_away = ratings[row.home_team], ratings[row.away_team]
        diff = r_home - r_away + (0.0 if row.neutral else hfa)
        p_home = elo_win_prob(diff)
        pre_home.append(r_home)
        pre_away.append(r_away)
        probs.append(p_home)

        # Margin-of-victory multiplier (FiveThirtyEight style) damps blowouts
        # by heavy favorites so ratings do not overreact.
        margin = row.home_pts - row.away_pts
        winner_diff = diff if margin > 0 else -diff
        mov_mult = ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)
        shift = k * mov_mult * ((1.0 if margin > 0 else 0.0) - p_home)
        ratings[row.home_team] = r_home + shift
        ratings[row.away_team] = r_away - shift

    out = games.copy()
    out["elo_home_pre"] = pre_home
    out["elo_away_pre"] = pre_away
    out["p_elo"] = probs
    return out


def add_form_and_schedule(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Team-level rolling stats, shifted by one game so they stay pre-game."""
    home = games[["game_id", "season", "date", "home_team", "home_pts", "away_pts"]].rename(
        columns={"home_team": "team", "home_pts": "pts_for", "away_pts": "pts_against"})
    away = games[["game_id", "season", "date", "away_team", "away_pts", "home_pts"]].rename(
        columns={"away_team": "team", "away_pts": "pts_for", "home_pts": "pts_against"})
    long = pd.concat([home.assign(is_home=1), away.assign(is_home=0)], ignore_index=True)
    long = long.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    long["margin"] = long["pts_for"] - long["pts_against"]
    long["win"] = (long["margin"] > 0).astype(int)

    grouped = long.groupby(["team", "season"], sort=False)
    long["margin_last10"] = grouped["margin"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()).fillna(0.0)
    prior_wins = grouped["win"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    prior_games = grouped.cumcount()
    long["winpct_shrunk"] = (prior_wins + 5) / (prior_games + 10)
    rest = grouped["date"].diff().dt.days
    long["rest_days"] = rest.clip(upper=7).fillna(7)
    long["b2b"] = (long["rest_days"] == 1).astype(int)

    cols = ["game_id", "margin_last10", "winpct_shrunk", "rest_days", "b2b"]
    h = long[long["is_home"] == 1][cols].add_prefix("home_").rename(columns={"home_game_id": "game_id"})
    a = long[long["is_home"] == 0][cols].add_prefix("away_").rename(columns={"away_game_id": "game_id"})
    out = games.merge(h, on="game_id").merge(a, on="game_id")
    out["margin10_diff"] = out["home_margin_last10"] - out["away_margin_last10"]
    out["winpct_diff"] = out["home_winpct_shrunk"] - out["away_winpct_shrunk"]
    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]
    return out.sort_values("game_id").reset_index(drop=True)


def build_features(games: pd.DataFrame, k: float, hfa: float) -> pd.DataFrame:
    return add_form_and_schedule(add_elo(games, k=k, hfa=hfa))
