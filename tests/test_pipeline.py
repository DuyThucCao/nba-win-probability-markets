"""Leakage and arithmetic tests. Run with: python -m unittest discover -s tests -v"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import brier, payout  # noqa: E402
from features import build_features  # noqa: E402
from model import log_loss, walk_forward  # noqa: E402
from parse import check_schedule, moneyline_to_prob, signed_home_spread  # noqa: E402

PREGAME = ["elo_home_pre", "elo_away_pre", "p_elo", "home_margin_last10", "away_margin_last10",
           "home_winpct_shrunk", "away_winpct_shrunk", "home_rest_days", "away_rest_days"]


def synthetic_games(seasons=("2007-08", "2008-09", "2009-10", "2010-11", "2011-12",
                             "2012-13", "2013-14"), per_season=120, seed=0):
    """Random league of 8 teams; each team plays at most once per date."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    strength = dict(zip(teams, rng.normal(0, 5, len(teams))))
    rows, start = [], pd.Timestamp("2007-11-01")
    for s_idx, season in enumerate(seasons):
        day = start + pd.DateOffset(years=s_idx)
        for g in range(per_season):
            if g % 4 == 0:
                order = rng.permutation(teams)
                day += pd.Timedelta(days=int(rng.integers(1, 3)))
            home, away = order[2 * (g % 4)], order[2 * (g % 4) + 1]
            margin = strength[home] - strength[away] + 3 + rng.normal(0, 12)
            hp = int(105 + max(margin, 1) if margin > 0 else 105)
            ap = int(105 if margin > 0 else 105 + max(-margin, 1))
            p = 1 / (1 + np.exp(-(strength[home] - strength[away] + 3) / 8))
            ml_h = -round(100 * p / (1 - p)) if p > 0.5 else round(100 * (1 - p) / p)
            ml_a = round(100 * p / (1 - p)) if p > 0.5 else -round(100 * (1 - p) / p)
            rows.append({"season": season, "date": day, "away_team": away, "home_team": home,
                         "away_pts": ap, "home_pts": hp, "away_ml": ml_a, "home_ml": ml_h,
                         "neutral": False})
    games = pd.DataFrame(rows).sort_values(["date", "home_team"]).reset_index(drop=True)
    games["game_id"] = np.arange(len(games))
    games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)
    raw_h, raw_a = moneyline_to_prob(games["home_ml"]), moneyline_to_prob(games["away_ml"])
    games["overround"] = raw_h + raw_a
    games["p_market"] = raw_h / games["overround"]
    games["market_ok"] = True
    return games


class LeakageTests(unittest.TestCase):
    def test_current_result_does_not_change_its_own_pregame_features(self):
        games = synthetic_games()
        i = 300
        before = build_features(games, k=20, hfa=80).loc[i, PREGAME]
        games.loc[i, ["home_pts", "away_pts"]] = [160, 60]
        games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)
        after = build_features(games, k=20, hfa=80).loc[i, PREGAME]
        np.testing.assert_allclose(before.to_numpy(float), after.to_numpy(float))

    def test_future_results_do_not_change_earlier_features(self):
        games = synthetic_games()
        cut = 400
        before = build_features(games, k=20, hfa=80).loc[:cut - 1, PREGAME]
        games.loc[cut:, ["home_pts", "away_pts"]] = [50, 150]
        games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)
        after = build_features(games, k=20, hfa=80).loc[:cut - 1, PREGAME]
        np.testing.assert_allclose(before.to_numpy(float), after.to_numpy(float))

    def test_first_game_of_season_has_neutral_form(self):
        feats = build_features(synthetic_games(), k=20, hfa=80)
        first = feats.groupby("season").head(1).iloc[0]
        self.assertEqual(first["home_margin_last10"], 0.0)
        self.assertAlmostEqual(first["home_winpct_shrunk"], 0.5)
        self.assertEqual(first["home_rest_days"], 7)

    def test_walk_forward_predictions_ignore_future_seasons(self):
        games = synthetic_games()
        base = walk_forward(build_features(games, k=20, hfa=80))
        last = games["season"].max()
        games.loc[games["season"] == last, ["home_pts", "away_pts"]] = [40, 140]
        games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)
        changed = walk_forward(build_features(games, k=20, hfa=80))
        keep = base["season"] < last
        np.testing.assert_allclose(base.loc[keep, "p_model"], changed.loc[keep, "p_model"])
        np.testing.assert_allclose(base.loc[keep, "p_stack"], changed.loc[keep, "p_stack"])

    def test_predictions_are_probabilities(self):
        preds = walk_forward(build_features(synthetic_games(), k=20, hfa=80))
        self.assertTrue(preds["p_model"].between(0, 1, inclusive="neither").all())


class ArithmeticTests(unittest.TestCase):
    def test_moneyline_conversion_and_devig(self):
        p = moneyline_to_prob(pd.Series([-150, 150, 100]))
        np.testing.assert_allclose(p, [0.6, 0.4, 0.5])
        home, away = moneyline_to_prob(pd.Series([-200])), moneyline_to_prob(pd.Series([170]))
        self.assertAlmostEqual(float((home / (home + away)).iloc[0] + (away / (home + away)).iloc[0]), 1.0)

    def test_signed_home_spread(self):
        self.assertEqual(signed_home_spread("220.5", "5"), -5.0)   # home favored by 5
        self.assertEqual(signed_home_spread("3.5", "210"), 3.5)    # away favored by 3.5
        self.assertEqual(signed_home_spread("pk", "215"), 0.0)

    def test_payout(self):
        np.testing.assert_allclose(payout(np.array([150, -200])), [1.5, 0.5])

    def test_scores(self):
        self.assertAlmostEqual(brier([0, 1], [0.2, 0.8]), 0.04)
        self.assertAlmostEqual(log_loss(np.array([0, 1]), np.array([0.2, 0.8])), -np.log(0.8))

    def test_schedule_check_catches_double_booking(self):
        bad = pd.DataFrame({"date": pd.to_datetime(["2020-12-28", "2020-12-28"]),
                            "home_team": ["LALakers", "LALakers"], "away_team": ["Minnesota", "Portland"]})
        with self.assertRaises(ValueError):
            check_schedule(bad)


class DataTests(unittest.TestCase):
    """Checks on the real processed data, skipped if it has not been built."""
    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "games.csv"

    @unittest.skipUnless(path.exists(), "run run_all.py first")
    def test_processed_games(self):
        g = pd.read_csv(self.path, parse_dates=["date"])
        check_schedule(g)
        self.assertFalse(g.duplicated(["date", "home_team", "away_team"]).any())
        self.assertTrue((g["home_team"] != g["away_team"]).all())
        ok = g[g["market_ok"]]
        self.assertTrue(ok["p_market"].between(0, 1, inclusive="neither").all())
        # Favorites by the spread should also be favorites by the moneyline.
        self.assertLess(ok[["p_market", "home_spread_close"]].corr().iloc[0, 1], -0.85)


if __name__ == "__main__":
    unittest.main()
