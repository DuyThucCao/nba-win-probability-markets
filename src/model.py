"""Walk-forward models: every test season is predicted by a model fit only
on earlier seasons, so no future information leaks into a forecast."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features

FEATURES = ["elo_logit", "margin10_diff", "winpct_diff", "rest_diff",
            "home_b2b", "away_b2b", "neutral"]
BURN_IN = "2007-08"                       # Elo warm-up only, never scored
TUNING = ["2008-09", "2009-10", "2010-11", "2011-12"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune_elo(games: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Pick K and home-court advantage on the tuning seasons only."""
    rows = []
    for k in (15, 20, 25, 30):
        for hfa in (60, 80, 100, 120):
            feats = build_features(games, k=k, hfa=hfa)
            tune = feats[feats["season"].isin(TUNING)]
            rows.append({"k": k, "hfa": hfa,
                         "log_loss": log_loss(tune["home_win"], tune["p_elo"])})
    grid = pd.DataFrame(rows).sort_values("log_loss")
    best = grid.iloc[0]
    return float(best["k"]), float(best["hfa"]), grid


def walk_forward(feats: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window forecasts.

    Every season from the second tuning season on is predicted by a model
    fit only on earlier seasons, so each row's p_model is out of sample.
    The stacked market + model forecast for a test season is then fit on
    those earlier out-of-sample predictions, never on in-sample fits.
    """
    feats = feats.copy()
    feats["elo_logit"] = logit(feats["p_elo"])
    feats["neutral"] = feats["neutral"].astype(int)
    seasons = sorted(s for s in feats["season"].unique() if s != BURN_IN)
    test_seasons = [s for s in seasons if s not in TUNING]

    oos = []
    for season in seasons[1:]:
        train = feats[(feats["season"] < season) & (feats["season"] != BURN_IN)]
        target = feats[feats["season"] == season].copy()
        assert train["date"].max() < target["date"].min()
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(train[FEATURES], train["home_win"])
        target["p_model"] = model.predict_proba(target[FEATURES])[:, 1]
        oos.append(target)
    oos = pd.concat(oos, ignore_index=True)

    preds = []
    for season in test_seasons:
        history = oos[(oos["season"] < season) & oos["market_ok"]]
        test = oos[oos["season"] == season].copy()
        stack = LogisticRegression(C=1e6, max_iter=1000)
        stack.fit(np.column_stack([logit(history["p_market"]), logit(history["p_model"])]),
                  history["home_win"])
        ok = test["market_ok"]
        test["p_stack"] = np.nan
        test.loc[ok, "p_stack"] = stack.predict_proba(np.column_stack(
            [logit(test.loc[ok, "p_market"]), logit(test.loc[ok, "p_model"])]))[:, 1]
        preds.append(test)
    return pd.concat(preds, ignore_index=True)
