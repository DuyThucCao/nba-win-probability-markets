"""Score the forecasts against outcomes and against the betting market."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from model import log_loss, logit

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
FORECASTS = {"Elo": "p_elo", "Logistic model": "p_model",
             "Market (closing, no-vig)": "p_market", "Market + model": "p_stack"}


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def score_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in FORECASTS.items():
        y, p = df["home_win"], df[col]
        rows.append({"forecast": name, "games": len(df), "brier": brier(y, p),
                     "log_loss": log_loss(y, p), "accuracy": float(np.mean((p > 0.5) == y))})
    return pd.DataFrame(rows)


def encompassing_test(df: pd.DataFrame):
    """Logit of outcome on logit(market) and logit(model). If the market
    price is efficient, the model coefficient should be zero."""
    X = sm.add_constant(np.column_stack([logit(df["p_market"]), logit(df["p_model"])]))
    fit = sm.Logit(df["home_win"].values, X).fit(disp=0, cov_type="HC1")
    names = ["const", "logit_market", "logit_model"]
    return {n: {"coef": float(c), "se": float(s), "p_value": float(pv)}
            for n, c, s, pv in zip(names, fit.params, fit.bse, fit.pvalues)}


def payout(ml):
    """Profit on a 1-unit stake that wins at American odds ml."""
    return np.where(ml > 0, ml / 100.0, 100.0 / np.abs(ml))


def backtest(df: pd.DataFrame, threshold: float, rng: np.random.Generator):
    edge = df["p_model"] - df["p_market"]
    bet_home, bet_away = edge > threshold, edge < -threshold
    profit = np.where(bet_home, np.where(df["home_win"] == 1, payout(df["home_ml"]), -1.0), 0.0)
    profit = profit + np.where(bet_away, np.where(df["home_win"] == 0, payout(df["away_ml"]), -1.0), 0.0)
    placed = bet_home | bet_away
    bets = profit[placed]
    boot = [rng.choice(bets, size=len(bets), replace=True).mean() for _ in range(2000)] if len(bets) else [np.nan]
    return {"threshold": threshold, "bets": int(placed.sum()),
            "roi": float(bets.mean()) if len(bets) else np.nan,
            "roi_ci_low": float(np.percentile(boot, 2.5)), "roi_ci_high": float(np.percentile(boot, 97.5)),
            "hit_rate": float((bets > 0).mean()) if len(bets) else np.nan}, pd.Series(profit, index=df.index)


def line_movement(df: pd.DataFrame, sigma: float) -> dict:
    """Does the model anticipate how the line moves from open to close?

    Spreads are mapped to win probabilities with a normal margin model,
    P(home win) = Phi(-spread / sigma). If the model holds information the
    opening price lacks, its disagreement with the open (signal) should
    predict the later move toward the close (movement).
    """
    d = df.dropna(subset=["home_spread_open", "home_spread_close"]).copy()
    d = d[(d["home_spread_open"].abs() < 30) & (d["home_spread_close"].abs() < 30)]
    d["p_open"] = norm.cdf(-d["home_spread_open"] / sigma)
    d["p_close"] = norm.cdf(-d["home_spread_close"] / sigma)
    d["signal"] = d["p_model"] - d["p_open"]
    d["movement"] = d["p_close"] - d["p_open"]
    fit = sm.OLS(d["movement"], sm.add_constant(d["signal"])).fit(cov_type="HC1")
    # Robustness: opening lines can be posted before a team's previous game
    # ends, so drop games where either team played the day before.
    rested = d[(d["home_rest_days"] >= 2) & (d["away_rest_days"] >= 2)]
    fit_rested = sm.OLS(rested["movement"], sm.add_constant(rested["signal"])).fit(cov_type="HC1")
    moved = d[d["movement"] != 0]
    same_dir = float((np.sign(moved["signal"]) == np.sign(moved["movement"])).mean())
    big = d[d["signal"].abs() > 0.04]
    clv = np.sign(big["signal"]) * big["movement"]   # gain in probability points
    return {"sigma": sigma, "games": int(len(d)), "games_line_moved": int(len(moved)),
            "slope": float(fit.params["signal"]), "slope_se": float(fit.bse["signal"]),
            "slope_p_value": float(fit.pvalues["signal"]), "r_squared": float(fit.rsquared),
            "share_same_direction": same_dir,
            "rested_games": int(len(rested)), "rested_slope": float(fit_rested.params["signal"]),
            "rested_slope_se": float(fit_rested.bse["signal"]),
            "rested_slope_p_value": float(fit_rested.pvalues["signal"]),
            "clv_bets_edge_gt_4pct": int(len(big)),
            "clv_mean_prob_points": float(clv.mean()),
            "clv_share_positive": float((clv > 0).mean()),
            "clv_share_negative": float((clv < 0).mean())}


def margin_sigma(feats: pd.DataFrame, seasons: list[str]) -> float:
    """Std. dev. of the home margin around the closing spread (tuning seasons)."""
    t = feats[feats["season"].isin(seasons)].dropna(subset=["home_spread_close"])
    t = t[t["home_spread_close"].abs() < 30]
    return float(np.std(t["home_pts"] - t["away_pts"] + t["home_spread_close"]))


def calibration(df: pd.DataFrame, col: str, bins: int = 10) -> pd.DataFrame:
    cut = pd.cut(df[col], np.linspace(0, 1, bins + 1), include_lowest=True)
    return df.groupby(cut, observed=True).agg(mean_pred=(col, "mean"), freq=("home_win", "mean"),
                                              n=("home_win", "size")).reset_index(drop=True)


def run(preds: pd.DataFrame, elo_params: dict, sigma: float) -> dict:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    scored = preds[preds["market_ok"]].copy()   # same games for every forecast

    pooled = score_table(scored)
    by_season = pd.concat([score_table(g).assign(season=s) for s, g in scored.groupby("season")])
    pooled.to_csv(RESULTS / "scores_pooled.csv", index=False)
    by_season.to_csv(RESULTS / "scores_by_season.csv", index=False)

    enc = encompassing_test(scored)
    moves = line_movement(scored, sigma)
    rng = np.random.default_rng(42)
    bt_rows, curves = [], {}
    for thr in (0.02, 0.04, 0.06, 0.08):
        row, prof = backtest(scored, thr, rng)
        bt_rows.append(row)
        curves[thr] = prof.cumsum()
    bt = pd.DataFrame(bt_rows)
    bt.to_csv(RESULTS / "backtest.csv", index=False)

    # Figure 1: calibration of the model and the market.
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, col in [("Logistic model", "p_model"), ("Market (no-vig)", "p_market")]:
        c = calibration(scored, col)
        ax.plot(c["mean_pred"], c["freq"], marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set(xlabel="Forecast home-win probability", ylabel="Observed home-win rate",
           title="Calibration, out-of-sample 2012-13 to 2022-23")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "calibration.png", dpi=150); plt.close(fig)

    # Figure 2: Brier score by season.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name in ["Elo", "Logistic model", "Market (closing, no-vig)"]:
        s = by_season[by_season["forecast"] == name]
        ax.plot(s["season"], s["brier"], marker="o", label=name)
    ax.set(ylabel="Brier score (lower is better)", title="Out-of-sample Brier score by season")
    ax.tick_params(axis="x", rotation=45); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "brier_by_season.png", dpi=150); plt.close(fig)

    # Figure 3: cumulative profit of the edge strategy at each threshold.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for thr, curve in curves.items():
        ax.plot(range(len(curve)), curve.values, label=f"edge > {thr:.0%}")
    ax.axhline(0, color="k", lw=1)
    ax.set(xlabel="Games (chronological)", ylabel="Cumulative profit (units)",
           title="Betting the model's edge vs. closing moneylines (vig included)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "edge_backtest.png", dpi=150); plt.close(fig)

    summary = {"elo_params": elo_params, "test_games": int(len(scored)),
               "test_seasons": sorted(scored["season"].unique().tolist()),
               "avg_overround": float(scored["overround"].mean()),
               "pooled_scores": pooled.to_dict(orient="records"),
               "encompassing_test": enc, "line_movement": moves,
               "backtest": bt.to_dict(orient="records")}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
