"""Reproduce the whole project: download, parse, model, evaluate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

import evaluate  # noqa: E402
import ingest  # noqa: E402
import parse  # noqa: E402
from features import build_features  # noqa: E402
from model import TUNING, tune_elo, walk_forward  # noqa: E402


def main():
    ingest.download_all()
    games = parse.build_games()
    parse.PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(parse.PROCESSED, index=False)

    k, hfa, grid = tune_elo(games)
    grid.to_csv(evaluate.RESULTS / "elo_tuning_grid.csv", index=False)
    feats = build_features(games, k=k, hfa=hfa)
    preds = walk_forward(feats)
    preds.to_csv(evaluate.RESULTS / "predictions.csv", index=False)
    sigma = evaluate.margin_sigma(feats, TUNING)
    summary = evaluate.run(preds, {"k": k, "hfa": hfa}, sigma)

    print(pd.DataFrame(summary["pooled_scores"]).round(4).to_string(index=False))
    print("Encompassing test:", summary["encompassing_test"])
    print("Line movement:", summary["line_movement"])
    print(pd.DataFrame(summary["backtest"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
