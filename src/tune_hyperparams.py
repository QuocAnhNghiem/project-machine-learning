import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tìm kiếm siêu tham số để tối ưu accuracy")
    parser.add_argument("--data", type=str, default="data/epl_real_features.csv")
    parser.add_argument("--output", type=str, default="outputs/hyperparam_search_accuracy.csv")
    parser.add_argument("--summary", type=str, default="outputs/hyperparam_search_accuracy_best.json")
    return parser.parse_args()


def run_one(config: dict) -> dict:
    from evaluate_thresholds import main as sweep_main
    import sys

    args = [
        "evaluate_thresholds.py",
        "--data",
        config["data"],
        "--split-mode",
        "season",
        "--start",
        str(config["start"]),
        "--stop",
        str(config["stop"]),
        "--step",
        str(config["step"]),
        "--l2-lambda",
        str(config["l2_lambda"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--epochs",
        str(config["epochs"]),
        "--train-window-seasons",
        str(config["train_window_seasons"]),
        "--name",
        config["name"],
        "--output-dir",
        "outputs",
    ]

    old_argv = sys.argv
    try:
        sys.argv = args
        sweep_main()
    finally:
        sys.argv = old_argv

    csv_path = Path("outputs") / f"{config['name']}_season.csv"
    df = pd.read_csv(csv_path)
    model_rows = df[df["label"] == "model"].copy()
    best = model_rows.sort_values(["accuracy", "precision", "recall"], ascending=False).iloc[0].to_dict()
    best.update(
        {
            "learning_rate": config["learning_rate"],
            "epochs": config["epochs"],
            "l2_lambda": config["l2_lambda"],
            "train_window_seasons": config["train_window_seasons"],
        }
    )
    return best


def main() -> None:
    args = parse_args()

    learning_rates = [0.01, 0.03, 0.05, 0.08]
    epochs_list = [4000, 6000, 8000]
    l2_list = [0.0, 0.05, 0.1, 0.5, 1.0]
    window_list = [0, 5, 8]

    results = []

    for idx, (lr, ep, l2, window) in enumerate(product(learning_rates, epochs_list, l2_list, window_list), start=1):
        config = {
            "data": args.data,
            "learning_rate": lr,
            "epochs": ep,
            "l2_lambda": l2,
            "train_window_seasons": window,
            "start": 0.55,
            "stop": 0.75,
            "step": 0.001,
            "name": f"deep_sweep_{idx}",
        }
        best = run_one(config)
        results.append(best)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(["accuracy", "precision", "recall"], ascending=False).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    best = result_df.iloc[0].to_dict()
    with open(args.summary, "w", encoding="utf-8") as file:
        json.dump(best, file, indent=2)

    print(f"Đã lưu bảng tìm kiếm: {output_path}")
    print(f"Đã lưu cấu hình tốt nhất: {args.summary}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
