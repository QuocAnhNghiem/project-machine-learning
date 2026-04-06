import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import classification_metrics
from model import LogisticRegressionGD
from preprocessing import MinMaxNormalizer
from train import choose_feature_columns, train_test_split_season, train_test_split_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quét ngưỡng phân loại cho Logistic Regression")
    parser.add_argument("--data", type=str, default="data/epl_real_features.csv", help="Đường dẫn CSV")
    parser.add_argument("--target", type=str, default="target", help="Tên cột nhãn")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Tốc độ học")
    parser.add_argument("--epochs", type=int, default=5000, help="Số vòng lặp")
    parser.add_argument("--l2-lambda", type=float, default=0.0, help="Hệ số L2")
    parser.add_argument("--test-size", type=float, default=0.2, help="Tỉ lệ test khi chia time")
    parser.add_argument(
        "--train-window-seasons",
        type=int,
        default=0,
        help="Nếu >0 và split-mode=season, chỉ dùng N mùa gần nhất để train.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="season",
        choices=["time", "season"],
        help="Cách chia tập",
    )
    parser.add_argument("--start", type=float, default=0.35, help="Ngưỡng bắt đầu")
    parser.add_argument("--stop", type=float, default=0.70, help="Ngưỡng kết thúc")
    parser.add_argument("--step", type=float, default=0.05, help="Bước nhảy ngưỡng")
    parser.add_argument("--name", type=str, default="threshold_sweep", help="Tiền tố file đầu ra")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Thư mục đầu ra")
    return parser.parse_args()


def metric_row(label: str, threshold: float | None, metrics: dict) -> dict:
    cm = metrics["confusion_matrix"]
    return {
        "label": label,
        "threshold": threshold,
        "log_loss": metrics["log_loss"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "tp": cm["tp"],
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
    }


def get_best_rows(result_df: pd.DataFrame) -> dict:
    threshold_rows = result_df[result_df["label"] == "model"].copy()

    best_f1 = threshold_rows.sort_values(["f1_score", "precision", "recall"], ascending=False).iloc[0]
    best_precision = threshold_rows.sort_values(["precision", "recall", "f1_score"], ascending=False).iloc[0]
    best_recall = threshold_rows.sort_values(["recall", "precision", "f1_score"], ascending=False).iloc[0]

    return {
        "best_f1": best_f1.to_dict(),
        "best_precision": best_precision.to_dict(),
        "best_recall": best_recall.to_dict(),
    }


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Không tìm thấy cột nhãn '{args.target}'.")

    feature_cols = choose_feature_columns(df, args.target)
    if not feature_cols:
        raise ValueError("Không tìm thấy feature số hợp lệ.")

    if args.split_mode == "season":
        train_df, test_df = train_test_split_season(df, train_window_seasons=args.train_window_seasons)
    else:
        train_df, test_df = train_test_split_time(df, args.test_size)

    x_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[args.target].to_numpy(dtype=float)
    x_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[args.target].to_numpy(dtype=float)

    scaler = MinMaxNormalizer()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegressionGD(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2_lambda=args.l2_lambda,
    )
    model.fit(x_train_scaled, y_train)

    y_proba = model.predict_proba(x_test_scaled)

    rows = []
    thresholds = np.arange(args.start, args.stop + 1e-9, args.step)
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = classification_metrics(y_test, y_pred, y_proba)
        rows.append(metric_row("model", float(round(threshold, 4)), metrics))

    # Baselines
    y_pred_zero = np.zeros_like(y_test)
    zero_metrics = classification_metrics(y_test, y_pred_zero, np.full_like(y_test, np.mean(y_test), dtype=float))
    rows.append(metric_row("always_0", None, zero_metrics))

    y_pred_one = np.ones_like(y_test)
    one_metrics = classification_metrics(y_test, y_pred_one, np.full_like(y_test, np.mean(y_test), dtype=float))
    rows.append(metric_row("always_1", None, one_metrics))

    result_df = pd.DataFrame(rows)

    csv_path = output_dir / f"{args.name}_{args.split_mode}.csv"
    json_path = output_dir / f"{args.name}_{args.split_mode}_best.json"

    result_df.to_csv(csv_path, index=False)

    summary = {
        "split_mode": args.split_mode,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": feature_cols,
        "l2_lambda": args.l2_lambda,
        "train_window_seasons": args.train_window_seasons,
        "best": get_best_rows(result_df),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Đã lưu bảng ngưỡng: {csv_path}")
    print(f"Đã lưu tổng hợp ngưỡng tốt nhất: {json_path}")
    print(json.dumps(summary["best"], indent=2))


if __name__ == "__main__":
    main()
