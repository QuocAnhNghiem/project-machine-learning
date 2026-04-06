import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import classification_metrics
from model import LogisticRegressionGD
from preprocessing import MinMaxNormalizer


DEFAULT_FEATURES = [
    "RankDiff",
    "WinRateDiff",
    "HomeAdvantage",
    "PPGDiff",
    "ELODiff",
    "RecentFormDiff",
    "RestDaysDiff",
    "MarketImpliedProbDiff",
]


def parse_args() -> argparse.Namespace:
    # Tham số dòng lệnh để điều khiển đường dẫn dữ liệu, cách chia train/test, và siêu tham số.
    parser = argparse.ArgumentParser(description="Huấn luyện Logistic Regression bằng Gradient Descent")
    parser.add_argument(
        "--data",
        type=str,
        default="data/epl_50_matches_sample.csv",
        help="Đường dẫn tới file CSV",
    )
    parser.add_argument("--target", type=str, default="target", help="Tên cột nhãn")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Tốc độ học")
    parser.add_argument("--epochs", type=int, default=5000, help="Số vòng lặp")
    parser.add_argument("--l2-lambda", type=float, default=0.0, help="Hệ số L2")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng phân loại")
    parser.add_argument("--test-size", type=float, default=0.2, help="Tỉ lệ chia test")
    parser.add_argument(
        "--train-window-seasons",
        type=int,
        default=0,
        help="Nếu >0 và split-mode=season, chỉ dùng N mùa gần nhất để train.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="time",
        choices=["time", "season"],
        help="Cách chia: time (theo tỉ lệ) hoặc season (mùa cuối làm test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Thư mục lưu kết quả mô hình",
    )
    return parser.parse_args()


def choose_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    # Ưu tiên dùng các cột feature đã định nghĩa sẵn; nếu thiếu thì lấy tất cả cột số (trừ target).
    available = [col for col in DEFAULT_FEATURES if col in df.columns]
    if available:
        return available

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col != target_col]


def train_test_split_time(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Chia dữ liệu theo thời gian: sắp theo Date rồi cắt theo tỉ lệ.
    if "Date" in df.columns:
        df = df.sort_values("Date")

    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def train_test_split_season(df: pd.DataFrame, train_window_seasons: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Chia dữ liệu theo mùa: lấy mùa mới nhất làm test, các mùa trước làm train.
    if "Season" not in df.columns:
        raise ValueError("Cần cột 'Season' khi dùng split-mode=season.")

    season_num = pd.to_numeric(df["Season"], errors="coerce")
    valid_seasons = sorted(season_num.dropna().astype(int).unique().tolist())

    if len(valid_seasons) < 2:
        raise ValueError("Cần ít nhất 2 mùa để chia theo season.")

    test_season = valid_seasons[-1]
    is_test = season_num.eq(test_season)
    train_df = df[~is_test.fillna(False)].copy()
    test_df = df[is_test.fillna(False)].copy()

    if train_window_seasons > 0:
        # Chỉ giữ N mùa gần nhất trong tập train nếu người dùng yêu cầu.
        train_seasons = [season for season in valid_seasons if season != test_season]
        keep = set(train_seasons[-train_window_seasons:])
        train_mask = pd.to_numeric(train_df["Season"], errors="coerce").isin(keep)
        train_df = train_df[train_mask].copy()

    if "Date" in train_df.columns:
        train_df = train_df.sort_values("Date")
    if "Date" in test_df.columns:
        test_df = test_df.sort_values("Date")

    return train_df, test_df


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu đầu vào.
    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Không tìm thấy cột nhãn '{args.target}' trong dữ liệu.")

    feature_cols = choose_feature_columns(df, args.target)
    if not feature_cols:
        raise ValueError("Không tìm thấy cột feature dạng số.")

    # Chọn cách chia train/test theo thời gian hoặc theo mùa.
    if args.split_mode == "season":
        train_df, test_df = train_test_split_season(df, train_window_seasons=args.train_window_seasons)
    else:
        train_df, test_df = train_test_split_time(df, args.test_size)

    x_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[args.target].to_numpy(dtype=float)
    x_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[args.target].to_numpy(dtype=float)

    # Chuẩn hóa Min-Max dựa trên tập train, sau đó áp dụng cho test.
    scaler = MinMaxNormalizer()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Huấn luyện Logistic Regression bằng Gradient Descent.
    model = LogisticRegressionGD(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2_lambda=args.l2_lambda,
    )
    model.fit(x_train_scaled, y_train)

    y_proba = model.predict_proba(x_test_scaled)
    y_pred = model.predict(x_test_scaled, threshold=args.threshold)

    # Tính các chỉ số đánh giá.
    metrics = classification_metrics(y_test, y_pred, y_proba)

    weights_payload = {
        "feature_columns": feature_cols,
        "weights": model.weights.tolist(),
        "bias": model.bias,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "l2_lambda": args.l2_lambda,
    }

    # Lưu trọng số mô hình và các siêu tham số.
    with open(output_dir / "model_weights.json", "w", encoding="utf-8") as file:
        json.dump(weights_payload, file, indent=2)

    # Lưu các chỉ số đánh giá.
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Lưu dự đoán chi tiết để kiểm tra sau.
    prediction_df = pd.DataFrame(
        {
            "y_true": y_test.astype(int),
            "y_proba": y_proba,
            "y_pred": y_pred,
        }
    )
    prediction_df.to_csv(output_dir / "predictions.csv", index=False)

    # Vẽ đường loss theo epoch để kiểm tra hội tụ.
    plt.figure(figsize=(8, 4))
    plt.plot(model.loss_history)
    plt.title("Training Log Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)

    print("Huấn luyện xong.")
    print(f"Chế độ chia: {args.split_mode}")
    if args.split_mode == "season" and args.train_window_seasons > 0:
        print(f"Số mùa train gần nhất: {args.train_window_seasons}")
    print(f"Số dòng train: {len(train_df)}, test: {len(test_df)}")
    print(f"Danh sách feature: {feature_cols}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
