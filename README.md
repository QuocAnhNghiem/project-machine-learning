# Dự đoán kết quả EPL với Logistic Regression

## Mục tiêu

Dự đoán nhị phân kết quả trận cho đội chủ nhà:

- `1`: Chủ nhà thắng
- `0`: Chủ nhà không thắng (hòa hoặc thua)

Các thành phần chính trong pipeline:

- Chuẩn hóa Min-Max
- Logistic Regression
- Sigmoid
- Gradient Descent với Log Loss

## Cấu trúc dự án

- `data/`: dữ liệu đầu vào và dữ liệu đã xử lý
- `src/`: mã nguồn huấn luyện và tiền xử lý
- `outputs/`: kết quả mô hình (weights, metrics, predictions, loss curve)
- `report/`: báo cáo

## Yêu cầu môi trường

- Python 3.10+ (khuyến nghị 3.12)
- Các thư viện trong `requirements.txt`

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

### 1) Dữ liệu mẫu (nếu có sẵn)

Bạn có thể chạy nhanh với file mẫu đã có:

```bash
cd src
python train.py --data ../data/epl_50_matches_sample.csv --output-dir ../outputs
```

### 2) Dữ liệu EPL thật (football-data.co.uk)

1. Tải các file CSV EPL theo mùa (mã giải E0). Ví dụ: `E0_2425.csv`, `E0_2526.csv`.
2. Đặt các file vào thư mục `data/data-raw/`.
3. Tạo dataset đã xử lý (feature) bằng script:

```bash
cd src
python prepare_real_data.py --input-dir ../data/data-raw --output ../data/epl_real_features.csv
```

Tùy chọn lọc mùa:

```bash
python prepare_real_data.py \
	--input-dir ../data/data-raw \
	--output ../data/epl_real_features.csv \
	--min-season 2223 \
	--max-season 2526
```

## Huấn luyện mô hình

### Chạy bằng Docker Compose (EPL, chạy 1 lần rồi thoát)

```bash
docker compose run --rm epl
```

### 1) Huấn luyện cơ bản

```bash
cd src
python train.py --data ../data/epl_real_features.csv --output-dir ../outputs
```

### 2) Chia train/test theo mùa

```bash
python train.py \
	--data ../data/epl_real_features.csv \
	--split-mode season \
	--train-window-seasons 8 \
	--output-dir ../outputs
```

Tham số thường dùng:

- `--learning-rate`: tốc độ học
- `--epochs`: số vòng lặp
- `--l2-lambda`: hệ số L2
- `--threshold`: ngưỡng phân loại

## Quét ngưỡng (threshold sweep)

```bash
cd src
python evaluate_thresholds.py \
	--data ../data/epl_real_features.csv \
	--split-mode season \
	--start 0.55 \
	--stop 0.75 \
	--step 0.01 \
	--output-dir ../outputs
```

## Tìm kiếm siêu tham số (grid search)

```bash
cd src
python tune_hyperparams.py --data ../data/epl_real_features.csv
```

## Đầu ra

Sau khi chạy, các file kết quả nằm trong `outputs/`:

- `model_weights.json`: trọng số và siêu tham số
- `metrics.json`: accuracy, precision, recall, f1, log loss
- `predictions.csv`: y_true, y_proba, y_pred
- `loss_curve.png`: đường loss theo epoch
- `threshold_sweep_*.csv`, `*_best.json`: kết quả quét ngưỡng
- `hyperparam_search_accuracy.csv`, `hyperparam_search_accuracy_best.json`: kết quả grid search

##

## Ghi chú

- Dataset `epl_real_features.csv` đã có các feature như `RankDiff`, `WinRateDiff`, `PPGDiff`, `ELODiff`, `RecentFormDiff`, `RestDaysDiff`, `MarketImpliedProbDiff`, `HomeAdvantage`.
- Khi dùng `split-mode=season`, mùa mới nhất trong dữ liệu sẽ được dùng làm tập test.
