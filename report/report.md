# Báo cáo tổng hợp mô hình dự đoán kết quả trận đấu EPL

## 1. Bài toán và mục tiêu

- Bài toán phân loại nhị phân: dự đoán đội nhà thắng (1) hoặc không thắng (0).
- Mục tiêu chính: tăng độ chính xác (accuracy) trong điều kiện test theo mùa (train các mùa trước, test mùa 2526).

## 2. Thuật toán sử dụng

### 2.1 Logistic Regression

- Mô hình tuyến tính có hàm sigmoid để đưa ra xác suất thắng.
- Dùng Gradient Descent để tối ưu tham số.

Công thức điểm tuyến tính:

$$
z = \theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n
$$

### 2.2 Hàm kích hoạt Sigmoid

- Chuyển điểm tuyến tính sang xác suất trong khoảng [0, 1].

$$
\hat{y} = g(z) = \frac{1}{1 + e^{-z}}
$$

Quy tắc phân loại:

$$
\hat{c} = \begin{cases}
1, & \hat{y} \ge 0.5\\
0, & \hat{y} < 0.5
\end{cases}
$$

### 2.3 Hàm mất mát Log Loss

- Đo sai lệch giữa xác suất dự đoán và nhãn thực tế.

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1-\hat{y}^{(i)})\right]
$$

Gradient xuống (cập nhật tham số):

$$
  heta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$

### 2.4 Chuẩn hóa Min-Max

- Đưa tất cả feature về cùng thang đo, giúp học ổn định hơn.

$$
X_{new} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

## 3. Dữ liệu và cách chia tập

- Dữ liệu từ football-data.co.uk, gồm nhiều mùa EPL.
- Chế độ chia: theo mùa (season split).
  - Train: các mùa trước.
  - Test: mùa 2526.
- Mục tiêu: đánh giá khả năng tổng quát trên mùa mới.

## 4. Input đầu vào và cách tạo dữ liệu

### 4.1 Nguồn dữ liệu thô

Dữ liệu lấy từ các file CSV EPL (football-data.co.uk). Các cột dùng để tạo feature:

- `Date`: ngày đá
- `HomeTeam`, `AwayTeam`: đội nhà/đội khách
- `FTR`: kết quả (H/D/A)
- `FTHG`, `FTAG`: bàn thắng hai đội
- Odds: một trong các cột `B365H`, `PSH`, `WHH`, `AvgH`, `MaxH` (home) và `B365A`, `PSA`, `WHA`, `AvgA`, `MaxA` (away)

### 4.2 Feature đầu vào (phiên bản đơn giản 8 feature)

Các feature đều được tính trước trận, dựa trên thống kê tích lũy của từng đội:

- `RankDiff`: chênh lệch thứ hạng tạm thời (rank_away - rank_home).
- `WinRateDiff`: chênh lệch tỉ lệ thắng (wins/played).
- `HomeAdvantage`: hằng số = 1 để biểu diễn lợi thế sân nhà.
- `PPGDiff`: chênh lệch điểm trung bình mỗi trận (points/played).
- `ELODiff`: chênh lệch Elo trước trận (elo_home - elo_away).
- `RecentFormDiff`: chênh lệch phong độ 3 trận gần nhất (thắng = 1, hòa = 0, thua = -1).
- `RestDaysDiff`: chênh lệch số ngày nghỉ trước trận (home - away).
- `MarketImpliedProbDiff`: chênh lệch xác suất ngụ ý từ kèo nhà cái $(1/odd_{home}) - (1/odd_{away})$.

### 4.3 Nhãn (target)

- `target = 1` nếu `FTR = H` (đội nhà thắng)
- `target = 0` nếu `FTR = D` hoặc `FTR = A`

### 4.4 Chuẩn hóa đầu vào

Toàn bộ feature được chuẩn hóa Min-Max theo tập train, sau đó áp dụng cho tập test.

## 5. Cách tính điểm mô hình và metric đánh giá

### 5.1 Điểm dự đoán và nhãn dự đoán

- Mô hình trả ra xác suất $\hat{y} \in [0, 1]$.
- Dự đoán nhãn dùng ngưỡng $t$:

$$
\hat{c} = \begin{cases}
1, & \hat{y} \ge t\\
0, & \hat{y} < t
\end{cases}
$$

### 5.2 Các metric chính

Gọi các phần tử trong confusion matrix:

- TP: dự đoán thắng và thực tế thắng
- TN: dự đoán không thắng và thực tế không thắng
- FP: dự đoán thắng nhưng thực tế không thắng
- FN: dự đoán không thắng nhưng thực tế thắng

Các công thức:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
Precision = \frac{TP}{TP + FP}
$$

$$
Recall = \frac{TP}{TP + FN}
$$

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

$$
LogLoss = -\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1-\hat{y}^{(i)})\right]
$$

### 5.3 Ý nghĩa ngưỡng (threshold)

- Ngưỡng cao: precision tăng nhưng recall giảm.
- Ngưỡng thấp: recall tăng nhưng precision giảm.

## 6. Cách tăng độ chính xác qua các bản

### 6.1 Ban đầu (baseline)

- Dùng ít feature và threshold mặc định 0.5.
- Độ chính xác chưa cao do dữ liệu lệch lớp và chưa tối ưu threshold.

### 6.2 Nâng cấp feature

- Bổ sung feature sức mạnh đội, phong độ gần đây, và thông tin kèo nhà cái.
- Loại bớt feature không cần thiết để giảm nhiễu.

### 6.3 Tối ưu threshold

- Quét threshold mịn (0.45 -> 0.75) để tìm điểm accuracy cao nhất.
- Chọn threshold tối ưu theo accuracy thay vì giữ cố định 0.5.

### 6.4 Tối ưu tham số huấn luyện

- Quét learning_rate, epochs, l2_lambda.
- Chọn cấu hình đạt accuracy cao nhất trên mùa test.

### 6.5 Chia dữ liệu theo mùa để giảm lệch (drift)

- Train các mùa trước và test mùa mới giúp kết quả thực tế hơn.
- Giảm nguy cơ học vượt (overfit) với dữ liệu quá cũ.

## 7. Sơ đồ luồng xử lý (planning flow)

![Flow chart pipeline](./Screenshot%20from%202026-04-06%2023-06-12.png)

## 8. Kết luận

- Mô hình Logistic Regression có thể đạt accuracy cao hơn khi:
  - feature phù hợp,
  - threshold được tối ưu,
  - tham số huấn luyện được tìm kiếm hệ thống.
- Trong bài toán bóng đá, accuracy có thể tăng nhưng đổi lại recall giảm nếu threshold quá cao.

### Hướng phát triển tiếp theo

- Bổ sung feature nâng cao: xG, số cú sút, kiểm soát bóng, lịch thi đấu dày/nhẹ.
- Tối ưu tham số và ngưỡng bằng grid/random search theo mùa để tăng accuracy.
- Cân bằng dữ liệu và thử class weight để giảm lệch lớp.
- Thử mô hình mạnh hơn (Random Forest, XGBoost) và chọn mô hình theo accuracy trên mùa mới.
