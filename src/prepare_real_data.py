import argparse
from collections import deque
from pathlib import Path
import re

import pandas as pd


REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]
HOME_ODDS_COLUMNS = ["B365H", "PSH", "WHH", "AvgH", "MaxH"]
AWAY_ODDS_COLUMNS = ["B365A", "PSA", "WHA", "AvgA", "MaxA"]
TEAM_ALIAS = {
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "man city": "Manchester City",
    "nott'm forest": "Nottingham Forest",
    "spurs": "Tottenham",
    "wolves": "Wolverhampton",
    "newcastle": "Newcastle United",
    "west brom": "West Bromwich Albion",
    "sheffield utd": "Sheffield United",
    "leeds": "Leeds United",
    "ipswich": "Ipswich Town",
}


def parse_args() -> argparse.Namespace:
    # Tham số dòng lệnh: thư mục dữ liệu thô và phạm vi mùa muốn giữ.
    parser = argparse.ArgumentParser(
        description="Tạo dữ liệu EPL sẵn sàng huấn luyện từ file thô football-data.co.uk"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/data-raw",
        help="Thư mục chứa file dạng E0_2425.csv, E0_2526.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/epl_real_features.csv",
        help="Đường dẫn CSV đầu ra sau khi tạo feature",
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=0,
        help="Mã mùa nhỏ nhất (vd: 2223). Dùng 0 để bỏ lọc.",
    )
    parser.add_argument(
        "--max-season",
        type=int,
        default=0,
        help="Mã mùa lớn nhất (vd: 2526). Dùng 0 để bỏ lọc.",
    )
    return parser.parse_args()


def season_from_filename(path: Path) -> str:
    stem = path.stem  # E0_2425
    parts = stem.split("_")
    return parts[-1] if len(parts) > 1 else stem


def normalize_team_name(name: str) -> str:
    # Chuẩn hóa tên đội để tránh lệch giữa các mùa.
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    if key in TEAM_ALIAS:
        return TEAM_ALIAS[key]
    return str(name).strip()


def parse_date_column(series: pd.Series) -> pd.Series:
    # Ưu tiên dayfirst, nếu lỗi thì thử lại theo monthfirst.
    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], dayfirst=False, errors="coerce")
    return parsed


def result_to_points(result: str, side: str) -> int:
    if result == "D":
        return 1
    if (result == "H" and side == "home") or (result == "A" and side == "away"):
        return 3
    return 0


def result_to_form_score(result: str, side: str) -> int:
    if result == "D":
        return 0
    if (result == "H" and side == "home") or (result == "A" and side == "away"):
        return 1
    return -1


def initialize_team_state(teams: list[str]) -> dict:
    # Trạng thái lưu thông tin tích lũy trước trận.
    state = {}
    for team in teams:
        state[team] = {
            "played": 0,
            "wins": 0,
            "points": 0,
            "goal_diff": 0,
            "recent_form": deque(maxlen=3),
            "last_match_date": None,
            "elo": 1500.0,
        }
    return state


def compute_rankings(team_state: dict) -> dict:
    # Tính thứ hạng tạm thời dựa trên điểm, hiệu số, số trận thắng.
    ordered = sorted(
        team_state.items(),
        key=lambda kv: (kv[1]["points"], kv[1]["goal_diff"], kv[1]["wins"]),
        reverse=True,
    )
    rankings = {}
    for idx, (team, _) in enumerate(ordered, start=1):
        rankings[team] = idx
    return rankings


def safe_win_rate(team_info: dict) -> float:
    if team_info["played"] == 0:
        return 0.5
    return team_info["wins"] / team_info["played"]


def safe_points_per_game(team_info: dict) -> float:
    if team_info["played"] == 0:
        return 1.5
    return team_info["points"] / team_info["played"]


def safe_elo(team_info: dict) -> float:
    return float(team_info["elo"])


def safe_recent_form(team_info: dict) -> int:
    if not team_info["recent_form"]:
        return 0
    return int(sum(team_info["recent_form"]))


def safe_rest_days(team_info: dict, match_date: pd.Timestamp) -> int:
    # Số ngày nghỉ từ trận trước, dùng để đo độ mệt.
    if team_info["last_match_date"] is None:
        return 7
    delta = (match_date - team_info["last_match_date"]).days
    return int(max(delta, 0))


def pick_first_valid_odds(row: pd.Series, columns: list[str]) -> float | None:
    for col in columns:
        if col in row.index and pd.notna(row[col]):
            value = float(row[col])
            if value > 1.0:
                return value
    return None


def market_implied_prob_diff(row: pd.Series) -> float:
    # Xác suất ngụ ý từ kèo nhà cái (chênh lệch home vs away).
    home_odd = pick_first_valid_odds(row, HOME_ODDS_COLUMNS)
    away_odd = pick_first_valid_odds(row, AWAY_ODDS_COLUMNS)
    if home_odd is None or away_odd is None:
        return 0.0
    return (1.0 / home_odd) - (1.0 / away_odd)


def expected_score(elo_a: float, elo_b: float, home_advantage: float = 50.0) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo_a + home_advantage - elo_b) / 400.0))


def update_elo(home_info: dict, away_info: dict, ftr: str, k: float = 20.0) -> None:
    # Cập nhật Elo sau khi biết kết quả trận.
    exp_home = expected_score(home_info["elo"], away_info["elo"])
    exp_away = 1.0 - exp_home
    actual_home = 1.0 if ftr == "H" else 0.5 if ftr == "D" else 0.0
    actual_away = 1.0 - actual_home
    home_info["elo"] += k * (actual_home - exp_home)
    away_info["elo"] += k * (actual_away - exp_away)


def process_one_season(csv_path: Path) -> pd.DataFrame:
    # Xử lý một mùa: làm sạch dữ liệu và tạo feature tiền trận.
    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong {csv_path.name}: {missing}")

    season = season_from_filename(csv_path)
    optional_cols = [col for col in HOME_ODDS_COLUMNS + AWAY_ODDS_COLUMNS if col in df.columns]
    # Chỉ lấy cột cần thiết + odds nếu có.
    season_df = df[REQUIRED_COLUMNS + optional_cols].copy()
    season_df["Date"] = parse_date_column(season_df["Date"])
    season_df["HomeTeam"] = season_df["HomeTeam"].map(normalize_team_name)
    season_df["AwayTeam"] = season_df["AwayTeam"].map(normalize_team_name)
    season_df = season_df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"])
    season_df = season_df.sort_values("Date").reset_index(drop=True)

    teams = sorted(set(season_df["HomeTeam"]).union(set(season_df["AwayTeam"])))
    team_state = initialize_team_state(teams)

    rows = []

    for _, row in season_df.iterrows():
        # Tạo feature trước trận từ trạng thái tích lũy.
        match_date = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        ftr = row["FTR"]
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])

        rankings = compute_rankings(team_state)
        home_rank = rankings.get(home, len(teams))
        away_rank = rankings.get(away, len(teams))

        home_info = team_state[home]
        away_info = team_state[away]

        rank_diff = away_rank - home_rank
        win_rate_diff = safe_win_rate(home_info) - safe_win_rate(away_info)
        recent_form_diff = safe_recent_form(home_info) - safe_recent_form(away_info)
        rest_days_diff = safe_rest_days(home_info, match_date) - safe_rest_days(away_info, match_date)
        ppg_diff = safe_points_per_game(home_info) - safe_points_per_game(away_info)
        market_prob_diff = market_implied_prob_diff(row)
        elo_home = safe_elo(home_info)
        elo_away = safe_elo(away_info)
        elo_diff = elo_home - elo_away

        rows.append(
            {
                "Date": match_date.date().isoformat(),
                "Season": season,
                "HomeTeam": home,
                "AwayTeam": away,
                "RankDiff": rank_diff,
                "WinRateDiff": round(win_rate_diff, 4),
                "HomeAdvantage": 1,
                "RecentFormDiff": int(recent_form_diff),
                "RestDaysDiff": int(rest_days_diff),
                "PPGDiff": round(ppg_diff, 4),
                "MarketImpliedProbDiff": round(market_prob_diff, 6),
                "ELODiff": round(elo_diff, 4),
                "target": 1 if ftr == "H" else 0,
            }
        )

        # Cập nhật trạng thái sau khi tạo feature.
        home_points = result_to_points(ftr, "home")
        away_points = result_to_points(ftr, "away")
        home_form = result_to_form_score(ftr, "home")
        away_form = result_to_form_score(ftr, "away")

        home_info["played"] += 1
        away_info["played"] += 1

        home_info["points"] += home_points
        away_info["points"] += away_points

        if home_points == 3:
            home_info["wins"] += 1
        if away_points == 3:
            away_info["wins"] += 1

        home_info["goal_diff"] += fthg - ftag
        away_info["goal_diff"] += ftag - fthg

        home_info["recent_form"].append(home_form)
        away_info["recent_form"].append(away_form)

        home_info["last_match_date"] = match_date
        away_info["last_match_date"] = match_date
        update_elo(home_info, away_info, ftr)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("E0_*.csv"))
    if not csv_files:
        raise ValueError(f"Không tìm thấy file E0_*.csv trong {input_dir}")

    all_frames = [process_one_season(path) for path in csv_files]
    final_df = pd.concat(all_frames, ignore_index=True)
    final_df = final_df.sort_values(["Date", "Season", "HomeTeam"]).reset_index(drop=True)

    # Lọc theo phạm vi mùa nếu người dùng yêu cầu.
    if args.min_season or args.max_season:
        season_num = pd.to_numeric(final_df["Season"], errors="coerce")
        mask = pd.Series(True, index=final_df.index)
        if args.min_season:
            mask &= season_num.ge(args.min_season)
        if args.max_season:
            mask &= season_num.le(args.max_season)
        final_df = final_df[mask.fillna(False)].copy()

    final_df.to_csv(output_path, index=False)
    print(f"Đã tạo {output_path} với {len(final_df)} dòng từ {len(csv_files)} file mùa.")


if __name__ == "__main__":
    main()
