import pandas as pd
import numpy as np
from pathlib import Path
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils import setup_logger

logger = setup_logger()

class DataPreprocessor:
    PLAYER_FILES = [
        "player_standard.parquet",
        "player_shooting.parquet",
        "player_playing_time.parquet",
        "player_misc.parquet"
    ]

    TEAM_FILES = [
        "team_standard.parquet",
        "team_shooting.parquet",
        "team_misc.parquet",
    ]

    # basic cleaning
    @staticmethod
    def flatten_columns(df: pd.DataFrame):
        # merge multiindex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(x).strip() for x in col if x and str(x) != "nan"])
                for col in df.columns
            ]
        
        return df
    
    @staticmethod
    def clean_column_names(df: pd.DataFrame):
        # standardise column names
        df.columns = (
            df.columns
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("%", "pct")
            .str.replace("/", "_per_")
            .str.replace(r"[^\w]", "", regex=True)
            .str.rstrip("_")
        )

        return df

    @staticmethod
    def remove_duplicate_columns(df: pd.DataFrame):
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    @staticmethod
    def convert_numeric_columns(df: pd.DataFrame):
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

        return df

    # player dataset
    def build_player_dataset(self):
        logger.info("Building player dataset...")

        dfs = [self.load_and_preprocess_file(file) for file in self.PLAYER_FILES]

        merged_df = dfs[0].copy()
        merge_keys = ["player", "team", "season"]

        for df in dfs[1:]:
            valid_keys = [key for key in merge_keys if key in df.columns]
            df = self.prepare_for_merge(merged_df, df, valid_keys)
            merged_df = merged_df.merge(df, on=valid_keys, how="left")

        merged_df = self.remove_duplicate_columns(merged_df)
        merged_df = merged_df.drop_duplicates()
        
        replacements = {
            "Manchester Utd": "Manchester United"
        }

        merged_df["team"] = merged_df["team"].replace(replacements)

        self.save_dataset(merged_df, PROCESSED_DATA_DIR / "players.parquet")

        logger.info(f"Saved players dataset: {merged_df.shape}")

        return merged_df

    # team dataset
    def build_team_dataset(self):
        logger.info("Building team dataset...")

        opp_stats = self.prepare_opponent_stats()

        dfs = [self.load_and_preprocess_file(file) for file in self.TEAM_FILES]
        dfs.append(opp_stats)

        merged_df = dfs[0].copy()
        merge_keys = ["team", "season"]

        for df in dfs[1:]:
            valid_keys = [key for key in merge_keys if key in df.columns]
            df = self.prepare_for_merge(merged_df, df, valid_keys)
            merged_df = merged_df.merge(df, on=valid_keys, how="left")

        merged_df = self.remove_duplicate_columns(merged_df)
        merged_df = merged_df.drop_duplicates()

        replacements = {
            "Manchester Utd": "Manchester United"
        }

        merged_df["team"] = merged_df["team"].replace(replacements)

        self.save_dataset(merged_df, PROCESSED_DATA_DIR / "teams.parquet")
        logger.info(f"Saved teams dataset: {merged_df.shape}")

        return merged_df

    # match dataset
    def build_match_dataset(self):
        logger.info("Building match dataset...")

        df = pd.read_parquet(RAW_DATA_DIR / "matches_api.parquet")
        df = self.clean_column_names(df)

        important_columns = [
            "utcdate",
            "status",
            "matchday",
            "hometeamname",
            "awayteamname",
            "scorefulltimehome",
            "scorefulltimeaway",
            "competitionname",
        ]

        df = df[important_columns].copy().rename(columns={
            "utcdate": "date",
            "hometeamname": "home_team",
            "awayteamname": "away_team",
            "scorefulltimehome": "home_goals",
            "scorefulltimeaway": "away_goals",
            "competitionname": "league",
        })

        df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)
        df["draw"] = (df["home_goals"] == df["away_goals"]).astype(int)
        df["away_win"] = (df["home_goals"] < df["away_goals"]).astype(int)
        df["total_goals"] = (df["home_goals"] + df["away_goals"])
        df["over_25"] = (df["total_goals"] > 2.5).astype(int)

        self.save_dataset(df, PROCESSED_DATA_DIR / "matches.parquet")

        logger.info(f"Saved matches dataset: {df.shape}")

        return df
    
    def clear_elo_rating(self):
        elo_df = pd.read_parquet(RAW_DATA_DIR / "elo_rating.parquet")

        leagues = [
            "ENG-Premier League", "GER-Bundesliga", "FRA-Ligue 1", "ESP-La Liga", "ITA-Serie A"
        ]

        # leave only teams from big 5 leagues
        elo_big5 = elo_df[elo_df["league"].isin(leagues)].copy()

        # data clearing
        elo_big5 = elo_big5.drop(columns="level")
        elo_big5["elo"] = elo_big5["elo"].round()
        
        for col in ["from", "to"]:
            elo_big5[col] = pd.to_datetime(elo_big5[col]).dt.strftime('%Y-%m-%d')
        
        elo_big5 = elo_big5.sort_values(by="team").reset_index(drop=True)

        logger.info(f"Elo ratings cleared: {elo_big5.shape}")
        elo_big5.to_parquet(PROCESSED_DATA_DIR / "elo_rating_big5.parquet")

    def prepare_opponent_stats(self):
        # standardise team names and add column prefix for proper merge
        opp_stats = self.load_and_preprocess_file(RAW_DATA_DIR / "opponent_standard.parquet")

        opp_stats["team"] = opp_stats["team"].str.replace(r'^vs\s+', '', regex=True).str.strip()
        opp_stats = opp_stats.drop(columns="url")

        exclude = {"league", "season", "team", "players_used", "age"}
        cols_to_rename = [col for col in opp_stats.columns if col not in exclude]

        opp_stats = opp_stats.rename(columns={col: f"opp_{col}" for col in cols_to_rename})

        return opp_stats

    def load_and_preprocess_file(self, file):
        path = RAW_DATA_DIR / file

        df = pd.read_parquet(path).reset_index()
        df = df.drop(columns=["index", "level_0"], errors="ignore") # drop redundant multiindex column
        df = self.flatten_columns(df)
        df = self.clean_column_names(df)
        df = self.remove_duplicate_columns(df)
        df = self.convert_numeric_columns(df)

        return df
    
    def prepare_for_merge(self, left, right, keys):
        # drop redundant columns
        right = right.drop(
            columns=[c for c in right.columns if c in left.columns and c not in keys],
            errors="ignore"
        )

        return right

    def save_dataset(self, df, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.build_player_dataset()
    processor.build_team_dataset()
    processor.build_match_dataset()
    processor.clear_elo_rating()