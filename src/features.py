import pandas as pd
import numpy as np
import soccerdata as sd
from rapidfuzz import process, fuzz
from config import PROCESSED_DATA_DIR
from utils import setup_logger

logger = setup_logger()

class FeatureEngineer():
    def __init__(self):
        self.players = pd.read_parquet(PROCESSED_DATA_DIR / "players.parquet")
        self.teams = pd.read_parquet(PROCESSED_DATA_DIR / "teams.parquet")
        self.matches = pd.read_parquet(PROCESSED_DATA_DIR / "matches.parquet")
        self.elo = pd.read_parquet(PROCESSED_DATA_DIR / "elo_rating_big5.parquet")

    def create_percentile(self):
        stat_cols = [
            col for col in self.players.columns
            if (
                col.startswith("performance_")
                or col.startswith("standard_")
                or col.startswith("per_90_")
            )
        ]

        excluded = ["standard_sotpct", "team_success__per", "team_success__per_90"]

        stat_cols = [col for col in stat_cols if col not in excluded]
        
        for col in stat_cols:
            self.players[f"{col}_pct"] = (self.players[col].rank(pct=True) * 100).round(2)

        logger.info("Created percentile features")

    def create_team_style(self):
        # based on scored goals and assists
        # more = better
        self.teams["attacking_rating"] = (
            self.teams["per_90_minutes_gls"] + self.teams["per_90_minutes_ast"]
        ).round(2)

        # based on conceded goals 
        # result closer to 1 = better
        self.teams["defensive_rating"] = (
            1 / (self.teams["opp_per_90_minutes_gls"] + 1)
        ).round(2)

        # based on card recieved
        # less = better
        self.teams["discipline_rating"] = (
            self.teams["performance_crdy"] + self.teams["performance_crdr"]
        )

        logger.info("Created team style features")

    def create_rolling_form(self):
        # avg points per match in last 5 matches
        self.matches = self.matches.sort_values("date")

        self.matches["home_points"] = np.select(
            [
                self.matches["home_goals"] > self.matches["away_goals"],
                self.matches["home_goals"] == self.matches["away_goals"]
            ],
            [3, 1],
            default=0
        )

        self.matches["away_points"] = np.select(
            [
                self.matches["away_goals"] > self.matches["home_goals"],
                self.matches["away_goals"] == self.matches["home_goals"]
            ],
            [3, 1],
            default=0
        )

        self.matches["home_form_5"] = (
            self.matches.groupby("home_team")["home_points"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean().round(2))
        )

        self.matches["away_form_5"] = (
            self.matches.groupby("away_team")["away_points"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean().round(2))
        )

        logger.info("Created rolling form features")

    def standardise_team_names(self):
        # manual names fix
        replacements = {
            "Man United": "Manchester United",
            "Man City": "Manchester City",
            "Wolverhampton Wanderers FC": "Wolves",
            "Stade Rennais FC 1901": "Rennes",
            "Paris SG": "Paris Saint-Germain",
            "Club Atlético de Madrid": "Atlético Madrid",
            "Atletico": "Atlético Madrid",
            "RC Celta de Vigo": "Celta Vigo",
            "RCD Espanyol de Barcelona": "Espanyol",
            "Cadiz": "Cádiz",
            "Almeria": "Almería",
            "Alaves": "Alavés",
            "Leganes": "Leganés",
            "Bilbao": "Athletic Club",
            "FC St. Pauli 1910": "St. Pauli",
            "FC Bayern München": "Bayern Munich",
            "Borussia Mönchengladbach": "Gladbach",
            "Koeln": "Köln"
        }

        for col in ["home_team", "away_team"]:
            self.matches[col] = self.matches[col].replace(replacements)

        self.elo["team"] = self.elo["team"].replace(replacements)

        canonical_teams = pd.unique(self.teams["team"])

        # calc similarity between team names
        def map_name(name):
            match = process.extractOne(name, canonical_teams, scorer=fuzz.WRatio)

            if match and match[1] >= 90:
                return match[0]
            
            return name
        
        self.matches["home_team"] = self.matches["home_team"].apply(map_name)
        self.matches["away_team"] = self.matches["away_team"].apply(map_name)

        self.elo["team"] = self.elo["team"].apply(map_name)

        logger.info("Standardised team names")

    def assign_elo_rating(self):
        self.matches["date"] = pd.to_datetime(self.matches["date"])
        self.elo["from"] = pd.to_datetime(self.elo["from"])

        self.matches = self.matches.sort_values("date").reset_index(drop=True)
        self.elo = self.elo.sort_values("from").reset_index(drop=True)

        home_matches = self.matches[["date", "home_team"]].copy()
        home_matches = home_matches.rename(columns={"home_team": "team"})

        # find nearest elo rating to the match date and merge
        home_merged = pd.merge_asof(
            home_matches,
            self.elo[["team", "from", "elo"]],
            left_on="date",
            right_on="from",
            by="team",
            direction="nearest"
        )

        away_matches = self.matches[["date", "away_team"]].copy()
        away_matches = away_matches.rename(columns={"away_team": "team"})

        # find nearest elo rating to the match date and merge
        away_merged = pd.merge_asof(
            away_matches,
            self.elo[["team", "from", "elo"]],
            left_on="date",
            right_on="from",
            by="team",
            direction="nearest"
        )

        self.matches["home_elo"] = home_merged["elo"]
        self.matches["away_elo"] = away_merged["elo"]

        self.matches["date"] = pd.to_datetime(self.matches["date"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        self.elo["from"] = pd.to_datetime(self.elo["from"]).dt.strftime('%Y-%m-%d')

        self.matches["elo_diff"] = self.matches["home_elo"] - self.matches["away_elo"]

        logger.info("Assigned ELO ratings to match results")

    def save_datasets(self):
        self.players.to_parquet(PROCESSED_DATA_DIR / "players_features.parquet", index=False)
        self.teams.to_parquet(PROCESSED_DATA_DIR / "teams_features.parquet", index=False)
        self.matches.to_parquet(PROCESSED_DATA_DIR / "matches_features.parquet", index=False)
        self.elo.to_parquet(PROCESSED_DATA_DIR / "elo_rating_big5.parquet", index=False)

        logger.info("Saved feature datasets")

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.create_percentile()
    engineer.create_team_style()
    engineer.create_rolling_form()
    engineer.standardise_team_names()
    engineer.assign_elo_rating()
    engineer.save_datasets()