import pandas as pd
import soccerdata as sd
import requests
import time
from src.utils import setup_logger
from src.config import RAW_DATA_DIR, FOOTBALL_DATA_API_KEY

logger = setup_logger()

class FootballDataLoader:
    PLAYER_STAT_TYPES = [
        "standard",
        "shooting",
        "playing_time",
        "misc"
    ]

    TEAM_STAT_TYPES = [
        "standard",
        "shooting",
        "misc"
    ]

    def __init__(self):
        self.fbref = sd.FBref(
            leagues="Big 5 European Leagues Combined",
            seasons=["2023-2024","2024-2025","2025-2026"]
        )

    def save_parquet(self, df: pd.DataFrame, filename: str):
        output_path = RAW_DATA_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path, index=False)

        logger.info(f"Saved: {output_path}")

    # player stats
    def load_player_stats(self):
        logger.info(f"Loading player stats...")

        for stat_type in self.PLAYER_STAT_TYPES:
            try:
                logger.info(f"Downloading player {stat_type} stats...")

                # download player data via fbref
                df = self.fbref.read_player_season_stats(
                    stat_type=stat_type
                ).reset_index()

                # standardise age column format
                df["age"] = df["age"].astype(str).str.split("-").str[0]
                df["age"] = pd.to_numeric(df["age"], errors="coerce")

                # fix bundesliga league name (loads with null for some reason)
                df["league"] = df["league"].fillna("GER-Bundesliga")

                filename = f"player_{stat_type}.parquet"
                self.save_parquet(df, filename)

            except Exception as e:
                logger.error(f"Failed player {stat_type}: {e}")

    # team stats
    def load_team_stats(self):
        logger.info("Loading team stats...")

        for stat_type in self.TEAM_STAT_TYPES:
            try:
                logger.info(f"Downloading team {stat_type} stats...")

                # download team data via fbref
                df = self.fbref.read_team_season_stats(
                    stat_type=stat_type,
                ).reset_index()

                # fix bundesliga league name (loads with null for some reason)
                df["league"] = df["league"].fillna("GER-Bundesliga")

                filename = f"team_{stat_type}.parquet"
                self.save_parquet(df, filename)

                # opponent data to provide conceded goals
                if stat_type == "standard":
                    df_opp = self.fbref.read_team_season_stats(
                        stat_type=stat_type,
                        opponent_stats=True
                    ).reset_index()

                    df_opp["league"] = df_opp["league"].fillna("GER-Bundesliga")

                    self.save_parquet(df_opp, "opponent_standard.parquet")

            except Exception as e:
                logger.error(f"Failed team {stat_type}: {e}")

    # match results API
    def load_matches_api(self):
        logger.info("Loading match data...")

        headers = {
            "X-Auth-Token": FOOTBALL_DATA_API_KEY
        }

        competitions = {
            "PL": "Premier League",
            "PD": "La Liga",
            "BL1": "Bundesliga",
            "SA": "Serie A",
            "FL1": "Ligue 1"
        }

        all_matches = []

        for code, league_name in competitions.items():
            try:
                # download only desired seasons
                for season in range(2023, 2026):
                    url = f"https://api.football-data.org/v4/competitions/{code}/matches?season={season}"

                    response = requests.get(url, headers=headers, timeout=30)
                    time.sleep(6) # free API max 10 calls per minute ;)
                    response.raise_for_status()

                    data = response.json()

                    # simplify json
                    matches = pd.json_normalize(data["matches"])
                    matches["league"] = league_name

                    all_matches.append(matches)

                    logger.info(f"Downloaded {league_name}, season {season}")

            except Exception as e:
                logger.error(f"Failed {league_name}: {e}")
            
        if not all_matches:
            logger.warning("No match data collected")
            return

        final_df = pd.concat(all_matches, ignore_index=True)

        self.save_parquet(final_df, "matches_api.parquet")

    def load_elo_rating(self):
        logger.info("Loading ELO rating...")

        clubelo = sd.ClubElo()

        # 3 dates from each season - start, middle and end
        dates = [
            "2023-09-15", "2024-01-15", "2024-05-15", 
            "2024-09-15", "2025-01-15", "2025-05-15",
            "2025-09-15", "2026-01-15", "2026-05-15",
        ]
        
        elo_df = [clubelo.read_by_date(date).reset_index() for date in dates]
        elo_df = pd.concat(elo_df, ignore_index=True)

        self.save_parquet(elo_df, "elo_rating.parquet")
        logger.info("Downloaded ELO rating")

if __name__ == "__main__":
    loader = FootballDataLoader()
    loader.load_player_stats()
    loader.load_team_stats()
    loader.load_matches_api()
    loader.load_elo_rating()