import joblib
import shap
import pandas as pd
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils import setup_logger

logger = setup_logger()

FEATURES = [
    "home_form_5",
    "away_form_5",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_goals_for_5",
    "home_goals_against_5",
    "away_goals_for_5",
    "away_goals_against_5",
    "home_goal_diff_5",
    "away_goal_diff_5"
]

class ModelExplainer():
    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "match_predictor.pkl")

        self.df = pd.read_parquet(PROCESSED_DATA_DIR / "matches_features.parquet")
        self.df = self.df.dropna(subset=FEATURES)

        self.X = self.df[FEATURES]

    def create_shap_values(self):
        explainer = shap.TreeExplainer(self.model)

        shap_values = explainer.shap_values(self.X)

        logger.info("Created SHAP values")

        return explainer, shap_values
    
    def create_summary_plot(self):
        _, shap_values = self.create_shap_values()

        shap.summary_plot(shap_values, self.X)

        logger.info("Created SHAP summary plot")

if __name__ == "__main__":
    explainer = ModelExplainer()
    explainer.create_summary_plot()