import joblib
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
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

class MatchPredictor:
    def __init__(self):
        self.df = pd.read_parquet(PROCESSED_DATA_DIR / "matches_features.parquet")

    def create_targets(self):
        self.df["target"] = np.select(
            [self.df["home_win"] == 1, self.df["draw"] == 1], ["H", "D"], default="A"
        )

        logger.info("Created targets")

    def prepare_data(self):
        # train data - older seasons
        # test data - 'recent' matches
        self.df = self.df.dropna(subset=FEATURES)
        self.df = self.df.sort_values("date")

        X = self.df[FEATURES]
        y = self.df["target"]

        split_idx = int(len(self.df) * 0.8)

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test
    
    def objective(self, trial):
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42
        }

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        return accuracy_score(y_test, preds)
    
    def tune_model(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=50)

        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params


    def train_model(self):
        best_params = self.tune_model()

        X_train, X_test, y_train, y_test = self.prepare_data()

        model = LGBMClassifier(**best_params)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        logger.info(f"Accuracy: {acc:.4f}")

        joblib.dump(model, MODELS_DIR / "match_predictor.pkl")

        logger.info("Saved match predictor model")

if __name__ == "__main__":
    predictor = MatchPredictor()
    predictor.create_targets()
    predictor.train_model()