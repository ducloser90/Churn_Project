import optuna

from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


def tune_model(features, target):
    """
    Optimize XGBoost hyperparameters using Optuna.

    The optimization objective is the mean recall obtained from
    3-fold cross-validation.

    Args:
        features: Feature dataset used for model training.
        target: Target variable.

    Returns:
        Dictionary containing the best hyperparameters found by Optuna.
    """

    def objective(trial):
        hyperparameters = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                300,
                800
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.2
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.5,
                1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0
            ),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }

        classifier = XGBClassifier(**hyperparameters)

        validation_scores = cross_val_score(
            classifier,
            features,
            target,
            cv=3,
            scoring="recall"
        )

        return validation_scores.mean()

    optimization_study = optuna.create_study(
        direction="maximize"
    )

    optimization_study.optimize(
        objective,
        n_trials=20
    )

    best_parameters = optimization_study.best_params

    print("Best parameters:", best_parameters)

    return best_parameters


if __name__ == "__main__":
    from src.data.load_data import load_csv
    from src.data.preprocess_data import preprocess_data
    from src.features.build_features import build_features

    dataset = load_csv("data/raw/data.csv")
    cleaned_data = preprocess_data(dataset)
    feature_data = build_features(cleaned_data)

    features = feature_data.drop(columns=["Churn"])
    target = feature_data["Churn"]

    best_params = tune_model(features, target)

    print("\nOptimization completed.")
    print("Best parameters:")
    print(best_params)