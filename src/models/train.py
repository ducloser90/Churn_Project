import mlflow
import mlflow.xgboost
import pandas as pd

from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_model(
    dataframe: pd.DataFrame,
    target_column: str
) -> None:
    """
    Train an XGBoost classifier and log the run to MLflow.

    Args:
        dataframe: Feature dataset containing the target column.
        target_column: Name of the target variable.
    """
    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    train_features, test_features, train_target, test_target = (
        train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42
        )
    )

    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    with mlflow.start_run():
        # Train the model
        classifier.fit(train_features, train_target)

        predictions = classifier.predict(test_features)

        accuracy = accuracy_score(test_target, predictions)
        recall = recall_score(test_target, predictions)

        # Log model parameters
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 6)

        # Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)

        # Log trained model
        mlflow.xgboost.log_model(classifier, "model")

        # Log training dataset as an MLflow input
        training_dataset = mlflow.data.from_pandas(
            dataframe,
            source="training_data"
        )

        mlflow.log_input(
            training_dataset,
            context="training"
        )

        print(
            f"Model trained successfully. "
            f"Accuracy: {accuracy:.4f}, "
            f"Recall: {recall:.4f}"
        )


if __name__ == "__main__":
    from src.data.load_data import load_csv
    from src.data.preprocess_data import preprocess_data
    from src.features.build_features import build_features

    dataset = load_csv("data/raw/data.csv")
    cleaned_data = preprocess_data(dataset)
    feature_data = build_features(cleaned_data)

    train_model(feature_data, target_column="Churn")