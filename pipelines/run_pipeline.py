import argparse
import json
import os
import sys
import time

import joblib
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


# Allow imports from the project root
PROJECT_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_DIRECTORY)

from src.data.load_data import load_csv
from src.data.preprocess_data import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


def main(arguments):
    """
    Run the complete Telco Customer Churn training pipeline.

    The pipeline performs data loading, validation, preprocessing,
    feature engineering, train/test splitting, model training,
    evaluation, and MLflow model logging.
    """

    # Configure MLflow
    tracking_uri = (
        arguments.mlflow_uri
        or f"file://{PROJECT_DIRECTORY}/mlruns"
    )
    

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(arguments.experiment)

    with mlflow.start_run():

        # Log pipeline configuration
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", arguments.threshold)
        mlflow.log_param("test_size", arguments.test_size)

        # Load data
        print("Loading data...")

        dataset = load_csv(arguments.input)

        print(
            f"Data loaded: {dataset.shape[0]} rows, "
            f"{dataset.shape[1]} columns"
        )

        # Validate data quality
        print("Validating data quality with Great Expectations...")

        is_valid, failed_expectations = validate_telco_data(dataset)

        mlflow.log_metric(
            "data_quality_pass",
            int(is_valid)
        )

        if not is_valid:
            mlflow.log_text(
                json.dumps(failed_expectations, indent=2),
                artifact_file="failed_expectations.json"
            )

            raise ValueError(
                "Data quality validation failed. "
                f"Issues: {failed_expectations}"
            )

        print("Data validation passed.")

        # Preprocess data
        print("Preprocessing data...")

        cleaned_data = preprocess_data(dataset)

        processed_file = os.path.join(
            PROJECT_DIRECTORY,
            "data",
            "processed",
            "telco_churn_processed.csv"
        )

        os.makedirs(
            os.path.dirname(processed_file),
            exist_ok=True
        )

        cleaned_data.to_csv(
            processed_file,
            index=False
        )

        print(
            f"Processed dataset saved to {processed_file} | "
            f"Shape: {cleaned_data.shape}"
        )

        # Feature engineering
        print("Building features...")

        target_column = arguments.target

        if target_column not in cleaned_data.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                "not found in dataset."
            )

        feature_data = build_features(
            cleaned_data,
            target_column=target_column
        )

        # Ensure boolean columns are compatible with XGBoost
        boolean_columns = feature_data.select_dtypes(
            include=["bool"]
        ).columns

        if len(boolean_columns) > 0:
            feature_data[boolean_columns] = (
                feature_data[boolean_columns].astype(int)
            )

        print(
            f"Feature engineering completed: "
            f"{feature_data.shape[1]} features"
        )

        # Save feature metadata
        artifacts_directory = os.path.join(
            PROJECT_DIRECTORY,
            "artifacts"
        )

        os.makedirs(
            artifacts_directory,
            exist_ok=True
        )

        feature_columns = list(
            feature_data.drop(
                columns=[target_column]
            ).columns
        )

        feature_metadata_file = os.path.join(
            artifacts_directory,
            "feature_columns.json"
        )

        with open(feature_metadata_file, "w") as metadata_file:
            json.dump(
                feature_columns,
                metadata_file
            )

        mlflow.log_text(
            "\n".join(feature_columns),
            artifact_file="feature_columns.txt"
        )

        preprocessing_metadata = {
            "feature_columns": feature_columns,
            "target": target_column,
        }

        preprocessing_file = os.path.join(
            artifacts_directory,
            "preprocessing.pkl"
        )

        joblib.dump(
            preprocessing_metadata,
            preprocessing_file
        )

        mlflow.log_artifact(preprocessing_file)

        print(
            f"Saved {len(feature_columns)} feature columns "
            "for serving consistency."
        )

        # Train/test split
        print("Splitting data...")

        features = feature_data.drop(
            columns=[target_column]
        )

        target = feature_data[target_column]

        train_features, test_features, train_target, test_target = (
            train_test_split(
                features,
                target,
                test_size=arguments.test_size,
                stratify=target,
                random_state=42,
            )
        )

        print(
            f"Train: {train_features.shape[0]} samples | "
            f"Test: {test_features.shape[0]} samples"
        )

        # Handle class imbalance
        positive_class_weight = (
            (train_target == 0).sum()
            / (train_target == 1).sum()
        )

        print(
            f"Class imbalance ratio: "
            f"{positive_class_weight:.2f}"
        )

        # Train XGBoost classifier
        print("Training XGBoost model...")

        classifier = XGBClassifier(
            n_estimators=301,
            learning_rate=0.034,
            max_depth=7,
            subsample=0.95,
            colsample_bytree=0.98,
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=positive_class_weight,
        )

        training_start = time.time()

        classifier.fit(
            train_features,
            train_target
        )

        training_duration = time.time() - training_start

        mlflow.log_metric(
            "train_time",
            training_duration
        )

        print(
            f"Model trained in "
            f"{training_duration:.2f} seconds"
        )

        # Model evaluation
        print("Evaluating model performance...")

        prediction_start = time.time()

        churn_probabilities = classifier.predict_proba(
            test_features
        )[:, 1]

        predictions = (
            churn_probabilities >= arguments.threshold
        ).astype(int)

        prediction_duration = time.time() - prediction_start

        mlflow.log_metric(
            "pred_time",
            prediction_duration
        )

        # Calculate evaluation metrics
        precision = precision_score(
            test_target,
            predictions
        )

        recall = recall_score(
            test_target,
            predictions
        )

        f1 = f1_score(
            test_target,
            predictions
        )

        roc_auc = roc_auc_score(
            test_target,
            churn_probabilities
        )

        # Log metrics to MLflow
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print("Model performance:")
        print(
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f}"
        )
        print(
            f"F1 Score: {f1:.3f} | "
            f"ROC AUC: {roc_auc:.3f}"
        )

        # Log model to MLflow
        print("Saving model to MLflow...")

        mlflow.xgboost.log_model(
            xgb_model=classifier,
            artifact_path="model"
        )

        print("Model saved to MLflow.")

        # Performance summary
        samples_per_second = (
            len(test_features) / prediction_duration
        )

        print("\nPerformance summary:")
        print(
            f"Training time: "
            f"{training_duration:.2f}s"
        )
        print(
            f"Inference time: "
            f"{prediction_duration:.4f}s"
        )
        print(
            f"Samples per second: "
            f"{samples_per_second:.0f}"
        )

        print("\nDetailed classification report:")
        print(
            classification_report(
                test_target,
                predictions,
                digits=3
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the Telco churn training pipeline "
            "with XGBoost and MLflow."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file."
    )

    parser.add_argument(
        "--target",
        type=str,
        default="Churn",
        help="Name of the target column."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Classification threshold."
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data used for testing."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="Telco Churn",
        help="MLflow experiment name."
    )

    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help=(
            "MLflow tracking URI. "
            "Defaults to the project's mlruns directory."
        )
    )

    cli_arguments = parser.parse_args()

    main(cli_arguments)