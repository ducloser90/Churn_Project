"""
Production inference pipeline for the Telco Customer Churn model.

This module loads the trained MLflow model and applies the same feature
transformations used during training. It ensures that inference data follows
the same feature schema and column ordering expected by the model.
"""

import glob
import os

import mlflow
import pandas as pd


# Model loading configuration
MODEL_DIRECTORY = os.path.join(os.path.dirname(__file__), "model")


def _load_model():
    """
    Load the trained model from the configured model directory.

    If the production model directory is unavailable, the function attempts
    to load the most recently modified model from the local MLflow artifacts.

    Returns:
        Loaded MLflow pyfunc model.

    Raises:
        Exception: If no model can be loaded.
    """
    try:
        loaded_model = mlflow.pyfunc.load_model(MODEL_DIRECTORY)

        print(
            f"Model loaded successfully from {MODEL_DIRECTORY}"
        )

        return loaded_model

    except Exception as model_error:
        print(
            f"Failed to load model from "
            f"{MODEL_DIRECTORY}: {model_error}"
        )

        try:
            local_model_paths = glob.glob(
                "./mlruns/*/*/artifacts/model"
            )

            if not local_model_paths:
                raise Exception(
                    "No model found in local MLflow artifacts."
                )

            latest_model_path = max(
                local_model_paths,
                key=os.path.getmtime
            )

            loaded_model = mlflow.pyfunc.load_model(
                latest_model_path
            )

            print(
                f"Fallback model loaded from "
                f"{latest_model_path}"
            )

            return loaded_model

        except Exception as fallback_error:
            raise Exception(
                f"Failed to load model: {model_error}. "
                f"Fallback failed: {fallback_error}"
            )


model = _load_model()


# Feature schema configuration
try:
    feature_schema_path = os.path.join(
        MODEL_DIRECTORY,
        "feature_columns.txt"
    )

    with open(feature_schema_path) as schema_file:
        FEATURE_COLUMNS = [
            line.strip()
            for line in schema_file
            if line.strip()
        ]

    print(
        f"Loaded {len(FEATURE_COLUMNS)} feature columns "
        "from training schema."
    )

except Exception as schema_error:
    raise Exception(
        f"Failed to load feature columns: {schema_error}"
    )


# Deterministic binary mappings used during training
BINARY_ENCODINGS = {
    "gender": {
        "Female": 0,
        "Male": 1
    },
    "Partner": {
        "No": 0,
        "Yes": 1
    },
    "Dependents": {
        "No": 0,
        "Yes": 1
    },
    "PhoneService": {
        "No": 0,
        "Yes": 1
    },
    "PaperlessBilling": {
        "No": 0,
        "Yes": 1
    },
}


# Numeric columns requiring type conversion
NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def _transform_for_serving(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Transform raw customer data into the model's expected feature format.

    The transformation must remain consistent with the training pipeline.

    Processing steps:
        1. Clean column names.
        2. Convert numeric columns to numeric types.
        3. Apply deterministic binary encoding.
        4. One-hot encode remaining categorical columns.
        5. Convert boolean columns to integers.
        6. Align columns with the training feature schema.

    Args:
        dataframe: Raw customer data.

    Returns:
        Transformed DataFrame ready for model inference.
    """
    transformed_data = dataframe.copy()

    # Clean column names
    transformed_data.columns = (
        transformed_data.columns.str.strip()
    )

    # Convert numeric columns
    for column_name in NUMERIC_COLUMNS:
        if column_name in transformed_data.columns:
            transformed_data[column_name] = pd.to_numeric(
                transformed_data[column_name],
                errors="coerce"
            )

            transformed_data[column_name] = (
                transformed_data[column_name].fillna(0)
            )

    # Apply deterministic binary encoding
    for column_name, encoding_map in BINARY_ENCODINGS.items():
        if column_name in transformed_data.columns:
            transformed_data[column_name] = (
                transformed_data[column_name]
                .astype(str)
                .str.strip()
                .map(encoding_map)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    # One-hot encode remaining categorical columns
    categorical_columns = (
        transformed_data
        .select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    if categorical_columns:
        transformed_data = pd.get_dummies(
            transformed_data,
            columns=categorical_columns,
            drop_first=True
        )

    # Convert boolean columns to integers
    boolean_columns = (
        transformed_data
        .select_dtypes(include=["bool"])
        .columns
    )

    if len(boolean_columns) > 0:
        transformed_data[boolean_columns] = (
            transformed_data[boolean_columns].astype(int)
        )

    # Align inference features with the training schema
    transformed_data = transformed_data.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    return transformed_data


def predict(input_data: dict) -> str:
    """
    Predict customer churn from raw customer information.

    Args:
        input_data: Dictionary containing customer information.

    Returns:
        Human-readable churn prediction:
            - "Likely to churn"
            - "Not likely to churn"
    """
    # Convert input dictionary to a single-row DataFrame
    customer_data = pd.DataFrame([input_data])

    # Apply training-compatible transformations
    model_features = _transform_for_serving(
        customer_data
    )

    # Generate prediction
    try:
        predictions = model.predict(model_features)

        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()

        if isinstance(predictions, (list, tuple)):
            prediction = predictions[0]
        else:
            prediction = predictions

    except Exception as prediction_error:
        raise Exception(
            f"Model prediction failed: {prediction_error}"
        )

    # Convert model output into business-friendly result
    if prediction == 1:
        return "Likely to churn"

    return "Not likely to churn"


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "InternetService": "DSL",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    prediction = predict(sample_customer)

    print(f"Prediction: {prediction}")