import pandas as pd


def _encode_binary_column(series: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to features with two categories.

    Known Yes/No and Male/Female features use explicit mappings. Other
    binary features use alphabetical ordering to ensure consistent encoding.

    Args:
        series: Categorical pandas Series to encode.

    Returns:
        Encoded Series with binary integer values when applicable.
    """
    unique_values = list(
        pd.Series(series.dropna().unique()).astype(str)
    )
    value_set = set(unique_values)

    # Apply explicit mapping for Yes/No features
    if value_set == {"Yes", "No"}:
        return series.map({"No": 0, "Yes": 1}).astype("Int64")

    # Apply explicit mapping for gender
    if value_set == {"Male", "Female"}:
        return series.map({"Female": 0, "Male": 1}).astype("Int64")

    # Apply deterministic alphabetical mapping to other binary features
    if len(unique_values) == 2:
        ordered_values = sorted(unique_values)

        binary_mapping = {
            ordered_values[0]: 0,
            ordered_values[1]: 1
        }

        return (
            series.astype(str)
            .map(binary_mapping)
            .astype("Int64")
        )

    # Leave non-binary features unchanged
    return series


def build_features(
    dataframe: pd.DataFrame,
    target_column: str = "Churn"
) -> pd.DataFrame:
    """
    Transform customer data into machine-learning-ready features.

    Categorical features with two categories are binary encoded, while
    features with more than two categories are one-hot encoded. The same
    deterministic transformations must be used during model serving.

    Args:
        dataframe: Input customer dataset.
        target_column: Name of the target variable.

    Returns:
        DataFrame containing the engineered features.
    """
    dataframe = dataframe.copy()

    print(
        f"Starting feature engineering on "
        f"{dataframe.shape[1]} columns..."
    )

    # Identify categorical and numerical columns
    categorical_columns = [
        column
        for column in dataframe.select_dtypes(include=["object"]).columns
        if column != target_column
    ]

    numerical_columns = dataframe.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    print(
        f"Found {len(categorical_columns)} categorical and "
        f"{len(numerical_columns)} numeric columns"
    )

    # Separate binary and multi-category features
    binary_columns = [
        column
        for column in categorical_columns
        if dataframe[column].dropna().nunique() == 2
    ]

    multi_category_columns = [
        column
        for column in categorical_columns
        if dataframe[column].dropna().nunique() > 2
    ]

    print(
        f"Binary features: {len(binary_columns)} | "
        f"Multi-category features: {len(multi_category_columns)}"
    )

    if binary_columns:
        print(f"Binary columns: {binary_columns}")

    if multi_category_columns:
        print(
            f"Multi-category columns: "
            f"{multi_category_columns}"
        )

    # Apply deterministic binary encoding
    for column in binary_columns:
        original_dtype = dataframe[column].dtype

        dataframe[column] = _encode_binary_column(
            dataframe[column].astype(str)
        )

        print(
            f"{column}: {original_dtype} -> binary (0/1)"
        )

    # Convert boolean columns to integers
    boolean_columns = dataframe.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    if boolean_columns:
        dataframe[boolean_columns] = (
            dataframe[boolean_columns].astype(int)
        )

        print(
            f"Converted {len(boolean_columns)} boolean "
            f"columns to integers: {boolean_columns}"
        )

    # Apply one-hot encoding to multi-category features
    if multi_category_columns:
        print(
            f"Applying one-hot encoding to "
            f"{len(multi_category_columns)} multi-category columns..."
        )

        previous_shape = dataframe.shape

        dataframe = pd.get_dummies(
            dataframe,
            columns=multi_category_columns,
            drop_first=True
        )

        created_features = (
            dataframe.shape[1]
            - previous_shape[1]
            + len(multi_category_columns)
        )

        print(
            f"Created {created_features} new features from "
            f"{len(multi_category_columns)} categorical columns"
        )

    # Convert nullable integer columns to standard integers
    for column in binary_columns:
        if pd.api.types.is_integer_dtype(dataframe[column]):
            dataframe[column] = (
                dataframe[column]
                .fillna(0)
                .astype(int)
            )

    print(
        f"Feature engineering complete: "
        f"{dataframe.shape[1]} final features"
    )

    return dataframe


if __name__ == "__main__":
    from src.data.load_data import load_csv
    from src.data.preprocess_data import preprocess_data

    dataset = load_csv("data/raw/data.csv")
    processed_dataset = preprocess_data(dataset)

    feature_dataset = build_features(processed_dataset)

    print("\nFeature engineering test completed.")
    print(f"Final shape: {feature_dataset.shape}")
    print("\nFinal columns:")
    print(feature_dataset.columns.tolist())