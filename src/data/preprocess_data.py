import pandas as pd


def preprocess_data(
    dataframe: pd.DataFrame,
    target_column: str = "Churn"
) -> pd.DataFrame:
    """
    Clean and prepare the Telco Customer Churn dataset.

    The preprocessing includes:
    - Cleaning column names.
    - Removing customer identifier columns.
    - Converting the target variable from Yes/No to 0/1.
    - Converting TotalCharges to a numeric type.
    - Ensuring SeniorCitizen contains integer values.
    - Filling missing numeric values with 0.

    Args:
        dataframe: Input dataset as a pandas DataFrame.
        target_column: Name of the target variable.

    Returns:
        Preprocessed pandas DataFrame.
    """
    # Clean column names
    dataframe.columns = dataframe.columns.str.strip()

    # Remove customer identifier columns if present
    identifier_columns = ["customerID", "CustomerID", "customer_id"]

    for column_name in identifier_columns:
        if column_name in dataframe.columns:
            dataframe = dataframe.drop(columns=[column_name])

    # Convert target variable from Yes/No to 0/1
    if (
        target_column in dataframe.columns
        and dataframe[target_column].dtype == "object"
    ):
        dataframe[target_column] = (
            dataframe[target_column]
            .str.strip()
            .map({"No": 0, "Yes": 1})
        )

    # Convert TotalCharges to numeric values
    if "TotalCharges" in dataframe.columns:
        dataframe["TotalCharges"] = pd.to_numeric(
            dataframe["TotalCharges"],
            errors="coerce"
        )

    # Convert SeniorCitizen to integer values
    if "SeniorCitizen" in dataframe.columns:
        dataframe["SeniorCitizen"] = (
            dataframe["SeniorCitizen"]
            .fillna(0)
            .astype(int)
        )

    # Fill missing numeric values with 0
    numeric_columns = dataframe.select_dtypes(
        include=["number"]
    ).columns

    dataframe[numeric_columns] = dataframe[numeric_columns].fillna(0)

    return dataframe


if __name__ == "__main__":
    from load_data import load_csv

    dataset = load_csv("data/raw/data.csv")

    print("Original dataset shape:", dataset.shape)

    processed_dataset = preprocess_data(dataset)

    print("Processed dataset shape:", processed_dataset.shape)
    print("\nProcessed data:")
    print(processed_dataset.head())
    print("\nMissing values:")
    print(processed_dataset.isnull().sum())