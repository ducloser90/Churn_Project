from pathlib import Path

import pandas as pd


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        csv_path: Location of the CSV file.

    Returns:
        A DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    data_file = Path(csv_path)

    if not data_file.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    return pd.read_csv(data_file)


if __name__ == "__main__":
    file_path = "data/raw/data.csv"

    dataframe = load_csv(file_path)

    print("Data loaded successfully!")
    print(dataframe.head())
    print(f"\nShape: {dataframe.shape}")