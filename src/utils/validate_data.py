import great_expectations as ge
import great_expectations.dataset
from typing import List, Tuple


def validate_telco_data(dataframe) -> Tuple[bool, List[str]]:
    """
    Validate the Telco Customer Churn dataset using Great Expectations.

    The validation covers required columns, business rules, numeric ranges,
    statistical constraints, and data consistency checks.

    Args:
        dataframe: Telco Customer Churn dataset as a pandas DataFrame.

    Returns:
        A tuple containing:
            - A boolean indicating whether all checks passed.
            - A list containing the names of failed expectations.
    """
    print("Starting data validation with Great Expectations...")

    # Convert pandas DataFrame to Great Expectations Dataset
    validation_dataset = ge.dataset.PandasDataset(dataframe)

    # Schema validation
    print("Validating schema and required columns...")

    validation_dataset.expect_column_to_exist("customerID")
    validation_dataset.expect_column_values_to_not_be_null("customerID")

    validation_dataset.expect_column_to_exist("gender")
    validation_dataset.expect_column_to_exist("Partner")
    validation_dataset.expect_column_to_exist("Dependents")

    validation_dataset.expect_column_to_exist("PhoneService")
    validation_dataset.expect_column_to_exist("InternetService")
    validation_dataset.expect_column_to_exist("Contract")

    validation_dataset.expect_column_to_exist("tenure")
    validation_dataset.expect_column_to_exist("MonthlyCharges")
    validation_dataset.expect_column_to_exist("TotalCharges")

    # Business logic validation
    print("Validating business logic constraints...")

    validation_dataset.expect_column_values_to_be_in_set(
        "gender",
        ["Male", "Female"]
    )

    validation_dataset.expect_column_values_to_be_in_set(
        "Partner",
        ["Yes", "No"]
    )

    validation_dataset.expect_column_values_to_be_in_set(
        "Dependents",
        ["Yes", "No"]
    )

    validation_dataset.expect_column_values_to_be_in_set(
        "PhoneService",
        ["Yes", "No"]
    )

    validation_dataset.expect_column_values_to_be_in_set(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    validation_dataset.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )

    # Numeric range validation
    print("Validating numeric ranges and business constraints...")

    validation_dataset.expect_column_values_to_be_between(
        "tenure",
        min_value=0
    )

    validation_dataset.expect_column_values_to_be_between(
        "MonthlyCharges",
        min_value=0
    )

    validation_dataset.expect_column_values_to_be_between(
        "TotalCharges",
        min_value=0
    )

    # Statistical validation
    print("Validating statistical properties...")

    validation_dataset.expect_column_values_to_be_between(
        "tenure",
        min_value=0,
        max_value=120
    )

    validation_dataset.expect_column_values_to_be_between(
        "MonthlyCharges",
        min_value=0,
        max_value=200
    )

    validation_dataset.expect_column_values_to_not_be_null("tenure")
    validation_dataset.expect_column_values_to_not_be_null("MonthlyCharges")

    # Data consistency validation
    print("Validating data consistency...")

    validation_dataset.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95
    )

    # Run validation suite
    print("Running complete validation suite...")
    validation_results = validation_dataset.validate()

    # Process validation results
    failed_expectations = [
        result["expectation_config"]["expectation_type"]
        for result in validation_results["results"]
        if not result["success"]
    ]

    total_checks = len(validation_results["results"])
    passed_checks = sum(
        1 for result in validation_results["results"]
        if result["success"]
    )
    failed_checks = total_checks - passed_checks

    if validation_results["success"]:
        print(
            f"Data validation passed: "
            f"{passed_checks}/{total_checks} checks successful"
        )
    else:
        print(
            f"Data validation failed: "
            f"{failed_checks}/{total_checks} checks failed"
        )
        print(f"Failed expectations: {failed_expectations}")

    return validation_results["success"], failed_expectations


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("data/raw/data.csv", na_values=[" ", ""])

    is_valid, failed_checks = validate_telco_data(data)

    if is_valid:
        print("Dataset validation completed successfully.")
    else:
        print("Dataset validation failed.")
        print(f"Failed checks: {failed_checks}")