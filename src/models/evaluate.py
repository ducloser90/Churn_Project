from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, test_features, test_target):
    """
    Evaluate a trained classification model on test data.

    Args:
        model: Trained classification model.
        test_features: Test feature dataset.
        test_target: True test labels.
    """
    predictions = model.predict(test_features)

    print(
        "Classification Report:\n",
        classification_report(test_target, predictions)
    )

    print(
        "Confusion Matrix:\n",
        confusion_matrix(test_target, predictions)
    )

