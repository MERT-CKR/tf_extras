import matplotlib.pyplot as plt


def plot_predictions(train_data=None,
                     train_labels=None,
                     test_data=None,
                     test_labels=None,
                     predictions=None):
    """
    Plots training data test data and compares predictions to ground truth labels.

    Args:
        train_data: Training feature values.
        train_labels: Training labels.
        test_data: Test feature values.
        test_labels: Test labels.
        predictions: Model predictions for the test data.

    Raises:
        ValueError: If any of the required arguments are None.
    """
    if any(arg is None for arg in [train_data, train_labels, test_data, test_labels, predictions]):
        raise ValueError("All arguments must be provided (train/test data and predictions).")

    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", label="Training data")

    # Testing data
    plt.scatter(test_data, test_labels, c="g", label="Test data")

    # Predictions
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend()
    plt.xlabel("Data")
    plt.ylabel("Labels")
    plt.title("Prediction Visualization")
    plt.show()
