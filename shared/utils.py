import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split


def load_dataset(dataset: str = "iris", test_size: float = 0.2, random_state: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split a supported dataset into training and test sets.

    Parameters
    ----------
        dataset : str
            Name of the dataset to load. Currently supports: 'iris'.
        test_size : float
            Fraction of data to reserve for testing.
        random_state : int
            Seed for reproducibility.

    Returns
    -------
        X_train : np.ndarray
            Training features
        X_test : np.ndarray
            Testing features
        y_train : np.ndarray
            Training labels
        y_test : np.ndarray
            Testing labels
    """
    if dataset == "iris":
        data = datasets.load_iris()
        X = data.data
        y = data.target
    else:
        raise NotImplementedError(f"Dataset '{dataset}' is not yet supported.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=14)

    print(f"Loaded '{dataset}' with {X.shape[0]} samples and {X.shape[1]} features each."
          f"\nTraining set: {X_train.shape[0]} samples"
          f"\nTesting set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def calculate_mean(data: np.ndarray) -> list[float]:
    """Compute the column-wise mean of a 2D NumPy array (manual implementation)."""
    means = [0.0] * data.shape[1]
    for j in range(data.shape[1]):
        total = 0
        for i in range(data.shape[0]):
            total += data[i][j]
        means[j] = float(total / data.shape[0])
    return means


def calculate_variance(data: np.ndarray, sample: bool = True) -> list[float]:
    """Compute the column-wise variance of a 2D NumPy array."""
    variances = [0.0] * data.shape[1]
    means = calculate_mean(data)
    n = data.shape[0] - 1 if sample else data.shape[0]
    for j in range(data.shape[1]):
        squared_diff_summ = 0
        for i in range(data.shape[0]):
            squared_diff_summ += ((data[i][j] - means[j]) ** 2)
        variances[j] = float(squared_diff_summ / n)
    return variances


# TODO refactor with list comprehension
def calculate_std_deviation(data: np.ndarray, sample: bool = True) -> list[float]:
    """Compute the column-wise standard deviation of a 2D NumPy array."""
    std_deviations = [0.0] * data.shape[1]
    variances = calculate_variance(data, sample)
    for i in range(len(variances)):
        std_deviations[i] = variances[i] ** 0.5
    return std_deviations
