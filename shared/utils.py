import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import seaborn as sns
from sklearn.model_selection import train_test_split


def load_dataset(
    dataset: str = "iris", test_size: float = 0.2, random_state: int = 14
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        df : pd.DataFrame
            Original dataset
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
        bunch = datasets.load_iris()  # sklearn bunch
        X = bunch.data
        y = bunch.target
        df = bunch.frame
    elif dataset == "titanic":
        df = sns.load_dataset(dataset)  # Pandas df
        X = df.iloc[:, 1:]
        y = df.iloc[:, :1]
    else:
        raise NotImplementedError(f"Dataset '{dataset}' is not yet supported.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=14
    )

    print(
        f"Loaded '{dataset}' with {X.shape[0]} samples and {X.shape[1]} features each."
        f"\nTraining set: {X_train.shape[0]} samples"
        f"\nTesting set: {X_test.shape[0]} samples\n"
    )
    return df, X_train, X_test, y_train, y_test


def calculate_mean(
    data: np.ndarray, columns: list[str] | None = None
) -> dict[str, float]:
    """
    Compute the column-wise mean of a NumPy array, ignoring NaN values.

    Parameters
    ----------
        data : np.ndarray
            Input array of shape (n_samples, n_features) or (n_samples,).
            If a 1D array is provided, it is reshaped into a column vector.

    Returns
    -------
        means : dict[str, float]
            Dict containing the mean of each column. Columns with no valid
            (non-NaN) values return NaN.
    """

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if columns is None:
        columns = [f"col_{i}" for i in range(data.shape[1])]
    means = {}
    for j, name in enumerate(columns):
        total = 0.0
        count = 0
        for i in range(data.shape[0]):
            x = data[i, j]
            if not np.isnan(x):
                total += x
                count += 1
        means[name] = total / count if count > 0 else float("nan")
    return means


def calculate_variance(
    data: np.ndarray, columns: list[str] | None = None, sample: bool = True
) -> dict[str, float]:
    """
    Compute the column-wise variance of a NumPy array, ignoring NaN values.

    Parameters
    ----------
        data : np.ndarray
            Input array of shape (n_samples, n_features) or (n_samples,).
            If a 1D array is provided, it is reshaped into a column vector.
        columns : list[str], optional
            Column names corresponding to each feature. If None, generic
            column names are assigned.
        sample : bool, default=True
            If True, computes sample variance (divides by n - 1).
            If False, computes population variance (divides by n).

    Returns
    -------
        variances : dict[str, float]
            Dict containing the variance of each column. Columns with fewer
            than two valid (non-NaN) values return NaN.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if columns is None:
        columns = [f"col_{i}" for i in range(data.shape[1])]
    means = calculate_mean(data, columns)  # <-- dict[str, float]
    variances: dict[str, float] = {}
    for j, col in enumerate(columns):
        squared_diff_sum = 0.0
        count = 0
        for i in range(data.shape[0]):
            x = data[i, j]
            if not np.isnan(x):
                squared_diff_sum += (x - means[col]) ** 2
                count += 1
        denom = (count - 1) if sample else count
        variances[col] = (squared_diff_sum / denom) if denom > 0 else float("nan")
    return variances


def calculate_std_deviation(
    data: np.ndarray, columns: list[str] | None = None, sample: bool = True
) -> dict[str, float]:
    """
    Compute the column-wise standard deviation of a NumPy array, ignoring NaN values.

    Parameters
    ----------
        data : np.ndarray
            Input array of shape (n_samples, n_features) or (n_samples,).
            If a 1D array is provided, it is reshaped into a column vector.
        columns : list[str], optional
            Column names corresponding to each feature. If None, generic
            column names are assigned.
        sample : bool, default=True
            If True, computes sample standard deviation.
            If False, computes population standard deviation.

    Returns
    -------
        std_deviations : dict[str, float]
            Dict containing the standard deviation of each column. Columns
            with fewer than two valid (non-NaN) values return NaN.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if columns is None:
        columns = [f"col_{i}" for i in range(data.shape[1])]
    variances = calculate_variance(data, columns, sample)
    # List comprehension / dict comprehension version
    std_deviations = {
        col: np.sqrt(var) if not np.isnan(var) else float("nan")
        for col, var in variances.items()
    }
    return std_deviations
