# ============================================================
# 📊 EDA + Feature Analysis Template
# Author: Fredy Mata
# Purpose: Reusable notebook for structured Exploratory Data Analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats

# ============================================================
# 1. BASIC DATA OVERVIEW
# ============================================================


def data_overview(df: pd.DataFrame):
    """
    Quick overview of dataset:
    - Shape
    - Data types
    - Missing values
    - Summary statistics
    """
    print("Shape:", df.shape)
    print("\nData Types:", f"\n{df.dtypes}")
    print("\nMissing Values (isnull):", f"\n{df.isnull().sum()}")
    print("\nSummary Statistics:", f"\n{df.describe().T}")
    return df.head()


# ============================================================
# 2. VISUAL EXPLORATION (RAW FEATURES)
# ============================================================
def plot_target_relationships(df: pd.DataFrame, target="price", ncols=3):
    """
    For each feature:
    - Scatter plot with regression line if continuous
    - Boxplot if categorical/discrete
    Helps visually identify trends, overlaps, and outliers.
    """
    features = [col for col in df.columns if col != target]
    n = len(features)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(6*ncols, 4*nrows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        unique_vals = df[col].nunique()

        if pd.api.types.is_numeric_dtype(df[col]) and unique_vals > 10:
            sns.scatterplot(x=df[col], y=df[target],
                            ax=ax, alpha=0.6, color="blue")
            sns.regplot(x=df[col], y=df[target], ax=ax,
                        scatter=False, color="red", line_kws={"linewidth": 2})
            ax.set_title(f"{target} vs {col} (scatter)")
        else:
            sns.boxplot(x=df[col], y=df[target], ax=ax)
            ax.set_title(f"{target} vs {col} (boxplot)")

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ============================================================
# 3. PLOT FEATURE VS TARGET // simpler version of function 2
# ============================================================
def plot_feature_vs_target(df: pd.DataFrame, feature: str, target="price"):
    """
    Plot relationship between a single feature and target.

    - Continuous features → scatter with regression line
    - Discrete/categorical features → boxplot

    Parameters:
    df : pandas.DataFrame
        Dataset
    feature : str
        Feature column name
    target : str
        Target column name (default: "price")
    """
    plt.figure(figsize=(6, 4))
    unique_vals = df[feature].nunique()

    if pd.api.types.is_numeric_dtype(df[feature]) and unique_vals > 10:
        # Scatter + regression line
        sns.scatterplot(x=df[feature], y=df[target], alpha=0.6, color="blue")
        sns.regplot(x=df[feature], y=df[target], scatter=False,
                    color="red", line_kws={"linewidth": 2})
        plt.title(f"{target} vs {feature} (scatter)")
    else:
        # Boxplot for categorical/discrete
        sns.boxplot(x=df[feature], y=df[target])
        plt.title(f"{target} vs {feature} (boxplot)")

    plt.xlabel(feature)
    plt.ylabel(target)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. CORRELATION HEATMAP (PROCESSED NUMERIC FEATURES)
# ============================================================
def correlation_heatmap(df: pd.DataFrame, target="price"):
    """
    Pearson correlation matrix for numeric features.
    Highlights linear relationships.
    """
    corr = df.corr()
    plt.figure(figsize=(4, 2))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap")
    plt.show()
    print("\nCorrelations with target:\n",
          corr[target].sort_values(ascending=False))

    # self
    target_corr = corr[target].sort_values(ascending=False)

    plt.figure(figsize=(2, 4))
    sns.heatmap(target_corr.to_frame(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation of Features with SalePrice")
    plt.show()


# ============================================================
# 5. ANOVA TESTS (CATEGORICAL FEATURES VS TARGET)
# ============================================================
def anova_tests(df: pd.DataFrame, target="price"):
    """
    One-way ANOVA for each categorical feature:
    Tests whether mean target differs significantly across groups.
    High F-statistic + low p-value → strong relationship.
    """
    cat_features = [col for col in df.columns if df[col].dtype ==
                    "object" or df[col].nunique() < 10]

    results = {}
    for col in cat_features:
        groups = [df[target][df[col] == val] for val in df[col].unique()]
        f_stat, p_val = stats.f_oneway(*groups)
        results[col] = {"F-stat": f_stat, "p-value": p_val}

    return pd.DataFrame(results).T.sort_values("F-stat", ascending=False)


# ============================================================
# 6. MUTUAL INFORMATION (NONLINEAR RELATIONSHIPS)
# ============================================================
def mutual_information_analysis(X, y):
    """
    Computes mutual information scores:
    - Captures nonlinear dependencies
    - Robust alternative to correlation
    """
    mi = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    print("\nMutual Information Scores:\n", mi_series)


# ============================================================
# 7. FEATURE IMPORTANCE (TREE-BASED MODELS)
# ============================================================
def feature_importance_forest(X, y):
    """
    Trains a RandomForestRegressor and extracts feature importances.
    Measures how much each feature reduces error across splits.
    """
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_,
                            index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=importances.index)
    plt.title("Feature Importances (RandomForest)")
    plt.show()

    return importances


# ============================================================
# 8. RESIDUAL ANALYSIS (BASELINE MODEL CHECK)
# ============================================================
def residual_analysis(model, X, y, feature=None):
    """
    Residual analysis for regression models.

    If feature=None:
        - Plots residuals vs predicted values (default behavior).
    If feature is specified (categorical):
        - Plots residuals grouped by that feature (boxplot).
    """
    preds = model.predict(X)
    residuals = y - preds

    if feature is None:
        # Original residual vs predictions plot
        plt.scatter(preds, residuals, alpha=0.6, color="blue")
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()
    else:
        # Group residuals by a categorical feature
        df_res = X.copy()
        df_res["residuals"] = residuals

        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df_res[feature], y=df_res["residuals"])
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"Residuals grouped by {feature}")
        plt.show()
