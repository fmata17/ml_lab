# 🧠 Machine Learning Study Lab

A structured and reproducible framework for studying, experimenting with, and comparing machine learning paradigms across multiple datasets.

Each dataset (for example, `iris`, `mnist`, `housing_prices`, etc.) is treated as its own experimental environment. Inside each dataset directory, you’ll find a data exploration notebook and multiple modeling notebooks (classical, Bayesian, neural, etc.), all following the same structure to make comparisons meaningful and direct.

## 🎯 Purpose

This repository is designed to make machine learning experiments comparable and reproducible. Each dataset folder contains a consistent workflow and structure so that models can be evaluated under the same conditions. The goal is not only to benchmark performance but also to understand how different paradigms reason about data, uncertainty, and model structure.

## ⚙️ Workflow Overview

Every dataset folder follows this standardized progression:

1. **`data_exploration.ipynb`**  
   Understand and preprocess the dataset.
   - Load and preview data  
   - Handle missing values and outliers  
   - Split into training and test sets (with fixed random seed)  
   - Basic EDA: distributions, correlations, target balance  
   - Save standardized train/test splits into `results/processed/`

2. **Modeling Notebooks** (`model_classical.ipynb`, `model_bayesian.ipynb`, `model_neural.ipynb`, etc.)  
   Each modeling notebook follows the same internal structure so that results are directly comparable across paradigms.

   **Notebook Structure:**  
   1. Notebook header (objective, assumptions, date)  
   2. Setup and imports (with same random seed and style)  
   3. Data loading (from processed results)  
   4. Optional EDA recap  
   5. Model definition (core differences between paradigms)  
   6. Training or inference  
   7. Evaluation (same metrics and plots)  
   8. Uncertainty or interpretability analysis  
   9. Discussion and insights

3. **`results/` Directory**  
   Stores outputs in a consistent format:

## 🧩 Modeling Paradigms

### Classical Machine Learning
Frequentist, point-estimate approaches using deterministic optimization.

**Examples:** Linear Regression, Logistic Regression, Decision Trees, Random Forests, SVM, Gradient Boosting.  
**Characteristics:**  
- Parameters estimated via MLE or optimization  
- Regularization interpreted as prior-like constraint  
- Outputs point estimates (no uncertainty quantification)

### Bayesian Machine Learning
Probabilistic models that incorporate prior beliefs and yield full posterior distributions.

**Examples:** Bayesian Linear or Logistic Regression, Hierarchical Models, Gaussian Mixtures, Probabilistic Graphical Models.  
**Characteristics:**  
- Parameters treated as random variables  
- Priors express domain knowledge or regularization beliefs  
- Posteriors quantify uncertainty in predictions and parameters  
- Inference via MCMC or Variational Inference

### Neural Models
Highly flexible, parameter-rich models trained with gradient descent.

**Examples:** Feedforward MLPs, CNNs (for image data), RNNs or Transformers (for sequence data).  
**Characteristics:**  
- High representational power  
- Require large data and regularization  
- Can be extended into Bayesian NNs for uncertainty estimation

## 📏 Evaluation Protocol

To maintain comparability across paradigms, every model uses the same evaluation framework. All metrics are saved as JSON for aggregation and visualization.

| Metric | Type | Description |
|:--------|:-----|:-------------|
| Accuracy | Classification | Proportion of correct predictions |
| F1-Score | Classification | Harmonic mean of precision & recall |
| RMSE | Regression | Root mean squared error |
| Log-Likelihood | Probabilistic | Model fit measure under the data |
| Calibration | Bayesian | Reliability of predicted probabilities |

**Example JSON Output:**
```json
{
"accuracy": 0.874,
"f1_score": 0.865,
"log_likelihood": -123.45
}
