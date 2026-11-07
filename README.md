# Fraud Detection — ML Algorithms from Scratch

A fraud classification pipeline on a real, severely imbalanced dataset (~100k transactions, ~14% fraud), comparing from-scratch implementations of three classic ML algorithms against scikit-learn.

**[Live demo](https://fraud-detection-using-machine-learn.vercel.app)** — visualizes the preprocessing pipeline, the trained decision tree, and a side-by-side comparison against scikit-learn's output on the same data.

## Algorithms

- **Decision Tree (C4.5)** — numeric and categorical features, missing-value handling, splits chosen by information gain.
- **Logistic Regression** — gradient descent, L2 regularization, class weighting and undersampling to handle the imbalance.
- **K-Nearest Neighbors** — Euclidean, Manhattan, and cosine distance metrics, weighted voting so closer neighbors count more.

Each one is also implemented with scikit-learn's `DecisionTreeClassifier`, `LogisticRegression`, and `KNeighborsClassifier` as a baseline, evaluated on accuracy, precision, recall, F1, and ROC-AUC, plus a Kaggle submission.

## Results

Class weighting and SMOTE on the logistic regression model took fraud recall from 0.02 to 0.52 — a 25x cut in false negatives on an otherwise near-unusable baseline. Benchmarking the from-scratch implementations against scikit-learn also surfaced a real ~330x efficiency gap, from the non-vectorized loops in the from-scratch versions versus scikit-learn's compiled, vectorized code.

## Preprocessing

Missing-value handling, duplicate removal, categorical encoding, numeric scaling, imbalance handling, and feature selection, run as one pipeline before any model sees the data.

## Run it

```bash
git clone https://github.com/favizz/Fraud-Detection-using-Machine-Learning-Algorithm.git
```

Notebook: [`src/`](src) · Live demo source: [`web/`](web)

## Team

Built as a 4-person university group project (Team 31, "Welfarella"). My own part: the preprocessing pipeline, exploratory data analysis, the scikit-learn benchmark implementations, and the live demo web app ([`web/`](web)) that visualizes the pipeline and results. Teammates implemented the from-scratch Decision Tree, Logistic Regression, and KNN algorithms.

| Contributor | Contribution |
|---|---|
| Favian Izza Diasputra | Preprocessing pipeline, feature encoding/scaling, EDA, scikit-learn implementations, live demo web app |
| Henrycus Hugatama Risaldy | Feature engineering, dimensionality reduction, from-scratch Logistic Regression |
| Farella Kamala Budianto | Train/val/test split, missing-data handling, EDA, from-scratch K-Nearest Neighbors |
| Kenlyn Tesalonika Winata | Outlier handling, duplicate removal, imbalance handling, from-scratch Decision Tree |
