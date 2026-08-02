"""
Reproduces the notebook's run_universal_pipeline -> DTLWrapper / CustomLogisticRegression
training flow (see src/IF3070_..._Notebook_Template (4).ipynb) WITHOUT pandas, because
pandas is blocked by an Application Control policy on this machine. Uses only csv + numpy
+ scikit-learn's OrdinalEncoder/OneHotEncoder/StandardScaler (which operate on plain numpy
arrays and don't need pandas).

Trains the "from scratch" DTL (C4.5-style) and Logistic Regression models on the full
train.csv and exports everything a browser needs to reproduce preprocessing + inference
client-side, with no backend:

  web/data/pipeline.json   feature order, impute stats, outlier transform, encoding maps, scaler
  web/data/logreg.json     weights + bias
  web/data/dtl_tree.json   decision tree as nested JSON
  web/data/samples.json    sample rows (with ground-truth label) for the demo's "load sample" button
  web/data/metrics.json    validation metrics for both models, for display in the UI
"""
import csv
import json
import math
import os
import random
import time

import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(ROOT, "src", "assets", "train.csv")
OUT_DIR = os.path.join(ROOT, "web", "data")

ID_COLS = ["ID", "transaction_id", "user_id"]
TARGET_COL = "is_fraud"

BIN_COLS = ["gender", "has_chargeback_history", "is_new_country"]
NOM_COLS = ["country", "device_type", "device_os", "merchant_category", "transaction_type", "day_of_week"]
# Numeric columns present in the raw CSV (before feature engineering adds 4 more).
NUM_COLS = [
    "age", "transaction_amount", "time_of_day", "transaction_duration",
    "num_prev_transactions", "avg_transaction_amount", "std_transaction_amount",
    "transactions_last_24h", "transactions_last_1h", "failed_login_attempts",
    "ip_risk_score", "device_trust_score", "account_age_days",
    "shared_ip_users", "shared_device_users", "merchant_risk", "country_risk",
    "distance_from_home",
]
ENGINEERED_NUM_COLS = ["risk_log_amount", "amount_ratio", "hour_sin", "hour_cos"]

RNG_SEED = 42


# ---------------------------------------------------------------------------
# CSV loading (no pandas)
# ---------------------------------------------------------------------------
def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def to_float_or_none(v):
    if v is None or v == "":
        return None
    return float(v)


def to_str_or_none(v):
    if v is None or v == "":
        return None
    return v


# ---------------------------------------------------------------------------
# Stats helpers matching pandas semantics
# ---------------------------------------------------------------------------
def pandas_skew(x):
    """Adjusted Fisher-Pearson skewness, matching pandas Series.skew()."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = x.mean()
    diffs = x - mean
    m2 = np.mean(diffs ** 2)
    m3 = np.mean(diffs ** 3)
    if m2 == 0:
        return 0.0
    g1 = m3 / (m2 ** 1.5)
    g1_corrected = (math.sqrt(n * (n - 1)) / (n - 2)) * g1
    return g1_corrected


def mode_of(values):
    """values: list of hashable, non-None. Ties broken by smallest value (matches
    pandas Series.mode()[0], which returns sorted-ascending modes and we take [0])."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best_count = max(counts.values())
    candidates = [v for v, c in counts.items() if c == best_count]
    return min(candidates)


def entropy_from_counts(pos, count):
    if count == 0:
        return 0.0
    p1 = pos / count
    p0 = 1.0 - p1
    eps = 1e-12
    return -(p0 * math.log2(p0 + eps) + p1 * math.log2(p1 + eps))


def entropy(y):
    n = len(y)
    if n == 0:
        return 0.0
    pos = float(np.sum(y == 1))
    return entropy_from_counts(pos, n)


# ---------------------------------------------------------------------------
# Fast numeric best-split search (matches notebook's best_split_numeric exactly,
# just computed with sorted-array prefix sums instead of O(thresholds * n) masks).
# ---------------------------------------------------------------------------
def best_split_numeric(col, y):
    n = len(col)
    if n < 2:
        return None, -999.0

    order = np.argsort(col, kind="mergesort")
    col_sorted = col[order]
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = cum_pos[-1] if n > 0 else 0

    unique_vals = np.unique(col)
    if len(unique_vals) < 50:
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
    else:
        thresholds = np.percentile(col, np.linspace(5, 95, 50))

    parent_entropy = entropy(y)
    best_t, best_gain = None, -999.0

    for t in thresholds:
        left_count = int(np.searchsorted(col_sorted, t, side="right"))
        right_count = n - left_count
        if left_count == 0 or right_count == 0:
            continue
        left_pos = cum_pos[left_count - 1]
        right_pos = total_pos - left_pos
        e_left = entropy_from_counts(left_pos, left_count)
        e_right = entropy_from_counts(right_pos, right_count)
        weighted = (left_count / n) * e_left + (right_count / n) * e_right
        gain = parent_entropy - weighted
        if gain > best_gain:
            best_gain = gain
            best_t = float(t)

    return best_t, best_gain


def find_best_split(X, y, feature_names):
    best_feature = None
    best_threshold = None
    best_score = -999.0
    for j, fname in enumerate(feature_names):
        t, score = best_split_numeric(X[:, j], y)
        if t is not None and score > best_score:
            best_score = score
            best_threshold = t
            best_feature = fname
    return best_feature, best_threshold, best_score


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def to_json(self):
        if self.value is not None:
            return {"value": self.value}
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left.to_json(),
            "right": self.right.to_json(),
        }


def build_tree(X, y, feature_names, depth, max_depth, min_samples, min_samples_leaf):
    if len(np.unique(y)) == 1:
        return Node(value=float(y.mean()))
    if depth >= max_depth or len(y) < min_samples:
        return Node(value=float(y.mean()))

    feature, threshold, gain = find_best_split(X, y, feature_names)
    if feature is None or gain <= 0:
        return Node(value=float(y.mean()))

    j = feature_names.index(feature)
    left_mask = X[:, j] <= threshold
    right_mask = ~left_mask

    if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
        return Node(value=float(y.mean()))

    left = build_tree(X[left_mask], y[left_mask], feature_names, depth + 1, max_depth, min_samples, min_samples_leaf)
    right = build_tree(X[right_mask], y[right_mask], feature_names, depth + 1, max_depth, min_samples, min_samples_leaf)
    return Node(feature=feature, threshold=threshold, left=left, right=right)


def count_nodes(node):
    if node.value is not None:
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)


# ---------------------------------------------------------------------------
# Logistic regression (copied verbatim in spirit from the notebook's
# CustomLogisticRegression - pandas-free already in the original).
# ---------------------------------------------------------------------------
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=3000, lambda_reg=0.05, use_class_weights=True):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.lambda_reg = lambda_reg
        self.use_class_weights = use_class_weights
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        class_weights = {}
        if self.use_class_weights:
            for cls in np.unique(y):
                count_cls = (y == cls).sum()
                class_weights[cls] = n_samples / (2.0 * count_cls)

        for i in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            A = self._sigmoid(linear_model)
            error = A - y

            if self.use_class_weights:
                sample_weights = np.array([class_weights[y_i] for y_i in y])
                weighted_error = error * sample_weights
            else:
                weighted_error = error

            dw = (1 / n_samples) * np.dot(X.T, weighted_error)
            db = (1 / n_samples) * np.sum(weighted_error)
            dw += (self.lambda_reg / n_samples) * self.weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._sigmoid(np.dot(X, self.weights) + self.bias)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Loading CSV...")
    rows = load_rows(TRAIN_CSV)
    print(f"  {len(rows)} rows loaded")

    # --- classify_features (verified to exactly match BIN_COLS/NOM_COLS/NUM_COLS above) ---

    # --- drop id cols, pop target ---
    y_all = np.array([int(r[TARGET_COL]) for r in rows])
    feat_rows = rows  # keep raw dict rows around, we'll pull columns as needed

    # --- remove_duplicates: dedupe on (all feature columns) raw string values, keep first ---
    feature_cols_raw_order = [c for c in rows[0].keys() if c not in ID_COLS and c != TARGET_COL]
    seen = set()
    keep_idx = []
    for i, r in enumerate(feat_rows):
        key = tuple(r[c] for c in feature_cols_raw_order)
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)
    print(f"  {len(rows) - len(keep_idx)} duplicate rows removed -> {len(keep_idx)} rows")

    feat_rows = [feat_rows[i] for i in keep_idx]
    y_all = y_all[keep_idx]

    # --- parse raw columns ---
    num_raw = {c: np.array([to_float_or_none(r[c]) for r in feat_rows], dtype=object) for c in NUM_COLS}
    cat_raw = {c: [to_str_or_none(r[c]) for r in feat_rows] for c in (BIN_COLS + NOM_COLS)}

    # --- impute_missing_values: numeric -> median, categorical -> mode ---
    impute_medians = {}
    for c in NUM_COLS:
        vals = num_raw[c]
        non_null = np.array([v for v in vals if v is not None], dtype=float)
        median = float(np.median(non_null))
        impute_medians[c] = median
        num_raw[c] = np.array([v if v is not None else median for v in vals], dtype=float)

    impute_modes = {}
    for c in (BIN_COLS + NOM_COLS):
        vals = cat_raw[c]
        non_null = [v for v in vals if v is not None]
        if not non_null:
            continue
        m = mode_of(non_null)
        impute_modes[c] = m
        cat_raw[c] = [v if v is not None else m for v in vals]

    print(f"  imputation done ({time.time()-t0:.1f}s)")

    # --- generate_new_features ---
    transaction_amount = num_raw["transaction_amount"]
    avg_transaction_amount = num_raw["avg_transaction_amount"]
    ip_risk_score = num_raw["ip_risk_score"]
    time_of_day = num_raw["time_of_day"]

    safe_amt = np.clip(transaction_amount, 0, None)
    risk_log_amount = ip_risk_score * np.log1p(safe_amt)
    amount_ratio = transaction_amount / (np.abs(avg_transaction_amount) + 1e-6)
    hour_sin = np.sin(2 * np.pi * time_of_day / 24.0)
    hour_cos = np.cos(2 * np.pi * time_of_day / 24.0)

    num_raw["risk_log_amount"] = risk_log_amount
    num_raw["amount_ratio"] = amount_ratio
    num_raw["hour_sin"] = hour_sin
    num_raw["hour_cos"] = hour_cos

    all_num_cols = NUM_COLS + ENGINEERED_NUM_COLS

    # --- handle_outliers_and_skew ---
    outlier_info = {}  # col -> {"log1p": bool, "lower": float, "upper": float}
    for c in all_num_cols:
        vals = num_raw[c]
        skew = pandas_skew(vals)
        do_log = bool(abs(skew) > 1.0 and np.all(vals >= 0))
        if do_log:
            vals = np.log1p(vals)
            num_raw[c] = vals
        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        num_raw[c] = np.clip(vals, lower, upper)
        outlier_info[c] = {"log1p": do_log, "lower": lower, "upper": upper}

    print(f"  outlier handling done ({time.time()-t0:.1f}s)")

    # --- perform_encoding ---
    ordinal_maps = {}
    bin_encoded = {}
    for c in BIN_COLS:
        col_vals = np.array(cat_raw[c]).reshape(-1, 1)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded = enc.fit_transform(col_vals).ravel()
        cats = enc.categories_[0].tolist()
        ordinal_maps[c] = {str(cat): float(i) for i, cat in enumerate(cats)}
        bin_encoded[c] = encoded

    onehot_maps = {}
    onehot_encoded = {}  # col -> {category_col_name: np.array}
    for c in NOM_COLS:
        col_vals = np.array(cat_raw[c]).reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = enc.fit_transform(col_vals)
        cats = enc.categories_[0].tolist()
        onehot_maps[c] = [str(cat) for cat in cats]
        for i, cat in enumerate(cats):
            onehot_encoded[f"{c}_{cat}"] = encoded[:, i]

    print(f"  encoding done ({time.time()-t0:.1f}s)")

    # --- assemble final feature matrix, column order matches pandas concat order:
    #     [retained original-order cols (num + bin, nominal cols removed)] + [OHE cols per NOM_COLS order] ---
    retained_cols = [c for c in feature_cols_raw_order if c not in NOM_COLS] + ENGINEERED_NUM_COLS
    feature_order = []
    columns = {}
    for c in retained_cols:
        if c in num_raw:
            columns[c] = num_raw[c]
            feature_order.append(c)
        elif c in bin_encoded:
            columns[c] = bin_encoded[c]
            feature_order.append(c)
    for c in NOM_COLS:
        for cat in onehot_maps[c]:
            name = f"{c}_{cat}"
            columns[name] = onehot_encoded[name]
            feature_order.append(name)

    n_rows = len(feat_rows)
    X = np.column_stack([columns[c] for c in feature_order]).astype(float)
    print(f"  final feature matrix: {X.shape}")

    # --- perform_scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaling = {
        feature_order[j]: {"mean": float(scaler.mean_[j]), "std": float(scaler.scale_[j])}
        for j in range(len(feature_order))
    }

    print(f"  scaling done ({time.time()-t0:.1f}s)")

    # --- train/val split (matches run_evaluation_cycle's stratified 80/20, seed 42) ---
    rng = np.random.RandomState(RNG_SEED)
    idx_pos = np.where(y_all == 1)[0]
    idx_neg = np.where(y_all == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_val_pos = int(len(idx_pos) * 0.2)
    n_val_neg = int(len(idx_neg) * 0.2)
    val_idx = np.concatenate([idx_pos[:n_val_pos], idx_neg[:n_val_neg]])
    train_idx = np.concatenate([idx_pos[n_val_pos:], idx_neg[n_val_neg:]])
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)

    X_tr, y_tr = X_scaled[train_idx], y_all[train_idx]
    X_val, y_val = X_scaled[val_idx], y_all[val_idx]

    def eval_metrics(y_true, y_prob, threshold=0.5):
        y_pred = (y_prob >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def best_f1_threshold(y_true, y_prob):
        """Sweep thresholds and pick the one maximizing F1. With ~14% fraud
        prevalence, 0.5 is a poor operating point (the notebook's own DTL
        exploration cell uses 0.15 for the same reason) -- this generalizes
        that idea instead of hardcoding a threshold."""
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            m = eval_metrics(y_true, y_prob, threshold=t)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_t = float(t)
        return best_t

    metrics = {}

    # --- train LogReg on validation split first (for metrics), then refit on full data ---
    print("Training Logistic Regression (validation split)...")
    logreg_val = CustomLogisticRegression(learning_rate=0.01, n_iterations=3000, lambda_reg=0.05)
    logreg_val.fit(X_tr, y_tr)
    val_probs = logreg_val.predict_proba(X_val)
    logreg_threshold = best_f1_threshold(y_val, val_probs)
    metrics["logreg"] = eval_metrics(y_val, val_probs, threshold=logreg_threshold)
    print(f"  LogReg val metrics @ t={logreg_threshold:.2f}: {metrics['logreg']}  ({time.time()-t0:.1f}s)")

    print("Training Logistic Regression (full data, for deployment)...")
    logreg_full = CustomLogisticRegression(learning_rate=0.01, n_iterations=3000, lambda_reg=0.05)
    logreg_full.fit(X_scaled, y_all)
    print(f"  done ({time.time()-t0:.1f}s)")

    # --- train DTL similarly ---
    MAX_DEPTH, MIN_SAMPLES, MIN_SAMPLES_LEAF = 10, 200, 50
    print(f"Training Decision Tree (validation split, max_depth={MAX_DEPTH})...")
    tree_val = build_tree(X_tr, y_tr, feature_order, 0, MAX_DEPTH, MIN_SAMPLES, MIN_SAMPLES_LEAF)

    feat_index = {f: j for j, f in enumerate(feature_order)}

    def predict_tree_proba_fast(X, node_root):
        out = np.empty(len(X))
        for i in range(len(X)):
            node = node_root
            row = X[i]
            while node.value is None:
                j = feat_index[node.feature]
                node = node.left if row[j] <= node.threshold else node.right
            out[i] = node.value
        return out

    val_probs_tree = predict_tree_proba_fast(X_val, tree_val)
    dtl_threshold = best_f1_threshold(y_val, val_probs_tree)
    metrics["dtl"] = eval_metrics(y_val, val_probs_tree, threshold=dtl_threshold)
    print(f"  DTL val metrics @ t={dtl_threshold:.2f}: {metrics['dtl']}  nodes={count_nodes(tree_val)}  ({time.time()-t0:.1f}s)")

    print("Training Decision Tree (full data, for deployment)...")
    tree_full = build_tree(X_scaled, y_all, feature_order, 0, MAX_DEPTH, MIN_SAMPLES, MIN_SAMPLES_LEAF)
    print(f"  done, nodes={count_nodes(tree_full)}  ({time.time()-t0:.1f}s)")

    # ---------------------------------------------------------------------
    # Export artifacts
    # ---------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    pipeline = {
        "raw_feature_order": feature_cols_raw_order,
        "num_cols": NUM_COLS,
        "engineered_num_cols": ENGINEERED_NUM_COLS,
        "bin_cols": BIN_COLS,
        "nom_cols": NOM_COLS,
        "impute_medians": impute_medians,
        "impute_modes": impute_modes,
        "outlier": outlier_info,
        "ordinal_maps": ordinal_maps,
        "onehot_maps": onehot_maps,
        "scaling": scaling,
        "feature_order": feature_order,
    }
    with open(os.path.join(OUT_DIR, "pipeline.json"), "w", encoding="utf-8") as f:
        json.dump(pipeline, f)

    with open(os.path.join(OUT_DIR, "logreg.json"), "w", encoding="utf-8") as f:
        json.dump({"weights": logreg_full.weights.tolist(), "bias": float(logreg_full.bias)}, f)

    with open(os.path.join(OUT_DIR, "dtl_tree.json"), "w", encoding="utf-8") as f:
        json.dump(tree_full.to_json(), f)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "logreg": metrics["logreg"],
            "dtl": metrics["dtl"],
            "logreg_threshold": logreg_threshold,
            "dtl_threshold": dtl_threshold,
            "dtl_nodes": count_nodes(tree_full),
            "dtl_hyperparams": {"max_depth": MAX_DEPTH, "min_samples": MIN_SAMPLES, "min_samples_leaf": MIN_SAMPLES_LEAF},
            "logreg_hyperparams": {"learning_rate": 0.01, "n_iterations": 3000, "lambda_reg": 0.05},
            "n_train_rows": n_rows,
            "fraud_prevalence": float(np.mean(y_all)),
        }, f, indent=2)

    # --- sample rows for the demo's "load a sample" button (raw values + ground truth label) ---
    random.seed(RNG_SEED)
    sample_indices = random.sample(range(len(feat_rows)), 24)
    samples = []
    for i in sample_indices:
        raw = {c: rows[keep_idx[i]][c] for c in feature_cols_raw_order}
        samples.append({"raw": raw, "is_fraud": int(y_all[i])})
    with open(os.path.join(OUT_DIR, "samples.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    # --- parity fixtures: raw input + expected model outputs, for JS<->Python cross-check ---
    parity_indices = random.sample(range(len(feat_rows)), 10)
    parity = []
    for i in parity_indices:
        raw = {c: rows[keep_idx[i]][c] for c in feature_cols_raw_order}
        x_row = X_scaled[i]
        logreg_p = float(logreg_full.predict_proba(x_row.reshape(1, -1))[0])
        dtl_p = float(predict_tree_proba_fast(x_row.reshape(1, -1), tree_full)[0])
        parity.append({"raw": raw, "logreg_proba": logreg_p, "dtl_proba": dtl_p, "is_fraud": int(y_all[i])})
    with open(os.path.join(OUT_DIR, "parity.json"), "w", encoding="utf-8") as f:
        json.dump(parity, f, indent=2)

    print(f"\nDone in {time.time()-t0:.1f}s. Artifacts written to {OUT_DIR}")
    for fname in ["pipeline.json", "logreg.json", "dtl_tree.json", "metrics.json", "samples.json", "parity.json"]:
        p = os.path.join(OUT_DIR, fname)
        print(f"  {fname}: {os.path.getsize(p)/1024:.1f} KB")


if __name__ == "__main__":
    main()
