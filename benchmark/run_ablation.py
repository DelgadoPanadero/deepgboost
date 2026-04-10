"""
Ablation runner: runs classification datasets only with fresh model instances per dataset.
Usage: python -m benchmark.run_ablation --n_layers N --n_trees T --tag LABEL
"""
import os
import json
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# Parse args first
parser = argparse.ArgumentParser()
parser.add_argument("--n_layers", type=int, required=True)
parser.add_argument("--n_trees", type=int, required=True)
parser.add_argument("--tag", type=str, required=True)
args = parser.parse_args()

assert args.n_layers * args.n_trees <= 100, "Tree budget exceeded! n_layers*n_trees must be <=100"

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
DATA_DIR = os.path.join(BENCH_DIR, "data")

# Classification datasets only
DATASETS = [
    {"name": "Adult", "file": "adult.csv", "target": "income"},
    {"name": "BankMarketing", "file": "bank_full.csv", "target": "y"},
    {"name": "Abalone", "file": "abalone.csv", "target": "sex"},
    {"name": "Penguins", "file": "penguins.csv", "target": "species"},
]

N_RUNS = 5
TEST_SIZE = 0.25


def load_dataset(file, target):
    path = os.path.join(DATA_DIR, file)
    data = pd.read_csv(path).dropna().reset_index(drop=True)
    X = data.drop(target, axis=1)
    if cat_cols := X.select_dtypes(exclude=["number"]).columns.tolist():
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        new_data = enc.fit_transform(X[cat_cols].astype(str).apply(lambda col: col.str.strip()))
        X = X.drop(cat_cols, axis=1)
        feat_df = pd.DataFrame(new_data, columns=enc.get_feature_names_out())
        X = pd.concat([X.reset_index(drop=True), feat_df], axis=1)
    X = X.values.astype(float)
    y_raw = data[target].astype(str).apply(lambda col: col.strip())
    y = LabelEncoder().fit_transform(y_raw)
    return X, y


def make_models(n_layers, n_trees):
    """Create fresh model instances — critical to avoid XGBoost state carry-over."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier
    from deepgboost import DeepGBoostMultiClassifier

    return {
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric="logloss", verbosity=0),
        "DeepGBoost": DeepGBoostMultiClassifier(
            n_layers=n_layers,
            n_trees=n_trees,
            learning_rate=0.1,
            n_jobs=8,
        ),
    }


results_summary = {}

for ds_info in DATASETS:
    name = ds_info["name"]
    print(f"\n--- Dataset: {name} ---")
    X, y = load_dataset(ds_info["file"], ds_info["target"])
    n_classes = len(np.unique(y))
    print(f"  Shape: {X.shape}, Classes: {n_classes}, Counts: {np.bincount(y)}")

    run_scores = {m: [] for m in ["GradientBoosting", "RandomForest", "XGBoost", "DeepGBoost"]}

    for run_idx in tqdm(range(N_RUNS)):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=run_idx * 7 + 42
        )
        # Fresh models each dataset-run to avoid state carry-over
        models = make_models(args.n_layers, args.n_trees)

        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                run_scores[model_name].append(score)
            except Exception as e:
                print(f"  ERROR {model_name} run {run_idx}: {e}")
                run_scores[model_name].append(float("nan"))

    results_summary[name] = {m: run_scores[m] for m in run_scores}

    # Print per-dataset summary
    print(f"\n  Results (n_layers={args.n_layers}, n_trees={args.n_trees}):")
    for model_name, scores in run_scores.items():
        arr = np.array(scores)
        valid = arr[~np.isnan(arr)]
        print(f"    {model_name:20s}: {np.mean(valid):.4f} ± {np.std(valid):.4f}")
    dg = np.array(run_scores["DeepGBoost"])
    xgb = np.array(run_scores["XGBoost"])
    gap = np.mean(dg) - np.mean(xgb)
    print(f"    Gap DeepGBoost vs XGBoost: {gap:+.4f}")


# Final summary
print(f"\n\n=== FINAL SUMMARY: n_layers={args.n_layers}, n_trees={args.n_trees} (tag={args.tag}) ===")
final = {}
for name, run_scores in results_summary.items():
    dg = np.array(run_scores["DeepGBoost"])
    xgb = np.array(run_scores["XGBoost"])
    gb = np.array(run_scores["GradientBoosting"])
    rf = np.array(run_scores["RandomForest"])
    final[name] = {
        "DeepGBoost_mean": float(np.nanmean(dg)),
        "DeepGBoost_std": float(np.nanstd(dg)),
        "XGBoost_mean": float(np.nanmean(xgb)),
        "GradientBoosting_mean": float(np.nanmean(gb)),
        "RandomForest_mean": float(np.nanmean(rf)),
        "gap_vs_XGBoost": float(np.nanmean(dg) - np.nanmean(xgb)),
        "gap_vs_GradientBoosting": float(np.nanmean(dg) - np.nanmean(gb)),
    }
    print(f"\n{name}:")
    print(f"  DeepGBoost:       {final[name]['DeepGBoost_mean']:.4f} ± {final[name]['DeepGBoost_std']:.4f}")
    print(f"  XGBoost:          {final[name]['XGBoost_mean']:.4f}")
    print(f"  GradientBoosting: {final[name]['GradientBoosting_mean']:.4f}")
    print(f"  Gap vs XGBoost:   {final[name]['gap_vs_XGBoost']:+.4f}")

# Save ablation results
ablation_file = os.path.join(RESULTS_DIR, f"ablation_{args.tag}.json")
with open(ablation_file, "w") as f:
    json.dump({
        "config": {"n_layers": args.n_layers, "n_trees": args.n_trees, "tag": args.tag},
        "results": final,
    }, f, indent=2)
print(f"\nResults saved to {ablation_file}")
