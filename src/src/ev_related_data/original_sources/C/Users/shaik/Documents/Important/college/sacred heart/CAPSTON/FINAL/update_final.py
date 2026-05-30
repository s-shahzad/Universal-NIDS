"""End-to-end EVCS DoS model comparison with saved reports and plots."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


RANDOM_STATE = 42
DROP_COLUMNS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"]
TARGET_COLUMN = "Label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate EVCS DoS detection models."
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/CAPSTON/network_traffic.csv",
        help="Path to the CSV dataset (default: network_traffic.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for metrics and confusion matrix plots.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction (default: 0.2).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Cross-validation folds for GridSearchCV (default: 3).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of rows to sample (stratified) before training.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display confusion matrix plots interactively.",
    )
    return parser.parse_args()


def _sanitize_file_part(text: str) -> str:
    return text.lower().replace(" ", "_")


def load_and_prepare_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)

    if TARGET_COLUMN not in df.columns:
        if "Attack" in df.columns:
            df[TARGET_COLUMN] = df["Attack"].map({"Benign": 0, "DoS": 1})
        else:
            raise ValueError(
                f"Dataset must include '{TARGET_COLUMN}' or 'Attack' column."
            )

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    features = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS, errors="ignore")
    numeric_features = features.select_dtypes(include=["number"]).copy()
    numeric_features = numeric_features.fillna(numeric_features.median(numeric_only=True))

    if numeric_features.empty:
        raise ValueError("No numeric feature columns available after preprocessing.")

    y = df[TARGET_COLUMN]
    class_count = y.nunique()
    if class_count < 2:
        raise ValueError(
            "Target column has fewer than two classes after preprocessing."
        )

    return numeric_features, y


def build_model_configs() -> dict[str, tuple[object, dict[str, list]]]:
    configs: dict[str, tuple[object, dict[str, list]]] = {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        ),
        "SVM": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("svc", SVC(probability=True, random_state=RANDOM_STATE)),
                ]
            ),
            {
                "svc__C": [0.1, 1.0, 10.0],
                "svc__gamma": ["scale", "auto"],
                "svc__kernel": ["rbf"],
            },
        ),
    }

    if XGBClassifier is not None:
        configs["XGBoost"] = (
            XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                tree_method="hist",
            ),
            {
                "learning_rate": [0.1, 0.01],
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
            },
        )
    else:
        print("XGBoost is not installed. Skipping XGBoost model.")

    return configs


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    cv: int,
    show_plots: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    models = build_model_configs()

    for name, (model, param_grid) in models.items():
        print(f"\nModel: {name}")
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
            verbose=1,
            n_jobs=-1,
        )

        train_start = time.time()
        grid.fit(X_train, y_train)
        train_time = time.time() - train_start

        best_model = grid.best_estimator_
        print("Best Params   :", grid.best_params_)
        print(f"Training Time : {train_time:.2f} sec")

        test_start = time.time()
        y_pred = best_model.predict(X_test)
        test_time = time.time() - test_start
        print(f"Testing Time  : {test_time:.2f} sec")

        y_proba = None
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "best_params": grid.best_params_,
            "training_time_sec": round(train_time, 4),
            "testing_time_sec": round(test_time, 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        }

        if y_proba is not None:
            metrics["auroc"] = round(roc_auc_score(y_test, y_proba), 4)

        print(f"Accuracy      : {metrics['accuracy']:.4f}")
        print(f"Precision     : {metrics['precision']:.4f}")
        print(f"Recall        : {metrics['recall']:.4f}")
        print(f"F1 Score      : {metrics['f1_score']:.4f}")
        if "auroc" in metrics:
            print(f"AUROC         : {metrics['auroc']:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        display = ConfusionMatrixDisplay(cm, display_labels=["Benign", "DoS"])
        fig, ax = plt.subplots(figsize=(6, 6))
        display.plot(ax=ax, cmap="Blues", values_format=",")
        ax.set_title(f"{name} - Confusion Matrix")
        ax.grid(False)

        plot_file = output_dir / f"{_sanitize_file_part(name)}_confusion_matrix.png"
        fig.tight_layout()
        fig.savefig(plot_file, dpi=150)
        print(f"Saved plot     : {plot_file}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        results.append(metrics)

    return results


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_and_prepare_data(data_path)

    if args.sample_size is not None:
        if args.sample_size <= 1:
            raise ValueError("--sample-size must be greater than 1.")
        if args.sample_size < len(X):
            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=args.sample_size,
                stratify=y,
                random_state=RANDOM_STATE,
            )
            print(f"Using sample size: {args.sample_size}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )

    print(f"Loaded dataset  : {data_path.resolve()}")
    print(f"Output folder   : {output_dir.resolve()}")
    print(f"Train/Test size : {X_train.shape[0]}/{X_test.shape[0]}")
    print(f"Feature count   : {X.shape[1]}")

    results = evaluate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=output_dir,
        cv=args.cv,
        show_plots=args.show_plots,
    )

    results_df = pd.DataFrame(results)
    summary_csv = output_dir / "model_summary.csv"
    summary_json = output_dir / "model_summary.json"
    results_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nSaved summary  : {summary_csv}")
    print(f"Saved summary  : {summary_json}")


if __name__ == "__main__":
    main()


