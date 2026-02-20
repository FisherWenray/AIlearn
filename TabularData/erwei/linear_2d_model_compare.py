#!/usr/bin/env python3
"""Compare multiple classifiers on a synthetic 2D linearly separable dataset."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

HAS_MATPLOTLIB = True
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False
    plt = None


def build_models() -> dict[str, object]:
    return {
        "Perceptron(single-layer)": make_pipeline(
            StandardScaler(),
            Perceptron(max_iter=1000, eta0=0.01, tol=1e-3, random_state=42),
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        "SVM(Linear)": make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=1.0, random_state=42),
        ),
        "SVM(RBF)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        ),
        "KNN(k=7)": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7),
        ),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LDA": make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(),
        ),
    }


def evaluate_models(
    models: dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[list[dict[str, float | str]], dict[str, object]]:
    results: list[dict[str, float | str]] = []
    fitted_models: dict[str, object] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        result = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
        }
        results.append(result)
        fitted_models[name] = model

    results.sort(key=lambda x: (x["accuracy"], x["f1"]), reverse=True)
    return results, fitted_models


def print_results_table(results: list[dict[str, float | str]]) -> None:
    name_width = max(len("Model"), max(len(str(row["model"])) for row in results))
    header = (
        f"{'Model':<{name_width}} | {'Accuracy':>8} | {'Precision':>9} | "
        f"{'Recall':>6} | {'F1':>6}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)
    for row in results:
        print(
            f"{str(row['model']):<{name_width}} | "
            f"{float(row['accuracy']):>8.4f} | "
            f"{float(row['precision']):>9.4f} | "
            f"{float(row['recall']):>6.4f} | "
            f"{float(row['f1']):>6.4f}"
        )
    print(sep)


def save_generated_dataset(X: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    data = np.column_stack((X, y))
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="feature_1,feature_2,label",
        comments="",
        fmt=["%.6f", "%.6f", "%d"],
    )


def plot_decision_boundaries(
    fitted_models: dict[str, object],
    results: list[dict[str, float | str]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    n_models = len(results)
    cols = 3
    rows = ceil(n_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    X_all = np.vstack([X_train, X_test])
    x_min, x_max = X_all[:, 0].min() - 1.0, X_all[:, 0].max() + 1.0
    y_min, y_max = X_all[:, 1].min() - 1.0, X_all[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for idx, row in enumerate(results):
        name = str(row["model"])
        model = fitted_models[name]
        ax = axes[idx]

        zz = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")

        ax.scatter(
            X_train[y_train == 0, 0],
            X_train[y_train == 0, 1],
            c="tab:blue",
            s=20,
            label="class 0 train",
            alpha=0.85,
        )
        ax.scatter(
            X_train[y_train == 1, 0],
            X_train[y_train == 1, 1],
            c="tab:orange",
            s=20,
            label="class 1 train",
            alpha=0.85,
        )
        ax.scatter(
            X_test[y_test == 0, 0],
            X_test[y_test == 0, 1],
            c="tab:blue",
            marker="x",
            s=30,
            label="class 0 test",
            alpha=0.9,
        )
        ax.scatter(
            X_test[y_test == 1, 0],
            X_test[y_test == 1, 1],
            c="tab:orange",
            marker="x",
            s=30,
            label="class 1 test",
            alpha=0.9,
        )

        ax.set_title(f"{name}\\nacc={row['accuracy']:.3f}, f1={row['f1']:.3f}")
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")

    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("2D Decision Boundaries on the Same Split", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # 1) Generate 2D linearly-separable-like data
    X, y = make_classification(
        n_samples=600,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        flip_y=0.01,
        random_state=42,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "linear_2d_dataset.csv"
    save_generated_dataset(X, y, dataset_path)

    # 2) Same split for all algorithms
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # 3) Train and compare
    models = build_models()
    results, fitted_models = evaluate_models(models, X_train, y_train, X_test, y_test)

    print("=== Synthetic 2D Classification Benchmark ===")
    print("Data: make_classification(n_features=2, n_classes=2, class_sep=2.0)")
    print("Split: test_size=0.3, random_state=42, stratify=y")
    print_results_table(results)
    print(f"Best model: {results[0]['model']} (acc={results[0]['accuracy']:.4f})")
    print(f"Saved dataset: {dataset_path.resolve()}")

    # 4) Optional visualization
    if not HAS_MATPLOTLIB:
        print("Matplotlib is not installed. Skip boundary plot.")
        return

    output_path = output_dir / "linear_2d_decision_boundaries.png"
    plot_decision_boundaries(
        fitted_models=fitted_models,
        results=results,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_path=output_path,
    )
    print(f"Saved boundary plot: {output_path.resolve()}")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
