#!/usr/bin/env python3
"""Iris 多模型对比：同一训练/测试划分，一键评测表。"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _build_models() -> list[tuple[str, object]]:
    return [
        (
            "Perceptron",
            make_pipeline(
                StandardScaler(),
                Perceptron(max_iter=1000, eta0=0.01, tol=1e-3, random_state=42),
            ),
        ),
        (
            "LogisticRegression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ),
        (
            "SVM(RBF)",
            make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
            ),
        ),
        (
            "KNN(k=5)",
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        ),
        ("DecisionTree", DecisionTreeClassifier(random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("GaussianNB", GaussianNB()),
        (
            "LDA",
            make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        ),
    ]


def _print_table(rows: Iterable[tuple[str, float, float]]) -> None:
    rows = list(rows)
    name_width = max(len("Model"), max(len(r[0]) for r in rows))
    header = f"{'Model':<{name_width}} | {'Accuracy':>8} | {'Macro-F1':>8}"
    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)
    for name, acc, macro_f1 in rows:
        print(f"{name:<{name_width}} | {acc:>8.4f} | {macro_f1:>8.4f}")
    print(divider)


def main() -> None:
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 同一训练/测试划分，保证多模型公平对比
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    results: list[tuple[str, float, float]] = []
    for model_name, model in _build_models():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        results.append((model_name, acc, macro_f1))

    results.sort(key=lambda x: (x[1], x[2]), reverse=True)

    print("=== Iris 多模型对比 (同一 train/test split) ===")
    print("split: test_size=0.3, random_state=42, stratify=y")
    _print_table(results)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
