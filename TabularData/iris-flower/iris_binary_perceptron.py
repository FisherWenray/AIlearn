#!/usr/bin/env python3
"""Iris 配对二分类：同一感知机配置做三次测试。"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_pair_test(
    X_all: np.ndarray,
    y_all: np.ndarray,
    target_names: np.ndarray,
    class_neg: int,
    class_pos: int,
) -> tuple[float, int]:
    mask = (y_all == class_neg) | (y_all == class_pos)
    X = X_all[mask]
    y = y_all[mask]

    # 映射成二分类标签：{-1, +1}
    y_bin = np.where(y == class_neg, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_bin,
        test_size=0.3,
        random_state=42,
        stratify=y_bin,
    )

    # 同一个感知机配置，针对每一对类别单独训练并测试
    model = make_pipeline(
        StandardScaler(),
        Perceptron(max_iter=1000, eta0=0.01, tol=1e-3, random_state=42),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    pair_names = [target_names[class_neg], target_names[class_pos]]
    perceptron = model.named_steps["perceptron"]

    print("=" * 60)
    print("Iris 配对二分类(sklearn 单层感知机)")
    print(f"类别: {pair_names[0]}(-1) vs {pair_names[1]}(+1)")
    print(f"测试集准确率: {acc:.4f}")
    print(f"训练轮数: {perceptron.n_iter_}\n")

    print("混淆矩阵 (行=真实, 列=预测):")
    print(cm, "\n")

    print("分类报告:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[-1, 1],
            target_names=pair_names,
        )
    )

    print("前 10 条测试样本预测(真实 -> 预测):")
    label_to_name = {-1: pair_names[0], 1: pair_names[1]}
    for yt, yp in list(zip(y_test, y_pred))[:10]:
        print(f"{label_to_name[yt]} -> {label_to_name[yp]}")

    return acc, perceptron.n_iter_


def main() -> None:
    iris = load_iris()
    X_all = iris.data
    y_all = iris.target

    summary: list[tuple[str, float, int]] = []
    for class_neg, class_pos in combinations(range(len(iris.target_names)), 2):
        acc, n_iter = run_pair_test(
            X_all=X_all,
            y_all=y_all,
            target_names=iris.target_names,
            class_neg=class_neg,
            class_pos=class_pos,
        )
        pair_label = f"{iris.target_names[class_neg]} vs {iris.target_names[class_pos]}"
        summary.append((pair_label, acc, n_iter))

    print("\n" + "=" * 60)
    print("三组配对测试汇总")
    for pair_label, acc, n_iter in summary:
        print(f"{pair_label}: accuracy={acc:.4f}, n_iter={n_iter}")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
