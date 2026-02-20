#!/usr/bin/env python3
"""Iris 数据集感知机分类示例。"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from iris_visualization import generate_visualizations


def _build_baseline_model() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        Perceptron(max_iter=1000, eta0=0.01, random_state=42, tol=1e-3),
    )


def _tune_perceptron_model(X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scaler", StandardScaler()),
            ("clf", Perceptron(random_state=42, tol=1e-3)),
        ]
    )
    param_grid = {
        "poly__degree": [1, 2],
        "clf__eta0": [0.001, 0.01, 0.1, 1.0],
        "clf__max_iter": [1000, 3000],
        "clf__penalty": [None, "l2"],
        "clf__alpha": [1e-5, 1e-4, 1e-3],
        "clf__fit_intercept": [True, False],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search


def main() -> None:
    # 1) 加载数据
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2) 切分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # 3) 基线模型（你之前的配置）
    baseline_model = _build_baseline_model()
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    # 4) 改进：交叉验证调参 + 可选二次特征
    search = _tune_perceptron_model(X_train, y_train)
    model = search.best_estimator_
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=== Iris 感知机分类结果 ===")
    print(f"基线模型准确率: {baseline_acc:.4f}")
    print(f"调参后准确率: {acc:.4f}")
    print(f"交叉验证最佳分数: {search.best_score_:.4f}")
    print(f"最佳参数: {search.best_params_}\n")
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm, "\n")
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    print("前 5 条测试样本预测(真实 -> 预测):")
    for true_label, pred_label in list(zip(y_test, y_pred))[:5]:
        print(f"{iris.target_names[true_label]} -> {iris.target_names[pred_label]}")

    # 6) 生成更详细的 Iris 可视化结果
    generate_visualizations(
        X=X,
        y=y,
        feature_names=iris.feature_names,
        target_names=iris.target_names,
        cm=cm,
    )



if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
