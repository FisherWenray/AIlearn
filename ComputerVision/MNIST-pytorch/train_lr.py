"""
MNIST 手写数字识别 - Logistic Regression (逻辑回归) 算法
"""

import os
import struct
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==================== 配置 ====================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== 数据加载 ====================

def read_idx_images(filepath):
    """读取 MNIST 图像文件"""
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
    return images


def read_idx_labels(filepath):
    """读取 MNIST 标签文件"""
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_data():
    """加载训练集和测试集"""
    # 训练集
    train_images = read_idx_images(os.path.join(DATA_DIR, 'train-images.idx3-ubyte'))
    train_labels = read_idx_labels(os.path.join(DATA_DIR, 'train-labels.idx1-ubyte'))
    
    # 测试集
    test_images = read_idx_images(os.path.join(DATA_DIR, 't10k-images.idx3-ubyte'))
    test_labels = read_idx_labels(os.path.join(DATA_DIR, 't10k-labels.idx1-ubyte'))
    
    # 归一化到 [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    return train_images, train_labels, test_images, test_labels


# ==================== Logistic Regression ====================

def train_lr(X_train, y_train, max_iter=1000):
    """
    训练 Logistic Regression 模型
    """
    print(f"📦 Logistic Regression 模型初始化")
    print(f"   最大迭代次数: {max_iter}")
    print(f"   训练样本数: {len(X_train)}")
    
    model = LogisticRegression(
        max_iter=max_iter,
        solver='lbfgs',        # 优化器
        multi_class='multinomial',  # 多分类
        n_jobs=-1,
        verbose=1
    )
    
    print("⏳ 正在训练...")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"✅ 训练完成! 耗时: {elapsed:.2f}秒")
    
    return model


def evaluate(model, X_test, y_test):
    """评估模型"""
    print("\n🔍 正在测试...")
    
    start = time.time()
    y_pred = model.predict(X_test)
    elapsed = time.time() - start
    
    print(f"⏱️ 推理耗时: {elapsed:.2f}秒")
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 测试准确率: {accuracy * 100:.2f}%")
    
    # 详细报告
    print("\n📊 各类别准确率:")
    report = classification_report(y_test, y_pred, digits=2)
    print(report)
    
    return accuracy


def main():
    print("=" * 50)
    print("MNIST Logistic Regression 分类器")
    print("=" * 50)
    
    # 加载数据
    print("\n📂 加载数据...")
    train_images, train_labels, test_images, test_labels = load_data()
    print(f"   训练集: {len(train_images)} 样本")
    print(f"   测试集: {len(test_images)} 样本")
    print(f"   特征维度: {train_images.shape[1]}")
    
    # 训练
    model = train_lr(train_images, train_labels)
    
    # 评估
    accuracy = evaluate(model, test_images, test_labels)
    
    # 参数量
    # Logistic Regression: 每个类别一个权重向量 (784 维) + 偏置 (10 类)
    num_params = train_images.shape[1] * 10 + 10
    print(f"\n📦 模型参数量: {num_params:,}")
    print(f"   权重: {train_images.shape[1]} × 10 = {train_images.shape[1] * 10:,}")
    print(f"   偏置: 10")
    
    # 模型大小
    model_size_kb = num_params * 4 / 1024  # float32 = 4 bytes
    print(f"   模型大小: {model_size_kb:.2f} KB")
    
    print("\n" + "=" * 50)
    print("完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
