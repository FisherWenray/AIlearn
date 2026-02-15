"""
MNIST 手写数字识别 - KNN (K-Nearest Neighbors) 算法
"""

import os
import struct
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==================== 配置 ====================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# KNN 超参数
K = 3 # 近邻数量
METRIC = 'euclidean'  # 距离度量: euclidean(欧氏), manhattan(曼哈顿)


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


# ==================== KNN 算法 ====================

def train_knn(X_train, y_train, k=3, metric='euclidean'):
    """
    训练 KNN 模型
    注意：KNN 没有真正的"训练"过程，只是存储数据
    """
    print(f"📦 KNN 模型初始化: k={k}, metric={metric}")
    print(f"   训练样本数: {len(X_train)}")
    
    model = KNeighborsClassifier(
        n_neighbors=k,
        metric=metric,
        n_jobs=-1  # 使用所有 CPU 核心
    )
    
    print("⏳ 正在加载训练数据到模型...")
    start = time.time()
    model.fit(X_train, y_train)
    print(f"✅ 模型加载完成! 耗时: {time.time() - start:.2f}秒")
    
    return model


def evaluate(model, X_test, y_test):
    """评估模型"""
    print("\n🔍 正在测试...")
    
    # 为了进度显示，分批预测
    batch_size = 1000
    all_preds = []
    
    start = time.time()
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        preds = model.predict(batch)
        all_preds.extend(preds)
        
        # 显示进度
        progress = min(i + batch_size, len(X_test)) / len(X_test) * 100
        print(f"   进度: {progress:.1f}%", end='\r')
    
    elapsed = time.time() - start
    print(f"\n⏱️ 推理耗时: {elapsed:.2f}秒 ({len(X_test)/elapsed:.1f} 样本/秒)")
    
    y_pred = np.array(all_preds)
    
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
    print("MNIST KNN 分类器")
    print("=" * 50)
    
    # 加载数据
    print("\n📂 加载数据...")
    train_images, train_labels, test_images, test_labels = load_data()
    print(f"   训练集: {len(train_images)} 样本")
    print(f"   测试集: {len(test_images)} 样本")
    print(f"   特征维度: {train_images.shape[1]}")
    
    # 训练（实际是加载数据）
    model = train_knn(train_images, train_labels, k=K, metric=METRIC)
    
    # 评估
    accuracy = evaluate(model, test_images, test_labels)
    
    # 参数量（KNN 没有参数，存储的是训练数据）
    model_size_mb = train_images.nbytes / (1024 * 1024)
    print(f"\n📦 模型大小: {model_size_mb:.1f} MB (存储所有训练数据)")
    print(f"   训练样本数: {len(train_images)}")
    print(f"   每个样本维度: {train_images.shape[1]}")
    
    print("\n" + "=" * 50)
    print("完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
