"""
CIFAR-10 手写数字识别 - HOG + KNN 算法
"""

import os
import pickle
import numpy as np
import time
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ==================== 配置 ====================

DATA_DIR = "/Users/fisher/Documents/workspace/mycode/AI/cifar-10-batches-py"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# HOG 参数
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# KNN 参数
K = 3


# ==================== 数据加载 ====================

def unpickle(file_path):
    """读取 CIFAR-10 pickle 文件"""
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")
    return data_dict


def load_data():
    """加载训练集和测试集"""
    images_list = []
    labels_list = []
    
    # 训练集 (5个batch)
    for i in range(1, 6):
        batch_file = os.path.join(DATA_DIR, f"data_batch_{i}")
        batch = unpickle(batch_file)
        images_list.append(batch[b"data"])
        labels_list.extend(batch[b"labels"])
    
    train_images = np.vstack(images_list)
    train_labels = np.array(labels_list)
    
    # 测试集
    test_batch = unpickle(os.path.join(DATA_DIR, "test_batch"))
    test_images = test_batch[b"data"]
    test_labels = np.array(test_batch[b"labels"])
    
    print(f"   训练集: {len(train_images)} 样本")
    print(f"   测试集: {len(test_images)} 样本")
    
    return train_images, train_labels, test_images, test_labels


def images_to_hog_features(images):
    """将图片转换为HOG特征"""
    features = []
    total = len(images)
    
    for i, img in enumerate(images):
        # 原始格式: (3072,) -> (32, 32, 3)
        img = img.reshape(3, 32, 32).transpose(1, 2, 0)
        
        # 转灰度图
        gray = rgb2gray(img)
        
        # 提取HOG特征
        hog_feat = hog(gray, 
                      orientations=HOG_ORIENTATIONS,
                      pixels_per_cell=HOG_PIXELS_PER_CELL,
                      cells_per_block=HOG_CELLS_PER_BLOCK,
                      block_norm='L2-Hys')
        
        features.append(hog_feat)
        
        # 进度显示
        if (i + 1) % 5000 == 0:
            print(f"   进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
    
    return np.array(features)


# ==================== KNN 分类 ====================

def train_knn(X_train, y_train, k=3):
    """训练KNN模型"""
    print(f"📦 KNN 模型初始化: k={k}")
    print(f"   训练样本数: {len(X_train)}")
    
    model = KNeighborsClassifier(
        n_neighbors=k,
        metric='euclidean',
        n_jobs=-1
    )
    
    print("⏳ 正在加载训练数据到模型...")
    start = time.time()
    model.fit(X_train, y_train)
    print(f"✅ 模型加载完成! 耗时: {time.time() - start:.2f}秒")
    
    return model


def evaluate(model, X_test, y_test):
    """评估模型"""
    print("\n🔍 正在测试...")
    
    batch_size = 1000
    all_preds = []
    
    start = time.time()
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        preds = model.predict(batch)
        all_preds.extend(preds)
        
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
    for i, name in enumerate(CLASSES):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == i).sum() / mask.sum()
            print(f"   {name:12s}: {class_acc*100:.1f}%")
    
    return accuracy


def main():
    print("=" * 50)
    print("CIFAR-10 HOG + KNN 分类器")
    print("=" * 50)
    
    # 加载数据
    print("\n📂 加载数据...")
    train_images, train_labels, test_images, test_labels = load_data()
    
    # 提取HOG特征
    print("\n🔧 提取HOG特征 (训练集)...")
    start = time.time()
    train_hog = images_to_hog_features(train_images)
    print(f"   训练集HOG特征维度: {train_hog.shape}")
    print(f"   耗时: {time.time() - start:.1f}秒")
    
    print("\n🔧 提取HOG特征 (测试集)...")
    start = time.time()
    test_hog = images_to_hog_features(test_images)
    print(f"   测试集HOG特征维度: {test_hog.shape}")
    print(f"   耗时: {time.time() - start:.1f}秒")
    
    # 标准化
    print("\n📏 特征标准化...")
    scaler = StandardScaler()
    train_hog = scaler.fit_transform(train_hog)
    test_hog = scaler.transform(test_hog)
    
    # 训练KNN
    model = train_knn(train_hog, train_labels, k=K)
    
    # 评估
    accuracy = evaluate(model, test_hog, test_labels)
    
    # 模型信息
    model_size_mb = train_hog.nbytes / (1024 * 1024)
    print(f"\n📦 模型大小: {model_size_mb:.1f} MB (存储所有训练HOG特征)")
    print(f"   特征维度: {train_hog.shape[1]}")
    print(f"   训练样本数: {len(train_hog)}")
    
    print("\n" + "=" * 50)
    print("完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
