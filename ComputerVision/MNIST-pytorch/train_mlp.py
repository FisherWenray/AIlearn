"""
MNIST 手写数字识别 - ANN(MLP)模型训练
"""

import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==================== 配置 ====================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_mlp.pth")

# 超参数
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available()
                     else "cpu")


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


# ==================== MLP 模型 ====================

class MLP(nn.Module):
    """多层感知机 (ANN)"""
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x


# ==================== 训练 ====================

def train():
    print(f"🖥  设备: {DEVICE}")
    print(f"📊 超参数: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, hidden={HIDDEN_SIZE}")
    print("=" * 60)
    
    # 加载数据
    train_images, train_labels, test_images, test_labels = load_data()
    print(f"📂 训练集: {len(train_images)} 张, 测试集: {len(test_images)} 张")
    
    # 转换为 Tensor
    train_x = torch.tensor(train_images, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    test_x = torch.tensor(test_images, dtype=torch.float32)
    test_y = torch.tensor(test_labels, dtype=torch.long)
    
    # DataLoader
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 模型
    model = MLP(input_size=784, hidden_size=HIDDEN_SIZE, num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数量: {total_params:,}")
    print()
    
    # 训练
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        # 每 5 轮评估一次
        if epoch % 5 == 0 or epoch == EPOCHS:
            test_acc = evaluate(model, test_x, test_y)
            print(f"Epoch [{epoch:2d}/{EPOCHS}]  Loss: {epoch_loss:.4f}  "
                  f"Train Acc: {epoch_acc:.2f}%  Test Acc: {test_acc:.2f}%")
        else:
            print(f"Epoch [{epoch:2d}/{EPOCHS}]  Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n💾 模型已保存至: {MODEL_PATH}")
    
    # 最终测试
    test_acc = evaluate(model, test_x, test_y)
    print(f"🏆 最终测试准确率: {test_acc:.2f}%")


def evaluate(model, test_x, test_y):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
        outputs = model(test_x)
        _, predicted = outputs.max(1)
        correct = predicted.eq(test_y).sum().item()
    return 100.0 * correct / len(test_y)


# ==================== 测试 ====================

def test():
    print(f"🖥  设备: {DEVICE}")
    print(f"💾 模型路径: {MODEL_PATH}")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print("❌ 模型文件不存在，请先训练！")
        return
    
    # 加载数据
    _, _, test_images, test_labels = load_data()
    test_x = torch.tensor(test_images, dtype=torch.float32).to(DEVICE)
    test_y = torch.tensor(test_labels, dtype=torch.long).to(DEVICE)
    
    # 加载模型
    model = MLP(input_size=784, hidden_size=HIDDEN_SIZE, num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 预测
    with torch.no_grad():
        outputs = model(test_x)
        _, predicted = outputs.max(1)
        correct = predicted.eq(test_y).sum().item()
    
    acc = 100.0 * correct / len(test_y)
    print(f"📊 测试准确率: {acc:.2f}% ({correct}/{len(test_y)})")


# ==================== 主入口 ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MNIST MLP 训练")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    else:
        test()
