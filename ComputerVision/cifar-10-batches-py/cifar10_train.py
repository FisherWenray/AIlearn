"""
CIFAR-10 分类模型训练与测试脚本
基于 PyTorch，使用 ResNet-18 进行训练

使用方式:
  训练:       python cifar10_train.py --mode train
  测试:       python cifar10_train.py --mode test
  训练+测试:  python cifar10_train.py --mode both
  单张预测:   python cifar10_train.py --mode predict --index 42
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ==================== 配置 ====================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cifar10_resnet18.pth")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 超参数
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")


# ==================== 数据集 ====================

def unpickle(file_path):
    """读取 CIFAR-10 pickle 文件"""
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")
    return data_dict


class CIFAR10Dataset(Dataset):
    """自定义 CIFAR-10 数据集"""

    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        images_list = []
        labels_list = []

        if train:
            for i in range(1, 6):
                batch_file = os.path.join(data_dir, f"data_batch_{i}")
                batch = unpickle(batch_file)
                images_list.append(batch[b"data"])
                labels_list.extend(batch[b"labels"])
            self.images = np.vstack(images_list)
        else:
            batch = unpickle(os.path.join(data_dir, "test_batch"))
            self.images = batch[b"data"]
            labels_list = batch[b"labels"]

        self.labels = np.array(labels_list)
        # CIFAR-10 原始格式: (N, 3072) -> (N, 3, 32, 32)
        self.images = self.images.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label


# ==================== 数据增强 ====================

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# ==================== 模型 (ResNet-18 for CIFAR-10) ====================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """适配 CIFAR-10 (32x32) 的 ResNet-18"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # CIFAR-10 用 3x3 卷积代替 7x7，不用 MaxPool
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ==================== 训练 ====================

def train():
    print(f"🖥  设备: {DEVICE}")
    print(f"📂 数据集: {DATA_DIR}")
    print(f"📊 超参数: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print("=" * 60)

    # 数据
    train_dataset = CIFAR10Dataset(DATA_DIR, train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)

    # 模型
    model = ResNet18(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数量: {total_params:,}")
    print()

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
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

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:2d}/{EPOCHS}]  "
              f"Loss: {epoch_loss:.4f}  "
              f"Acc: {epoch_acc:.2f}%  "
              f"LR: {lr:.6f}  "
              f"Time: {elapsed:.0f}s")

        # 每 5 轮在测试集上评估一次
        if epoch % 5 == 0 or epoch == EPOCHS:
            test_acc = evaluate(model)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  ✅ 最佳模型已保存 (测试准确率: {best_acc:.2f}%)")
            print()

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"🎉 训练完成！总耗时: {total_time:.0f}s")
    print(f"🏆 最佳测试准确·率: {best_acc:.2f}%")
    print(f"💾 模型保存至: {MODEL_PATH}")


def evaluate(model):
    """在测试集上评估模型"""
    test_dataset = CIFAR10Dataset(DATA_DIR, train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    return acc


# ==================== 测试 ====================

def test():
    import time  # 导入 time 模块

    print(f"🖥  设备: {DEVICE}")
    print(f"📂 数据集: {DATA_DIR}")
    print(f"💾 模型路径: {MODEL_PATH}")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print("❌ 模型文件不存在，请先运行训练！")
        print(f"   python {__file__} --mode train")
        return

    # 加载模型
    model = ResNet18(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("✅ 模型加载成功")
    print()

    # 加载测试数据
    test_dataset = CIFAR10Dataset(DATA_DIR, train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    # 记录开始时间
    start_time = time.time()

    # 整体准确率
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    # 计算耗时
    elapsed_time = time.time() - start_time
    throughput = total / elapsed_time if elapsed_time > 0 else 0
    avg_infer_time = (elapsed_time / total * 1000) if total > 0 else 0  # 毫秒/张

    overall_acc = 100.0 * correct / total

    # 打印结果
    print(f"📊 测试结果 (共 {total} 张图片)")
    print(f"   总体准确率: {overall_acc:.2f}%")
    print(f"   总耗时: {elapsed_time:.2f}s")
    print(f"   推理速度: {throughput:.1f} images/s")
    print(f"   平均推理时间: {avg_infer_time:.2f} ms/image")
    print()
    print("   各类别准确率:")
    print(f"   {'类别':<14} {'正确':>6} / {'总数':>6}   {'准确率':>7}")
    print("   " + "-" * 42)
    for i in range(10):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"   {CLASSES[i]:<14} {class_correct[i]:>5} / {class_total[i]:>5}   "
              f"{acc:>6.2f}%  {bar}")
    print()


# ==================== 单张图片预测 ====================

def predict(index):
    """从测试集中选一张图片进行预测"""
    from PIL import Image

    print(f"🖥  设备: {DEVICE}")
    print(f"💾 模型路径: {MODEL_PATH}")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print("❌ 模型文件不存在，请先运行训练！")
        return

    # 加载测试数据（不做 normalize，用于保存原始图片）
    test_dataset_raw = CIFAR10Dataset(DATA_DIR, train=False, transform=None)
    total_images = len(test_dataset_raw)

    if index < 0 or index >= total_images:
        print(f"❌ 索引超出范围！测试集共 {total_images} 张图片 (索引 0-{total_images - 1})")
        return

    # 获取原始图片和标签
    raw_image, true_label = test_dataset_raw[index]

    # 保存原始图片为 PNG
    img_array = raw_image.numpy()  # (3, 32, 32)
    img_array = np.transpose(img_array, (1, 2, 0))  # -> (32, 32, 3)
    img_array = (img_array * 255).astype(np.uint8)
    save_path = os.path.join(DATA_DIR, f"predict_{index}.png")
    Image.fromarray(img_array).resize((256, 256), Image.NEAREST).save(save_path)

    # 加载模型
    model = ResNet18(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 预测（需要 normalize）
    image_normalized = test_transform(raw_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_normalized)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_label = outputs.argmax(1).item()

    # 打印结果
    is_correct = predicted_label == true_label
    status = "✅ 正确" if is_correct else "❌ 错误"

    print(f"🖼  测试集第 {index} 张图片")
    print(f"   真实标签: {CLASSES[true_label]}")
    print(f"   预测结果: {CLASSES[predicted_label]}  {status}")
    print()
    print("   各类别置信度:")

    # 按概率排序显示 Top-5
    probs_sorted, indices_sorted = probabilities.sort(descending=True)
    for rank in range(5):
        cls_idx = indices_sorted[rank].item()
        prob = probs_sorted[rank].item() * 100
        marker = " 👈" if cls_idx == true_label else ""
        bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
        print(f"   {rank + 1}. {CLASSES[cls_idx]:<12} {prob:>6.2f}%  {bar}{marker}")

    print()
    print(f"💾 图片已保存至: {save_path}")


# ==================== 主入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 分类模型训练与测试")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["train", "test", "both", "predict"],
                        help="运行模式: train(训练), test(测试), both(训练+测试), predict(单张预测)")
    parser.add_argument("--index", type=int, default=0,
                        help="predict 模式下的测试集图片索引 (0-9999)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"训练轮数 (默认: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"批大小 (默认: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"学习率 (默认: {LEARNING_RATE})")
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr

    if args.mode in ("train", "both"):
        train()

    if args.mode in ("test", "both"):
        test()

    if args.mode == "predict":
        predict(args.index)
