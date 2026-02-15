#!/usr/bin/env python3
"""
ONNX Runtime Python 推理示例

Usage:
    python infer_onnx.py <model.onnx> <image.png>
    python infer_onnx.py models/cifar10_resnet18.onnx data/test.png
"""

import sys
import os
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed")
    print("  Run: pip install onnxruntime")
    sys.exit(1)

# CIFAR-10 classes
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Normalization parameters
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def preprocess(image_path):
    """
    图片预处理

    Steps:
        1. 读取图片
        2. Resize to 32x32
        3. Convert to numpy array
        4. Normalize to [0, 1]
        5. Standardize (subtract mean, divide by std)
        6. HWC -> CHW
        7. Add batch dimension
    """
    # 1. 读取图片
    img = Image.open(image_path)

    # 2. Resize to 32x32
    img = img.resize((32, 32))

    # 3. Convert to numpy array (HWC format)
    img_array = np.array(img, dtype=np.float32)

    # 4. Normalize to [0, 1]
    img_array = img_array / 255.0

    # 5. Standardize
    img_array = (img_array - MEAN) / STD

    # 6. HWC -> CHW (C=3, H=32, W=32)
    img_array = np.transpose(img_array, (2, 0, 1))

    # 7. Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def softmax(logits):
    """
    Softmax 激活函数

    Args:
        logits: 模型原始输出 (1, 10)

    Returns:
        probabilities: 概率分布 (1, 10)
    """
    exp_logits = np.exp(logits - np.max(logits))  # numerical stability
    return exp_logits / np.sum(exp_logits)


def print_top5(probs):
    """打印 Top-5 预测结果"""
    # 获取排序后的索引（降序）
    sorted_indices = np.argsort(probs[0])[::-1]

    print()
    print("Top-5 predictions:")
    print("-" * 40)

    for i in range(5):
        idx = sorted_indices[i]
        prob = probs[0][idx] * 100.0
        bar_len = int(prob / 5)
        bar = "#" * bar_len + "-" * (20 - bar_len)

        print(f"  {i + 1}. {CLASSES[idx]:<12}: {prob:>6.2f}%  {bar}")


def main():
    # 检查参数
    if len(sys.argv) != 3:
        print("Usage: python infer_onnx.py <model.onnx> <image.png>")
        print()
        print("Example:")
        print("  python infer_onnx.py models/cifar10_resnet18.onnx data/test.png")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    # 检查文件
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)

    print("=" * 50)
    print("CIFAR-10 ONNX Runtime Python Inference")
    print("=" * 50)
    print()

    # 1. 加载模型
    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path)
    print("   Model loaded successfully")
    print(f"   Inputs: {[input.name for input in session.get_inputs()]}")
    print(f"   Outputs: {[output.name for output in session.get_outputs()]}")
    print()

    # 2. 预处理图片
    print(f"Processing image: {image_path}")
    input_data = preprocess(image_path)
    print(f"   Input shape: {input_data.shape}")
    print()

    # 3. 推理
    print("Running inference...")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run(None, {input_name: input_data})
    logits = outputs[0]
    print(f"   Output shape: {logits.shape}")
    print()

    # 4. 后处理
    probs = softmax(logits)
    pred_class = np.argmax(probs[0])

    # 5. 输出结果
    print("=" * 50)
    print(f"Prediction: {CLASSES[pred_class]}")
    print(f"Confidence: {probs[0][pred_class] * 100:.2f}%")
    print("=" * 50)

    print_top5(probs)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
