#!/usr/bin/env python3
"""
Export CIFAR-10 ResNet-18 model to ONNX format

Usage:
    python export_onnx.py

Requirements:
    pip install onnx onnxruntime
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os
import sys

# CIFAR-10 classes
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "cifar10_resnet18.pth")
ONNX_PATH = os.path.join(SCRIPT_DIR, "cifar10_resnet18.onnx")


# ==================== Model Definition ====================

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
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 for CIFAR-10 (32x32)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

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


# ==================== Export Function ====================

def export_onnx():
    """Export PyTorch model to ONNX format"""

    print("=" * 60)
    print("CIFAR-10 ONNX Export")
    print("=" * 60)
    print()

    # Check model file
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("   Please train first: python cifar10_train.py --mode train")
        return False

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = ResNet18(num_classes=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print("   Model loaded successfully")
    print(f"   Classes: {len(CLASSES)}")
    print()

    # Create example input (batch_size=1, channels=3, height=32, width=32)
    example_input = torch.randn(1, 3, 32, 32)
    print(f"Input shape: {example_input.shape}")
    print()

    # Export to ONNX
    print(f"Exporting to ONNX: {ONNX_PATH}")
    torch.onnx.export(
        model,
        example_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=11,
        do_constant_folding=True,
        export_params=True
    )
    print("   Export successful!")
    print()

    # Verify exported model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("   ONNX model is valid!")
    print()

    # Test with ONNX Runtime
    print("Testing with ONNX Runtime...")
    ort_session = ort.InferenceSession(ONNX_PATH)

    # Run inference
    ort_inputs = {"input": example_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    output = ort_outputs[0]

    print(f"   ONNX output shape: {output.shape}")
    print(f"   PyTorch output shape: {model(example_input).shape}")
    print()

    # Check outputs match
    torch_output = model(example_input).detach().numpy()
    max_diff = np.max(np.abs(output - torch_output))
    print(f"   Max difference between ONNX and PyTorch: {max_diff:.10f}")

    if max_diff < 1e-5:
        print("   ✓ Outputs match!")
    else:
        print("   ⚠ Outputs differ (may be due to different implementations)")
    print()

    # Print model info
    print("=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print()
    print(f"ONNX model: {ONNX_PATH}")
    print()
    print("Model Information:")
    print(f"   - Input:  (batch_size, 3, 32, 32)")
    print(f"   - Output: (batch_size, {len(CLASSES)})")
    print(f"   - Opset:  11")
    print()
    print("Classes:")
    for i, cls in enumerate(CLASSES):
        print(f"   {i}: {cls}")
    print()

    return True


# ==================== Main ====================

if __name__ == "__main__":
    success = export_onnx()
    sys.exit(0 if success else 1)
