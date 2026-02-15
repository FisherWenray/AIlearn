#!/usr/bin/env python
# coding: utf-8

# ### AI培训课程系列课后作业

# ### 第四周课后作业，所有作业分为七周理论课的课后作业和一个实践项目大作业
# #### 个人课后作业：跑通cifar10分类训练，并基于上一周的充气拱门数据，整理并训练一个充气拱门的分类模型(几分类不限定),并测试点数
# #### 小组课后作业: 基于上一周八选一的数据，整理并训练一个充气拱门的分类模型(几分类不限定),并测试点数
# #### 课后选做作业: 分别使用softmax loss和cosface loss训练，并自己调节不同的𝛼和𝑚，训练模型，看看cosface是否确实能取得比softmax loss更好的结果

# ![image.png](attachment:0cff23ec-2bc2-4d49-9c80-0efd3b7c53c4.png)

# #### 作业参考：[https://studio.brainpp.com/project/55?name=MegEngine%20%E7%9A%84%20CIFA-10%20%E8%AE%AD%E7%BB%83 ]
# #### 作业参考：[https://studio.brainpp.com/project/7517?name=AI%E5%9F%B9%E8%AE%AD%E8%AF%BE%E5%90%8E%E4%BD%9C%E4%B8%9A%283_4%29_%E4%BD%9C%E4%B8%9A%E5%8F%82%E8%80%83 ]

# In[1]:


import os

from megengine.data.dataset import CIFAR10


# In[2]:


train_dataset = CIFAR10(root="./dataset", train=True, download=True)
test_dataset = CIFAR10(root="./dataset", train=False, download=False)


# In[3]:


import megengine as mge
train_on_gpu =  mge.is_cuda_available()
if not train_on_gpu:
    print('CUDA is not available!')
    mge.set_default_device('cpux')
else:
    print('CUDA is available!')
    mge.set_default_device('gpux')


# In[4]:


import megengine as mge
import megengine.module as M


class ConvBN(M.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = M.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        self.bn = M.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicResidualBlock(M.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.conv_bn1 = ConvBN(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv_bn2 = ConvBN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = M.ReLU()
        if self.stride == 1 and in_channels == out_channels:
            self.identity = M.Identity()
        else:
            self.identity = ConvBN(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        identity = self.identity(x)
        x = self.conv_bn1(x)
        x = self.relu(x)
        x = self.conv_bn2(x)
        x = x + identity
        x = self.relu(x)
        return x


class ResNet20(M.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = M.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn0 = M.BatchNorm2d(16)
        self.relu0 = M.ReLU()

        def make_layer(in_channels, out_channels, num_blocks, stride):
            blocks = []
            for i in range(num_blocks):
                blocks.append(BasicResidualBlock(in_channels, out_channels, 
                                                 stride=1 if i > 0 else stride))
                in_channels = out_channels
            return M.Sequential(*blocks)

        self.layer1 = make_layer(16, 16, 3, 1)
        self.layer2 = make_layer(16, 32, 3, 2)
        self.layer3 = make_layer(32, 64, 3, 2)
        self.pool = M.AvgPool2d(8)
        self.classifier = M.Linear(64, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

resnet20 = ResNet20()


# In[5]:


from megengine.data import DataLoader, RandomSampler
from megengine.data import transform

batch_size = 128
sampler = RandomSampler(dataset=train_dataset, batch_size=batch_size, drop_last=True)
# mean 和 std 的值来自于： 
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/cifar10.lua#L39-L40
mean = [125.3, 123.0, 113.9]
std = [63.0,  62.1,  66.7]
transform = transform.Compose([
                transform.RandomHorizontalFlip(),
                transform.RandomCrop(32, padding_size=4),
                transform.Normalize([0.,0.,0.], [255.,255.,255.]),
                transform.ToMode("CHW"),
            ])
train_dataloader = DataLoader(
    train_dataset,
    sampler=sampler,
    transform=transform,
)


# In[6]:


from megengine.jit import trace
import megengine.functional as F


# 定义静态图训练函数
@trace
def train_func(data, label, *, net, gm):
    net.train() # 网络设置成训练模式
    with gm:
        pred = net(data)
        # 使用交叉熵损失
        loss = F.loss.cross_entropy(pred, label)
        gm.backward(loss)
    return pred, loss


# In[7]:


import megengine.optimizer as optim
from megengine.autodiff import GradManager

# 定义优化器
opt = optim.SGD(resnet20.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
gm = GradManager().attach(resnet20.parameters())


# In[8]:


import megengine as mge
import numpy as np

# set trace.enabled=False if you want to run eager mode
# trace.enabled = False

# 训练迭代，优化器更新参数
# 这里为了方便演示，只迭代 10 个 epochs 。
# 实际训练可以设成 200 个 epochs ，在第 100 和第 150 个 epoch 位置将 lr 分别降至 0.01 和 0.001 。
epochs = 10
for i in range(epochs):
    loss_rec = []
    for data, label in train_dataloader:
        opt.clear_grad()
        _, loss = train_func(mge.tensor(data), mge.tensor(label, dtype="int32"), net=resnet20, gm=gm)
        opt.step()
        loss_rec.append(loss.numpy().item())
    loss = sum(loss_rec) / len(loss_rec)
    print("[Epoch {}] loss: {}".format(i, loss))


# In[9]:


mge.save(resnet20.state_dict(), 'workspace/cifar10_resnet_static.mge')


# In[10]:


resnet20 = ResNet20()
state_dict = mge.load('workspace/cifar10_resnet_static.mge')
resnet20.load_state_dict(state_dict)


# In[11]:


from megengine.data import SequentialSampler
from megengine.data import transform

batch_size = 100
sampler_test = SequentialSampler(dataset=test_dataset, batch_size=batch_size)

transform_test = transform.Compose([
                transform.Normalize([0.,0.,0.], [255.,255.,255.]),
                transform.ToMode("CHW"),
])
    
test_dataloader = DataLoader(
    test_dataset,
    sampler=sampler_test,
    transform=transform_test,
)


# In[12]:


# 定义静态图测试函数
@trace
def eval_func(data, label, *, net):
    net.eval() # 网络设置成测试模式
    pred = net(data)
    loss = F.loss.cross_entropy(pred, label)
    return pred, loss

correct = 0
total = 0
for data, label in test_dataloader:
    label = label.astype("float32")
    data = mge.tensor(data)
    label = mge.tensor(label)
    pred, _ = eval_func(data, label, net=resnet20)
    pred_label = pred.numpy().argmax(axis=1)
    correct += (pred_label == label).sum().item()
    total += label.shape[0]

print("correct: {}, total: {}, accuracy: {:.2f}%".format(correct, total, correct * 100.0 / total))


# In[13]:


import os
import json
import cv2
import shutil
import time
import random
import numpy as np
import argparse
import tarfile
from IPython import embed


def cvt_rect_json(p, w, h):
    return [
        max(int(p[0][0] * w), 0),
        max(int(p[0][1] * h), 0),
        min(int((p[1][0] - p[0][0]) * w), int(w)),
        min(int((p[2][1] - p[1][1]) * h), int(h))
    ]

def get_iou(a, b):
    if a == [] or b == []:
        return 0
    startx, endx = min(a[0], b[0]), max(a[0] + a[2], b[0] + b[2])
    starty, endy = min(a[1], b[1]), max(a[1] + a[3], b[1] + b[3])
    width = a[2] + b[2] - (endx - startx)
    height = a[3] + b[3] - (endy - starty)
    if width <= 0 or height <= 0:
        return 0
    else:
        area = width * height

    return area * 1.0 / (a[2] * a[3] + b[2] * b[3] - area)


def is_negtive(box, gts):
    for gt in gts:
        if get_iou(box, gt) > 0.2:
            return False
    return True


def make_crop_resize_json(obj, out):
    out_negetive = out + "_negetive"
    name = os.path.join("workspace", obj["uris"][0])
    if not os.path.exists(name):
        print("path not exist:",name)
        return
    # img = cv2.imread(name)
    img = cv2.imdecode(np.fromfile(name, dtype=np.uint8), -1)
    if img is None:
        return

    w = obj["resources"][0]["size"]["width"]
    h = obj["resources"][0]["size"]["height"]
    if not obj["results"]:
        return
    
    gts = []
    for item in obj["results"]["rects"]:
        rect = cvt_rect_json(item["rect"], w, h)
        gts.append(rect)
        if rect[3] < 10 or rect[2] < 10:
            continue
        img_crop = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        try:
            img_resize = cv2.resize(img_crop, (224, 224))
        except Exception:
            print("resize image failed")
            embed()
        name = str(time.time()).replace(".", "") + "".join(
            random.sample('zyxwvutsrqponmlkjihgfedcba', 10)) + ".png"
        #img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(os.path.join(out, name), img_resize)
    _times = 0
    while True and _times < 10:
        left, right = np.random.randint(0, int(w/2)), np.random.randint(0,int(h/2))
        width, height = np.random.randint(int(w/4), int(w/2) - 1), np.random.randint(int(h/4), int(h/2) - 1)
        if is_negtive([left, right, width, height], gts):
            img_crop = img[left:left+width, right:right+height]
            if np.min(img_crop.shape[:2]) < 20:
                continue
            img_resize = cv2.resize(img_crop, (224, 224))
            name = str(time.time()).replace(".", "") + "".join(
                random.sample('zyxwvutsrqponmlkjihgfedcba', 10)) + ".png"
            cv2.imwrite(os.path.join(out_negetive, name), img_resize)
            break
        _times += 1

def parse_json(json_file, out):
    """
    input:  json
    output: crop pic
    """
    with open(json_file, "r") as f:
        objs = json.load(f)
    # print(len(objs["items"]))
    for obj in objs["items"]:
        make_crop_resize_json(obj, out)


def main():
    _out = "workspace/crop_image"
    _json = "workspace/job-467099-4.json"
    out_negetive = _out + "_negetive"
    if os.path.exists(_out):
        shutil.rmtree(_out)
    os.makedirs(_out)
    if os.path.exists(out_negetive):
        shutil.rmtree(out_negetive)
    os.makedirs(out_negetive)

    parse_json(_json, _out)
    print("positive num: ", len(os.listdir(_out)))
    print("negetive num: ", len(os.listdir(out_negetive)))

if __name__ == "__main__":
    main()


# In[14]:


import os

p_path = "workspace/crop_image"
n_path = "workspace/crop_image_negetive"

p_list = []
n_list = []

for _file in os.listdir(p_path):
    file_path = os.path.join(p_path, _file)
    p_list.append([file_path, 1])

for _file in os.listdir(n_path):
    file_path = os.path.join(n_path, _file)
    n_list.append([file_path, 0])


train_p_num = int(len(p_list) * 0.6)
train_n_num = int(len(n_list) * 0.6)

verify_p_num = int(len(p_list) * 0.2)
verify_n_num = int(len(n_list) * 0.2)

with open("workspace/train.list", "w") as f:
    for l in p_list[:train_p_num]:
        f.write(l[0] + "\t" + str(l[1]) + "\n")
    for l in n_list[:train_n_num]:
        f.write(l[0] + "\t" + str(l[1]) + "\n")

with open("workspace/verify.list", "w") as f:
    for l in p_list[train_p_num:train_p_num+verify_p_num]:
        f.write(l[0] + "\t" + str(l[1]) + "\n")
    for l in n_list[train_n_num:train_n_num+verify_n_num]:
        f.write(l[0] + "\t" + str(l[1]) + "\n")
        
with open("workspace/test.list", "w") as f:
    for l in p_list[train_p_num+verify_p_num:]:
        f.write(l[0] + "\t" + str(l[1]) + "\n")
    for l in n_list[train_n_num+verify_n_num:]:
        f.write(l[0] + "\t" + str(l[1]) + "\n")


# In[15]:


"""MegEngine ImageNet DataLoader"""
import io
import os

import cv2
import megengine.data as data
import megengine.data.transform as T
import numpy as np
from megengine.data.dataset import Dataset

__all__ = ["ImageNetLocalDataset"]


class LocalDataset(Dataset):
    def __init__(self, local_list):
        self.local_list = local_list
        self.decode_local_list()

    def __getitem__(self, index):
        file_path, target = self.samples[index]
        # sample = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1) # some read png is 4 channel
        sample = cv2.imread(file_path) 
        # print(file_path, " ", sample.shape[0], " ", sample.shape[1], " ",sample.shape[2],"_")
        assert sample.shape == (224, 224, 3)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    local List: {}\n".format(self.local_list)

    def decode_local_list(self, local_list):
        raise NotImplementedError


class ImageNetLocalDataset(LocalDataset):
    def __init__(self, local_list):
        super().__init__(local_list)

    def decode_local_list(self):
        self.samples = []
        with open(self.local_list, "r") as f:
            for line in f.readlines():
                file_path, target = line.strip().split()
                assert int(target) == 0 or int(target) == 1
                self.samples.append((file_path, int(target)))


# In[16]:


import math

import megengine.functional as F
import megengine.hub as hub
import megengine.module as M


class BasicBlock(M.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        norm=M.BatchNorm2d,
    ):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        self.conv1 = M.Conv2d(
            in_channels, channels, 3, stride, padding=dilation, bias=False
        )
        self.bn1 = norm(channels)
        self.conv2 = M.Conv2d(channels, channels, 3, 1, padding=1, bias=False)
        self.bn2 = norm(channels)
        self.downsample = (
            M.Identity()
            if in_channels == channels and stride == 1
            else M.Sequential(
                M.Conv2d(in_channels, channels, 1, stride, bias=False),
                norm(channels),
            )
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(identity)
        x += identity
        x = F.relu(x)
        return x


class Bottleneck(M.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        norm=M.BatchNorm2d,
    ):
        super().__init__()
        width = int(channels * (base_width / 64.0)) * groups
        self.conv1 = M.Conv2d(in_channels, width, 1, 1, bias=False)
        self.bn1 = norm(width)
        self.conv2 = M.Conv2d(
            width,
            width,
            3,
            stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = norm(width)
        self.conv3 = M.Conv2d(
            width, channels * self.expansion, 1, 1, bias=False
        )
        self.bn3 = norm(channels * self.expansion)
        self.downsample = (
            M.Identity()
            if in_channels == channels * self.expansion and stride == 1
            else M.Sequential(
                M.Conv2d(
                    in_channels,
                    channels * self.expansion,
                    1,
                    stride,
                    bias=False,
                ),
                norm(channels * self.expansion),
            )
        )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        identity = self.downsample(identity)

        x += identity
        x = F.relu(x)

        return x


class ResNet(M.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=2,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm=M.BatchNorm2d,
    ):
        super().__init__()
        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = M.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm(self.in_channels)
        self.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm=norm)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            norm=norm,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            norm=norm,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            norm=norm,
        )
        self.fc = M.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)
            elif isinstance(m, M.Linear):
                M.init.msra_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block
        # behaves like an identity. According to https://arxiv.org/abs/1706.02677
        # This improves the model by 0.2~0.3%.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    M.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    M.init.zeros_(m.bn2.weight)

    def _make_layer(
        self,
        block,
        channels,
        blocks,
        stride=1,
        dilate=False,
        norm=M.BatchNorm2d,
    ):
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        layers.append(
            block(
                self.in_channels,
                channels,
                stride,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm=norm,
            )
        )
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm=norm,
                )
            )

        return M.Sequential(*layers)

    def extract_features(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        outputs["stem"] = x

        x = self.layer1(x)
        outputs["res2"] = x
        x = self.layer2(x)
        outputs["res3"] = x
        x = self.layer3(x)
        outputs["res4"] = x
        x = self.layer4(x)
        outputs["res5"] = x
        return outputs

    def forward(self, x):
        x = self.extract_features(x)["res5"]

        x = F.avg_pool2d(x, 7)
        x = F.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# In[17]:


import argparse
import bisect
import os
import time

import megengine
import megengine.autodiff as autodiff
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.optimizer as optim

logging = megengine.logger.get_logger()

class param():
    def __init__(self):
        self.data = ""
        self.arch = "resnet50"
        self.ngpus = 0
        self.save = "workspace/output"
        self.epochs = 10
        self.batch_size = 8
        self.lr = 0.0125
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.workers = 4
        self.print_freq = 20


def worker(args):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        os.makedirs(os.path.join(args.save, args.arch), exist_ok=True)
        megengine.logger.set_log_file(os.path.join(args.save, args.arch, "log.txt"))

    # build dataset
    train_dataloader, valid_dataloader = build_dataset(args)
    train_queue = iter(train_dataloader)  # infinite
    steps_per_epoch = 6400 // (world_size * args.batch_size)

    '''
    for step in range(0, steps_per_epoch):
        image, label = next(train_queue)
        print("step:", step," label:",label)
    '''
    # build model
    model = resnet50(num_classes=2)

    # Sync parameters
    if world_size > 1:
        dist.bcast_list_(model.parameters(), dist.WORLD)

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
        callbacks=dist.make_allreduce_cb("SUM") if world_size > 1 else None,
    )
    # Optimizer
    opt = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay * world_size,  # scale weight decay in "SUM" mode
    )

    # train and valid func
    def train_step(image, label):
        with gm:
            logits = model(image)
            loss = F.nn.cross_entropy(logits, label)  # , label_smooth=0.1)
            acc1, acc2 = F.topk_accuracy(logits, label, topk=(1, 2))
            gm.backward(loss)
            opt.step().clear_grad()
        return loss, acc1, acc2

    def valid_step(image, label):
        logits = model(image)
        loss = F.nn.cross_entropy(logits, label)
        acc1, acc2 = F.topk_accuracy(logits, label, topk=(1, 2))
        # calculate mean values
        if world_size > 1:
            loss = F.distributed.all_reduce_sum(loss) / world_size
            acc1 = F.distributed.all_reduce_sum(acc1) / world_size
            acc2 = F.distributed.all_reduce_sum(acc2) / world_size
        return loss, acc1, acc2

    # multi-step learning rate scheduler
    def adjust_learning_rate(step):
        lr = args.lr * 0.1 ** bisect.bisect_right(
            [30 * steps_per_epoch, 60 * steps_per_epoch, 80 * steps_per_epoch],
            step,
        )
        # if step < 5 * steps_per_epoch:  # warmup
        #     lr = args.lr * (step / (5 * steps_per_epoch))
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        return lr

    # start training
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top2 = AverageMeter("Acc@2")
    clck = AverageMeter("Time")

    for step in range(0, args.epochs * steps_per_epoch):
        lr = adjust_learning_rate(step)

        t = time.time()

        image, label = next(train_queue)
        #print("step:", step," label:",label)
        image = megengine.tensor(image, dtype="float32")
        label = megengine.tensor(label, dtype="int32")

        loss, acc1, acc2 = train_step(image, label)

        objs.update(loss.item())
        top1.update(100 * acc1.item())
        top2.update(100 * acc2.item())
        clck.update(time.time() - t)

        if step % args.print_freq == 0 and rank == 0:
            logging.info(
                "epoch %d Step %d, LR %.4f, %s %s %s",
                step // steps_per_epoch,
                step,
                lr,
                objs,
                top1,
                clck,
            )
            objs.reset()
            top1.reset()
            top2.reset()
            clck.reset()

        if (step + 1) % (1 * steps_per_epoch) == 0:  # eval and save frequency
            model.eval()
            _, valid_acc1, valid_acc2 = valid(
                valid_step, valid_dataloader, args
            )
            model.train()
            logging.info(
                "epoch %d Test Acc@1 %.3f Acc@2 %.3f",
                (step + 1) // steps_per_epoch, valid_acc1, valid_acc2
            )
            megengine.save(
                {
                    "epoch": (step + 1) // steps_per_epoch,
                    "state_dict": model.state_dict(),
                },
                os.path.join(args.save, args.arch, "checkpoint.pkl"),
            ) if rank == 0 else None


def valid(func, data_queue, args):
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top2 = AverageMeter("Acc@2")
    clck = AverageMeter("Time")

    t = time.time()
    for step, (image, label) in enumerate(data_queue):
        image = megengine.tensor(image, dtype="float32")
        label = megengine.tensor(label, dtype="int32")

        n = image.shape[0]

        loss, acc1, acc2 = func(image, label)

        objs.update(loss.item(), n)
        top1.update(100 * acc1.item(), n)
        top2.update(100 * acc2.item(), n)
        clck.update(time.time() - t, n)
        t = time.time()

        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info(
                "Test step %d, %s %s %s %s", step, objs, top1, top2, clck
            )

    return objs.avg, top1.avg, top2.avg


def build_dataset(args):

    train_dataset = ImageNetLocalDataset(
        "workspace/train.list"
    )
    train_sampler = data.Infinite(
        data.RandomSampler(
            train_dataset, batch_size=args.batch_size, drop_last=True
        )
    )
    train_dataloader = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.Normalize(
                    mean=[103.530, 116.280, 123.675],
                    std=[57.375, 57.120, 58.395],
                ),  # BGR
                T.ToMode("CHW"),
            ]
        ),
        num_workers=args.workers,
    )
    valid_dataset = ImageNetLocalDataset(
        "workspace/test.list"
    )
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=args.batch_size, drop_last=False
    )
    valid_dataloader = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        transform=T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[103.530, 116.280, 123.675],
                    std=[57.375, 57.120, 58.395],
                ),  # BGR
                T.ToMode("CHW"),
            ]
        ),
        num_workers=args.workers,
    )
    return train_dataloader, valid_dataloader


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":.3f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

#检查GPU是否可用
# import megengine
train_on_gpu =  megengine.is_cuda_available()
if not train_on_gpu:
    print('CUDA is not available!')
    megengine.set_default_device('cpux')
else:
    print('CUDA is available!')
    megengine.set_default_device('gpux')
    
args = param()
if args.ngpus is None:
    args.ngpus = dist.helper.get_device_count_by_fork("gpu")
print("args.ngpus=",args.ngpus)
worker(args)


# In[18]:


import argparse
import bisect
import os
import time

import megengine
import megengine.autodiff as autodiff
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.optimizer as optim

logging = megengine.logger.get_logger()

class param():
    def __init__(self):
        self.data = ""
        self.arch = "resnet50"
        self.ngpus = 0
        self.save = "workspace/output"
        self.epochs = 10
        self.batch_size = 2
        self.lr = 0.0125
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.workers = 2
        self.print_freq = 20


def worker(args):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        os.makedirs(os.path.join(args.save, args.arch), exist_ok=True)
        megengine.logger.set_log_file(os.path.join(args.save, args.arch, "test_log.txt"))

    # build dataset
    valid_dataloader = build_dataset(args)

    # Load param to cpu
    #checkpoint = megengine.load('workspace/output/resnet50/checkpoint.pkl', map_location="cpu0")
    #device_save = megengine.get_default_device()
    #megengine.set_default_device("cpu0")
    checkpoint = megengine.load('workspace/output/resnet50/checkpoint.pkl')

    model = resnet50(num_classes=2)
    model.load_state_dict(checkpoint["state_dict"])

    def valid_step(image, label):
        logits = model(image)
        loss = F.nn.cross_entropy(logits, label)
        acc1, acc2 = F.topk_accuracy(logits, label, topk=(1, 2))
        # calculate mean values
        if world_size > 1:
            loss = F.distributed.all_reduce_sum(loss) / world_size
            acc1 = F.distributed.all_reduce_sum(acc1) / world_size
            acc2 = F.distributed.all_reduce_sum(acc2) / world_size
        return loss, acc1, acc2

    model.eval()
    _, valid_acc1, valid_acc2 = valid(
        valid_step, valid_dataloader, args
    )
    logging.info(
        "Test Acc@1 %.3f Acc@2 %.3f", valid_acc1, valid_acc2
    )
    #model.train()
    #megengine.save(model.state_dict(), 'workspace/resnet50_static.mge')

def valid(func, data_queue, args):
    objs = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top2 = AverageMeter("Acc@2")
    clck = AverageMeter("Time")

    t = time.time()
    for step, (image, label) in enumerate(data_queue):
        image = megengine.tensor(image, dtype="float32")
        label = megengine.tensor(label, dtype="int32")

        n = image.shape[0]

        loss, acc1, acc2 = func(image, label)

        objs.update(loss.item(), n)
        top1.update(100 * acc1.item(), n)
        top2.update(100 * acc2.item(), n)
        clck.update(time.time() - t, n)
        t = time.time()

        if step % args.print_freq == 0 and dist.get_rank() == 0:
            logging.info(
                "Test step %d, %s %s %s %s", step, objs, top1, top2, clck
            )

    return objs.avg, top1.avg, top2.avg


def build_dataset(args):
    valid_dataset = ImageNetLocalDataset(
        "workspace/test.list"
    )
    valid_sampler = data.SequentialSampler(
        valid_dataset, batch_size=args.batch_size, drop_last=False
    )
    valid_dataloader = data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        transform=T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[103.530, 116.280, 123.675],
                    std=[57.375, 57.120, 58.395],
                ),  # BGR
                T.ToMode("CHW"),
            ]
        ),
        num_workers=args.workers,
    )
    return valid_dataloader


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":.3f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

#检查GPU是否可用
# import megengine
train_on_gpu =  megengine.is_cuda_available()
if not train_on_gpu:
    print('CUDA is not available!')
    megengine.set_default_device('cpux')
else:
    print('CUDA is available!')
    megengine.set_default_device('gpux')

args = param()
if args.ngpus is None:
    args.ngpus = dist.helper.get_device_count_by_fork("gpu")
print("args.ngpus=",args.ngpus)
worker(args)    


# In[ ]:





# In[ ]:




