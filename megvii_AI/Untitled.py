#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import boto3
import cv2
import nori2 as nori
import numpy as np
from imgaug import augmenters as iaa
from meghair.utils.imgproc import imdecode


# In[ ]:


import boto3
import cv2
import nori2 as nori
import numpy as np
from imgaug import augmenters as iaa
from meghair.utils.imgproc import imdecode


# In[ ]:


import boto3


# In[ ]:


import boto3


# In[ ]:


import cv2


# In[ ]:


import nori2 as nori


# In[ ]:


import nori2 as nori


# In[ ]:


import nori2 as nori


# In[ ]:


import numpy as np


# In[ ]:


from imgaug import augmenters as iaa


# In[ ]:


from imgaug import augmenters as iaa


# In[ ]:


import boto3
import cv2
import nori2 as nori
import numpy as np
from imgaug import augmenters as iaa
from meghair.utils.imgproc import imdecode


# In[ ]:


import boto3
import cv2
import nori2 as nori
import numpy as np
from imgaug import augmenters as iaa
from meghair.utils.imgproc import imdecode


# In[ ]:


endpoint_url="http://oss.i.brainpp.cn"
bucket = "ai-cultivate"
key = "1percent_ImageNet.txt"
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")


# In[ ]:


# 获取图像
def read_images(bucket, key):
    txt_file = s3_client.get_object(Bucket=bucket, Key=key)
    data = txt_file['Body'].read().decode("utf8").split('\n')
    nori_ids = list(map(lambda x: x.split('\t')[0], data))[20:30]
    fetcher = nori.Fetcher()
    images = list(map(lambda x: imdecode(fetcher.get(x)), nori_ids))
    return images


# In[ ]:


# 获取图像统计数据
def get_statistics(images):
    height_list = []
    width_list = []
    for image in images:
        height_list.append(image.shape[0])
        width_list.append(image.shape[1])

    avg_height = np.mean(height_list)
    max_height = max(height_list)
    min_height = min(height_list)

    avg_width = np.mean(width_list)
    max_width = max(width_list)
    min_width = min(width_list)

    statistics = {
      'avg_height': avg_height,
      'max_height': max_height, 
      'min_height': min_height,
      'avg_width': avg_width, 
      'max_width': max_width, 
      'min_width': min_width}

    print("统计数据: {}".format(statistics))
    return


# In[ ]:


# 增强图像
def augment_image(images):
    H, W = 128, 128
    NUM = 5
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    ori_seq = iaa.Sequential([iaa.Resize({"height": H, "width": W})])

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 50%的图像镜像翻转
        iaa.Flipud(0.2),  # 20%的图像左右翻转
        iaa.Crop(percent=(0, 0.1)),  # 四边以0 - 0.1之间的比例像素剪裁
        sometimes(iaa.Affine(  # 对一部分图像做仿射变换
          scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # 图像缩放80%-120%
          rotate=(-45, 45)  # ±45度旋转
        )),
        iaa.GaussianBlur(sigma=(0, 2.0)),  # 高斯模糊
        iaa.Resize({"height": H, "width": W})
    ], random_order=True)

    res = np.zeros(shape=((H + 10) * len(images), (W + 10) * (NUM + 1), 3), dtype=np.uint8)

    for i, image in enumerate(images):
        image_array = np.array([image] * NUM, dtype=np.uint8)
        write_image = np.zeros(shape=(H, (W + 10) * (NUM + 1), 3), dtype=np.uint8)
        ori_image = ori_seq.augment_image(image)
        write_image[:, 0: W, :] = ori_image
        images_aug = seq.augment_images(images=image_array)
        for j, item in enumerate(images_aug):
            write_image[:, (j + 1) * (W + 10): (j + 1) * (W + 10) + W, :] = item

        res[i * (H + 10): i * (H + 10) + H, :, :] = write_image

    cv2.imwrite("results.jpg", res)


# In[ ]:


if __name__ == "__main__":
    images = read_images(bucket, key)
    get_statistics(images)
    augment_image(images)


# In[ ]:


images = read_images(bucket, key)


# In[ ]:


print(key)


# In[ ]:


print(read_images)


# In[ ]:


print(read_images)


# In[ ]:


import boto3
import cv2
import nori2 as nori
import numpy as np
from imgaug import augmenters as iaa
from meghair.utils.imgproc import imdecode


# In[ ]:


endpoint_url="http://oss.i.brainpp.cn"
bucket = "ai-cultivate"
key = "1percent_ImageNet.txt"
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")


# In[ ]:


# 获取图像
def read_images(bucket, key):
    txt_file = s3_client.get_object(Bucket=bucket, Key=key)
    data = txt_file['Body'].read().decode("utf8").split('\n')
    nori_ids = list(map(lambda x: x.split('\t')[0], data))[20:30]
    fetcher = nori.Fetcher()
    images = list(map(lambda x: imdecode(fetcher.get(x)), nori_ids))
    return images


# In[ ]:


# 获取图像统计数据
def get_statistics(images):
    height_list = []
    width_list = []
    for image in images:
        height_list.append(image.shape[0])
        width_list.append(image.shape[1])

    avg_height = np.mean(height_list)
    max_height = max(height_list)
    min_height = min(height_list)

    avg_width = np.mean(width_list)
    max_width = max(width_list)
    min_width = min(width_list)

    statistics = {
      'avg_height': avg_height,
      'max_height': max_height, 
      'min_height': min_height,
      'avg_width': avg_width, 
      'max_width': max_width, 
      'min_width': min_width}

    print("统计数据: {}".format(statistics))
    return


# In[ ]:


# 增强图像
def augment_image(images):
    H, W = 128, 128
    NUM = 5
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    ori_seq = iaa.Sequential([iaa.Resize({"height": H, "width": W})])

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 50%的图像镜像翻转
        iaa.Flipud(0.2),  # 20%的图像左右翻转
        iaa.Crop(percent=(0, 0.1)),  # 四边以0 - 0.1之间的比例像素剪裁
        sometimes(iaa.Affine(  # 对一部分图像做仿射变换
          scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # 图像缩放80%-120%
          rotate=(-45, 45)  # ±45度旋转
        )),
        iaa.GaussianBlur(sigma=(0, 2.0)),  # 高斯模糊
        iaa.Resize({"height": H, "width": W})
    ], random_order=True)

    res = np.zeros(shape=((H + 10) * len(images), (W + 10) * (NUM + 1), 3), dtype=np.uint8)

    for i, image in enumerate(images):
        image_array = np.array([image] * NUM, dtype=np.uint8)
        write_image = np.zeros(shape=(H, (W + 10) * (NUM + 1), 3), dtype=np.uint8)
        ori_image = ori_seq.augment_image(image)
        write_image[:, 0: W, :] = ori_image
        images_aug = seq.augment_images(images=image_array)
        for j, item in enumerate(images_aug):
            write_image[:, (j + 1) * (W + 10): (j + 1) * (W + 10) + W, :] = item

        res[i * (H + 10): i * (H + 10) + H, :, :] = write_image

    cv2.imwrite("results.jpg", res)


# In[ ]:


if __name__ == "__main__":
    images = read_images(bucket, key)
    get_statistics(images)
    augment_image(images)


# In[ ]:


images = read_images(bucket, key)


# In[ ]:


txt_file = s3_client.get_object(Bucket=bucket, Key=key)
data = txt_file['Body'].read().decode("utf8").split('\n')
nori_ids = list(map(lambda x: x.split('\t')[0], data))[20:30]
fetcher = nori.Fetcher()
images = list(map(lambda x: imdecode(fetcher.get(x)), nori_ids))
print(images)


# In[ ]:


endpoint_url="http://oss.i.brainpp.cn"
bucket = "ai-cultivate"
key = "1percent_ImageNet.txt"
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")


# In[ ]:


txt_file = s3_client.get_object(Bucket=bucket, Key=key)


# In[ ]:


import boto3
import boto3
import cv2
import nori2 as nori
import numpy as np
from imgaug import augmenters as iaa
from meghair.utils import io
from meghair.utils.imgproc import imdecode
from refile import smart_open


# In[ ]:


s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")
bucket = "ai-cultivate"
key = "1percent_ImageNet.txt"


# In[ ]:


# 读取
def read_img(bucket, key):
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    res = resp['Body'].read().decode("utf8")
    data = res.split('\n')
    nori_ids = list(map(lambda x: x.split('\t')[0], data))[-10::2]  
    fetcher = nori.Fetcher()
    img_list = list(map(lambda x: imdecode(fetcher.get(x)), nori_ids))
    print(type(img_list[0]))
    return img_list


# In[ ]:


images = read_img(bucket, key)


# In[ ]:


s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")
bucket = "ai-cultivate"
key = "chyh"


# In[ ]:


resp = s3_client.get_object(Bucket=bucket, Key=key)
res = resp['Body'].read().decode("utf8")
data = res.split('\n')
nori_ids = list(map(lambda x: x.split('\t')[0], data))[-10::2]  
fetcher = nori.Fetcher()
img_list = list(map(lambda x: imdecode(fetcher.get(x)), nori_ids))
print(type(img_list[0]))


# In[ ]:


resp = s3_client.get_object(Bucket="ai-cultivate", Key="1percent_ImageNet.txt")


# In[ ]:


# Client初始化
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")

#load 数据
resp = s3_client.get_object(Bucket="ai-cultivate", Key="1percent_ImageNet.txt") 
data_str = resp['Body'].read().decode("utf-8")
data = data_str.split('\n')


# In[ ]:


import boto3
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import nori2 as nori
import numpy as np
from meghair.utils import io
from meghair.utils.imgproc import imdecode
import matplotlib.pyplot as plt


# In[ ]:


# Client初始化
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")

#load 数据
resp = s3_client.get_object(Bucket="ai-cultivate", Key="1percent_ImageNet.txt") 
data_str = resp['Body'].read().decode("utf-8")
data = data_str.split('\n')


# In[ ]:


# Client初始化
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")

#load 数据
resp = s3_client.get_object(Bucket="ai-cultivate", Key="1percent_ImageNet.txt") 
data_str = resp['Body'].read().decode("utf-8")
data = data_str.split('\n')


# In[ ]:


import boto3
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import nori2 as nori
import numpy as np
from meghair.utils import io
from meghair.utils.imgproc import imdecode
import matplotlib.pyplot as plt


# In[ ]:


# Client初始化
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")

#load 数据
resp = s3_client.get_object(Bucket="ai-cultivate", Key="1percent_ImageNet.txt") 
data_str = resp['Body'].read().decode("utf-8")
data = data_str.split('\n')


# In[ ]:


# Client初始化
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")

#load 数据
resp = s3_client.get_object(Bucket="ai-cultivate", Key="1percent_ImageNet.txt") 
data_str = resp['Body'].read().decode("utf-8")
data = data_str.split('\n')


# In[ ]:


import boto3
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import nori2 as nori
import numpy as np
from meghair.utils import io
from meghair.utils.imgproc import imdecode
import matplotlib.pyplot as plt


# In[ ]:


# Client初始化
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")

#load 数据
resp = s3_client.get_object(Bucket="ai-cultivate", Key="1percent_ImageNet.txt") 
data_str = resp['Body'].read().decode("utf-8")
data = data_str.split('\n')


# In[ ]:


import boto3
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import nori2 as nori
import numpy as np
from meghair.utils import io
from meghair.utils.imgproc import imdecode
import matplotlib.pyplot as plt


# In[ ]:


# Client初始化
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")

#load 数据
resp = s3_client.get_object(Bucket="ai-cultivate", Key="1percent_ImageNet.txt") 
data_str = resp['Body'].read().decode("utf-8")
data = data_str.split('\n')


# In[ ]:


# Client初始化
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")

#load 数据
resp = s3_client.get_object(Bucket="ai-cultivate", Key="1percent_ImageNet.txt") 
data_str = resp['Body'].read().decode("utf-8")
data = data_str.split('\n')


# In[1]:


import boto3
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import nori2 as nori
import numpy as np
from meghair.utils import io
from meghair.utils.imgproc import imdecode
import matplotlib.pyplot as plt


# In[2]:


# Client初始化
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")

#load 数据
resp = s3_client.get_object(Bucket="ai-cultivate", Key="1percent_ImageNet.txt") 
data_str = resp['Body'].read().decode("utf-8")
data = data_str.split('\n')


# In[3]:


print(rsep)


# In[4]:


print(resp)


# In[ ]:


print(data_str)


# In[6]:


data_str=resp['Body'].read()


# In[7]:


print(data_str)


# In[8]:


data_str=resp['Body'].read().decode("utf-8")
print(data_str)


# In[9]:


print(resp)


# In[10]:


data_str=resp['Body'].read()


# In[11]:


print(data_str)


# In[12]:


data_str = resp['Body'].read().decode("utf-8")


# In[13]:


print(data_str)


# In[15]:


resp = s3_client.get_object(Bucket="ai-cultivate", Key="1percent_ImageNet.txt") 
res = resp['Body'].read().decode("utf8")
data = res.split('\n')
nori_ids = list(map(lambda x: x.split('\t')[0], data))[-10::2]  
fetcher = nori.Fetcher()
img_list = list(map(lambda x: imdecode(fetcher.get(x)), nori_ids))
print(type(img_list[0]))


# In[ ]:


print(res)


# In[ ]:


print(data)


# In[18]:


print(resp)


# In[19]:


resp['Body'].read().decode("utf8")


# In[20]:


resp=['Body'].read().decode("utf8")


# In[25]:


img=cv2.imread(res[1])


# In[26]:


print(img)


# In[27]:


endpoint_url="http://oss.i.brainpp.cn"
bucket = "ai-cultivate"
key = "1percent_ImageNet.txt"
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")


# In[28]:


txt_file = s3_client.get_object(Bucket=bucket, Key=key)
data = txt_file['Body'].read().decode("utf8").split('\n')
nori_ids = list(map(lambda x: x.split('\t')[0], data))[20:30]
fetcher = nori.Fetcher()
images = list(map(lambda x: imdecode(fetcher.get(x)), nori_ids))


# In[ ]:


print(images)


# In[30]:


print(images.shape)


# In[31]:


print(images[0].shape)


# In[34]:


cv2.imwrite('1.jgp',images[0])


# In[35]:


cv2.imshow(images[0])


# In[1]:


cv2.imshow('img',images[0])


# In[2]:


import boto3
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import nori2 as nori
import numpy as np
from meghair.utils import io
from meghair.utils.imgproc import imdecode
import matplotlib.pyplot as plt


# In[3]:


endpoint_url="http://oss.i.brainpp.cn"
bucket = "ai-cultivate"
key = "1percent_ImageNet.txt"
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")


# In[4]:


txt_file = s3_client.get_object(Bucket=bucket, Key=key)
data = txt_file['Body'].read().decode("utf8").split('\n')
nori_ids = list(map(lambda x: x.split('\t')[0], data))[20:30]
fetcher = nori.Fetcher()
images = list(map(lambda x: imdecode(fetcher.get(x)), nori_ids))


# In[ ]:


cv2.imshow('img',images[0])


# In[1]:


print(images[0])


# In[2]:


import boto3
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import nori2 as nori
import numpy as np
from meghair.utils import io
from meghair.utils.imgproc import imdecode
import matplotlib.pyplot as plt


# In[39]:


# 设置基本参数
endpoint_url="http://oss.i.brainpp.cn"
bucket = "ai-cultivate"
key = "1percent_ImageNet.txt"
s3_client = boto3.client('s3', endpoint_url="http://oss.i.brainpp.cn")


# In[20]:


# 导入数据
txt_file = s3_client.get_object(Bucket=bucket, Key=key)
data = txt_file['Body'].read().decode("utf8").split('\n')
nori_ids = list(map(lambda x: x.split('\t')[0], data))[20:30]
fetcher = nori.Fetcher()
images = list(map(lambda x: imdecode(fetcher.get(x)), nori_ids))


# In[36]:


#  计算图集的最大宽度、高度、最小宽度、高度、平均宽度、高度

height_list = []
width_list = []

# 遍历数据集，打印每张图片，以及图片的高度和宽度
for image in images:
    height_list.append(image.shape[0])
    width_list.append(image.shape[1])
    plt.imshow(image)
    print("高度",image.shape[0],"宽度",image.shape[1])
    plt.show()

# 计算图集的最大高度、最小高度、平均高度

avg_height = np.mean(height_list)
max_height = max(height_list)
min_height = min(height_list)

# 计算图集的最大宽度、最小宽度、平均宽度

avg_width = np.mean(width_list)
max_width = max(width_list)
min_width = min(width_list)

print('统计数据：''平均高度',avg_height,'最大高度',max_height,'最小高度',min_height,'平均宽度',avg_width,'最大宽度',max_width,'最小宽度',min_width)


# In[40]:


# 针对图集做一系列的仿射变换
H, W = 128, 128
NUM = 5
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

ori_seq = iaa.Sequential([iaa.Resize({"height": H, "width": W})])


seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50%的图像镜像翻转
    iaa.Flipud(0.2),  # 20%的图像左右翻转
    iaa.Crop(percent=(0, 0.1)),  # 四边以0 - 0.1之间的比例像素剪裁
    sometimes(iaa.Affine(  # 对一部分图像做仿射变换
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # 图像缩放80%-120%
        rotate=(-45, 45)  # ±45度旋转
    )),
    iaa.GaussianBlur(sigma=(0, 2.0)),  # 高斯模糊
    iaa.Resize({"height": H, "width": W})
], random_order=True)

res = np.zeros(shape=((H + 10) * len(images), (W + 10) * (NUM + 1), 3), dtype=np.uint8)

for i, image in enumerate(images):
    image_array = np.array([image] * NUM, dtype=np.uint8)
    write_image = np.zeros(shape=(H, (W + 10) * (NUM + 1), 3), dtype=np.uint8)
    ori_image = ori_seq.augment_image(image)
    write_image[:, 0: W, :] = ori_image
    images_aug = seq.augment_images(images=image_array)
    for j, item in enumerate(images_aug):
        write_image[:, (j + 1) * (W + 10): (j + 1) * (W + 10) + W, :] = item

    res[i * (H + 10): i * (H + 10) + H, :, :] = write_image

cv2.imwrite("results.jpg", res)


# In[ ]:




