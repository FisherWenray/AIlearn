#!/usr/bin/env python
# coding: utf-8

# ## AI培训课程系列课后作业

# ### 所有作业分为七周理论课的课后作业和一个实践项目大作业

# In[1]:


from megengine.data.dataset import MNIST


# In[2]:


train_dataset=MNIST(root="./dataset/MNIST",train=True,download=True)


# In[3]:


test_dataset=MNIST(root="./dataset/MNIST",train=False,download=False)


# In[ ]:





# In[4]:


import megengine as mge
from megengine.optimizer import SGD
import megengine.functional as F
from megengine.data import DataLoader
from megengine.data.transform import ToMode, Pad, Normalize, Compose
from megengine.data import RandomSampler
from megengine.data.dataset import MNIST
from megengine.jit import trace
import megengine.module as M
from megengine.autodiff import GradManager
import numpy as np
import time


# In[5]:


class Net(M.Module):
    def __init__(self):
        super().__init__()
        self.conv0 =M.Conv2d(1, 20, kernel_size=5, bias=False)
        self.bn0=M.BatchNorm2d(20)
        self.relu0= M.ReLU()
        self.pool0 =M.MaxPool2d(2)
        self.conv1 =M.Conv2d(20, 20, kernel_size=5, bias=False)
        self.bn1=M.BatchNorm2d(20)
        self.relul=M.ReLU()
        self.pool1 =M.MaxPool2d(2)
        self.fc0= M.Linear(500, 64, bias=True)
        self.relu2 = M.ReLU()
        self.fc1=M.Linear(64,10,bias=True)
    
    def forward(self, x):
        x =self.conv0(x)
        x=self.bn0(x)
        x =self.relu0(x)
        x =self.pool0(x)
        x =self.conv1(x)
        x =self.bn1(x)
        x =self.relul(x)
        x =self.pool1(x)
        x=F.flatten(x, 1)
        x= self.fc0(x)
        x =self.relu2(x)
        x= self.fc1(x)
        
        return x
from megengine.jit import trace

@trace(symbolic=True)
def train_func(data, label,gm,net):
    print(1)
    net.train()
    print(2)
    with gm:
        pred=net(data)
        loss =F.loss.cross_entropy(pred, label)
        gm.backward(loss)

    return pred, loss


    


# In[ ]:








# In[6]:


dataloader= DataLoader(
    train_dataset,
    transform=Compose([
        Normalize(mean=0.1307*255,std=0.3081*255),
        Pad(2),
        ToMode('CHW'),
    ]),
    sampler=RandomSampler(dataset=train_dataset, batch_size=64),
    )


# In[7]:



net=Net()



# In[8]:



optimizer=SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
gm= GradManager().attach(net.parameters())
total_epochs=10


# In[9]:


for epoch in range(total_epochs):
    total_loss=0
    for step,(batch_data,batch_label) in enumerate(dataloader):
        batch_label= batch_label.astype(np.int32)
        optimizer.clear_grad()#将参数的梯度置零
#       pred,loss=train_func(mge.tensor(batch_data),mge.tensor(batch_label),gm=gm,net=net)
        net.train()
        with gm:
            pred = net(batch_data)
            loss = F.loss.cross_entropy(pred, batch_label)
            gm.backward(loss)
        optimizer.step()#根据梯度更新参数值
        total_loss += loss.numpy().item()
    print("epoch:",epoch,"loss:",total_loss/len(dataloader))


# In[10]:


total_epochs


# In[11]:


mge.save(net.state_dict(),'mnist_net.mge')


# In[12]:


net=Net()
state_dict=mge.load("mnist_net.mge")
net.load_state_dict(state_dict)


# In[18]:


from megengine.data.sampler import SequentialSampler

test_sampler=SequentialSampler(test_dataset, batch_size=500)
dataloader_test= DataLoader(
    test_dataset,
    sampler=test_sampler,
    transform=Compose([
        Normalize(mean=0.1307*255,std=0.3081*255),
        Pad(2),
        ToMode('CHW'),
    ]),
)
correct=0
total =0
for idx,(batch_data, batch_label) in enumerate(dataloader_test):
    batch_label=batch_label.astype(np.int32)
    net.eval()
    pred=net(mge.tensor(batch_data))
    loss=F.loss.cross_entropy(pred,mge.tensor(batch_label))
    predicted=pred.numpy().argmax(axis=1)
    correct +=(predicted ==batch_label).sum().item()
    total += batch_label.shape[0]
print("correct: {},total:{}, accuracy: {}".format(correct, total,float(correct)/total))


# In[ ]:




