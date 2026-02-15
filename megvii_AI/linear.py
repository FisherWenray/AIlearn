import numpy as np
import matplotlib.pyplot as plt

# 生成随机数
def generate_randow(n,noise):

    w=np.random.randint(5,10)
    b=np.random.randint(-10,10)
    data=np.zeros((n,))
    label=np.zeros((n,))

    for i in range(n):
        data[i]=np.random.uniform(-10,10)
        label[i]=w*data[i]+b+np.random.uniform(-noise,noise)
        plt.scatter(data[i], label[i],marker=".")

    plt.plot()
    plt.show()
    return data,label

gen_data,gen_label=generate_randow(100, 5)

#设置超参数
epochs =30
lr=0.01
#获取数据
data= gen_data
label =gen_label
n= len(data)
#参数初始化
w=0
b=0
#定义模型
def linear_model(x):
 return w*x+b
#模型训练
for epoch in range(epochs):
#初始化loss
    Loss =0
#梯度初始化
    sum_grad_w=0
    sum_grad_b=0
    for i in range(n):
#前向传播,主要计算预测值pred以及损失值loss
        pred =linear_model(data[i])
#通常pred为将data代入“模型”得到的输出,即
        Loss += (pred- label[i])**2
#反向传播,根据均方误差损失计算参数的梯度
        sum_grad_w +=2*(pred- label[i])*data[i]
        sum_grad_b +=2*(pred- label[i])
#计算平均损失
    Loss =Loss/n
#计算平均梯度
    grad_w= sum_grad_w/n
    grad_b= sum_grad_b/n

#更新w和b的参数
    w=w-lr*grad_w
    b=b-lr*grad_b
#查看参数和损失
    print("epoch:",epoch,"w:" ,w,"b:" ,b,"loss",Loss)
#绘图查看拟合情况
    x=np.array( [-10, 10])
    y=w*x+b
    plt.scatter(data, label,marker=".")
    plt.plot(x, y,"-b")
    plt.show()



