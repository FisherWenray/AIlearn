import numpy as np
from scipy.optimize import minimize

# 定义一个目标函数
def objective(x):
    return x**2 + 2*x + 1

# 最小化目标函数
result = minimize(objective, 0)
print(result.x)  # 打印最优解
