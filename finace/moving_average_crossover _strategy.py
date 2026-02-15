import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 下载苹果股票数据
data = yf.download("AAPL", start="2023-01-01", end="2024-12-31")

# 计算移动平均线
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# 创建买卖信号
buy_signal = (data['MA10'] > data['MA50']) & (data['MA10'].shift(1) <= data['MA50'].shift(1))
sell_signal = (data['MA10'] < data['MA50']) & (data['MA10'].shift(1) >= data['MA50'].shift(1))

# 提取信号点
buy_dates = data[buy_signal].index
sell_dates = data[sell_signal].index

# 作图
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='AAPL Close', color='black', alpha=0.5)
plt.plot(data.index, data['MA10'], label='MA10 (短期)', color='orange')
plt.plot(data.index, data['MA50'], label='MA50 (长期)', color='blue')

# 添加买卖点
plt.scatter(buy_dates, data.loc[buy_dates, 'Close'], marker='^', color='green', label='Buy (金叉)', s=100)
plt.scatter(sell_dates, data.loc[sell_dates, 'Close'], marker='v', color='red', label='Sell (死叉)', s=100)

plt.title("Apple (AAPL) 移动平均线策略信号图")
plt.xlabel("日期")
plt.ylabel("股价")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
