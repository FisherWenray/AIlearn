import backtrader as bt
import datetime

# 定义一个简单的动量策略
class MomentumStrategy(bt.Strategy):
    params = (
        ('momentum_period', 20),  # 动量计算周期
        ('buy_threshold', 0.05),  # 超过5%收益率则买入
    )

    def __init__(self):
        # 计算过去 N 天的收益率：当前收盘价 / N 天前的收盘价 - 1
        self.momentum = self.data.close / self.data.close(-self.p.momentum_period) - 1

    def next(self):
        # 如果还未持仓，检查是否达到买入条件
        if not self.position:
            if self.momentum[0] > self.p.buy_threshold:
                self.buy()
                self.log(f"Buy ORDER executed, Price: {self.data.close[0]:.2f}")
        else:
            # 持仓时如果动量变为负，则平仓离场
            if self.momentum[0] < 0:
                self.close()
                self.log(f"Sell ORDER executed, Price: {self.data.close[0]:.2f}")
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} - {txt}")

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MomentumStrategy)

    # 使用 Yahoo Finance 数据作为示例
    data = bt.feeds.YahooFinanceData(
        dataname='AAPL',
        fromdate=datetime.datetime(2019, 1, 1),
        todate=datetime.datetime(2020, 12, 31)
    )
    cerebro.adddata(data)

    # 设置初始资金和下单规模
    cerebro.broker.setcash(100000)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # 绘制回测结果图
    cerebro.plot()