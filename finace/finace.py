import yfinance as yf
import backtrader as bt
import pandas as pd
from datetime import datetime

# 定义均线交叉策略
class MACrossStrategy(bt.Strategy):
    params = (
        ('fast_period', 15),     # 快速均线周期
        ('slow_period', 50),     # 慢速均线周期
        ('risk_ratio', 0.02),    # 单次风险占比
    )

    def __init__(self):
        # 计算移动平均线
        self.fast_ma = bt.indicators.SMA(
            self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(
            self.data.close, period=self.params.slow_period)
        
        # 计算均线交叉信号
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # 记录交易
        self.order = None
        self.price = None
        self.comm = None

    def next(self):
        # 如果有待执行订单，返回
        if self.order:
            return

        # 没有持仓
        if not self.position:
            # 当快线上穿慢线，买入
            if self.crossover > 0:
                # 计算购买数量
                risk_amount = self.broker.getvalue() * self.params.risk_ratio
                size = risk_amount / self.data.close[0]
                # 下单买入
                self.order = self.buy(size=int(size))
                self.log(f'买入订单: {self.data.close[0]:.2f}')
        
        # 持有仓位
        else:
            # 当快线下穿慢线，卖出
            if self.crossover < 0:
                self.order = self.close()
                self.log(f'卖出订单: {self.data.close[0]:.2f}')

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

def run_backtest(stock_code='AAPL'):
    # 初始化回测引擎
    cerebro = bt.Cerebro()
    
    # 下载股票数据
    data = yf.download(
        stock_code,
        start='2023-01-01',
        end=datetime.now().strftime('%Y-%m-%d'),
        progress=False
    )
    
    # 添加数据到回测引擎
    feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(feed)
    
    # 添加策略
    cerebro.addstrategy(MACrossStrategy)
    
    # 设置初始资金和手续费
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # 打印初始资金
    print(f'初始资金: {cerebro.broker.getvalue():.2f}')
    
    # 运行回测
    results = cerebro.run()
    strat = results[0]
    
    # 打印回测结果
    print(f'最终资金: {cerebro.broker.getvalue():.2f}')
    print(f'总收益率: {strat.analyzers.returns.get_analysis()["rtot"]*100:.2f}%')
    print(f'夏普比率: {strat.analyzers.sharpe.get_analysis()["sharperatio"]:.2f}')
    print(f'最大回撤: {strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]:.2f}%')
    
    # 绘制回测结果
    cerebro.plot(style='candle')

if __name__ == '__main__':
    # 运行回测
    run_backtest('AAPL')  # 可以更换其他股票代码