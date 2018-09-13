# -*- coding:utf-8 -*-
# By tostq <tostq216@163.com>
# Reference to hmmlearn.examples.plot_hmm_stock_analysis.py
# 博客: blog.csdn.net/tostq
import datetime

import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

from hmm import GaussianHMM
from sklearn.preprocessing import scale


###############################################################################
# 导入Yahoo金融数据
quotes = pd.read_csv('data/yahoofinance-INTC-19950101-20040412.csv')

dates = quotes.index.values
close_v = quotes[["Close"]].values.flatten()
volume = quotes[["Volume"]].values.flatten()
# diff：out[n] = a[n+1] - a[n] 得到价格变化
diff = np.diff(close_v)
dates = dates[1:]
close_v = close_v[1:]
volume = volume[1:]

# scale归一化处理：均值为0和方差为1
# 将价格和交易数组成输入数据
X = np.column_stack([scale(diff), scale(volume)])

# 训练高斯HMM模型，这里假设隐藏状态4个
model = GaussianHMM(4,2,20)
model.train(X)

# 预测隐状态
hidden_states = model.decode(X)

# 打印参数
print("Transition matrix: ", model.transmat_prob)
print("Means and vars of each hidden state")
for i in range(model.n_state):
    print("{0}th hidden state".format(i))
    print("mean = ", model.emit_means[i])
    print("var = ", model.emit_covars[i])
    print()

# 画图描述
fig, axs = plt.subplots(model.n_state, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_state))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

plt.show()