from pandas_datareader import DataReader
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pyflux as pf

a = DataReader('AMZN',  'yahoo', datetime(2012,1,1), datetime(2016,6,1))
a_returns = pd.DataFrame(np.diff(np.log(a['Adj Close'].values)))
a_returns.index = a.index.values[1:a.index.values.shape[0]]
a_returns.columns = ["Amazon Returns"]

spy = DataReader('SPY',  'yahoo', datetime(2012,1,1), datetime(2016,6,1))
spy_returns = pd.DataFrame(np.diff(np.log(spy['Adj Close'].values)))
spy_returns.index = spy.index.values[1:spy.index.values.shape[0]]
spy_returns.columns = ['S&P500 Returns']

one_mon = DataReader('DGS1MO', 'fred',datetime(2012,1,1), datetime(2016,6,1))
one_day = np.log(1+one_mon)/365

returns = pd.concat([one_day,a_returns,spy_returns],axis=1).dropna()
excess_m = returns["Amazon Returns"].values - returns['DGS1MO'].values
excess_spy = returns["S&P500 Returns"].values - returns['DGS1MO'].values
final_returns = pd.DataFrame(np.transpose([excess_m,excess_spy, returns['DGS1MO'].values]))
final_returns.columns=["Amazon","SP500","Risk-free rate"]
final_returns.index = returns.index

plt.figure(figsize=(15,5))
plt.title("Excess Returns")
x = plt.plot(final_returns);
plt.legend(iter(x), final_returns.columns)
plt.show()


model = pf.DynReg('Amazon ~ SP500', data=final_returns)
x = model.fit()
x.summary()

model.plot_fit()


