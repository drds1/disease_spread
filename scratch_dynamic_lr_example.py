#from pandas_datareader import DataReader
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#import pyflux as pf

from pydlm import dlm, trend, seasonality, dynamic, autoReg, longSeason
data = np.array([0] * 100 + [3] * 100)
myDLM = dlm(data)
myDLM = myDLM + trend(degree=1, discount=0.95, name='trend1')
myDLM.fit()

coef = np.array(myDLM.getLatentState())
results = np.array(myDLM.result.predictedObs)[:,0,0]
results_var = np.array(myDLM.result.predictedObsVar )[:,0,0]


import matplotlib.pylab as plt
#fig = plt.figure()
#ax1 = fig.add_subplot(311)
#ax1.plot(coef[:,0])
#ax2 = fig.add_subplot(312)
#ax2.plot(coef[:,1])
#ax3 = fig.add_subplot(313)
#ax3.plot(results)
#plt.show()



'''


import pydlm
#data = np.random.random((1, 1000))
#myDlm = pydlm.dlm(data) + pydlm.trend(degree = 2, discount = 0.98)
#myDlm.fitForwardFilter()

data = np.random.random((1,100))
mydlm = pydlm.dlm(data) + pydlm.trend(degree=1, discount = 0.98, name='a') \
        + pydlm.dynamic(features=[[i] for i in range(100)], discount = 1, name='b')
mydlm.fit()
coef_a = mydlm.getLatentState('a')
coef_b = mydlm.getLatentState('b')


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
'''


