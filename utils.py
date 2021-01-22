import pandas as pd
import io
import requests
import calendar as cal
import pandas as pd
import numpy as np
import dateutil
import time
import copy
import requests
from datetime import datetime, timedelta, date
import sklearn

# some info on r
#https://en.wikipedia.org/wiki/Basic_reproduction_number



#monthsstr = list(cal.month_abbr)
#date_today = pd.Timestamp.today().date()
#filename = 'data_'+str(date_today.year)+'-'+monthsstr[date_today.month]+'-'+str(date_today.day)+'.csv'

# Download the global timeseries data
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))

# isolate UK data
uk = c[c['Country/Region'] == 'United Kingdom']
timeseries = uk.iloc[:,4:].sum(axis=0)
timeseries.index = pd.to_datetime(timeseries.index)
rates = timeseries.diff()

# isolate period of max peak to fit timeseries model
idxpeak = rates.argmax()
y = rates.iloc[idxpeak:].values
t = np.arange(len(y))
t_fc = np.arange(len(t)+60)
tfeatures = sklearn.preprocessing.PolynomialFeatures(degree=1,
                                                     interaction_only=False,
                                                     include_bias=True,
                                                     order='C').fit_transform(t_fc.reshape(-1,1))

dates = rates.index[idxpeak:]
dates_fc = pd.date_range(start = dates[0], freq='1D', nperiods = len(t_fc))
# fit exponential model N = N0 R^(t/th) (log linear in y)
yln = np.log(y)
parms, cov = np.polyfit(t, yln, 1, w=np.ones(len(y)), cov=True)

# multisample the covariance matrix
nsamples = 1000
parms_multisample = np.random.multivariate_normal(parms[-1::-1],cov.T, nsamples).T
yln_models = np.matmul(tfeatures,parms_multisample)
y_models = np.exp(yln_models)
y_proj = np.percentile(y_models,[25,50,75],axis=1).T


# plot the data and overlay the models with uncertainty snakes
ylo, ymed, yhi = y_proj[:,0], y_proj[:,1], yproj[:,2]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(dates_fc, ylo)
ax1.plot(dates_fc, ymed,color='b',label = 'forecast')
ax1.fill_between(dates_fc, ylo, yhi,color='b',alpha=0.2)








