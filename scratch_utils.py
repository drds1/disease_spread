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
import sklearn.preprocessing
import matplotlib.pylab as plt

# some info on r
#https://en.wikipedia.org/wiki/Basic_reproduction_number



#monthsstr = list(cal.month_abbr)
#date_today = pd.Timestamp.today().date()
#filename = 'data_'+str(date_today.year)+'-'+monthsstr[date_today.month]+'-'+str(date_today.day)+'.csv'

# Download the global timeseries data
#url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url = 'https://api.coronavirus.data.gov.uk/v2/data?areaType=overview&metric=newCasesBySpecimenDate&format=csv'
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
c['date'] = pd.to_datetime(c['date'])
timeseries = c[['date','newCasesBySpecimenDate']].sort_values(by='date').set_index('date')
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

dates = rates.index
dates_fc = pd.date_range(start = dates[idxpeak], freq='1D', periods = len(t_fc))
# fit exponential model N = N0 R^(t/th) (log linear in y)
yln = np.log(y)

# compute error bars using running average smoothing
window = 3
ylnwindow = np.log(rates.iloc[idxpeak-window:].values)
smooth = pd.Series(ylnwindow).rolling(window, win_type='triang').mean().bfill().values[window:]
res = yln - smooth
sd = np.ones(len(yln))*np.std(res)

# perform fit using inverse square residual weights (w in polyfit configured for 1/sd NOT 1/sd^2 for gaussian weights)
parms, cov = np.polyfit(t, yln, 1, w=1./sd, cov=True)

# multisample the covariance matrix
nsamples = 1000
parms_multisample = np.random.multivariate_normal(parms[-1::-1],cov.T, nsamples).T
yln_models = np.matmul(tfeatures,parms_multisample)
y_models = np.exp(yln_models)
y_proj = np.percentile(y_models,[25,50,75],axis=1).T


# plot the data and overlay the models with uncertainty snakes
ylo, ymed, yhi = y_proj[:,0], y_proj[:,1], y_proj[:,2]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.bar(dates, rates)
ax1.plot(dates_fc, ylo)
ax1.plot(dates_fc, ymed,color='b',label = 'forecast')
ax1.fill_between(dates_fc, ylo, yhi,color='b',alpha=0.2)
plt.show()








