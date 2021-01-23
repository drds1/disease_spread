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

class rmodel():
    def __init__(self):
        self.df_master = None

    def download(self,
                 url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
                       'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'):
        # Download the global timeseries data
        s=requests.get(url).content
        self.df_master = pd.read_csv(io.StringIO(s.decode('utf-8')))

    def prep_timeseries(self):
        '''
        isolate UK data and get rates timeseries
        :return:
        '''
        c = self.df_master
        uk = c[c['Country/Region'] == 'United Kingdom']
        self.timeseries = uk.iloc[:, 4:].sum(axis=0)
        self.timeseries.index = pd.to_datetime(self.timeseries.index)
        self.rates = self.timeseries.diff()


    def prep_features(self, forecast_length = 60):
        '''
        prep feature matrix for model fitting
        :return:
        '''
        # isolate period of max peak to fit timeseries model
        self.idxpeak = self.rates.argmax()
        self.y = self.rates.iloc[self.idxpeak:].values
        self.t = np.arange(len(self.y))
        self.t_fc = np.arange(len(self.t)+forecast_length)
        self.tfeatures = sklearn.preprocessing.PolynomialFeatures(degree=1,
                                                             interaction_only=False,
                                                             include_bias=True,
                                                             order='C').fit_transform(self.t_fc.reshape(-1,1))

    def prep_model(self):
        '''

        :return:
        '''
        self.dates = self.rates.index
        self.dates_fc = pd.date_range(start = self.dates[self.idxpeak], freq='1D', periods = len(self.t_fc))
        # fit exponential model N = N0 R^(t/th) (log linear in y)
        self.yln = np.log(self.y)

    def prep_weights(self,window = 3):
        '''

        :return:
        '''
        # compute error bars using running average smoothing
        self.ylnwindow = np.log(self.rates.iloc[self.idxpeak-window:].values)
        self.smooth = pd.Series(self.ylnwindow).rolling(window, win_type='triang').mean().bfill().values[window:]
        self.res = self.yln - self.smooth
        self.sd = np.ones(len(self.yln))*np.std(self.res)

    def fit(self):
        '''

        :return:
        '''
        # perform fit using inverse square residual weights (w in polyfit configured for 1/sd NOT 1/sd^2 for gaussian weights)
        parms, cov = np.polyfit(self.t, self.yln, 1, w=1./self.sd, cov=True)

        # multisample the covariance matrix
        nsamples = 1000
        parms_multisample = np.random.multivariate_normal(parms[-1::-1],cov.T, nsamples).T
        yln_models = np.matmul(self.tfeatures,parms_multisample)
        y_models = np.exp(yln_models)
        self.y_proj = np.percentile(y_models,[25,50,75],axis=1).T

    def plot(self):
        '''

        :return:
        '''
        # plot the data and overlay the models with uncertainty snakes
        ylo, ymed, yhi = self.y_proj[:,0], self.y_proj[:,1], self.y_proj[:,2]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.bar(self.dates, self.rates)
        ax1.plot(self.dates_fc, ylo)
        ax1.plot(self.dates_fc, ymed,color='b',label = 'forecast')
        ax1.fill_between(self.dates_fc, ylo, yhi,color='b',alpha=0.2)
        plt.show()





if __name__ == '__main__':

    x = rmodel()
    x.download()
    x.prep_timeseries()
    x.prep_features()
    x.prep_model()
    x.prep_weights()
    x.fit()
    x.plot()
