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
import corner
import matplotlib.backends.backend_pdf
#from matplotlib import rc
#rc('text', usetex=True)

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
        Nt = len(self.y)
        self.t = np.arange(len(self.y)) - Nt/2
        self.t_fc = np.append(self.t,np.arange(self.t[-1]+1, self.t[-1]+forecast_length,1))
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

    def fit(self, nsamples = 1000):
        '''

        :return:
        '''
        # perform fit using inverse square residual weights (w in polyfit configured for 1/sd NOT 1/sd^2 for gaussian weights)
        self.parms, self.cov = np.polyfit(self.t, self.yln, 1, w=1./self.sd, cov=True)
        self.parms = self.parms[-1::-1]
        self.cov = self.cov.T

        # multisample the covariance matrix
        self.parms_multisample = np.random.multivariate_normal(self.parms,self.cov, nsamples).T
        yln_models = np.matmul(self.tfeatures,self.parms_multisample)
        y_models = np.exp(yln_models)
        self.y_proj = np.percentile(y_models,[25,50,75],axis=1).T

    def plot_model(self,
                   file = None,
                   return_figure = True,
                   reference_date = pd.Timestamp(2020,9,1)):
        '''

        :return:
        '''
        # plot the data and overlay the models with uncertainty snakes
        ylo, ymed, yhi = self.y_proj[:,0], self.y_proj[:,1], self.y_proj[:,2]
        plt.close()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.bar(self.dates, self.rates,color='r')
        ax1.plot(self.dates_fc, ylo)
        Rlo, Rmed, Rhi = np.percentile(self.scatter_parms[:,1],[16,50,84])
        I0lo, I0med, I0hi = np.percentile(self.scatter_parms[:,0],[16,50,84])
        ann = 'Forecast \n '+r'$I=I_0\, R^{t / \tau}$'
        text = 'Infectious period '+r'$\tau = '+np.str(np.int(self.tau))+'$'+' days\n'+\
               r'$R_0 = e^{m\tau} = $('+\
               np.str(np.round(Rlo,2))+' < ' + np.str(np.round(Rmed,2))+' < '+ np.str(np.round(Rhi,2))+ ')\n' + \
                r'$I_0 = e^{c} = $(' + \
                np.str(np.int(I0lo)) + ' < ' + np.str(np.int(I0med)) + ' < ' + np.str(np.int(I0hi)) + ')'
        ax1.plot(self.dates_fc, ymed,color='b',label = ann)
        ax1.fill_between(self.dates_fc, ylo, yhi,color='b',alpha=0.2)
        yref = self.rates[self.dates > reference_date].values[0]
        ax1.axhline(yref, label='Reference level: '+str(reference_date.date()) ,
                    color='k',ls='--')
        ax1.annotate('Model Stats: ',
                     (0.05, 0.70),
                     xycoords = 'axes fraction',
                     va="top",
                     ha="left",
                     weight="bold")
        ax1.annotate(text,
                     (0.05, 0.65),
                     xycoords = 'axes fraction',
                     va="top",
                     ha="left")
        ax1.set_title('Cases by date reported: '+str(pd.Timestamp.today().date()) )
        plt.xticks(rotation=45)
        ax1.legend()
        plt.tight_layout()
        if return_figure is True:
            return fig
        elif file is not None:
            plt.savefig(file)
            plt.close()

    def get_output_parms(self, tau = 14):
        '''
        convert the log exponential fit to I0 and R parameters
        :return:
        '''
        parms, cov, parms_multisample = x.parms, x.cov, x.parms_multisample
        # set infection lifetime
        # and convert gradient parameter to R-value
        self.tau = tau
        gradient = parms_multisample[1, :]
        output_R = np.exp(gradient * tau)
        output_I0 = np.exp(parms_multisample[0, :])
        self.scatter_parms = np.array((output_I0, output_R)).T



    def plot_covariance(self, file = 'covariance_plot.pdf', return_figure = True):
        '''
        visualise paramter covariance matrix
        :return:
        '''
        fig = plt.figure()
        figure = corner.corner(self.scatter_parms, labels=[r"$I_0$", r"$R$"],
                               quantiles=[0.16, 0.5, 0.84],
                               fig=fig,
                               show_titles=True, title_kwargs={"fontsize": 12})
        figure.get_axes()[2].set_ylim([-0.5, 3.5])
        figure.get_axes()[3].set_xlim([-0.5, 3.5])
        if return_figure is True:
            return fig
        elif file is not None:
            plt.savefig(file)
            plt.close()





if __name__ == '__main__':

    x = rmodel()
    x.download()
    x.prep_timeseries()
    x.prep_features(forecast_length=120)
    x.prep_model()
    x.prep_weights()
    x.fit()
    x.get_output_parms()
    fig_plot = x.plot_model(file = 'rvalue_forecast.pdf',
                            return_figure = True,
                            reference_date = pd.Timestamp(2020,9,1))

    fig_covariance = x.plot_covariance(file='covariance_plot.pdf', return_figure=True)
    pdf = matplotlib.backends.backend_pdf.PdfPages("rmodel_outputs.pdf")
    pdf.savefig(fig_plot)
    pdf.savefig(fig_covariance)
    pdf.close()






