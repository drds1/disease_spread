import pandas as pd
import io
import numpy as np
import requests
import sklearn.preprocessing
import matplotlib.pylab as plt
import corner
import matplotlib.backends.backend_pdf
import os
import pickle
import matplotlib.gridspec as gridspec


np.random.seed(12345)

class rmodel():
    def __init__(self, url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
                   'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
                 plot_title='Cases by date reported: ' + str(pd.Timestamp.today().date()),
                 model_days = 14,
                 model_date = None,
                 forecast_length = 60):
        self.df_master = None
        self.url = url
        self.plot_title = plot_title
        self.model_days = model_days
        self.model_date = model_date
        self.forecast_length = forecast_length

    def download(self):
        '''

        :return:
        '''
        url = self.url
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


    def set_model_window(self):
        '''
        set the period to be modelled by the fit
        :return:
        '''
        if self.model_date is not None:
            self.idxpeak = np.where(self.rates.index >= self.model_date)[0][0]
        else:
            self.idxpeak = len(self.rates) - self.model_days


    def prep_features(self):
        '''
        prep feature matrix for model fitting
        :return:
        '''
        self.set_model_window()
        # isolate period of max peak to fit timeseries model
        self.y = self.rates.iloc[self.idxpeak:].values
        Nt = len(self.y)
        self.t = np.arange(len(self.y)) - Nt/2
        self.t_fc = np.append(self.t,np.arange(self.t[-1]+1, self.t[-1]+self.forecast_length,1))
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
        self.dates_modeled = self.dates_fc[:len(self.y)]
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

    def fit(self, nsamples = 5000):
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

    def fit_week_model(self):
        '''
        include features for the day of the week
        :return:
        '''

        pass


    def plot_model(self,
                   file = None,
                   return_figure = True,
                   figin = None,
                   axin=None,
                   reference_level = 2000):
        '''

        :return:
        '''
        # plot the data and overlay the models with uncertainty snakes
        ylo, ymed, yhi = self.y_proj[:,0], self.y_proj[:,1], self.y_proj[:,2]
        if axin is None and figin is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
        else:
            fig = figin
            ax1 = axin
        ax1.bar(self.dates, self.rates,color='r')
        Rlo, Rmed, Rhi = np.percentile(self.scatter_parms[:,1],[16,50,84])
        I0lo, I0med, I0hi = np.percentile(self.scatter_parms[:,0],[16,50,84])
        ann = 'Forecast \n '+r'$I=I_0\, R^{t / \tau}$'
        text = 'Assumed infectious period '+r'$\tau = '+np.str(np.int(self.tau))+'$'+' days\n'+\
               r'$R = e^{m\tau} = $('+\
               np.str(np.round(Rlo,2))+' < ' + np.str(np.round(Rmed,2))+' < '+ np.str(np.round(Rhi,2))+ ')\n' + \
               'Model period: '+str(self.dates_modeled[0].date())+' --> '+str(self.dates_modeled[-1].date())
                #r'$I_0 = e^{c} = $(' + \
                #np.str(np.int(I0lo)) + ' < ' + np.str(np.int(I0med)) + ' < ' + np.str(np.int(I0hi)) + ')'
        ymin, ymax = ax1.get_ylim()

        if Rmed < 1.0:
            ax1.plot(self.dates_fc, ymed,color='b',label = ann)
            ax1.fill_between(self.dates_fc, ylo, yhi,color='b',alpha=0.2)
            idx_ref = np.where(ymed > reference_level)[0][-1]
            date_ref = str(self.dates_fc[idx_ref].date())
            yref = self.rates[self.rates > reference_level].values[0]
            label = 'Arbitrary "safe" level\n' + str(reference_level) + ' cases'
            if (idx_ref < len(ymed) - 2): #do not annotate safe level forecast if beyond edge of forecast
                label += ' reached by: '+date_ref
            ax1.axhline(yref, label=label, color='k',ls='--')

        else:
            doubling_time = self.tau * np.log(2)/np.log(Rmed)
            ax1.plot(self.dates_fc, ymed, color='r', label=ann)
            ax1.fill_between(self.dates_fc, ylo, yhi, color='r', alpha=0.2)
            ax1.axhline(reference_level,
                        label='Arbitrary "safe" level\n' + str(reference_level) + ' unreachable (R > 1)\nCurrent doubling time = '+np.str(np.round(doubling_time,2))+' days',
                        color='k', ls='--')

        ax1.set_ylim((ymin,ymax))
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
        ax1.set_title(self.plot_title)
        plt.xticks(rotation=45)
        ax1.legend()
        plt.tight_layout()
        if axin is not None and figin is not None:
            return fig, ax1
        elif return_figure is True:
            return fig
        elif file is not None:
            plt.savefig(file)


    def get_output_parms(self, tau = 14):
        '''
        convert the log exponential fit to I0 and R parameters
        :return:
        '''
        parms, cov, parms_multisample = self.parms, self.cov, self.parms_multisample
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
        figure = corner.corner(self.scatter_parms, labels=[r"$I_0$", r"$R$"],
                               quantiles=[0.16, 0.5, 0.84],
                               show_titles=True, title_kwargs={"fontsize": 12})
        figure.get_axes()[2].set_ylim([-0.5, 3.5])
        figure.get_axes()[3].set_xlim([-0.5, 3.5])
        if return_figure is True:
            return figure
        elif file is not None:
            plt.savefig(file)
            plt.close()


class rmodel_govuk(rmodel):
    def __init__(self,forecast_length = 60,
                 model_days = 14,
                 model_date = None,
                 discount_incomplete_days = 4,
                 url = 'https://api.coronavirus.data.gov.uk/v2/data?areaType=overview&metric=newCasesBySpecimenDate&format=csv'):
        super().__init__(url = url,
                         plot_title='Cases by specimen date: ' + str(pd.Timestamp.today().date()),
                         forecast_length = forecast_length,
                         model_days = model_days,
                         model_date = model_date)
        self.discount_incomplete_days = discount_incomplete_days

    def check_todays_update(self):
        '''
        check to see if new data is available today
        :return:
        '''
        xtoday = rmodel_govuk(url = 'https://api.coronavirus.data.gov.uk/v2/data?areaType=overview&metric=newCasesByPublishDate&format=csv')
        xtoday.download()
        df = xtoday.df_master
        return str(df['date'].max()) == str(pd.Timestamp.today().date())

    def prep_timeseries(self):
        c = self.df_master
        c = c[self.discount_incomplete_days:]
        c['date'] = pd.to_datetime(c['date'])
        self.rates = c[['date', 'newCasesBySpecimenDate']].sort_values(by='date').set_index('date').iloc[:,0].astype(float)

    def multi_run(self,min_date = pd.Timestamp(2020,3,15)):
        '''
        run a rolling day set of predictions to get R value throughout pandemic
        :return:
        '''
        df_master = self.df_master.copy()
        df_master['date'] = pd.to_datetime(df_master['date'])
        end_date = df_master['date'].iloc[0]
        dates_running = pd.date_range(start = min_date,
                                      end = end_date,
                                      freq = '1D')
        model_days = 20
        multi_date_r = []
        multi_r_r = []
        multi_r_sd = []
        multi_r_lo = []
        multi_r_hi = []
        for date in list(dates_running):
            x1 = rmodel_govuk(model_days=model_days, forecast_length=150)
            x1.df_master = df_master.loc[df_master['date'] <= date]
            x1.prep_timeseries()
            x1.prep_features()
            x1.prep_model()
            x1.prep_weights()
            x1.fit()
            x1.get_output_parms()
            x1r = np.percentile(x1.scatter_parms[:,1],[16,50,84])
            multi_date_r.append(date - pd.Timedelta(str(model_days / 2) + 'D'))
            multi_r_r.append(x1r[1])
            multi_r_sd.append(0.5*(x1r[2]-x1r[0]))
            multi_r_lo.append(x1r[0])
            multi_r_hi.append(x1r[2])
        self.multi_r_r = np.array(multi_r_r)
        self.multi_r_lo = np.array(multi_r_lo)
        self.multi_r_hi = np.array(multi_r_hi)
        self.multi_r_sd = np.array(multi_r_sd)
        self.multi_date_r = pd.Series(multi_date_r)


    def plot_r_estimate(self, fig, ax2):
        '''

        :param fig:
        :param r:
        :return:
        '''
        for i in range(len(self.multi_date_r)-1):
            x = self.multi_date_r[i:i+1]
            y = self.multi_r_r[i:i+1]
            ylo = self.multi_r_lo[i:i+1]
            yhi = self.multi_r_hi[i:i+1]
            if y < 1:
                color = 'b'
            else:
                color = 'r'

            ax2.fill_between(x, ylo, yhi, color=color,alpha=0.2,label=None)
            ax2.scatter(x, y, color=color, label=None,alpha=1)
        ax2.axhline(1.0, color='k', ls='--', label='r=1')
        ax2.axhline(0.0, color='k', ls='-', label=None)
        ax2.set_title('Rolling R Calculation')
        plt.xticks(rotation=45)
        return fig, ax2


    def plot_multi(self,reference_level=2000):
        '''
        overplot the rolling r estimate
        :return:
        '''
        fig = plt.figure()
        gs = gridspec.GridSpec(5, 2)
        gs.update(hspace=0.05)  # set the spacing between axes.

        ax1 = fig.add_subplot(gs[:-2, :])
        fig, ax1 = self.plot_model(file=None,
                   return_figure=True,
                   figin=fig,
                   axin=ax1,
                   reference_level=reference_level)
        ax1.set_xticklabels('')

        #make running R plot
        ax2 = fig.add_subplot(gs[-2:, :])
        fig, ax2 = self.plot_r_estimate(fig, ax2)
        ax2.set_xlim(ax1.get_xlim())

        plt.tight_layout()
        return fig



def run_govukmodel():
    '''

    :return:
    '''
    x = rmodel_govuk(model_days=21, forecast_length=150)
    x.download()
    x.prep_timeseries()

    x.prep_features()
    x.prep_model()
    x.prep_weights()
    x.multi_run(min_date=pd.Timestamp(2020, 6, 15))
    x.fit()
    x.get_output_parms()
    fig_plot = x.plot_model(file='rvalue_forecast.pdf',
                            return_figure=True,
                            reference_level=1000)
    fig_covariance = x.plot_covariance(file='covariance_plot.pdf', return_figure=True)

    # save model and figures
    dirname = './results/rvalue_model_' + str(pd.Timestamp.today().date()).replace('-', '_')
    if os.path.exists(dirname) is False:
        os.system('mkdir ' + dirname)
    pdf = matplotlib.backends.backend_pdf.PdfPages(dirname + "/rmodel_outputs.pdf")
    pdf.savefig(fig_plot)
    pdf.savefig(fig_covariance)
    pdf.close()
    f = open(dirname + "/model.pkl", "wb")
    pickle.dump({'model': x}, f)
    f.close()

    return x




def perform_1it():
    '''
    perform 1 iteration of the model and save days results
    :return:
    '''
    x = rmodel(model_days=21)
    x.download()
    x.prep_timeseries()
    x.prep_features(forecast_length=120)
    x.prep_model()
    x.prep_weights()
    x.fit()
    x.get_output_parms()
    fig_plot = x.plot_model(file='rvalue_forecast.pdf',
                            return_figure=True,
                            reference_date=pd.Timestamp(2020, 9, 1))
    fig_covariance = x.plot_covariance(file='covariance_plot.pdf', return_figure=True)

    # save model and figures
    dirname = './results/rvalue_model_' + str(pd.Timestamp.today().date()).replace('-', '_')
    if os.path.exists(dirname) is False:
        os.system('mkdir ' + dirname)
    pdf = matplotlib.backends.backend_pdf.PdfPages(dirname + "/rmodel_outputs.pdf")
    pdf.savefig(fig_plot)
    pdf.savefig(fig_covariance)
    pdf.close()
    f = open(dirname + "/model.pkl", "wb")
    pickle.dump({'model': x}, f)
    f.close()

    # check results load correctly
    with open(dirname + '/model.pkl', 'rb') as handle:
        xload = pickle.load(handle)['model']
    f.close()
    return xload



if __name__ == '__main__':




    x = rmodel_govuk(model_days=21,
                     discount_incomplete_days = 4,
                     forecast_length=150)
    #new_data = x.check_todays_update()
    x.download()



    x.prep_timeseries()
#
#
    x.prep_features()
    x.prep_model()
    x.prep_weights()
    x.multi_run(min_date=pd.Timestamp(2020, 6, 15))
    x.fit()
    x.get_output_parms()
##
    plt.close()
    fig_plot = x.plot_multi(reference_level=2000)
    plt.savefig('./adhoc/rvalue_multi_forecast.pdf')
#

    #only plot the rolling r tracker
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x.plot_r_estimate(fig, ax1)
    ax1.set_title('Rolling Reproduction Factor Calculation')

    xann = {'date':[pd.Timestamp(2020, 3, 23),
            pd.Timestamp(2020, 11, 5),
            pd.Timestamp(2021, 1, 2)],
            'label':['Lockdown 1','2','3']}
    idx = 1
    for date, lab in zip(xann['date'],xann['label']):
        if idx == 1:
            label = 'UK Lockdowns'
        else:
            label = None
        ax1.axvline(date,ls=':',label=label,color='purple')
        idx += 1
    #add uk return to school
    ax1.axvline(pd.Timestamp(2021, 3, 8),ls='--',color='r',label='Schools re-open')

    plt.legend()
    plt.tight_layout()
    plt.savefig('./adhoc/rvalue_plot.pdf')





