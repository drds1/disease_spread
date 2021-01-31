#see here for math equn
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5348083/


import numpy as np
import matplotlib.pylab as plt
import itertools
import pandas as pd


def simple_growth(seed, day, doublingtime):
    '''
    calculate corona cases vs doubling timescale
    :param seed:
    :param day:
    :param doublingtime:
    :return:
    '''

    return seed * 2**(day/doublingtime)




def malthus_equation(day,seed, r, p):
    '''
    more accurate malthus equation describing disease spread
    :param day:
    :param seed:
    :param r:
    :param p:
    :return:
    '''
    m = 1./(1. - p)
    A = seed**(1./m)
    ct = (r/m * day + A)**m
    #remove small nan values
    ct[ct!=ct] = 0
    return ct









def example_case():
    # visualise the infection curve timeseries for various seeds
    t = np.arange(0, 35, 0.1)
    rlist = [0.2]
    plist = [0.6, 0.8, 0.9]
    seed = [1950]
    comb = list(itertools.product(*[rlist, plist, seed]))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    idx = 0
    for parms in comb:
        r, p, seed = parms
        # y = simple_growth(seed, t, doublingtime)
        y = malthus_equation(t, seed, r, p)
        l = 'C(0)=' + str(seed) + ', r=' + str(r) + ', p=' + str(p)
        ax1.plot(t, y, label=l)
        idx += 1
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('cases')
    ax1.set_title('Growth curve for different starting levels')
    # ax1.plot([t[0],t[-1]],[64.e6]*2,ls='--',label='UK population')
    plt.legend()
    plt.show()


class modelCV:

    def __init__(self,
                 parms = {'rlist':list(np.arange(0.1,1.0,0.1)),'plist':list(np.arange(0.1,1.0,0.1)) },
                 equn = malthus_equation()):
        self.parms = parms
        self.keys = list(self.parms.keys())
        parmslist = [self.parms[k] for k in self.keys]
        self.plist = list(itertools.product(*[rlist, plist]))

    def fit(self,t,y):

        





if __name__ == '__main__':

    #build a model to optimise the number of covid19 case forecasts
    #parameters
    dat = np.loadtxt('cumcases.dat')
    C0 = dat[-1]
    t = -1*np.arange(1,len(dat))[::-1]


    parms = list(itertools.product(*[rlist, plist]))
    bofgrid = []
    output = {'r':[],'p':[],'BOF':[]}
    for parmsnow in parms:
        r, p = parmsnow
        y = malthus_equation(t, C0, r, p)
        output['r'].append(r)
        output['p'].append(p)
        output['BOF'].append(np.sum((y - dat[:-1])**2))
    output = pd.DataFrame(output)
    output.sort_values(by='BOF',inplace=True)
    print(output)





