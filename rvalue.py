import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import datetime
import itertools
#https://en.wikipedia.org/wiki/Basic_reproduction_number



if __name__ == '__main__':

    #infectious lifetime (assume 3 weeks)
    tau = 14.0

    #test a variety of r
    r = list(np.arange(0.5,1.0,0.1))# + [1.75]
    t = np.arange(365)
    today = datetime.date.today()
    dates = pd.date_range(today,periods = len(t),freq='1D')
    annotation_fraction = 50.

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('')
    ax1.set_ylabel('Remaining Infected Population\n(% relative to today)')
    ax1.set_title('Infection reduction timescale\n Dependence on "r"')
    #compute the infected fraction (relative to current levels) vs time
    for rnow in r:
        f = 100*rnow**(t/tau)
        color = next(ax1._get_lines.prop_cycler)['color']

        #annotate plot with half lives
        if rnow < 0.9:
            idx_annotation_fraction = np.where(f < annotation_fraction)[0][0]
            date_af = dates[idx_annotation_fraction]
            weeks_away = int(np.round(pd.Timedelta(date_af - pd.Timestamp(today)).days/7))
            txt = str(int(annotation_fraction))+'% reduction in ' + str(weeks_away) +' weeks\n'
            ax1.plot(dates, f, label='r = ' + str(np.round(rnow, 2))+'\n'+txt, color=color)
            ax1.plot([date_af] * 2, [0, annotation_fraction], label=None, ls=':', color=color)
        else:
            ax1.plot(dates, f, label='r = ' + str(np.round(rnow, 2)) + '\n\n', color=color)

    plt.legend(fontsize='x-small')


    plt.savefig('rvalue_study.pdf')


