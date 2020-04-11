### −∗− mode : python ; −∗−
# @file EpiModel.py
# @author Bruno Goncalves
######################################################

#https://github.com/DataForScience/Epidemiology101/blob/master/EpiModel.py
#https://medium.com/data-for-science/epidemic-modeling-101-or-why-your-covid19-exponential-fits-are-wrong-97aa50c55f8

import networkx as nx
import numpy as np
import scipy.integrate
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


class EpiModel:
    """Simple Epidemic Model Implementation

        Provides a way to implement and numerically integrate
    """

    def __init__(self, compartments=None):
        self.transitions = nx.DiGraph()

        if compartments is not None:
            self.transitions.add_nodes_from([comp for comp in compartments])

    def add_interaction(self, source, target, agent, rate):
        self.transitions.add_edge(source, target, agent=agent, rate=rate)

    def add_spontaneous(self, source, target, rate):
        self.transitions.add_edge(source, target, rate=rate)

    def _new_cases(self, population, time, pos):
        """Internal function used by integration routine"""
        diff = np.zeros(len(pos))
        N = np.sum(population)

        for edge in self.transitions.edges(data=True):
            source = edge[0]
            target = edge[1]
            trans = edge[2]

            rate = trans['rate'] * population[pos[source]]

            if 'agent' in trans:
                agent = trans['agent']
                rate *= population[pos[agent]] / N

            diff[pos[source]] -= rate
            diff[pos[target]] += rate

        return diff

    def plot(self, title=None, normed=False):
        """Convenience function for plotting"""
        try:
            if normed:
                N = self.values_.iloc[0].sum()
                ax = (self.values_ / N).plot()
            else:
                ax = self.values_.plot()

            ax.set_xlabel('Time')
            ax.set_ylabel('Population')

            if title is not None:
                ax.set_title(title)

            return ax
        except:
            raise NotInitialized('You must call integrate() first')

    def __getattr__(self, name):
        """Dynamic method to return the individual compartment values"""
        if 'values_' in self.__dict__:
            return self.values_[name]
        else:
            raise AttributeError("'EpiModel' object has no attribute '%s'" % name)

    def integrate(self, timesteps, **kwargs):
        """Numerically integrate the epidemic model"""
        pos = {comp: i for i, comp in enumerate(kwargs)}
        population = np.zeros(len(pos))

        for comp in pos:
            population[pos[comp]] = kwargs[comp]

        time = np.arange(1, timesteps, 1)

        self.values_ = pd.DataFrame(scipy.integrate.odeint(self._new_cases, population, time, args=(pos,)),
                                    columns=pos.keys(), index=time)

    def __repr__(self):
        text = 'Epidemic Model with %u compartments and %u transitions:\n\n' % \
               (self.transitions.number_of_nodes(),
                self.transitions.number_of_edges())

        for edge in self.transitions.edges(data=True):
            source = edge[0]
            target = edge[1]
            trans = edge[2]

            rate = trans['rate']

            if 'agent' in trans:
                agent = trans['agent']
                text += "%s + %s = %s %f\n" % (source, agent, target, rate)
            else:
                text += "%s -> %s %f\n" % (source, target, rate)

        return text











class epi_gridsearch:

    def __init__(self,days = 365,parms = {'S':np.arange(10),
                     'I':np.arange(10),
                     'R':np.zeros(10),
                     'D':np.zeros(10),
                     'transmission_rate':np.array([0.2]),
                     'recovery_rate':np.array([0.1]),
                     'death_rate':np.array([0.1])}):
        self.data_cases = None
        self.data_deaths = None
        self.data_recovered = None
        self.days = days
        self.parms = parms

    def prepare_grid(self):
        keys = self.parms.keys()
        values = (self.parms[key] for key in keys)
        self.grid = pd.DataFrame([dict(zip(keys, combination)) for combination in itertools.product(*values)])



    def run_model(self,S=100000,I=1,R=0,D=0,
                  transmission_rate=0.2,
                  recovery_rate = 0.1,
                  death_rate = 0.1):
        SIR = EpiModel()
        SIR.add_interaction('S', 'I', 'I', transmission_rate)
        SIR.add_spontaneous('I', 'R', recovery_rate)
        #add deaths
        SIR.add_spontaneous('I', 'D', death_rate)
        SIR.integrate(self.days+1, S=S-I, I=I, R=R, D=D)
        x = SIR.values_
        self.Sts, self.Its, \
        self.Rts, self.Dts = x.values[:,0], \
                             x.values[:,1], \
                             x.values[:,2],\
                             x.values[:,3]

    def evaluate_performance(self,model_cases, model_deaths, model_recovered,
                    data_cases, data_deaths, data_recovered):

        Ndays = len(data_deaths)
        BOF_deaths = np.nan
        BOF_recovered = np.nan
        BOF_cases = np.nan
        BOF = 0.0
        if data_cases is not None:
            BOF_cases = np.sum((model_cases[:Ndays] - data_cases)**2)
            BOF = BOF + BOF_cases

        if data_deaths is not None:
            BOF_deaths = np.sum((model_deaths[:Ndays] - data_deaths)**2)
            BOF = BOF+BOF_deaths

        if data_recovered is not None:
            BOF_recovered = np.sum((model_recovered[:Ndays] - data_recovered)**2)
            BOF = BOF + BOF_recovered

        return BOF



    def grid_search(self):
        '''
        perform grid search for all combinations of parameters in grid
        :return:
        '''
        grid = self.grid.values
        BOF = []
        dead_ts = []
        infected_ts = []
        recovered_ts = []
        for i in range(len(grid)):
            Snow, Inow, Rnow, Dnow, tr, rr, dr = grid[i,:]
            self.run_model(S=Snow,I=Inow,R=Rnow,
                           D=Dnow,
                           transmission_rate=tr,
                           recovery_rate = rr,
                           death_rate = dr)
            BOF.append(self.evaluate_performance(self.Its,self.Dts,self.Rts,self.data_cases, self.data_deaths, None))
            dead_ts.append(self.Dts)
            recovered_ts.append(self.Rts)
            infected_ts.append(self.Its)
        self.grid['BOF'] = BOF
        self.grid['infected ts'] = infected_ts
        self.grid['recovered ts'] = recovered_ts
        self.grid['dead ts'] = dead_ts
        self.grid.sort_values(by='BOF',inplace=True)




if __name__ == '__main__':
    # load data
    data_cases = np.loadtxt('data_cases.dat')
    data_deaths = np.loadtxt('data_deaths.dat')
    Ndays = len(data_deaths)
    # run model
    x = epi_gridsearch(
        parms={'S': np.array([65.e6]),
               'I': np.arange(10),
               'R': np.array([0]),
               'D':np.array([0]),
               'death_rate':np.array([0.2]),
               'recovery_rate':np.array([0.2]),
               'transmission_rate':np.array([0.2])},
        days=Ndays
    )
    x.data_cases = data_cases
    x.data_deaths = data_deaths
    x.prepare_grid()
    x.grid_search()
    xgrid = x.grid



