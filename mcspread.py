import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import mcspread_utils as utils
from scipy.spatial import distance
import os

class movement:
    '''
    class to control movement around mc grid
    '''

    def __init__(self, Nx = 100, Ny = 100, N = 50, Ntimes = 100,seed=None):
        self.Nx = Nx
        self.Ny = Ny
        self.N = N
        self.Nt = Ntimes
        self.pos = np.zeros((N,Ntimes,2))
        np.random.seed(seed)

    def arrange_particles(self):
        '''
        set the starting positions for the particles
        :return:
        '''
        xpos = np.random.randint(0, self.Nx, self.N)
        ypos = np.random.randint(0, self.Ny, self.N)
        self.pos[:,0,0] = xpos
        self.pos[:,0,1] = ypos

    def random_walk(self):
        '''
        random walk all particles for N time steps
        :return:
        '''
        move_code = np.random.randint(0, 5, [self.N,self.Nt])
        #stay = move_code == 0
        right = move_code == 1
        left = move_code == 2
        up = move_code == 3
        down = move_code == 4
        diff_posx = utils.move(right, left)# - utils.move(left)
        diff_posy = utils.move(up, down)# - utils.move(down)
        N,Nt = np.shape(diff_posx)
        p0x = np.tile(self.pos[:, 0, 0],Nt).reshape(Nt,N).T
        p0y = np.tile(self.pos[:, 0, 1], Nt).reshape(Nt, N).T
        self.pos[:, :, 0] = p0x + diff_posx
        self.pos[:, :, 1] = p0y + diff_posy
        self.dpx = diff_posx
        self.dpy = diff_posy

    def compute_distances(self,transmission_distance = 2.0):
        '''
        compute the distances between all pairs
        of particles at each step in the simulation
        :return:
        '''
        N, Nt, Ndim = self.N,self.Nt,2
        self.dist = np.zeros((N,N,Nt))
        self.prox = np.zeros((N, N, Nt))
        for it in range(Nt):
            dnow = distance.cdist(self.pos[:, it, :], self.pos[:, it, :], 'euclidean')
            self.dist[:,:,it] = dnow
            self.prox[:,:,it] = dnow < transmission_distance

    def compute_infections(self,
                           default_immunity = 0.0,
                           default_infections = 0.2,
                           default_transmission = 0.8,
                           infection_length = 5,
                           death_rate = 0.01,
                           verbose = True):
        '''
        compute infections
        :return:
        '''
        self.transmit_prob = default_transmission
        self.alive = np.ones((self.N, self.Nt))
        self.days_infected = np.zeros((self.N,self.Nt))

        #default immunity
        if default_immunity < 1:
            self.N_immune = np.int(default_immunity*self.N)
        else:
            self.N_immune = default_immunity
        self.immune = np.zeros((self.N, self.Nt))
        idx = np.random.choice(np.arange(self.N), self.N_immune, replace=False)
        self.immune[idx, 0] = 1


        #default infections
        if default_infections < 1:
            self.N_start_infections = np.int(self.N*default_infections)
        else:
            self.N_start_infections = default_infections
        self.infected = np.zeros((self.N, self.Nt))
        idx = np.random.choice(np.arange(self.N),self.N_start_infections,replace=False)
        idx = idx[self.immune[idx,0] == 0]
        self.infected[idx,:] = 1


        #do it
        for it in range(1,self.Nt):
            if verbose:
                print('day '+str(it)+' of '+str(self.Nt))
            prox0 = self.prox[:, :, it]

            #new infections
            idx_infected = np.where(self.infected[:, it] == 1)[0]
            idxprox = np.array(np.where(prox0 == 1)).T
            idxprox = idxprox[idxprox[:, 0] != idxprox[:, 1], :]
            nprox = len(idxprox)
            for idx in range(nprox):
                idx1, idx2 = idxprox[idx,:]
                if idx1 in idx_infected and \
                        self.immune[idx2,it] == 0 and \
                        self.alive[idx2,it] == 1 and \
                        self.immune[idx1,it] == 0:
                    self.infected[idx2,it] = 1

            #days infected
            self.days_infected[idx_infected,it] = self.days_infected[idx_infected,it-1] + 1
            idx_notinfected = np.where(self.infected[:, it] == 0)[0]
            self.days_infected[idx_notinfected, it] = 0

            #new deaths
            idx_infectionend = np.where(self.days_infected[:,it] >= infection_length)[0]
            ninfectionend = len(idx_infectionend)
            ndie = np.int(death_rate*ninfectionend)
            idxdie = np.random.choice(idx_infectionend,ndie,replace=False)
            #idxsurvive = np.setdiff1d(idx_infectionend, idxdie)
            self.alive[idxdie,it:] = 0
            self.infected[idx_infectionend,it:] = 0
            self.days_infected[idx_infectionend,it] = 0
            self.immune[idx_infectionend,it:] = 1
        self.DeadNumber = self.N - np.sum(self.alive,axis=0)
        self.InfectedNumber = np.sum(self.infected,axis=0)
        self.ImmuneNumber = np.sum(self.immune,axis=0)




    def plot_timeseries(self,
                        quantity=['infections','deaths','immunity'],
                        save=None):
        '''
        plot key metric timeseries "infections","deaths","immunity"
        :return:
        '''
        plt.close()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        x = np.arange(self.Nt)
        for q in quantity:
            if q == 'deaths':
                y = self.DeadNumber/self.N
            if q == 'infections':
                y = self.InfectedNumber/self.N
            if q == 'immunity':
                y = self.ImmuneNumber/self.N
            ax1.plot(x,y,label=q)
        ax1.set_xlabel('days since start')
        ax1.set_ylabel('population fraction')
        plt.legend()
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
            plt.close()




    def plot_movements(self,file = 'annimation'):
        '''
        video of particle movements
        :return:
        '''
        os.system('rm -rf '+file)
        os.system('mkdir '+file)
        for it in range(self.Nt):
            plt.close()
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            x = self.pos[:, it, 0]
            y = self.pos[:, it, 1]

            alive = self.alive[:,it]
            idxalive = np.where(alive==1)[0]
            ndead = 100.*(1. - len(idxalive)/self.N)
            ax1.scatter(x[idxalive],y[idxalive],color='b',label='normal: ('+str(ndead)+' pc dead)')

            infected = self.infected[:, it]
            idxinfected = np.where(infected == 1)[0]
            ninfected = 100*len(idxinfected)/self.N
            ax1.scatter(x[idxinfected],y[idxinfected],color='r',label=str(ninfected)+' pc infected')

            immune = self.immune[:, it]
            idximmune = np.where(immune == 1)[0]
            nimmune = 100.*(len(idximmune)/self.N)
            ax1.scatter(x[idximmune],y[idximmune],color='green',label=str(nimmune)+' pc immune')

            plt.legend()
            plt.savefig(file+'/'+str(it)+'.pdf')
            #idxdead = np.where(alive == 0)[0]
            #ax1.scatter(x[idxdead], y[idxdead], color='b', label='normal')





def test_particle_movements(file = 'particle_movements.pdf'):
    x = movement()
    x.arrange_particles()
    x.random_walk()
    x.compute_distances()
    pos = x.pos
    dist = x.dist

    #plot particle movements
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    for i in range(5):
        ax1.plot(pos[i,:,0],pos[i,:,1],label='particle '+str(i))
    plt.legend()
    plt.savefig(file)


if __name__ == '__main__':
    x = movement(seed=12345)
    x.arrange_particles()
    x.random_walk()
    x.compute_distances()
    pos = x.pos
    dist = x.dist
    prox = x.prox
    x.compute_infections()

    x.plot_timeseries(save='simulation_timeseries.pdf')
    x.plot_movements(file='annimation')

    #check infection duration
    inf6 = x.infected[6, :]
    dist0 = dist[:,:,0]
    prox0 = prox[:,:,0]
    idxprox = np.array(np.where(prox0==1)).T
    idxprox = idxprox[idxprox[:,0] != idxprox[:,1],:]
