import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import mcspread_utils as utils
from scipy.spatial import distance


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
                           death_rate = 0.01):
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
        self.infected[idx,0] = 1


        #do it
        for it in range(1,self.Nt):
            prox0 = self.prox[:, :, it]
            infected0 = self.infected[:,it]
            immune0 = self.immune[:,it]
            alive0 = self.alive[:,it]


            #new infections
            idxprox = np.array(np.where(prox0 == 1)).T
            nprox = len(idxprox)/2
            for idx in range(nprox):
                idx1, idx2 = idxprox[idx,:]

            idxprox = idxprox[idxprox[:, 0] != idxprox[:, 1], :]
            idx_infected = np.where(self.infected[:, it] == 1)[0]


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


            #recieved from
            receive = prox0*infected0*immune0*alive0

            #transmitted to











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

    dist0 = dist[:,:,0]
    prox0 = prox[:,:,0]
    idxprox = np.array(np.where(prox0==1)).T
    idxprox = idxprox[idxprox[:,0] != idxprox[:,1],:]
