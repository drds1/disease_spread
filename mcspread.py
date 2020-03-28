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
        self.Ntimes = Ntimes
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
        move_code = np.random.randint(0, 5, [self.N,self.Ntimes])
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
        N, Nt, Ndim = np.shape(self.pos)
        self.dist = np.zeros((N,N,Nt))
        self.prox = np.zeros((N, N, Nt))
        for it in range(Nt):
            dnow = distance.cdist(self.pos[:, it, :], self.pos[:, it, :], 'euclidean')
            self.dist[:,:,it] = dnow
            self.prox[:,:,it] = dnow < transmission_distance

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
