import numpy as np

def move(right, left):
    l = np.zeros(np.shape(left))
    l[left] = 1
    r = np.zeros(np.shape(right))
    r[right] = 1
    return np.cumsum(r,axis=1)-np.cumsum(l,axis=1)