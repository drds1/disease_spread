import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import make_blobs

#mean = [0, 0]
#cov = [[1, 0], [0, 100]]  # diagonal covariance
#x, y = np.random.multivariate_normal(mean, cov, 5000).T
#plt.scatter(x,y)
#plt.show()


X, y = make_blobs(n_samples=10000, centers=3, cluster_std = 0.5, n_features=2,random_state=0)
plt.scatter(X[:,0],X[:,1],s=2)
plt.show()
hist = np.histogram2d(X[:,0],X[:,1])