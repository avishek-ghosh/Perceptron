import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.decomposition import kernel_pca
import scipy.sparse.linalg as slag
from sklearn import manifold
import numpy.linalg as la
from mpl_toolkits.mplot3d import axes3d, Axes3D

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def RBF_PCA(X, sigma):
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    W = exp( mat_sq_dists/(sigma))

    #centering
    N = W.shape[0]
    one_n = np.ones((N,N)) / N
    W = W - one_n.dot(W) - W.dot(one_n) + one_n.dot(W).dot(one_n)

    eval, evec = slag.eigsh(W, k=3, which='LA')
    X_ = evec

    return np.array(X_)


def Sigmoid(X):
        return  1/( 1 + np.exp(-X))

def update():
    w = np.ones((x_train.shape[1], 1))
    #for i in range(x_train.shape[0]):
    for i in range(100):
        result = np.dot(x_train, w)
        ypred = Sigmoid(result)
        error = (y_train - ypred)
        gradient = np.dot( x_train.T, error)
        w = w + 0.1 *gradient
    return w

def predit():
    weight = update()
    result = np.dot(x_train, weight)
    ypred = Sigmoid(result)
    ypred[ypred < 0.5] = 0
    ypred[ypred > 0.5] = 1
    return ypred

def accuracy(ypred):
    correct = 0
    for i in range(y_train.shape[0]):
        if (ypred[i] == y_train[i]):
            correct += 1
    return correct/float(len(ypred))

def Get_Graph(X):
    x = X[0:1]
    y = X[1:2]
    z = X[2:3]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.show()

if __name__ == '__main__':
    data = scipy.io.loadmat("A:/Spring 2017/ML in SP/Assignments/ass4/Assignments_export/Homework #4/concentric.mat")
    X = np.array(data['X'])
    X = np.array(X.T)
    X = RBF_PCA(X, 0.1)
    #Get_Graph(X.T)
    #add bias 1 to the matrix
    x_train = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))

    # x_train = Z
    y_train = np.zeros((x_train.shape[0], 1))

    for i in range(x_train.shape[0]):
        if (i < 51):
            y_train[i] = 0
        else:
            y_train[i] = 1

    yhat = predit()
    acc = accuracy(yhat)
    print(acc)





