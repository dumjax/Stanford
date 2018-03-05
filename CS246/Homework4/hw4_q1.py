from numpy import genfromtxt
import numpy as np


def f(w, b, X, y, C):
    n = X.shape[0]
    g = np.ones(n) - (np.dot(X, np.transpose(w))*y+b)
    g[g <= 0] = 0

    return 0.5*np.linalg.norm(w, 2)**2 + C*sum(g)


def batch_gradient(input_features, input_target, w_init, b_init, eta, eps):

    X = genfromtxt(input_features, delimiter=',')
    y = genfromtxt(input_target)
    d = X.shape[1]
    w_init = np.zeros(X.shape[1])
    b_init = 0
    w = w_init
    b = b_init

    cost = 0
    delta_cost =  float("inf")

    grad = np.ndarray(X.shape)
    for j in range(d):
        grad[j] = np.sum(X[:, j]*y, 1)

    while delta_cost > eps:
        for i in range(d):
            w[j] = w[j] - eta*(w[j] + X[:, j]*y)




