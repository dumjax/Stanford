from numpy import genfromtxt
import numpy as np

input_features = '/Users/maximedumonal/github/Stanford/CS246/Homework4/data/features.txt'
input_target = '/Users/maximedumonal/github/Stanford/CS246/Homework4/data/target.txt'

C = 100


# bgd
eta = 0.0000003
eps = 0.25
# costs = batch_gradient(input_features, input_target, C, eta, eps)

#sgd
eta = 0.0001
eps = 0.001


def f(w, b, X, y, C):
    n = X.shape[0]
    g = np.ones(n) - (np.dot(X, np.transpose(w)) + b) * y
    g[g <= 0] = 0

    return 0.5*np.linalg.norm(w, 2)**2 + C*sum(g)


def grad_w(w, b, X, y, C):
    n = X.shape[0]
    d = X.shape[1]
    grad_w_sum = np.ndarray(d)
    g = np.ones(n) - (np.dot(X, np.transpose(w)) + b) * y
    for j in range(d):
        tmp = - X[:, j] * y
        tmp[g <= 0] = 0
        grad_w_sum[j] = np.sum(tmp)

    return w + C*grad_w_sum


def grad_b(w, b, X, y, C):
    n = X.shape[0]
    g = np.ones(n) - (np.dot(X, np.transpose(w)) + b) * y
    tmp = -y
    tmp[g <= 0] = 0
    return C*np.sum(tmp)


def bgd(input_features, input_target, C, eta, eps):

    X = genfromtxt(input_features, delimiter=',')
    y = genfromtxt(input_target)

    w_init = np.zeros(X.shape[1])
    b_init = 0

    w = w_init
    b = b_init

    delta_cost = float("inf")

    f_1 = f(w, b, X, y, C)
    costs = [f_1]
    while delta_cost > eps:

        f_tmp = f_1
        grad_w_tmp = grad_w(w, b, X, y, C)
        grad_b_tmp = grad_b(w, b, X, y, C)

        w = w - eta * grad_w_tmp
        b = b - eta * grad_b_tmp

        f_1 = f(w, b, X, y, C)
        delta_cost = np.abs(f_1-f_tmp)*100/f_tmp
        print delta_cost
        costs.append(f_1)

    return costs


def sgd(input_features, input_target, C, eta, eps):

    X = genfromtxt(input_features, delimiter=',')
    y = genfromtxt(input_target)

    n = X.shape[0]
    d = X.shape[1]

    w_init = np.zeros(X.shape[1])
    b_init = 0

    w = w_init
    b = b_init

    delta_cost = float("inf")

    f_1 = f(w, b, X, y, C)
    costs = [f_1]
    delta_cost_init = 0
    i = 0
    k = 0
    while delta_cost > eps:

        if i == 0 and k == 0:
            delta_cost = delta_cost_init

        f_tmp = f_1

        g = 1 - (np.dot(X[i, :], np.transpose(w)) + b) * y[i]

        for j in range(d):
            if g <= 0:
                grad_w_tmp = w[j]
            else:
                grad_w_tmp = w[j]-C*y[i]*X[i, j]

            w[j] = w[j] - eta*grad_w_tmp

        if g <= 0:
            grad_b_tmp = 0
        else:
            grad_b_tmp = -y[i]*C

        b = b - eta*grad_b_tmp

        f_1 = f(w, b, X, y, C)
        delta_cost = 0.5*delta_cost+0.5*np.abs(f_1-f_tmp)*100/f_tmp
        print f_tmp
        print delta_cost
        costs.append(f_1)
        (k, i) = divmod(i+1, n-1)

        print i

    return costs
