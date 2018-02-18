from numpy import genfromtxt
import numpy as np


def problem4(input_file,name_shows_file):

    R = genfromtxt(input_file, delimiter=' ')
    movies = list()
    with open(name_shows_file) as f:
        for line in f:
            movies.append(line.strip())

    P = np.diag(R.sum(-1))
    Q = np.diag(R.T.sum(-1))

    T = np.transpose(R).dot(R)
    T2 = R.dot(R.T)

    Q_d = np.diag(np.diagonal(Q)**(-1.0/2))
    P_d = np.diag(np.diagonal(P)**(-1.0/2))

    Lambda_items = R.dot(Q_d).dot(T).dot(Q_d)
    Lambda_users = P_d.dot(T2).dot(P_d).dot(R)

    Alex_items = [list(np.array(movies)[np.argsort(-Lambda_items[499][:100])[:5]]),
                  list(-np.sort(-Lambda_items[499][:100])[:5])]
    Alex_users = [list(np.array(movies)[np.argsort(-Lambda_users[499][:100])[:5]]),
                       list(-np.sort(-Lambda_users[499][:100])[:5])]

    return Alex_items, Alex_users

