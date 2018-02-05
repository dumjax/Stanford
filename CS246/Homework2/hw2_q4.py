from numpy import genfromtxt
import numpy as np

R = genfromtxt('/Users/maximedumonal/Github/Stanford/CS246/Homework2/data/q1-dataset/q1-dataset/user-shows.txt',delimiter=' ')

P = np.diag(R.sum(-1))
Q = np.diag(R.T.sum(-1))

T = np.transpose(R).dot(R)
T2 = R.dot(R.T)

Q_d = np.diag(np.diagonal(Q)**(-1.0/2))
P_d = np.diag(np.diagonal(P)**(-1.0/2))


Lambda_movies = R.dot(Q_d).dot(T).dot(Q_d)
Lambda_users = P_d.dot(T2).dot(P_d).dot(R)
