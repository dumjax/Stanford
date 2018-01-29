from scipy.linalg import *
import numpy as np


def problem1():

    M = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    svd(M, full_matrices=False)
    Evals, Evecs = eigh(np.dot(np.transpose(M), M))
    Evals_index = np.flip(np.argsort(Evals),0)
    Evals = sorted(Evals, reverse=True)
    Evecs = Evecs[:, Evals_index]



