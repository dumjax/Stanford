import numpy as np
import csv
import math
from numpy import genfromtxt

list_param = [(3, 1561), (17, 277), (38, 394), (61, 13), (78, 246)]

delta = math.e**(-5)
eps = math.e * 0.0001
n_buckets = int(math.ceil(math.e / eps))
p = 123457


def hash_fun(a, b, p, n_buckets):

    def output(x):
        y = divmod(x, int(p))[1]
        hash_val = divmod(a*y+b, p)[1]
        return divmod(hash_val, n_buckets)[1]

    return output


def initiate_hash_fun(list_param, n_buckets):

    hash_funs = list()

    for i in range(5):
        hash_funs.append(hash_fun(list_param[i][0], list_param[i][1], p, n_buckets))

    return hash_funs


def algo():

    #input_stream = '/Users/maximedumonal/github/Stanford/CS246/Homework4/data/words_stream.txt'
    #input_count = '/Users/maximedumonal/github/Stanford/CS246/Homework4/data/counts.txt'

    input_stream = '/Users/maximedumonal/github/Stanford/CS246/Homework4/data/words_stream_tiny.txt'
    input_count = '/Users/maximedumonal/github/Stanford/CS246/Homework4/data/counts_tiny.txt'

    counts = genfromtxt(input_count)
    n = counts.shape[0]

    h_list = initiate_hash_fun(list_param, n_buckets)
    c_matrix = np.zeros((5, n_buckets))

    t = 0
    f = open(input_stream, 'rU')
    for a in f:
        t += 1
        k_mod = int(divmod(t, int(100000))[1])
        if k_mod == 0:
            print t
        for j in range(5):
            c_matrix[j, h_list[j](int(a))] += 1

    Err = list()
    c_row = list()
    for i in range(n):
        for j in range(5):
            c_row.append(c_matrix[j, h_list[j](i+1)])
        F_t = min(c_row)
        Err.append((float(F_t)/float(counts[i][1])-1))
        c_row = list()

    F_true = counts[:, 1] / t

    with open("output_hw4_q4_small.csv", 'wb') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        for i in range(n):
            wr.writerow([F_true[i], abs(Err[i])])




