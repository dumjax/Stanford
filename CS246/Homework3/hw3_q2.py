from pyspark import SparkConf, SparkContext
import numpy as np


def problem2_q1(file_input):

    conf = SparkConf()
    sc = SparkContext(conf=conf)
    lines = sc.textFile(file_input)
    degrees = lines.map(lambda g: (float(g.split("\t")[0])-1, float(g.split("\t")[1])-1)).distinct().countByKey()

    matrix = lines.map(lambda g: (float(g.split("\t")[1])-1,
                                  float(g.split("\t")[0])-1, 1.0/degrees[float(g.split("\t")[0])-1])).distinct()

    n = int(matrix.map(lambda g: (1, g[0])).reduceByKey(max).collect()[0][1])+1
    r = np.ones(n)/n
    ITER = 40
    beta = 0.8
    for _ in range(ITER):
        tmp = matrix.map(lambda g: (g[1], (g[0], g[2]*r[int(g[1])]))).map(lambda g: (g[1][0], g[1][1])).\
            reduceByKey(lambda n1, n2: n1+n2)
        for l in tmp.collect():
            r[int(l[0])] = l[1]*beta + (1-beta)/n

    sc.stop()

    top_5 = np.argsort(-r)[:5]
    bottom_5 = np.argsort(r)[:5]

    return top_5+1, -np.sort(-r)[:5], bottom_5+1, np.sort(r)[:5]


def problem2_q2(file_input):

    conf = SparkConf()
    sc = SparkContext(conf=conf)
    lines = sc.textFile(file_input)

    L = lines.map(lambda g: (float(g.split("\t")[0])-1,
                                  float(g.split("\t")[1])-1, 1.0)).distinct()
    L_t = lines.map(lambda g: (float(g.split("\t")[1])-1,
                                  float(g.split("\t")[0])-1, 1.0)).distinct()

    n = int(L.map(lambda g: (1, g[0])).reduceByKey(max).collect()[0][1])+1
    h = np.ones(n)
    a = np.zeros(n)
    ITER = 40
    lbda = 1
    mu = 1

    for _ in range(ITER):

        tmp = L_t.map(lambda g: (g[1], (g[0], g[2]*h[int(g[1])]))).map(lambda g: (g[1][0], g[1][1])).\
            reduceByKey(lambda n1, n2: n1+n2)
        for l in tmp.collect():
            a[int(l[0])] = l[1]*mu

        a = a/max(a)

        tmp = L.map(lambda g: (g[1], (g[0], g[2]*a[int(g[1])]))).map(lambda g: (g[1][0], g[1][1])).\
            reduceByKey(lambda n1, n2: n1+n2)
        for l in tmp.collect():
            h[int(l[0])] = l[1]*lbda

        h = h/max(h)

    sc.stop()

    top_5_h = np.argsort(-h)[:5]
    bottom_5_h = np.argsort(h)[:5]
    top_5_a = np.argsort(-a)[:5]
    bottom_5_a = np.argsort(a)[:5]

    return top_5_h+1, -np.sort(-h)[:5], bottom_5_h+1, np.sort(h)[:5], top_5_a+1, -np.sort(-a)[:5], bottom_5_a+1, np.sort(a)[:5]
