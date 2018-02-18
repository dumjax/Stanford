from pyspark import SparkConf, SparkContext
import sys
import itertools
import numpy as np
from scipy.spatial.distance import cdist


def l2(u, v):
    return np.linalg.norm(u - v, 2)


def l1(u, v):
    return np.linalg.norm(u - v, 1)


def manhattan_d(A, B):
    return cdist([A], B, 'cityblock')


def euclidean_d(A, B):
    return cdist([A], B, 'euclidean')


def manhattan_cost(X, centroids):
    distances = manhattan_d(X, centroids)
    return np.argmin(distances), np.min(distances)


def euclidean_cost(X, centroids):
    distances = euclidean_d(X, centroids)
    return np.argmin(distances), np.min(distances)


def problem2(file_input, centroids_init_file,norm='e'):

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    X = sc.textFile(file_input)
    C = np.genfromtxt(centroids_init_file)

    points = X.flatMap(lambda l: np.array([[float(n) for n in l.split(" ")]]))

    k = 20
    costs = []

    if norm == 'e':
        f = euclidean_cost
    elif norm == 'm':
        f = manhattan_cost
    else:
        return "er"

    for i in range(k):

        clusters_map = points.map(lambda p: (f(p, C), p)).map(lambda p: (p[0][0], p[1]))
        min_dist_map = points.map(lambda p: (f(p, C), p)).map(lambda p: (p[1], p[0][1]))

        # clusters_map = points.map(lambda p: (euclidean_cost(p, C), p)).map(lambda p: (p[0][0], p[1]))
        #     min_dist_map = points.map(lambda p: (manhattan_cost(p, C), p)).map(lambda p: (p[1], p[0][1]))
        #     min_dist_map = points.map(lambda p: (euclidean_cost(p, C), p)).map(lambda p: (p[1], p[0][1]))

        cost = 0
        for point in min_dist_map.collect():

            # Manhattan
            #cost += point[1]

            # euclidean
            if norm == 'e':
                cost += point[1]**2
            else:
                cost += point[1]

        costs.append(cost)

        C = clusters_map.groupByKey().map(lambda c: np.mean([x for x in c[1]], 0)).collect()

    sc.stop()

    return costs






