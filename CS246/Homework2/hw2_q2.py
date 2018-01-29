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
    return cdist(A, B, 'euclidean')


def manhattan_cost(X, centroids):
    distances = manhattan_d(X, centroids)
    return np.argmin(distances), np.min(distances)


def euclidean_cost(X, centroids):
    return np.argmin(manhattan_d(X, centroids)), np.min(manhattan_d(X, centroids))


def problem2(file_input, centroids_init_file):

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    X = sc.textFile(file_input)
    C = np.genfromtxt(centroids_init_file)

    points = X.flatMap(lambda l: np.array([[float(n) for n in l.split(" ")]]))

    clusters_map = points.map(lambda p: (manhattan_cost(p, C), p)).map(lambda p: (p[0][0], p[1]))

    min_manhattan_dist_map = points.map(lambda p: (manhattan_cost(p, C), p)).map(lambda p: (p[1], p[0][1]))

    clusters = clusters_map.groupByKey().map(lambda c: np.mean([x for x in c[1]])).collect()
    cost = 0
    for point in clusters_map.collect():
        cost += point[1][1]








