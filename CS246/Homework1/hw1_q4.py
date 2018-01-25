# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla

import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image


# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
# TODO: Implement this
def l1(u, v):
    # raise NotImplementedError
    return np.linalg.norm(u-v, 1)


# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')


# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f


# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0, 
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions


# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])


# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))


# Retrieve all of the points that hash to one of the same buckets 
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))


# Sets up the LSH.  You should try to call this function as few times as 
# possible, since it is expensive.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)


# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]


# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")


# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    rows_index = map(lambda v: l1(A[query_index, :], v), A)
    return np.argsort(rows_index)[1:num_neighbors+1]


# TODO: Write a function that computes the error measure
def error(A, lsh, lin, z):
    points = []
    i = 0
    for p in z:
        num = l1(A[lsh[i][0], :], A[p, :])+l1(A[lsh[i][1], :], A[p, :])+l1(A[lsh[i][2], :], A[p, :])
        den = l1(A[lin[i][0], :], A[p, :]) + l1(A[lin[i][1], :], A[p, :]) + l1(A[lin[i][2], :], A[p, :])
        points.append(num/den)
        i += 1
    return np.mean(points)



def error_lsh(A, images_index, k, L):
    # raise NotImplementedError

  #  found = False
  #  while found is False:
    functions, hashed_A = lsh_setup(A, k, L)
    lsh_output = []
    ts = []
    for i in images_index:
        ts_1 = time.time()
        tmp = lsh_search(A, hashed_A, functions, i, 3)
        # if len(tmp) < 3:
        #     print "retry"
        #     break
        ts.append(time.time()-ts_1)
        lsh_output.append(tmp)
   #     found = True
    print "Average time lsh : %s" % np.mean(ts)

    lin_output = []
    ts = []
    for i in images_index:
        ts_1 = time.time()
        tmp = linear_search(A, i, 3)
        ts.append(time.time()-ts_1)
        lin_output.append(tmp)
    print "Average time lin : %s" % np.mean(ts)

    return error(A, lsh_output, lin_output, images_index)


# TODO: Solve Problem 4
def problem4():

    A = load_data('CS246/data/patches.csv')
    images_index = range(99, 1000, 100)

    # errors calculations for the range L = [10-12-14-16-18-20]
    errors_L = []
    for L in range(10, 22, 2):
        errors_L.append(error_lsh(A, images_index, k=24, L=L))

    # errors calculations for the range k = [16-18-20-22-24]
    errors_k = []
    for k in range(16, 25, 2):
        errors_k.append(error_lsh(A, images_index, k=k, L=10))

    #10-nearest neighbor images
    functions, hashed_A = lsh_setup(A, k, L)
    lsh_10 = lsh_search(A, hashed_A, functions, 99, 10)
    lin_10 = linear_search(A, 99, 10)

    plot(A, [99], 'lsh_original')
    plot(A, lin_10, 'lin_res')
    plot(A, lsh_10, 'lsh_res')

    return errors_k, errors_L


#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))


    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)


if __name__ == '__main__':
    # unittest.main() ### TODO: Uncomment this to run tests
    er_k, er_l = problem4()
