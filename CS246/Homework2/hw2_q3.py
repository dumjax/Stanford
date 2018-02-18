import numpy as np


def problem3(input_file, eta, lbda, it_n):

    k = 20
    #lbda = 0.2
    #eta = 0.001

    # We find the dimensions first (we can't save the matrix R in memory)
    m = 0
    n = 0
    with open(input_file) as f:
        for line in f:
            l = line.strip().split("\t")
            if n < int(l[0]):
                n= int(l[0])
            if m < int(l[1]):
                m = int(l[1])

    Q = np.random.rand(m, k)*np.sqrt(5.0/k)
    P = np.random.rand(n, k)*np.sqrt(5.0/k)

    errors = list()
    for it in range(it_n):
        with open(input_file) as f:
            for line in f:
                l = line.strip().split("\t")
                u = int(l[0])-1
                i = int(l[1])-1
                R = float(l[2])
                p_tmp = P[u, :]
                q_tmp = Q[i, :]
                Err = 2*(R - np.dot(q_tmp, np.transpose(p_tmp)))
                P[u, :] = p_tmp + eta * (Err*q_tmp - 2*lbda*p_tmp)
                Q[i, :] = q_tmp + eta * (Err*p_tmp - 2*lbda*q_tmp)
        # print P[0, :]

        # E calculation
        E = 0
        # let's calculate the regularization term first
        E = lbda*(sum(np.diag(P.dot(P.T)))+sum(np.diag(Q.dot(Q.T))))

        with open(input_file) as f:
            for line in f:
                l = line.strip().split("\t")
                u = int(l[0])-1
                i = int(l[1])-1
                p_tmp = P[u, :]
                q_tmp = Q[i, :]
                E = E + (float(l[2]) - np.dot(q_tmp, np.transpose(p_tmp))) ** 2

        errors.append(E)
        print E

    return errors
