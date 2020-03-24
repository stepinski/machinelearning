import os
import sys
import time
import traceback
import numpy as np
import linear_regression
import svm
import softmax
import features
import kernel

sys.path.append("..")
import utils

verbose = False

epsilon = 1e-6

def green(s):
    return '\033[1;32m%s\033[m' % s

def yellow(s):
    return '\033[1;33m%s\033[m' % s

def red(s):
    return '\033[1;31m%s\033[m' % s

def log(*m):
    print(" ".join(map(str, m)))

def log_exit(*m):
    log(red("ERROR:"), *m)
    exit(1)


def check_real(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not np.isreal(res):
        log(red("FAIL"), ex_name, ": does not return a real number, type: ", type(res))
        return True
    if not -epsilon < res - exp_res < epsilon:
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True


def equals(x, y):
    if type(y) == np.ndarray:
        return (np.abs(x - y) < epsilon).all()
    return -epsilon < x - y < epsilon

def check_tuple(ex_name, f, exp_res, *args, **kwargs):
    try:
        res = f(*args, **kwargs)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == tuple:
        log(red("FAIL"), ex_name, ": does not return a tuple, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected a tuple of size ", len(exp_res), " but got tuple of size", len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True

def check_array(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == np.ndarray:
        log(red("FAIL"), ex_name, ": does not return a numpy array, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected an array of shape ", exp_res.shape, " but got array of shape", res.shape)
        return True
    if not equals(res, exp_res):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)

        return True

def check_list(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not type(res) == list:
        log(red("FAIL"), ex_name, ": does not return a list, type: ", type(res))
        return True
    if not len(res) == len(exp_res):
        log(red("FAIL"), ex_name, ": expected a list of size ", len(exp_res), " but got list of size", len(res))
        return True
    if not all(equals(x, y) for x, y in zip(res, exp_res)):
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True

def check_get_mnist():
    ex_name = "Get MNIST data"
    train_x, train_y, test_x, test_y = utils.get_MNIST_data()
    log(green("PASS"), ex_name, "")


def check_closed_form():
    ex_name = "Closed form"
    X = np.arange(1, 16).reshape(3, 5)
    Y = np.arange(1, 4)
    lambda_factor = 0.5
    exp_res = np.array([-0.03411225,  0.00320187,  0.04051599,  0.07783012,  0.11514424])
    if check_array(
            ex_name, linear_regression.closed_form,
            exp_res, X, Y, lambda_factor):
        return

    log(green("PASS"), ex_name, "")

def check_svm():
    ex_name = "One vs rest SVM"
    n, m, d = 5, 3, 7
    train_x = np.random.random((n, d))
    test_x = train_x[:m]
    train_y = np.zeros(n)
    train_y[-1] = 1
    exp_res = np.zeros(m)

    if check_array(
            ex_name, svm.one_vs_rest_svm,
            exp_res, train_x, train_y, test_x):
        return

    train_y = np.ones(n)
    train_y[-1] = 0
    exp_res = np.ones(m)

    if check_array(
            ex_name, svm.one_vs_rest_svm,
            exp_res, train_x, train_y, test_x):
        return

    log(green("PASS"), ex_name, "")


def check_compute_probabilities():
    ex_name = "Compute probabilities"
    n, d, k = 3, 5, 7
    X = np.arange(0, n * d).reshape(n, d)
    zeros = np.zeros((k, d))
    temp = 0.2
    exp_res = np.ones((k, n)) / k
    if check_array(
            ex_name, softmax.compute_probabilities,
            exp_res, X, zeros, temp):
        return

    theta = np.arange(0, k * d).reshape(k, d)
    softmax.compute_probabilities(X, theta, temp)
    exp_res = np.zeros((k, n))
    exp_res[-1] = 1
    if check_array(
            ex_name, softmax.compute_probabilities,
            exp_res, X, theta, temp):
        return

    log(green("PASS"), ex_name, "")

def check_compute_cost_function():
    ex_name = "Compute cost function"
    n, d, k = 3, 5, 7
    X = np.arange(0, n * d).reshape(n, d)
    Y = np.arange(0, n)
    zeros = np.zeros((k, d))
    temp = 0.2
    lambda_factor = 0.5
    exp_res = 1.9459101490553135
    if check_real(
            ex_name, softmax.compute_cost_function,
            exp_res, X, Y, zeros, lambda_factor, temp):
        return
    log(green("PASS"), ex_name, "")


def check_run_gradient_descent_iteration():
    ex_name = "Run gradient descent iteration"
    n, d, k = 3, 5, 7
    X = np.arange(0, n * d).reshape(n, d)
    Y = np.arange(0, n)
    zeros = np.zeros((k, d))
    alpha = 2
    temp = 0.2
    lambda_factor = 0.5
    exp_res = np.zeros((k, d))
    exp_res = np.array([
       [ -7.14285714,  -5.23809524,  -3.33333333,  -1.42857143, 0.47619048],
       [  9.52380952,  11.42857143,  13.33333333,  15.23809524, 17.14285714],
       [ 26.19047619,  28.0952381 ,  30.        ,  31.9047619 , 33.80952381],
       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286],
       [ -7.14285714,  -8.57142857, -10.        , -11.42857143, -12.85714286]
    ])

    if check_array(
            ex_name, softmax.run_gradient_descent_iteration,
            exp_res, X, Y, zeros, alpha, lambda_factor, temp):
        return
    softmax.run_gradient_descent_iteration(X, Y, zeros, alpha, lambda_factor, temp)
    log(green("PASS"), ex_name, "")

def check_update_y():
    ex_name = "Update y"
    train_y = np.arange(0, 10)
    test_y = np.arange(9, -1, -1)
    exp_res = (
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
            np.array([0, 2, 1, 0, 2, 1, 0, 2, 1, 0])
            )
    if check_tuple(
            ex_name, softmax.update_y,
            exp_res, train_y, test_y):
        return
    log(green("PASS"), ex_name, "")

###Correction note:  check_project_onto_PC fucntion have been modified since release.
def check_project_onto_PC():
    ex_name = "Project onto PC"
    X = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9],
        [4, 8, 12],
    ]);
    x_centered, feature_means = features.center_data(X)
    pcs = features.principal_components(x_centered)
    exp_res = np.array([
        [5.61248608, 0],
        [1.87082869, 0],
        [-1.87082869, 0],
        [-5.61248608, 0],
    ])
    n_components = 2
    if check_array(
            ex_name, features.project_onto_PC,
            exp_res, X, pcs, n_components, feature_means):
        return
    log(green("PASS"), ex_name, "")

def check_polynomial_kernel():
    ex_name = "Polynomial kernel"
    n, m, d = 3, 5, 7
    c = 1
    p = 2
    X = np.random.random((n, d))
    Y = np.random.random((m, d))
    try:
        K = kernel.polynomial_kernel(X, Y, c, d)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    for i in range(n):
        for j in range(m):
            exp = (X[i] @ Y[j] + c) ** d
            got = K[i][j]
            if (not equals(exp, got)):
                log(
                    red("FAIL"), ex_name,
                    ": values at ({}, {}) do not match. Expected {}, got {}"
                    .format(i, j, exp, got)
                )
    log(green("PASS"), ex_name, "")

def check_rbf_kernel():
    ex_name = "RBF kernel"
    n, m, d = 3, 5, 7
    gamma = 0.5
    X = np.random.random((n, d))
    Y = np.random.random((m, d))
    try:
        K = kernel.rbf_kernel(X, Y, gamma)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    for i in range(n):
        for j in range(m):
            exp = np.exp(-gamma * (np.linalg.norm(X[i] - Y[j]) ** 2))
            got = K[i][j]
            if (not equals(exp, got)):
                log(
                    red("FAIL"), ex_name,
                    ": values at ({}, {}) do not match. Expected {}, got {}"
                    .format(i, j, exp, got)
                )
    log(green("PASS"), ex_name, "")


def main():
    log(green("PASS"), "Import mnist project")
    try:
        check_get_mnist()
        check_closed_form()
        check_svm()
        check_compute_probabilities()
        check_compute_cost_function()
        check_run_gradient_descent_iteration()
        check_update_y()
        check_project_onto_PC()
        check_polynomial_kernel()
        check_rbf_kernel()
    except Exception:
        log_exit(traceback.format_exc())

if __name__ == "__main__":
    main()


X=np.array( [[ 1., 25., 74., 87., 93., 35., 57., 66.,  2., 36., 36.],
 [ 1., 74., 91., 70., 65., 14., 21., 42., 55., 53., 52.],
 [ 1., 91., 46., 98., 75., 35., 53.,  4., 62., 34., 39.],
 [ 1., 63., 59., 53., 23., 92., 43.,  3., 89., 86.,  6.],
 [ 1., 57., 89., 84., 66., 41., 59., 77., 79., 55., 38.],
 [ 1., 50., 75.,  7., 68., 91., 24., 12., 85., 27., 14.],
 [ 1., 49., 95., 37., 83., 93., 41., 64., 26., 59.,  5.],
 [ 1., 48., 59., 92.,  6., 32., 15., 48., 57., 98.,  1.],
 [ 1., 37., 46., 90., 20., 19., 27., 96., 86., 87., 50.],
 [ 1., 48., 51.,  4.,  3., 19., 73., 34., 36., 76., 15.]] )  
theta=np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
 [0. ,0. ,0. ,0. ,0., 0., 0., 0., 0., 0., 0.]])  
temp_parameter=1.0  
lambda_factor=0.0001
Y=np.array([1 ,1, 1, 1 ,1 ,1 ,1, 1, 1, 1])


X=np.array( [[ 1., 79., 82., 36., 13., 26., 61., 27., 29., 55.,  5.,],
 [ 1., 33., 87., 66., 22., 96., 22., 45., 27., 80., 13.],
 [ 1., 13., 88., 66., 48., 53., 42., 22., 18., 80., 71.],
 [ 1., 12., 10., 24., 28., 93., 15., 95., 92., 43., 25.],
 [ 1., 92., 66., 90., 10., 75., 32., 39.,  9., 39., 71.],
 [ 1., 61., 72., 91., 99., 85., 20., 96.,  7.,  6., 17.],
 [ 1.,  4., 25., 37., 65., 29., 21., 30., 21., 62., 94.],
 [ 1., 77., 59., 88., 61., 84., 95., 64., 29., 34., 27.],
 [ 1., 13., 75., 29., 39., 46., 77., 15., 20., 10., 40.],
 [ 1., 16., 32., 41., 89., 98., 19., 75.,  9.,  1., 31.]] )  
X=np.array( [[ 1., 14.,  3., 35.,  4., 86., 22., 66., 23., 49., 41.,],
 [ 1., 75., 86., 15., 52., 20., 30., 95., 58., 65., 80.,],
 [ 1., 79., 95., 33., 86., 68., 61., 86., 79., 43., 94.,],
 [ 1., 43., 41., 42., 41.,  1., 42., 20., 15., 46., 70.,],
 [ 1.,  7., 71., 32., 16., 97., 38., 46., 79., 77., 74.,],
 [ 1., 77., 22., 99., 93., 92., 33., 72., 60., 25., 57.,],
 [ 1., 89., 75., 68., 93., 32., 42., 96., 12., 61., 75.,],
 [ 1., 13., 15., 38., 89., 46., 88., 39., 14., 56., 34.,],
 [ 1., 98., 23., 78., 51., 48., 99., 50., 11., 79.,  4.,],
 [ 1., 60., 73., 49., 90., 66., 34., 34., 45.,  3., 47.]]) 
X=np.array([[ 1., 59., 44., 35., 12., 67.,  9., 91., 71., 71., 69.],
 [ 1., 20., 45., 24., 78., 74., 71., 84., 45.,  9., 26.],
 [ 1., 10., 48., 54., 42., 29., 93., 62., 85., 67., 16.],
 [ 1., 24., 32., 89., 97., 66., 37., 50., 29., 14., 14.],
 [ 1., 46., 55., 27., 38., 67., 42., 45., 39., 80., 29.],
 [ 1., 12., 62., 94., 87., 22.,  8., 58., 66., 69., 28.],
 [ 1., 98., 65., 33., 60., 79., 44., 11., 76., 44.,  4.],
 [ 1., 55., 40., 45., 83., 91., 56., 94., 16.,  5., 13.],
 [ 1., 89., 97., 17., 53., 42., 66., 17., 48., 59.,  5.],
 [ 1., 49., 33., 57., 85., 21., 59., 46., 53., 46.,  6.]])


