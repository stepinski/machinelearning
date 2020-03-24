import scipy.sparse as sparse
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import utils
import sys
sys.path.append("..")


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))


def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    # YOUR CODE HERE
    p1 = np.dot(theta, X.T)/temp_parameter
    c = p1.max(0)
    p1 = np.exp(np.dot(theta, X.T)/temp_parameter-c)
    norm = 1/p1.sum(0)
    return norm*p1


def chunking_dot(big_matrix, small_matrix, chunk_size=100):
    # Make a copy if the array is not already contiguous
    small_matrix = np.ascontiguousarray(small_matrix)
    R = np.empty((big_matrix.shape[0], small_matrix.shape[1]))
    for i in range(0, R.shape[0], chunk_size):
        end = i + chunk_size
        R[i:end] = np.dot(big_matrix[i:end], small_matrix)
    return R

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    import gc
    stheta=sparse.coo_matrix(theta)
    # YOUR CODE HERE
    ex = np.exp(np.dot(theta, X.T)/temp_parameter)
    ex2 = ex.sum(0)
    tmpx=ex/ex2
    p=tmpx > 0.0
    res=np.zeros_like(tmpx)
    res[p]=np.log(tmpx[p])

    # log = np.ma.log(ex/ex2)
    # prepare equality calculation
    tl = np.tile(np.arange(theta.shape[0]), (Y.shape[0], 1))
    ts2 = np.tile(Y, (theta.shape[0], 1))
    iseq = np.equal(ts2.T, tl)*1
  
    # final e
    # part1 =  -1/Y.shape[0]*np.dot(iseq, log)  if log.ndim==0 else 0
    del ex,ex2,theta,tl,ts2,tmpx,X
    gc.collect()
    iseqs=sparse.coo_matrix(iseq)
    ress=sparse.coo_matrix(res)

    tsta=chunking_dot(res,iseq)
    part1a = (-1/Y.shape[0])*tsta
    part2 = stheta.power(2).sum()*lambda_factor/2
    tmp1 = part1a  + part2

    res=tmp1.sum(1)[0]
    
    if np.isnan(res):
        res=part2

    return res

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    # YOUR CODE HERE
    p1 = -1/(Y.shape[0]*temp_parameter)
    ex = np.exp(np.dot(theta, X.T)/temp_parameter)
    ex2 = ex.sum(0)

    tmpx=ex/ex2
    p=tmpx > 0.0
    res=np.zeros_like(tmpx)
    res[p]=tmpx[p]

    tl = np.tile(np.arange(theta.shape[0]), (Y.shape[0], 1))
    ts2 = np.tile(Y, (theta.shape[0], 1))
    iseq = np.equal(ts2.T, tl)*1
    H=compute_probabilities(X,theta,temp_parameter)
    #(iseq-res.T)
    #H => H.T
    gradtheta = (p1*np.dot(X.T, (iseq-H.T))).T + lambda_factor*theta

    return theta - alpha*gradtheta


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    # YOUR CODE HERE
    return (train_y%3,test_y%3)


def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    # YOUR CODE HERE
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)%3
    return 1 - np.mean(assigned_labels == Y)


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(
            X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(
            X, Y, theta, alpha, lambda_factor, temp_parameter)
        print(i)
    return theta, cost_function_progression


def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis=0)


def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()


def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
