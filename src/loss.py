import numpy as np

def compute_loss(X, Y, R):
    """
    :param X: a matrix of dimension (m,n) where the columns are the english embeddings
    :param Y: a matrix of dimension (m,n) where the columns correspond to the french embeddings
    :param R: a matrix of dimension (n,n) transformation matrix from english to french vector space embeddings
    :return loss: a matrix of dimension (m,n) the value of the loss function for given X, Y and R
    """
    # number of rows in X
    m = X.shape[0]

    # XR - Y
    diff = np.dot(X, R) - Y

    # element-wise square of the difference
    diff_squared = np.square(diff)

    # sum of squared elements
    sum_diff_squared = np.sum(diff_squared)

    # loss i the sum_diff_squared divided by the number of examples
    loss = sum_diff_squared / m

    return loss

def compute_gradient(X, Y, R):
    """
    :param X: a matrix of dimension (m,n) where the columns are the english embeddings
    :param Y: a matrix of dimension (m,n) where the columns corresponding to the french embeddings
    :param R: a matrix of dimension (n,n) transformation matrix from english to french vector space embeddings
    :return gradient: a matrix of dimension (n,n) gradient of the loss function for given X, Y and R
    """
    # number of rows in X
    m = X.shape[0]

    # gradient is X^T(XR - Y) * 2/m
    gradient = np.dot(X.T, (np.dot(X,R)-Y)) * (2/m)

    return gradient