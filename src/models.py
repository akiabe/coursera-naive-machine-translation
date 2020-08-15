import numpy as np
import loss

def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003):
    """
    :param X: a matrix of dimension (m,n) where the columns are the english embeddings
    :param Y: a matrix of dimension (m,n) where the columns correspond to the french eembeddings
    :param train_steps: how many steps will gradient descent algorithm do
    :param learning_rate: how big steps will gradient descent algorithm do
    :return R: a matrix of dimension (n,n) the projection matrix that minimize that F norm ||X R -Y||^2
    """
    np.random.seed(129)

    # initialize R matrix
    R = np.random.rand(X.shape[1], X.shape[1])

    for i in range(train_steps):
        # print out loss at each 25 iterations
        if i % 25 == 0:
            print(f"loss at iteration {i} is: {loss.compute_loss(X, Y, R):.4f}")

        # compute gradient
        gradient = loss.compute_gradient(X, Y, R)

        # update R by subtraction the learning rate times gradient
        R -= learning_rate * gradient

    return R


